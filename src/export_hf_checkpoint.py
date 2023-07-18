import os
import json
import shutil
import subprocess

import torch
from accelerate import infer_auto_device_map, dispatch_model
from accelerate.utils import get_balanced_memory
from peft import PeftModel
from transformers import PreTrainedModel


def do_export():
    BASE_MODEL = 'tiiuae/falcon-40b'
    LORA_WEIGHTS = 'falcon-40b.h2oaiopenassistant_oasst1_h2ogpt.1_epochs.894d8450d35c180cd03222a45658d04c15b78d4b.9'
    OUTPUT_NAME = "h2ogpt-oasst1-2048-falcon-40b"

    base_model = os.getenv('BASE_MODEL')
    output = os.getenv('MODEL')
    # for testing
    if base_model and output:
        BASE_MODEL = base_model
        LORA_WEIGHTS = output + ".lora"
        OUTPUT_NAME = output

    llama_type = "llama" in BASE_MODEL
    as_pytorch = False  # False -> HF

    from loaders import get_loaders
    model_loader, tokenizer_loader = get_loaders(model_name=BASE_MODEL, reward_type=False, llama_type=llama_type)

    tokenizer = tokenizer_loader.from_pretrained(
        BASE_MODEL,
        local_files_only=False,
        resume_download=True,
    )
    tokenizer.save_pretrained(OUTPUT_NAME)

    base_model = model_loader(
        BASE_MODEL,
        load_in_8bit=False,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map={"": "cpu"},
    )

    print(base_model)
    if llama_type:
        layers = base_model.model.layers
        first_weight = layers[0].self_attn.q_proj.weight
    else:
        if any([x in BASE_MODEL.lower() for x in ["pythia", "h2ogpt", "gpt-neox"]]):
            layers = base_model.gpt_neox.base_model.layers
            first_weight = layers[0].attention.query_key_value.weight
        elif any([x in BASE_MODEL.lower() for x in ["falcon"]]):
            first_weight = base_model.transformer.h._modules['0'].self_attention.query_key_value.weight
        else:
            layers = base_model.transformer.base_model.h
            first_weight = layers[0].attn.q_proj.weight
    first_weight_old = first_weight.clone()

    lora_model = PeftModel.from_pretrained(
        base_model,
        LORA_WEIGHTS,
        device_map={"": "cpu"},
        torch_dtype=torch.float16,
    )

    assert torch.allclose(first_weight_old, first_weight)

    # merge weights TODO: include all lora_target_modules, not just default ones
    if llama_type:
        merged_model = lora_model.merge_and_unload()
        # for layer in lora_model.base_model.model.model.layers:
        #     layer.self_attn.q_proj.merge_weights = True
        #     layer.self_attn.k_proj.merge_weights = True
        #     layer.self_attn.v_proj.merge_weights = True
        #     layer.self_attn.o_proj.merge_weights = True
    else:
        if any([x in BASE_MODEL.lower() for x in ["pythia", "gpt-neox"]]):
            for layer in lora_model.base_model.gpt_neox.base_model.layers:
                layer.attention.query_key_value.merge_weights = True
            merged_model = lora_model
        else:
            merged_model = lora_model.merge_and_unload()
            # for layer in lora_model.base_model.transformer.base_model.h:
            #     layer.attn.q_proj.merge_weights = True
            #     layer.attn.v_proj.merge_weights = True

    # max_memory = get_balanced_memory(merged_model)
    # device_map = infer_auto_device_map(merged_model, max_memory=max_memory)
    # merged_model = dispatch_model(
    #     merged_model,
    #     device_map=device_map,
    # )
    merged_model.eval()
    print(merged_model)

    # did we do anything?
    assert not torch.allclose(first_weight_old, first_weight)

    merged_model_sd = merged_model.state_dict()

    if as_pytorch:
        # FIXME - might not be generic enough still
        params = {
            "dim": base_model.config.hidden_size,
            "n_heads": base_model.config.num_attention_heads,
            "n_layers": base_model.config.num_hidden_layers,
            "norm_eps": base_model.config.layer_norm_eps,
            "vocab_size": base_model.config.vocab_size,
        }
        n_layers = params["n_layers"]
        n_heads = params["n_heads"]
        dim = params["dim"]
        dims_per_head = dim // n_heads
        base = 10000.0
        inv_freq = 1.0 / (base ** (torch.arange(0, dims_per_head, 2).float() / dims_per_head))

        def permute(w):
            return (
                w.view(n_heads, dim // n_heads // 2, 2, dim).transpose(1, 2).reshape(dim, dim)
            )


        def unpermute(w):
            return (
                w.view(n_heads, 2, dim // n_heads // 2, dim).transpose(1, 2).reshape(dim, dim)
            )


        def translate_state_dict_key(k):
            if "gpt-neoxt" in BASE_MODEL.lower():
                k = k.replace("gpt_neox.model.", "")
            else:
                k = k.replace("base_model.model.", "")
            if k == "model.embed_tokens.weight":
                return "tok_embeddings.weight"
            elif k == "model.norm.weight":
                return "norm.weight"
            elif k == "lm_head.weight":
                return "output.weight"
            elif k.startswith("model.layers."):
                layer = k.split(".")[2]
                if k.endswith(".self_attn.q_proj.weight"):
                    return f"layers.{layer}.attention.wq.weight"
                elif k.endswith(".self_attn.k_proj.weight"):
                    return f"layers.{layer}.attention.wk.weight"
                elif k.endswith(".self_attn.v_proj.weight"):
                    return f"layers.{layer}.attention.wv.weight"
                elif k.endswith(".self_attn.o_proj.weight"):
                    return f"layers.{layer}.attention.wo.weight"
                elif k.endswith(".mlp.gate_proj.weight"):
                    return f"layers.{layer}.feed_forward.w1.weight"
                elif k.endswith(".mlp.down_proj.weight"):
                    return f"layers.{layer}.feed_forward.w2.weight"
                elif k.endswith(".mlp.up_proj.weight"):
                    return f"layers.{layer}.feed_forward.w3.weight"
                elif k.endswith(".input_layernorm.weight"):
                    return f"layers.{layer}.attention_norm.weight"
                elif k.endswith(".post_attention_layernorm.weight"):
                    return f"layers.{layer}.ffn_norm.weight"
                elif k.endswith("rotary_emb.inv_freq") or "lora" in k:
                    return None
                else:
                    print(layer, k)
                    raise NotImplementedError
            else:
                print(k)
                raise NotImplementedError


        new_state_dict = {}
        for k, v in merged_model_sd.items():
            new_k = translate_state_dict_key(k)
            if new_k is not None:
                if "wq" in new_k or "wk" in new_k:
                    new_state_dict[new_k] = unpermute(v)
                else:
                    new_state_dict[new_k] = v

        os.makedirs("./ckpt", exist_ok=True)

        torch.save(new_state_dict, "./ckpt/consolidated.00.pth")

        with open("./ckpt/params.json", "w") as f:
            json.dump(params, f)
    else:
        # deloreanized_sd = {
        #     k.replace("base_model.model.", ""): v
        #     for k, v in merged_model_sd.items()
        #     if "lora" not in k
        # }
        merged_model.config.custom_pipelines = {
            "text-generation": {
              "impl": "h2oai_pipeline.H2OTextGenerationPipeline",
              "pt": "AutoModelForCausalLM"
            }
        }
        PreTrainedModel.save_pretrained(
            merged_model,
            OUTPUT_NAME,
            # state_dict=deloreanized_sd,
            # max_shard_size="5GB",
        )

    do_copy(OUTPUT_NAME)
    test_copy()


def do_copy(OUTPUT_NAME):
    dest_file = os.path.join(OUTPUT_NAME, "h2oai_pipeline.py")
    shutil.copyfile("src/h2oai_pipeline.py", dest_file)
    os.system("""sed -i 's/from enums.*//g' %s""" % dest_file)
    os.system("""sed -i 's/from stopping.*//g' %s""" % dest_file)
    os.system("""sed -i 's/from prompter.*//g' %s""" % dest_file)
    os.system("""cat %s|grep -v "from enums import PromptType" >> %s""" % ('src/enums.py', dest_file))
    os.system("""cat %s|grep -v "from enums import PromptType" >> %s""" % ('src/prompter.py', dest_file))
    os.system("""cat %s|grep -v "from enums import PromptType" >> %s""" % ('src/stopping.py', dest_file))


TEST_OUTPUT_NAME = "test_output"


def test_copy():
    if os.path.isdir(TEST_OUTPUT_NAME):
        shutil.rmtree(TEST_OUTPUT_NAME)
    os.makedirs(TEST_OUTPUT_NAME, exist_ok=False)
    do_copy(TEST_OUTPUT_NAME)
    shutil.copy('src/export_hf_checkpoint.py', TEST_OUTPUT_NAME)
    os.environ['DO_COPY_TEST'] = '1'
    os.chdir(TEST_OUTPUT_NAME)
    output = subprocess.check_output(['python', 'export_hf_checkpoint.py'])
    print(output)


def inner_test_copy():
    """
    pytest -s -v export_hf_checkpoint.py::test_copy
    :return:
    """
    # test imports
    # below supposed to look bad in pycharm, don't fix!
    from h2oai_pipeline import get_stopping, get_prompt, H2OTextGenerationPipeline
    assert get_stopping
    assert get_prompt
    assert H2OTextGenerationPipeline


if __name__ == '__main__':
    if os.getenv('DO_COPY_TEST'):
        inner_test_copy()
    else:
        do_export()
    # uncomment for raw isolated test, but test is done every time for each export now
    # test_copy()
