import os
import json
import shutil

import torch
from peft import PeftModel
from transformers import PreTrainedModel
from finetune import get_loaders

BASE_MODEL = 'decapoda-research/llama-30b-hf'
LORA_WEIGHTS = 'llama-30b-hf.h2oaiopenassistant_oasst1_h2ogpt.8.0_epochs.31eef248d53c9f39e51c60b8b030c1e3cafc34b0.llama30b_7'
OUTPUT_NAME = "h2ogpt-research-oasst1-512-30b"
llama_type = "llama" in BASE_MODEL
as_pytorch = False  # False -> HF

model_loader, tokenizer_loader = get_loaders(llama_type=llama_type, model_name=BASE_MODEL, reward_type=False)

tokenizer = tokenizer_loader.from_pretrained(
    BASE_MODEL,
    local_files_only=True,
    resume_download=True,
)
tokenizer.save_pretrained(OUTPUT_NAME)

base_model = model_loader.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
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
    lora_model = lora_model.merge_and_unload()
    # for layer in lora_model.base_model.model.model.layers:
    #     layer.self_attn.q_proj.merge_weights = True
    #     layer.self_attn.k_proj.merge_weights = True
    #     layer.self_attn.v_proj.merge_weights = True
    #     layer.self_attn.o_proj.merge_weights = True
else:
    if any([x in BASE_MODEL.lower() for x in ["pythia", "h2ogpt", "gpt-neox"]]):
        for layer in lora_model.base_model.gpt_neox.base_model.layers:
            layer.attention.query_key_value.merge_weights = True
    else:
        # lora_model.merge_and_unload()  # might work sometimes
        for layer in lora_model.base_model.transformer.base_model.h:
            layer.attn.q_proj.merge_weights = True
            layer.attn.v_proj.merge_weights = True

lora_model.train(False)

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()

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
    for k, v in lora_model_sd.items():
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
    deloreanized_sd = {
        k.replace("base_model.model.", ""): v
        for k, v in lora_model_sd.items()
        if "lora" not in k
    }
    base_model.config.custom_pipeline = {
        "text-generation": {
          "impl": "h2oai_pipeline.H2OTextGenerationPipeline",
          "pt": "AutoModelForCausalLM"
        }
    }
    PreTrainedModel.save_pretrained(
        base_model,
        OUTPUT_NAME,
        state_dict=deloreanized_sd,
        max_shard_size="5GB",
    )
    shutil.copyfile("h2oai_pipeline.py", os.path.join(OUTPUT_NAME, "h2oai_pipeline.py"))
