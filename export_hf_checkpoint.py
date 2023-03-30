import os
import json
import torch
from peft import PeftModel

import transformers
from transformers import PreTrainedModel

from finetune import get_loaders

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"

BASE_MODEL = 'EleutherAI/gpt-j-6B'
BASE_MODEL = 'decapoda-research/llama-13b-hf'
LORA_WEIGHTS = "lora_6B_daidocs_alpaca_daifaq"
LORA_WEIGHTS = "llama-13b-hf.config.json.20_epochs.5e2efef6a3d2af21f217dd86f9d89c262877dbe2.20230329-022308"
OUTPUT_NAME = (BASE_MODEL + LORA_WEIGHTS).split("/")[-1]
llama_type = "llama" in BASE_MODEL
as_pytorch = True  # False -> HF

model_loader, _ = get_loaders(llama_type=llama_type)

base_model = model_loader.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

if llama_type:
    layers = base_model.model.layers
    first_weight = layers[0].self_attn.q_proj.weight
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

# merge weights
if llama_type:
    for layer in lora_model.base_model.model.model.layers:
        layer.self_attn.q_proj.merge_weights = True
        layer.self_attn.v_proj.merge_weights = True
else:
    for layer in lora_model.base_model.transformer.base_model.h:
        layer.attn.q_proj.merge_weights = True
        layer.attn.v_proj.merge_weights = True

lora_model.train(False)

# did we do anything?
assert not torch.allclose(first_weight_old, first_weight)

lora_model_sd = lora_model.state_dict()

if as_pytorch:
    # https://huggingface.co/decapoda-research/llama-13b-hf#quantitative-analysis
    params = {
        "dim": 5120,
        "multiple_of": 256,
        "n_heads": 40,
        "n_layers": 40,
        "norm_eps": 1e-06,
        "vocab_size": -1,
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

    PreTrainedModel.save_pretrained(
        base_model,
        OUTPUT_NAME,
        state_dict=deloreanized_sd,
        max_shard_size="5GB",
    )
