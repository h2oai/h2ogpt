import torch
from peft import PeftModel

import transformers
from transformers import PreTrainedModel

from finetune import get_loaders

assert (
    "LlamaTokenizer" in transformers._import_structure["models.llama"]
), "LLaMA is now in HuggingFace's main branch.\nPlease reinstall it: pip uninstall transformers && pip install git+https://github.com/huggingface/transformers.git"

BASE_MODEL = 'EleutherAI/gpt-j-6B'
OUTPUT_NAME = "gpt-j-6B_daidocs_alpaca"
llama_type = False

model_loader, _ = get_loaders(llama_type=llama_type)

base_model = model_loader.from_pretrained(
    BASE_MODEL,
    load_in_8bit=False,
    torch_dtype=torch.float16,
    device_map={"": "cpu"},
)

if llama_type:
    layers = base_model.model.model.layers
    first_weight = layers[0].self_attn.q_proj.weight
else:
    layers = base_model.transformer.base_model.h
    first_weight = layers[0].attn.q_proj.weight
first_weight_old = first_weight.clone()

lora_model = PeftModel.from_pretrained(
    base_model,
    "lora_6B_daidocs_alpaca",
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
