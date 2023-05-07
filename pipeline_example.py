import torch
from h2oai_pipeline import H2OTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
import textwrap as tr

tokenizer = AutoTokenizer.from_pretrained("h2oai/h2ogpt-oasst1-512-12b", padding_side="left")

# 8-bit will use much less memory, so set to True if
# e.g. with 512-12b load_in_8bit=True required for 24GB GPU
# if have 48GB GPU can do load_in_8bit=False for more accurate results
load_in_8bit = True
# device_map = 'auto' might work in some cases to spread model across GPU-CPU, but it's not supported
device_map = {"": 0}
model = AutoModelForCausalLM.from_pretrained("h2oai/h2ogpt-oasst1-512-12b", torch_dtype=torch.float16,
                                             device_map=device_map, load_in_8bit=load_in_8bit)

generate_text = H2OTextGenerationPipeline(model=model, tokenizer=tokenizer)

# generate
outputs = generate_text("Why is drinking water so healthy?", return_full_text=True, max_new_tokens=1000)

for output in outputs:
    print(tr.fill(output['generated_text'], width=40))

# Generated text should be similar to below.
"""
Drinking water is a healthy habit
because it helps to keep your body
hydrated and flush out toxins. It also
helps to keep your digestive system
running smoothly and can even help to
prevent certain diseases.
"""
