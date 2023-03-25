from transformers import AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXConfig
import torch

tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")
config = GPTNeoXConfig.from_pretrained("EleutherAI/gpt-neox-20b")
config.is_decoder = True
model = GPTNeoXForCausalLM.from_pretrained("EleutherAI/gpt-neox-20b", config=config)

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

prediction_logits = outputs.logits
print(prediction_logits)
