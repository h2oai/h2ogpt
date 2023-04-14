from transformers import AutoModelForCausalLM, AutoTokenizer
from h2oai_pipeline import H2OTextGenerationPipeline

tokenizer = AutoTokenizer.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("togethercomputer/GPT-NeoXT-Chat-Base-20B", device_map="auto")

generate_text = H2OTextGenerationPipeline(model=model, tokenizer=tokenizer)

input = "Explain to me the difference between nuclear fission and fusion."
output = generate_text(input, max_new_tokens=100)
print(output)
