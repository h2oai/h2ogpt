---
license: apache-2.0
language:
- en
library_name: transformers
inference: false
---
# h2oGPT Model Card
## Summary

H2O.ai's `<<MODEL_NAME>>` is a <<MODEL_SIZE>> billion parameter instruction-following large language model licensed for commercial use.

- Base model: <<BASE_MODEL>>
- Fine-tuning dataset: <<DATASET>>
- Data-prep and fine-tuning code: [H2O.ai Github](https://github.com/h2oai/h2ogpt)
- Training logs: [zip](<<TRAINING_LOGS>>)

## Usage

To use the model with the `transformers` library on a machine with GPUs, first make sure you have the `transformers` and `accelerate` libraries installed.

```bash
pip install transformers==4.28.1
```

```python
import torch
from transformers import pipeline

generate_text = pipeline(model="h2oai/<<MODEL_NAME>>", torch_dtype=torch.bfloat16, trust_remote_code=True, device_map="auto")

res = generate_text("Why is drinking water so healthy?", max_new_tokens=100)
print(res[0]["generated_text"])
```

Alternatively, if you prefer to not use `trust_remote_code=True` you can download [instruct_pipeline.py](https://huggingface.co/h2oai/<<MODEL_NAME>>/blob/main/h2oai_pipeline.py),
store it alongside your notebook, and construct the pipeline yourself from the loaded model and tokenizer:

```python
import torch
from h2oai_pipeline import H2OTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("h2oai/<<MODEL_NAME>>", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("h2oai/<<MODEL_NAME>>", torch_dtype=torch.bfloat16, device_map="auto")
generate_text = H2OTextGenerationPipeline(model=model, tokenizer=tokenizer)

res = generate_text("Why is drinking water so healthy?", max_new_tokens=100)
print(res[0]["generated_text"])
```

## Model Architecture

```
<<MODEL_ARCH>>
```

## Model Configuration

```json
<<MODEL_CONFIG>>
```
