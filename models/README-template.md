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
res = generate_text("Why is drinking water so healthy?")
print(res[0]["generated_text"])
```

Alternatively, if you prefer to not use `trust_remote_code=True` you can download [instruct_pipeline.py](https://huggingface.co/h2oai/<<MODEL_NAME>>/blob/main/h2oai_pipeline.py),
store it alongside your notebook, and construct the pipeline yourself from the loaded model and tokenizer:

```
import torch
from h2oai_pipeline import H2OTextGenerationPipeline
from transformers import AutoModelForCausalLM, AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("h2oai/<<MODEL_NAME>>", padding_side="left")
model = AutoModelForCausalLM.from_pretrained("h2oai/<<MODEL_NAME>>", device_map="auto", torch_dtype=torch.bfloat16)

generate_text = H2OTextGenerationPipeline(model=model, tokenizer=tokenizer)
```

### LangChain Usage

To use the pipeline with LangChain, you must set `return_full_text=True`, as LangChain expects the full text to be returned 
and the default for the pipeline is to only return the new text.

```
import torch
from transformers import pipeline

generate_text = pipeline(model="h2oai/<<MODEL_NAME>>", torch_dtype=torch.bfloat16,
                         trust_remote_code=True, device_map="auto", return_full_text=True)
```

You can create a prompt that either has only an instruction or has an instruction with context:

```
from langchain import PromptTemplate, LLMChain
from langchain.llms import HuggingFacePipeline

# template for an instrution with no input
prompt = PromptTemplate(
    input_variables=["instruction"],
    template="{instruction}")

# template for an instruction with input
prompt_with_context = PromptTemplate(
    input_variables=["instruction", "context"],
    template="{instruction}\n\nInput:\n{context}")

hf_pipeline = HuggingFacePipeline(pipeline=generate_text)

llm_chain = LLMChain(llm=hf_pipeline, prompt=prompt)
llm_context_chain = LLMChain(llm=hf_pipeline, prompt=prompt_with_context)
```

Example predicting using a simple instruction:

```
print(llm_chain.predict(instruction="Why is drinking water so healthy?").lstrip())
```

Example predicting using an instruction with context:

```
context = """Model A: AUC=0.8
Model from Driverless AI: AUC=0.95
Model C: AUC=0.6
Model D: AUC=0.7
"""

print(llm_context_chain.predict(instruction="Which model performs best?", context=context).lstrip())
```

## Model Architecture

```
<<MODEL_ARCH>>
```

## Model Configuration

```
<<MODEL_CONFIG>>
```
