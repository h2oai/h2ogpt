# Offline Mode:

Note, when running `generate.py` and asking your first question, it will download the model(s), which for the 6.9B model takes about 15 minutes per 3 pytorch bin files if have 10MB/s download.

If all data has been put into `~/.cache` by HF transformers, then these following steps (those related to downloading HF models) are not required.

1) Download model and tokenizer of choice

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
model_name = 'h2oai/h2ogpt-oasst1-512-12b'
model = AutoModelForCausalLM.from_pretrained(model_name)
model.save_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.save_pretrained(model_name)
```

2) Download reward model, unless pass `--score_model='None'` to `generate.py`
```python
# and reward model
reward_model = 'OpenAssistant/reward-model-deberta-v3-large-v2'
from transformers import AutoModelForSequenceClassification, AutoTokenizer
model = AutoModelForSequenceClassification.from_pretrained(reward_model)
model.save_pretrained(reward_model)
tokenizer = AutoTokenizer.from_pretrained(reward_model)
tokenizer.save_pretrained(reward_model)
```

3) For LangChain support, download embedding model:
```python
hf_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
model_kwargs = 'cpu'
from langchain.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name=hf_embedding_model, model_kwargs=model_kwargs)
```

4) For HF inference server and OpenAI, this downloads the tokenizers used for Hugging Face text generation inference server and gpt-3.5-turbo:
```python
import tiktoken
encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
```

5) Run generate with transformers in [Offline Mode](https://huggingface.co/docs/transformers/installation#offline-mode)

```bash
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python generate.py --base_model='h2oai/h2ogpt-oasst1-512-12b' --gradio_offline_level=2 --share=False
```

Some code is always disabled that involves uploads out of user control: Huggingface telemetry, gradio telemetry, chromadb posthog.

The additional option `--gradio_offline_level=2` changes fonts to avoid download of google fonts. This option disables google fonts for downloading, which is less intrusive than uploading, but still required in air-gapped case.  The fonts don't look as nice as google fonts, but ensure full offline behavior.

If the front-end can still access internet, but just backend should not, then one can use `--gradio_offline_level=1` for slightly better-looking fonts.

Note that gradio attempts to download [iframeResizer.contentWindow.min.js](https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.1/iframeResizer.contentWindow.min.js),
but nothing prevents gradio from working without this.  So a simple firewall block is sufficient.  For more details, see: https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/10324.

6. Disable access or port

To ensure nobody can access your gradio server, disable the port via firewall.  If that is a hassle, then one can enable authentication by adding to CLI when running `python generate.py`:
```
--auth=[('jon','password')]
```
with no spaces.  Run `python generate.py --help` for more details.

7. To avoid unauthorized telemetry, which document options still do not disable, run:
```bash
sp=`python -c 'import site; print(site.getsitepackages()[0])'`
sed -i 's/posthog\.capture/return\n            posthog.capture/' $sp/chromadb/telemetry/posthog.py
```
or the equivalent for windows/mac using.  Or edit the file manually to just return in the `capture` function.
