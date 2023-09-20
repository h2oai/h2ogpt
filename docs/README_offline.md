# Offline Mode:

## Easy Way:

Run h2oGPT as would in offline mode, ensuring to use LLM and upload docs using same parsers as would want in offline mode.  The `~/.cache` folder will be filled, and one can use that in offline mode.

## Moderately Easy Way:

If you can run on same (or better) system that will be like that in offline mode, you can run the below and collect all needed items in the `~/.cache/` and `~/nltk_data` folders, specifically:
* `~/.cache/selenium/`
* `~/.cache/huggingface/`
* `~/.cache/torch/`
* `~/.cache/clip/`
* `~/.cache/doctr/`
* `~/.cache/chroma/`
* `~/.cache/ms-playwright/`
* `~/.cache/selenium/`
* `~/nltk_data/`
```
python generate.py --score_model=None --gradio_size=small --model_lock="[{'base_model': 'h2oai/h2ogpt-4096-llama2-7b-chat'}]" --save_dir=save_fastup_chat --prepare_offline_level=2
# below are already in docker
python -m nltk.downloader all
playwright install --with-deps
```
Some of these locations can be controlled, but others not, so best to make local version of ~/.cache (e.g. move original out of way), run the above, archive it for offline system, restore old ~/.cache, then use offline.  If same system, then those steps aren't required, one can just go fully offline.

If you are only concerned with what h2oGPT needs, not any inference servers, you can run with `--prepare_offline_level=1` that will not obtain models associated with inference severs (e.g. vLLM or TGI).

If you have a GGML file, you should download it ahead of time and place it in some path you provide to `--llamacpp_dict` for its `model_path_llama` dict entry.

## Hard Way:

Identify all models needed and download each.  The below list is not exhaustive because the models added changes frequently and each uses different approach for downloading.

Note, when running `generate.py` and asking your first question, it will download the model(s), which for the 6.9B model takes about 15 minutes per 3 pytorch bin files if have 10MB/s download.

If all data has been put into `~/.cache` by HF transformers and GGML files downloaded already and one points to them (e.g. with `--model_path_llama=llama-2-7b-chat.ggmlv3.q8_0.bin`), then these following steps (those related to downloading HF models) are not required.

* Download model and tokenizer of choice
    
    ```python
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_name = 'h2oai/h2ogpt-oasst1-512-12b'
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.save_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(model_name)
    ```
    If using GGML files, those should be downloaded separately manually, and point to file path, e.g. `--base_model=llama --model_path_llama=llama-2-7b-chat.ggmlv3.q8_0.bin`.

* Download reward model, unless pass `--score_model='None'` to `generate.py`
    ```python
    # and reward model
    reward_model = 'OpenAssistant/reward-model-deberta-v3-large-v2'
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    model = AutoModelForSequenceClassification.from_pretrained(reward_model)
    model.save_pretrained(reward_model)
    tokenizer = AutoTokenizer.from_pretrained(reward_model)
    tokenizer.save_pretrained(reward_model)
    ```
    
* For LangChain support, download embedding model:
    ```python
    hf_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = 'cpu'
    from langchain.embeddings import HuggingFaceEmbeddings
    embedding = HuggingFaceEmbeddings(model_name=hf_embedding_model, model_kwargs=model_kwargs)
    ```
    
* For HF inference server and OpenAI, this downloads the tokenizers used for Hugging Face text generation inference server and gpt-3.5-turbo:
    ```python
    import tiktoken
    encoding = tiktoken.get_encoding("cl100k_base")
    encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
    ```

* Get gpt-2 tokenizer for summarization token counting
    ```python
    from transformers import AutoTokenizer
    model_name = 'gpt2'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.save_pretrained(model_name)
    ```

## Run h2oGPT in offline mode

```bash
HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python generate.py --base_model='h2oai/h2ogpt-oasst1-512-12b' --gradio_offline_level=2 --share=False
```
For more info for transformers, see [Offline Mode](https://huggingface.co/docs/transformers/installation#offline-mode).

Some code is always disabled that involves uploads out of user control: Huggingface telemetry, gradio telemetry, chromadb posthog.

The additional option `--gradio_offline_level=2` changes fonts to avoid download of google fonts. This option disables google fonts for downloading, which is less intrusive than uploading, but still required in air-gapped case.  The fonts don't look as nice as google fonts, but ensure full offline behavior.

If the front-end can still access internet, but just backend should not, then one can use `--gradio_offline_level=1` for slightly better-looking fonts.

Note that gradio attempts to download [iframeResizer.contentWindow.min.js](https://cdnjs.cloudflare.com/ajax/libs/iframe-resizer/4.3.1/iframeResizer.contentWindow.min.js),
but nothing prevents gradio from working without this.  So a simple firewall block is sufficient.  For more details, see: https://github.com/AUTOMATIC1111/stable-diffusion-webui/pull/10324.

### Disable access or port

To ensure nobody can access your gradio server, disable the port via firewall.  If that is a hassle, then one can enable authentication by adding to CLI when running `python generate.py`:
```
--auth=[('jon','password')]
```
with no spaces.  Run `python generate.py --help` for more details.

### To fully disable Chroma telemetry, which documented options still do not disable, run:

```bash
sp=`python -c 'import site; print(site.getsitepackages()[0])'`
sed -i 's/posthog\.capture/return\n            posthog.capture/' $sp/chromadb/telemetry/posthog.py
```
or the equivalent for windows/mac using.  Or edit the file manually to just return in the `capture` function.

To avoid h2oGPT monitoring which elements are clicked in UI, set the ENV `H2OGPT_ENABLE_HEAP_ANALYTICS=False` pass `--enable-heap-analytics=False` to `generate.py`.  Note that no data or user inputs are included, only raw svelte UI element IDs and nothing from the user inputs or data.
