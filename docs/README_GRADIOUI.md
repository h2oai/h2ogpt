### Gradio UI

`generate.py` by default runs a gradio server with a [UI (click for help with UI)](FAQ.md#explain-things-in-ui).  Key benefits of the UI include:
* Save, export, import chat histories and undo or regenerate last query-response pair
* Upload and control documents of various kinds for document Q/A
* Choose which specific collection to query, or just chat with LLM
* Choose specific documents out of collection for asking questions
* Side-by-side 2-model comparison view
* RLHF response score evaluation for every query-response

See how we compare to other tools like PrivateGPT, see our comparisons at [h2oGPT LangChain Integration FAQ](README_LangChain.md#what-is-h2ogpts-langchain-integration-like).

We disable background uploads by disabling telemetry for Hugging Face, gradio, and chroma, and one can additionally avoid downloads (of fonts) by running `generate.py` with `--gradio_offline_level=2`.  See [Offline Documentation](FAQ.md#offline-mode) for details.

