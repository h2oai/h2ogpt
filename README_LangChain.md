## h2oGPT integration with LangChain and Chroma/FAISS for Vector DB

Our goal is to make it easy to have private offline document question-answer using LLMs.

### Try h2oGPT now, with LangChain on example databases 

Live hosted instances:
- [![img-small.png](img-small.png) Latest LangChain-enabled h2oGPT (temporary link) 12B](https://179cd3463ebdd8e788.gradio.live/)
- [![img-small.png](img-small.png) LangChain-enabled h2oGPT (temporary link 1) 12B](https://9b1c74d9de90a71538.gradio.live/)
- [![img-small.png](img-small.png) LangChain-enabled h2oGPT (temporary link 2) 12B](https://30999ff1f3ff7577e0.gradio.live/)
- [![img-small.png](img-small.png) LangChain-enabled h2oGPT (temporary link 3) 12B](https://2855c4e61c677186aa.gradio.live/)

For questions, discussing, or just hanging out, come and join our <a href="https://discord.gg/WKhYMWcVbq"><b>Discord</b></a>!

## Supported Datatypes

Open-source data types are supported, .msg is not supported due to GPL-3 requirement.  Other meta types support other types inside them.  Special support for some behaviors is provided by the UI itself.

### Supported Native Datatypes

   - `.pdf`: Portable Document Format (PDF),
   - `.txt`: Text file (UTF-8),
   - `.csv`: CSV,
   - `.toml`: Toml,
   - `.py`: Python,
   - `.rst`: reStructuredText,
   - `.rtf`: Rich Text Format,
   - `.md`: Markdown,
   - `.html`: HTML File,
   - `.docx`: Word Document (optional),
   - `.doc`: Word Document (optional),
   - `.enex`: EverNote,
   - `.eml`: Email,
   - `.epub`: EPub,
   - `.odt`: Open Document Text,
   - `.pptx` : PowerPoint Document,
   - `.ppt` : PowerPoint Document,
   - `.png` : PNG Image (optional),
   - `.jpg` : JPEG Image (optional),
   - `.jpeg` : JPEG Image (optional).

To support image captioning, on Ubuntu run:
```bash
sudo apt-get install libmagic-dev poppler-utils tesseract-ocr
```
and ensure in `requirements_optional_langchain.txt` that `unstructured[local-inference]` and `pdf2image` are installed.  Otherwise, for no image support just `unstructured` is sufficient.

OCR is disabled by default, but can be enabled if making database via `make_db.py`, and then on Ubuntu run:
```bash
sudo apt-get install tesseract-ocr
```
and ensure you `pip install pytesseract`.

To support Microsoft Office docx and doc, on Ubuntu run:
```bash
sudo apt-get install libreoffice
```

### Supported Meta Datatypes

   - `.zip` : Zip File containing any native datatype,
   - `.urls` : Text file containing new-line separated URLs (to be consumed via download).

### Supported Datatypes in UI

   - `Files` : All Native and Meta DataTypes as file(s),
   - `URL` : Any URL,
   - `Text` : Paste Text into UI.

## Database creation

To use some example databases (will overwrite UserData make above unless change options) and run generate after, do:
```bash
python make_db.py --download_some=True
python generate.py --base_model=h2oai/h2ogpt-oasst1-512-12b --load_8bit=True --langchain_mode=UserData --visible_langchain_modes="['UserData', 'wiki', 'MyData', 'github h2oGPT', 'DriverlessAI docs']"
```
which downloads example databases.  This obtains files from some [pre-generated databases](https://huggingface.co/datasets/h2oai/db_dirs).  A large Wikipedia database is also available.

To build the database first outside chatbot, then run generate after, do:
```bash
python make_db.py
python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6.9b --langchain_mode=UserData
```

To add data to the existing database, then run generate after, do:
```bash
python make_db.py --add_if_exists=True
python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6.9b --langchain_mode=UserData
```

## FAQ

#### Why does the source link not work?

For links to direct to the document and download to your local machine, the original source documents must still be present on the host system where the database was created, e.g. `user_path` for `UserData` by default.  If the database alone is copied somewhere else, that host won't have access to the documents.  URL links like Wikipedia will still work normally on any host.


#### What is h2oGPT's LangChain integration like?

* [PrivateGPT](https://github.com/imartinez/privateGPT) but fully commercially viable by not using [GPT4All](https://github.com/nomic-ai/gpt4all) based upon [LLaMa](https://github.com/facebookresearch/llama) and used data from GPT3.5 (violation of ToS).

* [Vault-AI](https://github.com/pashpashpash/vault-ai) but fully private and open-source by not using OpenAI or [pinecone](https://www.pinecone.io/).

* [DB-GPT](https://github.com/csunny/DB-GPT) but fully commercially viable by not using [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) (LLaMa based with GPT3.5 training data).