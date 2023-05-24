## h2oGPT integration with LangChain and Chroma/FAISS for Vector DB

Our goal is to make it easy to have private offline document question-answer using LLMs.

### Try h2oGPT now, with LangChain on example databases 

Live hosted instances:
- [![img-small.png](img-small.png) Latest LangChain-enabled h2oGPT (temporary link) 12B](https://449b26467fe4cf2531.gradio.live)
- [![img-small.png](img-small.png) LangChain-enabled h2oGPT (temporary link 1) 12B](https://9b1c74d9de90a71538.gradio.live/)
- [![img-small.png](img-small.png) LangChain-enabled h2oGPT (temporary link 2) 12B](https://30999ff1f3ff7577e0.gradio.live/)
- [![img-small.png](img-small.png) LangChain-enabled h2oGPT (temporary link 3) 12B](https://2855c4e61c677186aa.gradio.live/)

For questions, discussing, or just hanging out, come and join our <a href="https://discord.gg/WKhYMWcVbq"><b>Discord</b></a>!

## Getting Started

To get started quickly to upload from Chatbot any docs, urls, etc. do:
```bash
grep -v '#\|peft' requirements.txt > req_constraints.txt
pip install -r requirements_optional_langchain.txt -c req_constraints.txt
python generate.py --base_model=h2oai/h2ogpt-oasst1-512-12b --load_8bit=True --langchain_mode=MyData
```
See below for additional instructions to add support for some file types.

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
   - `URL` : Any URL (i.e. `http://` or `https://`),
   - `ArXiv` : Any ArXiv name (e.g. `arXiv:1706.03762`),
   - `Text` : Paste Text into UI.

To support ArXiv API, do:
```bash
grep -v '#\|peft' requirements.txt > req_constraints.txt
pip install -r requirements_optional_langchain.gpllike.txt -c req_constraints.txt
```
but pymupdf is AGPL, requiring any source code be made available, which is not an issue directly for h2oGPT, but it's like GPL and too strong a constraint for general commercial use.

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
python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6_9b --langchain_mode=UserData
```

To add data to the existing database, then run generate after, do:
```bash
python make_db.py --add_if_exists=True
python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6_9b --langchain_mode=UserData
```

## FAQ

#### Why does the source link not work?

For links to direct to the document and download to your local machine, the original source documents must still be present on the host system where the database was created, e.g. `user_path` for `UserData` by default.  If the database alone is copied somewhere else, that host won't have access to the documents.  URL links like Wikipedia will still work normally on any host.


#### What is h2oGPT's LangChain integration like?

* [PrivateGPT](https://github.com/imartinez/privateGPT) but h2oGPT is fully commercially viable by not using [GPT4All](https://github.com/nomic-ai/gpt4all) based upon [LLaMa](https://github.com/facebookresearch/llama) and used data from GPT3.5 (violation of ToS).

* [Vault-AI](https://github.com/pashpashpash/vault-ai) but h2oGPT is fully private and open-source by not using OpenAI or [pinecone](https://www.pinecone.io/).

* [DB-GPT](https://github.com/csunny/DB-GPT) but h2oGPT is fully commercially viable by not using [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) (LLaMa based with GPT3.5 training data).

* [ChatPDF](https://www.chatpdf.com/) but h2oGPT is open-source and private and many more data types.

* [ChatDoc](https://chatdoc.com/) but h2oGPT is open-source and private. ChatDoc shows nice side-by-side view with doc on one side and chat in other.  Select specific doc or text in doc for question/summary.

* [Perplexity](https://www.perplexity.ai/) but h2oGPT is open-source and private, similar control over sources.

* [HayStack](https://github.com/deepset-ai/haystack) but h2oGPT is open-source and private.  Haystack is pivot to LLMs from NLP tasks, so well-developed documentation etc.  But mostly LangChain clone.

* [Empler](https://www.empler.ai/) but h2oGPT is open-source and private.  Empler has nice AI and content control, and focuses on use cases like marketing.

* [Writesonic](https://writesonic.com/) but h2oGPT is open-source and private.  Writesonic has better image/video control.

* [HuggingChat](https://huggingface.co/chat/) Not for commercial use, uses LLaMa and GPT3.5 training data, so violates ToS.

* [Bard](https://bard.google.com/) but h2oGPT is open-source and private.  Bard has better automatic link and image use.

* [ChatGPT](https://chat.openai.com/) but h2oGPT is open-source and private.  ChatGPT code interpreter has better image, video, etc. handling.

* [Bing](https://www.bing.com/) but h2oGPT is open-source and private.  Bing has excellent search queries and handling of results.

* [Bearly](https://bearly.ai/) but h2oGPT is open-source and private.  Bearly focuses on creative content creation.

* [Poe](https://poe.com/) but h2oGPT is open-source and private.  Poe also has immediate info-wall requiring phone number.

* [WiseOne](https://wiseone.io/) but h2oGPT is open-source and private.  WiseOne is reading helper.

* [Poet.ly or Aify](https://aify.co/) but h2oGPT is open-source and private.  Poet.ly focuses on writing articles.

* [PDFGPT.ai](https://pdfgpt.io/) but h2oGPT is open-source and private.  Only PDF and on expensive side.

* [BratGPT](https://bratgpt.com/) but h2oGPT is open-source and private.  Focuses on uncensored chat.

* [Halist](https://halist.ai/) but h2oGPT is open-source and private.  Uses ChatGPT but does not store chats, but can already do that now with ChatGPT.

* [UltimateGPT Toolkit](https://play.google.com/store/apps/details?id=com.neuralminds.ultimategptoolkit&ref=producthunt&pli=1) Android plugin for ChatGPT.

* [Intellibar](https://intellibar.app/) ChatGPT on iPhone.

* [GPTMana](https://play.google.com/store/apps/details?id=com.chatgpt.gptmana) Android Plugin.

* [Genie](https://www.genieai.co/) but h2oGPT is open-source and private.  Focuses on legal assistant.

* [ResearchAI](https://research-ai.io/) but h2oGPT is open-source and private.  Focuses on research helper with tools.

* [ChatOn](https://apps.apple.com/us/app/chaton) but h2oGPT is open-source and private.  ChatOn focuses on mobile, iPhone app.

* [Ask](https://iask.ai/) but h2oGPT is open-source and private.  Similar content control.

* [Petey](https://apps.apple.com/us/app/petey-ai-assistant/id6446047813) but h2oGPT is open-source and private.  Apple Watch.

* [QuickGPT](https://www.quickgpt.io/) but h2oGPT is open-source and private.  QuickGPT is ChatGPT for Whatsapp.

* [Raitoai](https://www.raitoai.com/) but h2oGPT is open-source and private.  Raito.ai focuses on helping writers.

* [AIChat](https://deepai.org/chat) but h2oGPT is open-source and private.  Heavy on ads, avoid.

* [AnonChatGPT](https://anonchatgpt.com/) but h2oGPT is open-source and private.  Anonymous use of ChatGPT, i.e. no account required.

* [GPTPro](https://play.google.com/store/apps/details?id=com.dfmv.gptpro&hl=en_US&gl=US) but h2oGPT is open-source and private.  GPTPro focuses on Android.

* [Rio](https://www.oziku.tech/rio-openai-chatgpt-assistant) but h2oGPT is open-source and private.  Browser-based assistant.

* [CommanderGPT](https://www.commandergpt.app/) but h2oGPT is open-source and private.  CommanderGPT focuses on MAC with a few tasks like image generation, translation, youtube query, etc.

* [ThreeSigma](https://www.threesigma.ai/) but h2oGPT is open-source and private.  Focuses on research tools, nice page linking.

* [LocalAI](https://github.com/go-skynet/LocalAI) but h2oGPT has document question/answer.  LocalAI has audio transcription, image generation, and a variety of models.

* [LocalLLaMa](https://github.com/jlonge4/local_llama) but h2oGPT has UI and GPU support. LocalLLaMa is command-line focused.  Like privateGPT.

* [ChartGPT](https://www.chartgpt.dev/) Focus on drawing charts.