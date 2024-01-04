## h2oGPT integration with LangChain and Chroma/FAISS/Weaviate for Vector DB

Our goal is to make it easy to have private offline document question-answer using LLMs.

## Get Started

Follow the [get started steps](../README.md#get-started) in the main README.  In this readme, we focus on other optional aspects.

To support GPU FAISS database, run:
```bash
pip install -r reqs_optional/requirements_optional_faiss.txt
```
or for CPU FAISS database, run:
```bash
pip install -r reqs_optional/requirements_optional_faiss_cpu.txt
```

or for Weaviate, run:
```bash
pip install -r reqs_optional/requirements_optional_langchain.txt
```
## Supported Data types

Open-source data types are supported, .msg is not supported due to GPL-3 requirement.  Other meta types support other types inside them.  Special support for some behaviors is provided by the UI itself.

### Supported Native Data types

   - `.pdf`: Portable Document Format (PDF),
   - `.txt`: Text file (UTF-8),
   - `.csv`: CSV,
   - `.toml`: TOML,
   - `.py`: Python,
   - `.rst`: reStructuredText,
   - `.rtf`: Rich Text Format,
   - `.md`: Markdown,
   - `.html`: HTML File,
   - `.mhtml`: MHTML File,
   - `.htm`: HTML File,
   - `.docx`: Word Document (optional),
   - `.doc`: Word Document (optional),
   - `.xlsx`: Excel Document (optional),
   - `.xls`: Excel Document (optional),
   - `.enex`: EverNote,
   - `.eml`: Email,
   - `.epub`: EPub,
   - `.odt`: Open Document Text,
   - `.pptx` : PowerPoint Document,
   - `.ppt` : PowerPoint Document,
   - `.xml`: XML,

   - `.apng` : APNG Image (optional),
   - `.blp` : BLP Image (optional),
   - `.bmp` : BMP Image (optional),
   - `.bufr` : BUFR Image (optional),
   - `.bw` : BW Image (optional),
   - `.cur` : CUR Image (optional),
   - `.dcx` : DCX Image (optional),
   - `.dds` : DDS Image (optional),
   - `.dib` : DIB Image (optional),
   - `.emf` : EMF Image (optional),
   - `.eps` : EPS Image (optional),
   - `.fit` : FIT Image (optional),
   - `.fits` : FITS Image (optional),
   - `.flc` : FLC Image (optional),
   - `.fli` : FLI Image (optional),
   - `.fpx` : FPX Image (optional),
   - `.ftc` : FTC Image (optional),
   - `.ftu` : FTU Image (optional),
   - `.gbr` : GBR Image (optional),
   - `.gif` : GIF Image (optional),
   - `.grib` : GRIB Image (optional),
   - `.h5` : H5 Image (optional),
   - `.hdf` : HDF Image (optional),
   - `.icb` : ICB Image (optional),
   - `.icns` : ICNS Image (optional),
   - `.ico` : ICO Image (optional),
   - `.iim` : IIM Image (optional),
   - `.im` : IM Image (optional),
   - `.j2c` : J2C Image (optional),
   - `.j2k` : J2K Image (optional),
   - `.jfif` : JFIF Image (optional),
   - `.jp2` : JP2 Image (optional),
   - `.jpc` : JPC Image (optional),
   - `.jpe` : JPE Image (optional),
   - `.jpeg` : JPEG Image (optional),
   - `.jpf` : JPF Image (optional),
   - `.jpg` : JPG Image (optional),
   - `.jpx` : JPX Image (optional),
   - `.mic` : MIC Image (optional),
   - `.mpeg` : MPEG Image (optional),
   - `.mpg` : MPG Image (optional),
   - `.msp` : MSP Image (optional),
   - `.pbm` : PBM Image (optional),
   - `.pcd` : PCD Image (optional),
   - `.pcx` : PCX Image (optional),
   - `.pgm` : PGM Image (optional),
   - `.png` : PNG Image (optional),
   - `.pnm` : PNM Image (optional),
   - `.ppm` : PPM Image (optional),
   - `.ps` : PS Image (optional),
   - `.psd` : PSD Image (optional),
   - `.pxr` : PXR Image (optional),
   - `.qoi` : QOI Image (optional),
   - `.ras` : RAS Image (optional),
   - `.rgb` : RGB Image (optional),
   - `.rgba` : RGBA Image (optional),
   - `.sgi` : SGI Image (optional),
   - `.tga` : TGA Image (optional),
   - `.tif` : TIF Image (optional),
   - `.tiff` : TIFF Image (optional),
   - `.vda` : VDA Image (optional),
   - `.vst` : VST Image (optional),
   - `.webp` : WEBP Image (optional),
   - `.wmf` : WMF Image (optional),
   - `.xbm` : XBM Image (optional),
   - `.xpm` : XPM Image (optional).

   - `.mp4` : MP4 Audio (optional).
   - `.mpeg` : MP4-based MPEG Audio (optional).
   - `.mpg` : MP4-based MPG Audio (optional).
   - `.mp3` : MP3 Audio (optional).
   - `.ogg` : OGG Audio (optional).
   - `.flac` : FLAC Audio (optional).
   - `.aac` : AAC Audio (optional).
   - `.au` : AU Audio (optional).


### Supported Meta Data types

   - `.zip` : Zip File containing any native datatype.
   - `.urls` : Text file containing new-line separated URLs (to be consumed via download).

Note: If you upload files and one of the files is a zip that contains images to be read by BLIP/DocTR or PDFs to be read by DocTR, this will currently fail with:
```text
Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
```
Please upload the zip separately for now.

### Supported Data Types in UI

   - `Files` : All Native and Meta Data Types as file(s),
   - `URL` : Any URL (i.e. `http://` or `https://`),
   - `ArXiv` : Any ArXiv name (e.g. `arXiv:1706.03762`),
   - `Text` : Paste Text into UI.

### Supported Meta Tasks

   - `ScrapeWithPlayWRight` : Async Web Scraping using headless Chromium via PlayWright
   - `ScrapeWithHttp` : Async Web Scraping using aiohttp (slower than PlayWright)

* Timing
  * Typical page like passing `https://github.com/h2oai/h2ogpt` takes about 300 seconds to process at default depth of 1 with about 140 pages.
  * No good progress indicators from these packages, so just have to wait.
* Depth:
  * Set env `CRAWL_DEPTH=<depth>` to control depth for some integer `<depth>`, where 0 means only actual page, 1 means that page + all links on that page, etc.  `CRAWL_DEPTH=1` by default to avoid excessive crawling.
  * Set env `ALL_CRAWL_DEPTH=<depth>` to force all url loaders to crawl at some depth (will be slower than async ones)
* BS4:
  * Set env `HTML_TRANS=BS4` to use `BS4` to transform instead of `Html2TextTransformer`.  Set `BS4_TAGS` env to some string of list to set [tags](https://python.langchain.com/docs/use_cases/web_scraping#quickstart).
    * e.g. `export BS4_TAGS="['span']"`
  * Scrape text content tags such as `<p>`, `<li>`, `<div>`, and `<a>` tags from the HTML content:
    * `<p>`: The paragraph tag. It defines a paragraph in HTML and is used to group together related sentences and/or phrases.
    * `<li>`: The list item tag. It is used within ordered (`<ol>`) and unordered (`<ul>`) lists to define individual items within the list.
    * `<div>`: The division tag. It is a block-level element used to group other inline or block-level elements.
    * `<a>`: The anchor tag. It is used to define hyperlinks.
    * `<span>`: an inline container used to mark up a part of a text, or a part of a document.
  For many news websites (e.g., WSJ, CNN), headlines and summaries are all in `<span>` tags.
* ScrapeWithHttp:
  * Can change code in src/gpt_langchain.py to change `requests_per_second=10` to some other value.

### Adding new file types

The function `file_to_doc` controls the ingestion, with [allowed ones listed](https://github.com/h2oai/h2ogpt/blob/1184f057088743599e2d5241329551b8f7f5320d/src/gpt_langchain.py#L1021-L1035).   If one wants to add a new file type, add it to the list `file_types`, and then add an entry in `file_to_doc()` function.

Metadata is added using `add_meta` function, and other metadata, like chunk_id, is added after chunking.  One could add a new step to add meta data to `page_content` to each langchain `Document`.

## Database creation

To use some example databases (will overwrite UserData make above unless change options) and run generate after, do:
```bash
python src/make_db.py --download_some=True
python generate.py --base_model=HuggingFaceH4/zephyr-7b-beta --langchain_mode=UserData --langchain_modes="['UserData', 'wiki', 'MyData', 'github h2oGPT', 'DriverlessAI docs']"
```
which downloads example databases.  This obtains files from some [pre-generated databases](https://huggingface.co/datasets/h2oai/db_dirs).  A large Wikipedia database is also available.

To build the database first outside chatbot, then run generate after, do:
```bash
python src/make_db.py
python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6_9b --langchain_mode=UserData
```

To add data to the existing database, then run generate after, do:
```bash
python src/make_db.py --add_if_exists=True
python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6_9b --langchain_mode=UserData
```

By default, `generate.py` will load an existing UserData database and add any documents added to user_path or change any files that have changed.  To avoid detecting any new files, just avoid passing --user_path=user_path, which sets it to None, i.e.:
```bash
python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6_9b --langchain_mode=UserData
```
which will avoid using `user_path` since it is no longer passed.  Otherwise when passed, any new files will be added or changed (by hash) files will be updated (delete old sources and add new sources).

If you have enough GPU memory for embedding, but not the LLM as well, then a less private mode is to use OpenAI model.
```bash
python generate.py  --inference_server=openai_chat --base_model=gpt-3.5-turbo --score_model=None --langchain_mode=LLM --langchain_modes="['LLM', 'UserData', 'MyData']"
```
and if you want to push image caption model to get better captions, this can be done if have enough GPU memory or if use OpenAI:
```bash
python generate.py  --inference_server=openai_chat --base_model=gpt-3.5-turbo --score_model=None --langchain_mode=LLM --langchain_modes="['LLM', 'UserData', 'MyData']" --captions_model=Salesforce/blip2-flan-t5-xl
```
Similar commands can be used for Azure OpenAI, e.g.
```bash
OPENAI_API_KEY=<key> python generate.py --inference_server="openai_azure_chat:<deployment_name>:<base_url>:<api_version>" --base_model=gpt-3.5-turbo --h2ocolors=False --langchain_mode=UserData
```


### Multiple embeddings and sources

We only support one embedding at a time for each database.

So you could use src/make_db.py to make the db for different embeddings (`--hf_embedding_model` like gen.py, any HF model) for each collection (e.g. UserData, UserData2) for each source folders (e.g. user_path, user_path2), and then at generate.py time you can specify those different collection names in `--langchain_modes` and `--langchain_modes` and `--langchain_mode_paths`.  For example:
```bash
python src/make_db.py --user_path=user_path --collection_name=UserData --langchain_type=shared --hf_embedding_model=hkunlp/instructor-large
python src/make_db.py --user_path=user_path2 --collection_name=UserData2 --langchain_type=shared --hf_embedding_model=sentence-transformers/all-MiniLM-L6-v2
```
Note that `shared` is default type already, but we show above to show what options are relevant if want to change them.
Then run:
```bash
python generate.py --base_model='llama' --prompt_type=llama2 --score_model=None --langchain_mode='UserData' --langchain_modes=['UserData','UserData2'] --langchain_mode_paths={'UserData':'user_path','UserData2':'user_path2'} --langchain_mode_types={'UserData':'shared','UserData2':'shared'} --model_path_llama=https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf --max_seq_len=4096
```
or choose 13B.  And watch-out for use of whitespace.  For `langchain_mode_paths` you can pass surrounded by "'s and have spaces.

### Choosing document types

```python
import sys
sys.path.append('src')
from src.gpt_langchain import get_supported_types
non_image_types, image_types, video_types = get_supported_types()
print(non_image_types)
print(image_types)
```
Select types, and pass to `make_db` like:
```bash
python src/make_db.py --user_path="/home/jon/Downloads/demo_data" --collection_name=VAData --enable_pdf_ocr='off' --selected_file_types="['pdf', 'html', 'htm']"
python generate.py  --base_model='llama' --prompt_type=llama2 --score_model=None --langchain_mode=VAData --langchain_modes=['VAData'] --model_path_llama=https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf --max_seq_len=4096
```
or choose 13B.

To ensure a collection is persisted even when not using any athentication, be sure it is shared type, e.g.:
```bash
python generate.py --base_model='llama' --prompt_type=llama2 --score_model=None --max_max_new_tokens=2048 --max_new_tokens=1024 \
       --visible_tos_tab=False --visible_hosts_tab=False --visible_models_tab=False \
       --langchain_modes="['LLM','PersistData']" --langchain_mode=PersistData \
       --langchain_mode_types="{'PersistData':'shared'}" \
       --top_k_docs=-1 --max_time=360 --save_dir=save \
       --model_path_llama=https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf \
       --max_seq_len=4096
```
or choose 13B.

### Note about Embeddings

The default embedding for GPU is `instructor-large` since most accurate, however it leads to excessively high scores for references due to its flat score distribution.  For CPU the default embedding is `all-MiniLM-L6-v2`, and it has a sharp distribution of scores, so references make sense, but it is less accurate.

### Note about FAISS

FAISS filtering is not supported in h2oGPT yet, ask if this is desired to be added.  So subset by document does not function for FAISS.

### Using Weaviate

#### About
[Weaviate](https://weaviate.io/) is an open-source vector database designed to scale seamlessly into billions of data objects. This implementation supports hybrid search out-of-the-box (meaning it will perform better for keyword searches).

You can run Weaviate in 5 ways:

- **SaaS** – with [Weaviate Cloud Services (WCS)](https://weaviate.io/pricing).

  WCS is a fully managed service that takes care of hosting, scaling, and updating your Weaviate instance. You can try it out for free with a sandbox that lasts for 14 days.

  To set up a SaaS Weaviate instance with WCS:

  1.  Navigate to [Weaviate Cloud Console](https://console.weaviate.cloud/).
  2.  Register or sign in to your WCS account.
  3.  Create a new cluster with the following settings:
      - `Subscription Tier` – Free sandbox for a free trial, or contact [hello@weaviate.io](mailto:hello@weaviate.io) for other options.
      - `Cluster name` – a unique name for your cluster. The name will become part of the URL used to access this instance.
      - `Enable Authentication?` – Enabled by default. This will generate a static API key that you can use to authenticate.
  4.  Wait for a few minutes until your cluster is ready. You will see a green tick ✔️ when it's done. Copy your cluster URL.

- **Hybrid SaaS**

  > If you need to keep your data on-premise for security or compliance reasons, Weaviate also offers a Hybrid SaaS option: Weaviate runs within your cloud instances, but the cluster is managed remotely by Weaviate. This gives you the benefits of a managed service without sending data to an external party.

  The Weaviate Hybrid SaaS is a custom solution. If you are interested in this option, please reach out to [hello@weaviate.io](mailto:hello@weaviate.io).

- **Self-hosted** – with a Docker container

  To set up a Weaviate instance with Docker:

  1. [Install Docker](https://docs.docker.com/engine/install/) on your local machine if it is not already installed.
  2. [Install the Docker Compose Plugin](https://docs.docker.com/compose/install/)
  3. Download a `docker-compose.yml` file with this `curl` command:

```bash
curl -o docker-compose.yml "https://configuration.weaviate.io/v2/docker-compose/docker-compose.yml?modules=standalone&runtime=docker-compose&weaviate_version=v1.19.6"
```

     Alternatively, you can use Weaviate's docker compose [configuration tool](https://weaviate.io/developers/weaviate/installation/docker-compose) to generate your own `docker-compose.yml` file.

4. Run `docker compose up -d` to spin up a Weaviate instance.

     > To shut it down, run `docker compose down`.

- **Self-hosted** – with a Kubernetes cluster

  To configure a self-hosted instance with Kubernetes, follow Weaviate's [documentation](https://weaviate.io/developers/weaviate/installation/kubernetes).|

- **Embedded** - start a weaviate instance right from your application code using the client library
   
  This code snippet shows how to instantiate an embedded weaviate instance and upload a document:

```python
  import weaviate
  from weaviate.embedded import EmbeddedOptions

  client = weaviate.Client(
    embedded_options=EmbeddedOptions()
  )

  data_obj = {
    "name": "Chardonnay",
    "description": "Goes with fish"
  }

  client.data_object.create(data_obj, "Wine")
```
  
  Refer to the [documentation](https://weaviate.io/developers/weaviate/installation/embedded) for more details about this deployment method.
## How To Use
Simply pass the `--db_type=weaviate` argument. For example:
```bash
python src/make_db.py --db_type=weaviate
python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6_9b \
   --langchain_mode=UserData \
   --db_type=weaviate
```
will use an embedded weaviate instance.

If you have a weaviate instance hosted at say http://localhost:8080, then you need to define the `WEAVIATE_URL` environment variable before running the scripts:
```
WEAVIATE_URL=http://localhost:8080 python src/make_db.py --db_type=weaviate
WEAVIATE_URL=http://localhost:8080 python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6_9b \
   --langchain_mode=UserData \
   --db_type=weaviate
```

Similarly, if you had set up your weaviate instance with a username and password using the [OIDC Resource Owner Password flow](https://weaviate.io/developers/weaviate/configuration/authentication#oidc---a-client-side-perspective), you will need to define the following additional environment variables:
* WEAVIATE_USERNAME: the username used for authentication
* WEAVIATE_PASSWORD: the password used for authentication
* WEAVIATE_SCOPE: optional, defaults to "offline_access"

Notes:

* Since h2oGPT is focused on privacy, connecting to weaviate via WCS is not supported as that will expose your data to a 3rd party
* Weaviate doesn't know about persistent directory throughout code, and maintains locations based upon collection name
* Weaviate doesn't support query of all metadata except via similarity search up to 10k documents, so full list of sources is not possible in h2oGPT UI for `Update UI with Document(s) from DB` or `Show Sources from DB`

## Document Question-Answer FAQ

### What is UserData and MyData?

UserData: Shared with anyone who is on your server.  Persisted across sessions in single location for entire server.  Control upload via allow_upload_to_user_data option.  Useful for collaboration.

MyData: Personal space that is inaccessible if one goes into a new browser session.  Useful for public demonstrations so that every instance is independent.  Or useful  if user is not allowed to upload to shared UserData and wants to do Q/A.

It's work in progress to add other persistent databases and to have MyData persisted across browser sessions via cookie or other authentication.

#### Why does the source link not work?

For links to direct to the document and download to your local machine, the original source documents must still be present on the host system where the database was created, e.g. `user_path` for `UserData` by default.  If the database alone is copied somewhere else, that host won't have access to the documents.  URL links like Wikipedia will still work normally on any host.


#### What is h2oGPT's LangChain integration like?

* [PrivateGPT](https://github.com/imartinez/privateGPT) .  By comparison, h2oGPT has:
  * UI with chats export, import, selection, regeneration, and undo
  * UI and document Q/A, upload, download, and list
  * Parallel ingest of documents, using GPUs if present for vector embeddings, with progress bar in stdout
  * Choose which specific collection
  * Choose to get response regarding all documents or specific selected document(s) out of a collection
  * Choose to chat with LLM, get one-off LLM response to a query, or talk to a collection
  * GPU support from any hugging face model for highest performance
  * Upload a many types of docs, from PDFs to images (caption or OCR), URLs, ArXiv queries, or just plain text inputs
  * Server-Client API through gradio client
  * RLHF score evaluation for every response
  * UI with side-by-side model comparisons against two models at a time with independent chat streams
  * Fine-tuning framework with QLORA 4-bit, 8-bit, 16-bit GPU fine-tuning or CPU fine-tuning

* [localGPT](https://github.com/PromtEngineer/localGPT).  By comparison, h2oGPT has similar benefits as compared to localGPT.  Both h2oGPT and localGPT can use GPUs for LLMs and embeddings, including latest Vicuna or WizardLM models.

* [Quiver](https://github.com/StanGirard/quivr). By comparison, Quiver requires docker but also supports audio and video and currently only supports OpenAI models and embeddings.

* [LM Studio](https://github.com/lmstudio-ai). Nice control over models and llama settings, good windows installer.

* [DocsGPT](https://github.com/arc53/DocsGPT).  More limited document support.

* [GPT4-PDF-Chatbot-LangChain](https://github.com/mayooear/gpt4-pdf-chatbot-langchain).  Uses OpenAI, pinecone, etc. No longer maintained.

* [Vault-AI](https://github.com/pashpashpash/vault-ai) but h2oGPT is fully private and open-source by not using OpenAI or [pinecone](https://www.pinecone.io/).

* [DB-GPT](https://github.com/csunny/DB-GPT) but h2oGPT is fully commercially viable by not using [Vicuna](https://lmsys.org/blog/2023-03-30-vicuna/) (LLaMa based with GPT3.5 training data).

* [ChatBox](https://github.com/Bin-Huang/chatbox) has ability to collaborate.

* [Chat2DB](https://github.com/alibaba/Chat2DB) like DB-GPT by Alibaba.

* [pdfGPT](https://github.com/bhaskatripathi/pdfGPT) like PrivateGPT but no longer maintained.

* [docquery](https://github.com/impira/docquery) like PrivateGPT but uses LayoutLM.

* [KhoJ](https://github.com/khoj-ai/khoj) but also access from emacs or Obsidian.

* [ChatPDF](https://www.chatpdf.com/) but h2oGPT is open-source and private and many more data types.

* [TryGloo](https://www.trygloo.com/) Semantic Search and Classification.

* [Cube](https://cube.dev/blog/introducing-the-langchain-integration).

* [RFPBot](https://www.datarobot.com/platform/generative-ai/).  Confidence score, slack integration.

* [Sharly](https://www.sharly.ai/) but h2oGPT is open-source and private and many more data types.  Sharly and h2oGPT both allow sharing work through UserData shared collection.

* [ChatDoc](https://chatdoc.com/) but h2oGPT is open-source and private. ChatDoc shows nice side-by-side view with doc on one side and chat in other.  Select specific doc or text in doc for question/summary.

* [Casalioy](https://github.com/su77ungr/casalioy) with focus on air-gap with docker, otherwise like older privateGPT.

* [Perplexity](https://www.perplexity.ai/) but h2oGPT is open-source and private, similar control over sources.

* [HayStack](https://github.com/deepset-ai/haystack) but h2oGPT is open-source and private.  Haystack is pivot to LLMs from NLP tasks, so well-developed documentation etc.  But mostly LangChain clone.

* [Empler](https://www.empler.ai/) but h2oGPT is open-source and private.  Empler has nice AI and content control, and focuses on use cases like marketing.

* [Writesonic](https://writesonic.com/) but h2oGPT is open-source and private.  Writesonic has better image/video control.

* [HuggingChat](https://huggingface.co/chat/) Not for commercial use, uses LLaMa and GPT3.5 training data, so violates ToS.

* [Bard](https://bard.google.com/) but h2oGPT is open-source and private.  Bard has better automatic link and image use.

* [ChatGPT](https://chat.openai.com/) but h2oGPT is open-source and private.  ChatGPT code interpreter has better image, video, etc. handling.

* [ChatGPT-Next-Web](https://github.com/Yidadaa/ChatGPT-Next-Web) like local ChatGPT.

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
