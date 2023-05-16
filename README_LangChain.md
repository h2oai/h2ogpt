## h2oGPT integration with LangChain and Chroma/FAISS for Vector DB

Our goal is to make it easy to have private offline document question-answer using LLMs.

### Try h2oGPT now, with LangChain on example databases 

Live hosted instances:
- [![img-small.png](img-small.png) LangChain-enabled h2oGPT (temporary link) 12B](https://9b1c74d9de90a71538.gradio.live/)

For questions, discussing, or just hanging out, come and join our <a href="https://discord.gg/WKhYMWcVbq"><b>Discord</b></a>!

### Database creation

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

