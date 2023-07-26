## CPU Details

Details that do not depend upon whether running on CPU for Linux, Windows, or MAC.

### LLaMa.cpp 
  
* Download from [TheBloke](https://huggingface.co/TheBloke).  For example, [13B WizardLM Quantized](https://huggingface.co/TheBloke/wizardLM-13B-1.0-GGML) or [7B WizardLM Quantized](https://huggingface.co/TheBloke/WizardLM-7B-uncensored-GGML).  TheBloke has a variety of model types, quantization bit depths, and memory consumption.  Choose what is best for your system's specs.  For 7B case, download [WizardLM-7B-uncensored.ggmlv3.q8_0.bin](https://huggingface.co/TheBloke/WizardLM-7B-uncensored-GGML/resolve/main/WizardLM-7B-uncensored.ggmlv3.q8_0.bin) into local path:
   ```bash
    wget https://huggingface.co/TheBloke/WizardLM-7B-uncensored-GGML/resolve/main/WizardLM-7B-uncensored.ggmlv3.q8_0.bin
   ```
* Change `.env_gpt4all` model name if desired.
   ```.env_gpt4all
   model_path_llama=WizardLM-7B-uncensored.ggmlv3.q8_0.bin
   ```
    Then one sets `model_path_llama` in `.env_gpt4all`, which is currently the default.

* When using `llama.cpp` based CPU models, for computers with low system RAM or slow CPUs, we recommend adding to `.env_gpt4all`:
   ```.env_gpt4all
   use_mlock=False
   n_ctx=1024
   ```
    where `use_mlock=True` is default to avoid slowness and `n_ctx=2048` is default for large context handling.  For computers with plenty of system RAM, we recommend adding to `.env_gpt4all`:
   ```.env_gpt4all
   n_batch=1024
   ```
    for faster handling.  On some systems this has no strong effect, but on others may increase speed quite a bit.

* Run LLaMa.cpp model:

    With documents in `user_path` folder, run:
   ```bash
   python generate.py --base_model='llama' --prompt_type=wizard2 --score_model=None --langchain_mode='UserData' --user_path=user_path
   ```

### GPT4ALL

* Choose Model from GPT4All Model explorer [GPT4All-J compatible model](https://gpt4all.io/index.html). One does not need to download manually, the GPT4ALL package will download at runtime and put it into `.cache` like Hugging Face would.
    
* Change `.env_gpt4all` model name if chose different model from GPT4All Model Explorer.
    ```.env_gpt4all
    model_path_gptj=ggml-gpt4all-j-v1.3-groovy.bin
    model_name_gpt4all_llama=ggml-wizardLM-7B.q4_2.bin
    ```
    However, `gpjt` model often gives [no output](FAQ.md#gpt4all-not-producing-output), even outside h2oGPT.  See [GPT4All](https://github.com/nomic-ai/gpt4all) for details on installation instructions if any issues encountered.

### Low-memory

See [Low Memory](FAQ.md#low-memory-mode) for more information about low-memory recommendations.

