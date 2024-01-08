@echo off
REM Install primary dependencies.
REM Remove any bad dependencies that existed (required for new transformers it seems):
pip uninstall -y flash-attn
pip install -r requirements.txt
REM Optional: for bitsandbytes 4-bit and 8-bit:
pip uninstall bitsandbytes -y
pip install https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.41.1-py3-none-win_amd64.whl
REM Bitsandbytes can be uninstalled (`pip uninstall bitsandbytes`) and still h2oGPT can be used if one does not pass `--load_8bit=True`.
REM When running windows on GPUs with bitsandbytes in 8-bit you should see something like the below in output:
REM C:\Users\pseud\.conda\envs\h2ogpt\lib\site-packages\bitsandbytes\libbitsandbytes_cuda118.dll

REM * Install document question-answer dependencies
REM     # Required for Doc Q/A: LangChain:
pip install -r reqs_optional/requirements_optional_langchain.txt
REM     # Required for CPU: LLaMa/GPT4All:
pip install -r reqs_optional/requirements_optional_gpt4all.txt
REM     # Optional: PyMuPDF/ArXiv:
@echo off
IF "%GPLOK%"=="1" (
    pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt
)
REM # Optional: FAISS (for AutoGPT agent)
pip install -r reqs_optional/requirements_optional_faiss_cpu.txt
REM     # Optional: Selenium/PlayWright:
pip install -r reqs_optional/requirements_optional_langchain.urls.txt
REM  # Optional: for supporting unstructured package
python -m nltk.downloader all
REM     # Optional but required for PlayWright
playwright install --with-deps
REM     # Note: for Selenium, we match versions of playwright so above installer will add chrome version needed

REM # Audio transcription from Youtube videos and local mp3 files:
REM Only for Microsoft TTS, not Coqui
pip install pydub==0.25.1 librosa==0.10.1 ffmpeg==1.4 yt_dlp==2023.10.13 wavio==0.0.8
pip install soundfile==0.12.1
REM pip install torchaudio soundfile==0.12.1

REM # Vision/Image packages
pip install fiftyone
pip install pytube
pip install diffusers==0.24.0

REM * AutoGPTQ support:
pip uninstall -y auto-gptq
REM     # GPU
pip install auto-gptq==0.6.0
REM     # in-transformers support of AutoGPTQ, requires also auto-gptq above to be installed since used internally by transformers/optimum
pip install optimum==1.16.1

REM * AutoAWQ support:
pip uninstall -y autoawq
pip install autoawq==0.1.8

REM  Exllama support (GPU only):
pip uninstall -y exllama
pip install https://github.com/jllllll/exllama/releases/download/0.0.18/exllama-0.0.18+cu118-cp310-cp310-win_amd64.whl --no-cache-dir

REM * SERP for search:
pip install -r reqs_optional/requirements_optional_agents.txt
REM   For more info see [SERP Docs](README_SerpAPI.md).

REM * For supporting Word and Excel documents, if you don't have Word/Excel already, then download and install libreoffice: https://www.libreoffice.org/download/download-libreoffice/ .
REM * To support OCR, download and install [tesseract](https://github.com/UB-Mannheim/tesseract/wiki), see also: [Tesseract Documentation](https://tesseract-ocr.github.io/tessdoc/Installation.html).  Please add the installation directories to your PATH.

REM * vLLM support:
pip install https://h2o-release.s3.amazonaws.com/h2ogpt/openvllm-1.3.7-py3-none-any.whl

REM * PDF Viewer support (only if using gradio4):
REM pip install https://h2o-release.s3.amazonaws.com/h2ogpt/gradio_pdf-0.0.3-py3-none-any.whl

