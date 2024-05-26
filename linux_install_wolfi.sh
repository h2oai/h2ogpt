#!/bin/bash
set -o pipefail
set -ex


# broad support, but no training-time or data creation dependencies
pip install -r requirements.txt -c reqs_optional/reqs_constraints.txt

# Required for Doc Q/A: LangChain:
pip install -r reqs_optional/requirements_optional_langchain.txt -c reqs_optional/reqs_constraints.txt

# Required for CPU: LLaMa/GPT4All:
C=gcc-11 CXX=g++-11 pip install -r reqs_optional/requirements_optional_llamacpp_gpt4all.txt -c reqs_optional/reqs_constraints.txt --no-cache-dir

# Optional: PyMuPDF/ArXiv:
pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt -c reqs_optional/reqs_constraints.txt

# Optional: FAISS
pip install -r reqs_optional/requirements_optional_gpu_only.txt -c reqs_optional/reqs_constraints.txt

# Optional: Selenium/PlayWright:
pip install -r reqs_optional/requirements_optional_langchain.urls.txt -c reqs_optional/reqs_constraints.txt

# Optional: For DocTR
pip install -r reqs_optional/requirements_optional_doctr.txt -c reqs_optional/reqs_constraints.txt

# For DocTR: go back to older onnx so Tesseract OCR still works
pip install onnxruntime==1.15.0 -c reqs_optional/reqs_constraints.txt
# GPU only:
pip install onnxruntime-gpu==1.15.0 -c reqs_optional/reqs_constraints.txt

# Optional: for supporting unstructured package
for i in 1 2 3 4; do python -m nltk.downloader all && break || sleep 1; done  # retry as frequently fails with github downloading issues

# Audio transcription from Youtube videos and local mp3 files:
pip install pydub==0.25.1 librosa==0.10.1 ffmpeg==1.4 yt_dlp==2023.10.13 wavio==0.0.8 -c reqs_optional/reqs_constraints.txt

# Audio speed-up and slowdown (best quality), if not installed can only speed-up with lower quality
# may not work as it needs its CLI, example from ubuntu: sudo apt-get install -y rubberband-cli
pip install pyrubberband==0.3.0 -c reqs_optional/reqs_constraints.txt
# https://stackoverflow.com/questions/75813603/python-working-with-sound-librosa-and-pyrubberband-conflict
pip uninstall -y pysoundfile soundfile
pip install soundfile==0.12.1 -c reqs_optional/reqs_constraints.txt

# for any TTS:
pip install torchaudio soundfile==0.12.1 -c reqs_optional/reqs_constraints.txt
# GPU Only: for Coqui XTTS (ensure CUDA_HOME set and consistent with added postfix for extra-index):
# relaxed versions to avoid conflicts
pip install TTS deepspeed noisereduce emoji ffmpeg-python==0.2.0 trainer pysbd coqpit -c reqs_optional/reqs_constraints.txt

# for Coqui XTTS language helpers (specific versions probably not required)
pip install cutlet==0.3.0 langid==1.1.6 g2pkk==0.1.2 jamo==0.4.1 gruut[de,es,fr]==2.2.3 jieba==0.42.1 -c reqs_optional/reqs_constraints.txt
# For faster whisper:
#pip install git+https://github.com/SYSTRAN/faster-whisper.git -c reqs_optional/reqs_constraints.txt
# needed for librosa/soundfile to work, but violates TTS, but that's probably just too strict as we have seen before)
pip install numpy==1.23.0 --no-deps --upgrade -c reqs_optional/reqs_constraints.txt
# TTS or other deps load old librosa, fix:
pip install librosa==0.10.1 --no-deps --upgrade -c reqs_optional/reqs_constraints.txt

#
#* STT and TTS Notes:
#
# STT: Ensure microphone is on and in browser go to http://localhost:7860 instead of http://0.0.0.0:7860 for microphone to be possible to allow in browser.
# TTS: For XTT models, ensure `CUDA_HOME` is set correctly, because deepspeed compiles at runtime using torch and nvcc.  Those must match CUDA version.  E.g. if used `--extra-index https://download.pytorch.org/whl/cu118`, then must have ENV `CUDA_HOME=/usr/local/cuda-11.7` or ENV from conda must be that version.  Since conda only has up to cuda 11.7 for dev toolkit, but H100+ need cuda 11.8, for those cases one should download the toolkit from NVIDIA.
# Vision/Image packages
pip install fiftyone -c reqs_optional/reqs_constraints.txt
pip install pytube -c reqs_optional/reqs_constraints.txt
pip install diffusers==0.24.0 -c reqs_optional/reqs_constraints.txt

#
#* HNSW issue:
#
# In some cases old chroma migration package will install old hnswlib and that may cause issues when making a database, then do:
pip uninstall -y hnswlib chroma-hnswlib
# restore correct version
pip install chroma-hnswlib==0.7.3 --upgrade -c reqs_optional/reqs_constraints.txt

#
#* GPU Optional: For AutoGPTQ support on x86_64 linux
#
# in-transformers support of AutoGPTQ, requires also auto-gptq above to be installed since used internally by transformers/optimum
pip install optimum==1.17.1 -c reqs_optional/reqs_constraints.txt
#    See [AutoGPTQ](README_GPU.md#autogptq) about running AutoGPT models.

#
#* GPU Optional: For AutoAWQ support on x86_64 linux
pip uninstall -y autoawq ; pip install autoawq -c reqs_optional/reqs_constraints.txt
# fix version since don't need lm-eval to have its version of 1.5.0
pip install sacrebleu==2.3.1 --upgrade -c reqs_optional/reqs_constraints.txt

# ensure not installed if remade env on top of old env
pip uninstall llama_cpp_python_cuda -y

#* GPU Optional: For exllama support on x86_64 linux
 #pip uninstall -y exllama ; pip install https://github.com/jllllll/exllama/releases/download/0.0.18/exllama-0.0.18+cu121-cp310-cp310-linux_x86_64.whl --no-cache-dir -c reqs_optional/reqs_constraints.txt
#    See [exllama](README_GPU.md#exllama) about running exllama models.
pip install autoawq-kernels -c reqs_optional/reqs_constraints.txt

pip install auto-gptq==0.7.1 exllamav2==0.0.16

#
#* GPU Optional: Support amazon/MistralLite with flash attention 2
#
pip install --upgrade pip
pip install flash-attn==2.4.2 --no-build-isolation --no-cache-dir -c reqs_optional/reqs_constraints.txt

#
#* Control Core Count for chroma < 0.4 using chromamigdb package:
#
# Duckdb used by Chroma < 0.4 uses DuckDB 0.8.1 that has no control over number of threads per database, `import duckdb` leads to all virtual cores as threads and each db consumes another number of threads equal to virtual cores.  To prevent this, one can rebuild duckdb using [this modification](https://github.com/h2oai/duckdb/commit/dcd8c1ffc53dd020623630efb99ba6a3a4cbc5ad) or one can try to use the prebuild wheel for x86_64 built on Ubuntu 20.
pip uninstall -y pyduckdb duckdb
pip install https://h2o-release.s3.amazonaws.com/h2ogpt/duckdb-0.8.2.dev4025%2Bg9698e9e6a8.d20230907-cp310-cp310-linux_x86_64.whl --no-cache-dir --force-reinstall --no-deps -c reqs_optional/reqs_constraints.txt

#
#* SERP for search:
#
pip install -r reqs_optional/requirements_optional_agents.txt -c reqs_optional/reqs_constraints.txt
#  For more info see [SERP Docs](README_SerpAPI.md).

# https://github.com/h2oai/h2ogpt/issues/1483
pip uninstall flash_attn autoawq autoawq-kernels -y
pip install flash_attn autoawq autoawq-kernels --no-cache-dir -c reqs_optional/reqs_constraints.txt


bash ./docs/run_patches.sh

# setup tiktoken cache
python3.10 -c "
import tiktoken
from tiktoken_ext import openai_public
# FakeTokenizer etc. needs tiktoken for general tasks
for enc in openai_public.ENCODING_CONSTRUCTORS:
    encoding = tiktoken.get_encoding(enc)
model_encodings = [
    'gpt-4',
    'gpt-4-0314',
    'gpt-4-32k',
    'gpt-4-32k-0314',
    'gpt-3.5-turbo',
    'gpt-3.5-turbo-16k',
    'gpt-3.5-turbo-0301',
    'text-ada-001',
    'ada',
    'text-babbage-001',
    'babbage',
    'text-curie-001',
    'curie',
    'davinci',
    'text-davinci-003',
    'text-davinci-002',
    'code-davinci-002',
    'code-davinci-001',
    'code-cushman-002',
    'code-cushman-001'
]
for enc in model_encodings:
    encoding = tiktoken.encoding_for_model(enc)
print('Done!')
"


# make main workspace writable
chmod -R a+rwx /workspace

# remove pip cache
rm -rf /workspace/.cache
rm -rf /usr/lib/python3.10/site-packages/future/backports/test
