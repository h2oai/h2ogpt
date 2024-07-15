#!/bin/bash
set -o pipefail
set -ex

shopt -s expand_aliases
if ! test -f /usr/bin/sudo; then
  echo "No sudo"
  alias sudo=' '
fi

#
#* Optional: For document Q/A and use of DocTR.  Install before other pips to avoid long conflict checks.
#
if [[ -z "${WOLFI_OS}" ]]; then
  conda install weasyprint pygobject -c conda-forge -y
  # Avoids library mismatch.
  # fix any bad env
  pip uninstall -y pandoc pypandoc pypandoc-binary flash-attn
else
  echo "pandoc is part of base wolfi-os image"
fi

# upgrade pip
pip install --upgrade pip wheel

# broad support, but no training-time or data creation dependencies
pip install -r requirements.txt -c reqs_optional/reqs_constraints.txt

if [[ -z "${WOLFI_OS}" ]]; then
  #
  #* Optional: Install document question-answer dependencies:
  #
  # May be required for jq package:
  sudo apt-get update -y
  sudo apt-get -y install autoconf libtool
fi
# Required for Doc Q/A: LangChain:
pip install -r reqs_optional/requirements_optional_langchain.txt -c reqs_optional/reqs_constraints.txt
# Required for CPU: LLaMa/GPT4All:
if [[ -z "${WOLFI_OS}" ]]; then
  pip install -r reqs_optional/requirements_optional_llamacpp_gpt4all.txt -c reqs_optional/reqs_constraints.txt --no-cache-dir
else
  C=gcc-11 CXX=g++-11 pip install -r reqs_optional/requirements_optional_llamacpp_gpt4all.txt -c reqs_optional/reqs_constraints.txt --no-cache-dir
fi
# Optional: PyMuPDF/ArXiv:
#   Note!! that pymupdf is AGPL, requiring any source code be made available, but it's like GPL and too strong a constraint for general commercial use.
if [ "${GPLOK}" -eq "1" ]
then
    pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt -c reqs_optional/reqs_constraints.txt
fi
# Optional: FAISS
pip install -r reqs_optional/requirements_optional_gpu_only.txt -c reqs_optional/reqs_constraints.txt
# Optional: Selenium/PlayWright:
pip install -r reqs_optional/requirements_optional_langchain.urls.txt -c reqs_optional/reqs_constraints.txt

# Optional: support docx, pptx, ArXiv, etc. required by some python packages
if [[ -z "${WOLFI_OS}" ]]; then
  sudo apt-get install -y libmagic-dev poppler-utils tesseract-ocr libtesseract-dev libreoffice
else
  echo "libmagic, tesseract, libreoffice are part of base wolfi-os image, but no poppler"
fi
# Optional: For DocTR
pip install -r reqs_optional/requirements_optional_doctr.txt -c reqs_optional/reqs_constraints.txt
# For DocTR: go back to older onnx so Tesseract OCR still works
pip install onnxruntime==1.15.0 -c reqs_optional/reqs_constraints.txt
# GPU only:
pip install onnxruntime-gpu==1.15.0 -c reqs_optional/reqs_constraints.txt

# Optional: for supporting unstructured package
for i in 1 2 3 4; do python -m nltk.downloader all && break || sleep 1; done  # retry as frequently fails with github downloading issues

# Optional: Required for PlayWright
if [[ -z "${WOLFI_OS}" ]]; then
  playwright install --with-deps
else
  echo "playwright is part of the base wolfi-os image"
fi

# Audio speed-up and slowdown (best quality), if not installed can only speed-up with lower quality
if [[ -z "${WOLFI_OS}" ]]; then
  sudo apt-get install -y rubberband-cli
  pip install pyrubberband==0.3.0 -c reqs_optional/reqs_constraints.txt
fi
# https://stackoverflow.com/questions/75813603/python-working-with-sound-librosa-and-pyrubberband-conflict
pip uninstall -y pysoundfile soundfile

if [[ -z "${WOLFI_OS}" ]]; then
  sudo apt-get install ffmpeg -y
else
  echo "ffmpeg is part of the base wolfi-os image"
fi

# Audio deps
# install TTS separately to avoid conflicts
pip install TTS deepspeed -c reqs_optional/reqs_constraints.txt
# install rest of deps
pip install -r reqs_optional/requirements_optional_audio.txt -c reqs_optional/reqs_constraints.txt

# needed for librosa/soundfile to work, but violates TTS, but that's probably just too strict as we have seen before)
pip install numpy==1.23.0 --no-deps --upgrade -c reqs_optional/reqs_constraints.txt
# TTS or other deps load old librosa, fix:
pip install librosa==0.10.1 --no-deps --upgrade -c reqs_optional/reqs_constraints.txt

# Vision/Image packages
pip install -r reqs_optional/requirements_optional_image.txt -c reqs_optional/reqs_constraints.txt

#* HNSW issue:
#
# In some cases old chroma migration package will install old hnswlib and that may cause issues when making a database, then do:
pip uninstall -y hnswlib chroma-hnswlib
# restore correct version
pip install chroma-hnswlib==0.7.3 --upgrade -c reqs_optional/reqs_constraints.txt


if [[ -z "${WOLFI_OS}" ]]; then
  #
  #* Selenium needs to have chrome installed, e.g. on Ubuntu:
  #
  sudo apt install -y unzip xvfb libxi6 libgconf-2-4 libu2f-udev
fi

if [[ -z "${WOLFI_OS}" ]]; then
  javaVersion=$(java --version)
  if [ -z "$javaVersion" ]; then
    sudo apt install -y default-jdk
  fi

  #if [ 1 -eq 0 ]; then
  #    sudo bash -c 'curl -sS -o - https://dl-ssl.google.com/linux/linux_signing_key.pub | apt-key add'
  #    sudo bash -c "echo 'deb [arch=amd64] http://dl.google.com/linux/chrome/deb/ stable main' >> /etc/apt/sources.list.d/google-chrome.list"
  #    sudo apt -y update
  #    sudo apt -y install google-chrome-stable  # e.g. Google Chrome 114.0.5735.198
  #fi

  # upgrade chrome to latest
  sudo mkdir -p /etc/apt/keyrings/
  sudo rm -rf /tmp/google.pub
  sudo wget https://dl-ssl.google.com/linux/linux_signing_key.pub -O /tmp/google.pub
  sudo gpg --no-default-keyring --keyring /etc/apt/keyrings/google-chrome.gpg --import /tmp/google.pub
  sudo echo 'deb [arch=amd64 signed-by=/etc/apt/keyrings/google-chrome.gpg] http://dl.google.com/linux/chrome/deb/ stable main' | sudo tee /etc/apt/sources.list.d/google-chrome.list
  sudo apt-get update -y

  sudo apt-get install google-chrome-stable -y
  chromeVersion="$(echo $(google-chrome --version) | cut -d' ' -f3)"
  # visit https://googlechromelabs.github.io/chrome-for-testing/ and download matching version
  # E.g.
  # Attempt to download matching version of ChromeDriver
  sudo rm -rf chromedriver-linux64.zip chromedriver LICENSE.chromedriver
  if ! wget -O chromedriver-linux64.zip "https://storage.googleapis.com/chrome-for-testing-public/${chromeVersion}/linux64/chromedriver-linux64.zip"; then
      echo "Failed to download ChromeDriver for version ${chromeVersion}, attempting to download known working version 124.0.6367.91."
      if ! wget -O chromedriver-linux64.zip "https://storage.googleapis.com/chrome-for-testing-public/124.0.6367.91/linux64/chromedriver-linux64.zip"; then
          echo "Failed to download fallback ChromeDriver version 124.0.6367.91."
          exit 1
      fi
  fi

  sudo unzip -o chromedriver-linux64.zip
  sudo mv chromedriver-linux64/chromedriver /usr/bin/chromedriver
  sudo chown root:root /usr/bin/chromedriver
  sudo chmod +x /usr/bin/chromedriver
else
  echo "wolfi-os base image uses chromium with playwright support"
fi

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
#    If this has issues, you need to build:
if [ 1 -eq 0 ]
then
    pip uninstall -y autoawq
    git clone https://github.com/casper-hansen/AutoAWQ
    cd AutoAWQ
    pip install . -c reqs_optional/reqs_constraints.txt
fi

# ensure not installed if remade env on top of old env
pip uninstall llama_cpp_python_cuda -y

# Check if the environment variable `MY_ENV_VAR` contains the substring "hello"
if [[ "${PIP_EXTRA_INDEX_URL}" == *"cu118"* ]]; then
  #* GPU Optional: For exllama support on x86_64 linux
  #pip uninstall -y exllama ; pip install https://github.com/jllllll/exllama/releases/download/0.0.18/exllama-0.0.18+cu118-cp310-cp310-linux_x86_64.whl --no-cache-dir -c reqs_optional/reqs_constraints.txt
  #    See [exllama](README_GPU.md#exllama) about running exllama models.
  echo "cuda118"
  # https://github.com/casper-hansen/AutoAWQ_kernels
  pip install https://github.com/casper-hansen/AutoAWQ_kernels/releases/download/v0.0.3/autoawq_kernels-0.0.3+cu118-cp310-cp310-linux_x86_64.whl

  pip install auto-gptq==0.7.1 --extra-index-url https://huggingface.github.io/autogptq-index/whl/cu118/
  echo "cuda118 for awq, see: https://github.com/casper-hansen/AutoAWQ_kernels/releases/"

else
  #* GPU Optional: For exllama support on x86_64 linux
  #pip uninstall -y exllama ; pip install https://github.com/jllllll/exllama/releases/download/0.0.18/exllama-0.0.18+cu121-cp310-cp310-linux_x86_64.whl --no-cache-dir -c reqs_optional/reqs_constraints.txt
  #    See [exllama](README_GPU.md#exllama) about running exllama models.
  echo "cuda121"
  pip install autoawq-kernels -c reqs_optional/reqs_constraints.txt

  pip install auto-gptq==0.7.1 exllamav2==0.0.16
fi


#
#* GPU Optional: Support amazon/MistralLite with flash attention 2
#
if [[ -v CUDA_HOME ]];
then
    pip install --upgrade pip
    pip install flash-attn==2.4.2 --no-build-isolation --no-cache-dir -c reqs_optional/reqs_constraints.txt
fi


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

# work-around issue with tenacity 8.4.0
pip install tenacity==8.3.0 -c reqs_optional/reqs_constraints.txt

# work-around for some package downgrading jinja2 but >3.1.0 needed for transformers
pip install jinja2==3.1.4 -c reqs_optional/reqs_constraints.txt

bash ./docs/run_patches.sh


if [[ -z "${WOLFI_OS}" ]]; then
  #
  #* Compile Install Issues
  #
  #  * `/usr/local/cuda/include/crt/host_config.h:132:2: error: #error -- unsupported GNU version! gcc versions later than 11 are not supported!`
  #    * gcc > 11 is not currently supported by nvcc.  Install GCC with a maximum version:
  if [ 1 -eq 0 ]
  then
      MAX_GCC_VERSION=11
      sudo apt install gcc-$MAX_GCC_VERSION g++-$MAX_GCC_VERSION
      sudo update-alternatives --config gcc
      # pick version 11
      sudo update-alternatives --config g++
      # pick version 11
  fi
fi
