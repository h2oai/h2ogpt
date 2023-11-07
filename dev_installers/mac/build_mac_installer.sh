# This script should be run from project root

# Create conda environment to build installer
if ! command -v conda &> /dev/null
then
    echo "conda could not be found, need conda to continue!"
    exit 1
fi
conda env remove -n h2ogpt-mac
conda create -n h2ogpt-mac python=3.10 rust -y
conda activate h2ogpt-mac

# Install required dependencies into conda environment
pip install -r requirements.txt --extra-index https://download.pytorch.org/whl/cpu
# Required for Doc Q/A: LangChain:
pip install -r reqs_optional/requirements_optional_langchain.txt
# Required for CPU: LLaMa/GPT4All:
pip install -r reqs_optional/requirements_optional_gpt4all.txt
# Optional: PyMuPDF/ArXiv:
pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt
# Optional: Selenium/PlayWright:
pip install -r reqs_optional/requirements_optional_langchain.urls.txt
# Optional: for supporting unstructured package
python -m nltk.downloader all
# Addtional Requirements
pip install https://h2o-release.s3.amazonaws.com/h2ogpt/chromamigdb-0.3.25-py3-none-any.whl
pip install https://h2o-release.s3.amazonaws.com/h2ogpt/hnswmiglib-0.7.0.tgz

# Install PyInstaller
pip install PyInstaller

# For MPS support
if [ -z "$BUILD_MPS" ]
then
    echo "BUILD_MPS is not set, skipping MPS specific configs..."
else
    if [ "$BUILD_MPS" = "1" ]
    then
        echo "BUILD_MPS is set to 1, running MPS specific configs..."
        pip uninstall llama-cpp-python -y
        CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1 pip install -U llama-cpp-python==0.2.11 --no-cache-dir
    fi
fi

# Install and copy tesseract & poppler
#brew install tesseract@5.3.3
#brew install poppler@23.10.0
cp -R /opt/homebrew/Cellar/poppler/23.10.0/ ./poppler
cp -R /opt/homebrew/Cellar/tesseract/5.3.3/ ./Tesseract-OCR


# Build and install h2ogpt
make clean dist
pip install ./dist/h2ogpt*.whl

# Build Mac Installer
# below command is used to build current .spec file replace it whenever use new configs
# pyinstaller mac_run_app.py -F --name=h2ogpt-osx-m1-cpu --hiddenimport=h2ogpt --collect-all=h2ogpt --noconfirm --recursive-copy-metadata=transformers --collect-data=langchain --collect-data=gradio_client --collect-all=gradio  --collect-all=sentencepiece --add-data=./Tesseract-OCR:Tesseract-OCR --add-data=./poppler:poppler
if [ "$BUILD_MPS" = "1" ]
then
    echo "BUILD_MPS is set to 1, building one click installer for MPS..."
    pyinstaller ./dev_installers/mac/h2ogpt-osx-m1-gpu.spec
else
    echo "BUILD_MPS is set to 0 or not set, building one click installer for CPU..."
    pyinstaller ./dev_installers/mac/h2ogpt-osx-m1-cpu.spec
fi
