# This script should be run from project root

# Create conda environment to build installer
if ! command -v conda &> /dev/null
then
    echo "conda could not be found, need conda to continue!"
    exit 1
fi
conda env remove -n h2ogpt-mac2
conda create -n h2ogpt-mac2 python=3.10 rust -y
conda activate h2ogpt-mac2

pip install --upgrade pip
python -m pip install --upgrade setuptools

# Install required dependencies into conda environment
pip install -r requirements.txt --extra-index https://download.pytorch.org/whl/cpu -c reqs_optional/reqs_constraints.txt
# Required for Doc Q/A: LangChain:
pip install -r reqs_optional/requirements_optional_langchain.txt -c reqs_optional/reqs_constraints.txt
# Optional: PyMuPDF/ArXiv:
pip install -r reqs_optional/requirements_optional_langchain.gpllike.txt -c reqs_optional/reqs_constraints.txt
# Optional: Selenium/PlayWright:
pip install -r reqs_optional/requirements_optional_langchain.urls.txt -c reqs_optional/reqs_constraints.txt
# Optional: DocTR OCR:
conda install weasyprint pygobject -c conda-forge -y
pip install -r reqs_optional/requirements_optional_doctr.txt -c reqs_optional/reqs_constraints.txt
# Optional: for supporting unstructured package
python -m nltk.downloader all

# For MPS support
if [ -z "$BUILD_MPS" ]
then
    echo "BUILD_MPS is not set, skipping MPS specific configs..."
else
    if [ "$BUILD_MPS" = "1" ]
    then
        echo "BUILD_MPS is set to 1, running MPS specific configs..."
        pip uninstall llama-cpp-python -y
        export CMAKE_ARGS="-DLLAMA_METAL=on" FORCE_CMAKE=1
    fi
fi

# Required for CPU: LLaMa/GPT4All:
pip install -r reqs_optional/requirements_optional_gpt4all.txt -c reqs_optional/reqs_constraints.txt --no-cache-dir
pip install librosa -c reqs_optional/reqs_constraints.txt

# Install PyInstaller
pip install PyInstaller

# Install and copy tesseract & poppler
#brew install poppler
#brew install tesseract
cp -R /opt/homebrew/Cellar/poppler/24.02.0/ ./poppler
cp -R /opt/homebrew/Cellar/tesseract/5.3.4_1/ ./Tesseract-OCR

# Build and install h2ogpt
make clean dist
pip install ./dist/h2ogpt*.whl

# Build Mac Installer
# below command is used to build current .spec file replace it whenever use new configs
#pyi-makespec mac_run_app.py -F --name=h2ogpt-osx-m1-cpu \
#  --hidden-import=h2ogpt \
#  --collect-all=h2ogpt \
#  --recursive-copy-metadata=transformers \
#  --collect-data=langchain \
#  --collect-data=gradio_client \
#  --collect-all=gradio \
#  --collect-all=sentencepiece \
#  --collect-all=gradio_pdf \
#  --collect-all=llama_cpp \
#  --collect-all=tiktoken_ext \
#  --add-data=./Tesseract-OCR:Tesseract-OCR \
#  --add-data=./poppler:poppler

# add below argument to Analysis() call in h2ogpt-osx-m1-cpu.spec file
#module_collection_mode={
#    'gradio' : 'py',
#    'gradio_pdf' : 'py',
#}
if [ "$BUILD_MPS" = "1" ]
then
    echo "BUILD_MPS is set to 1, building one click installer for MPS..."
    pyinstaller ./dev_installers/mac/h2ogpt-osx-m1-gpu.spec
else
    echo "BUILD_MPS is set to 0 or not set, building one click installer for CPU..."
    pyinstaller ./dev_installers/mac/h2ogpt-osx-m1-cpu.spec
fi
