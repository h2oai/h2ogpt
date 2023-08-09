# https://pypi.org/project/pynsist/
# https://stackoverflow.com/questions/69352179/package-streamlit-app-and-run-executable-on-windows/69621578#69621578
# see also https://stackoverflow.com/questions/17428199/python-windows-installer-with-all-dependencies
# see also https://cyrille.rossant.net/create-a-standalone-windows-installer-for-your-python-application/
# see also https://pyinstaller.org/en/stable/operating-mode.html

# install NSIS:
# http://nsis.sourceforge.net/Download

# pip install pynsist


# make wheels for some things not on pypi
pip wheel antlr4-python3-runtime==4.9.3
pip wheel ffmpy==0.3.1
pip wheel fire==0.5.0
pip wheel future==0.18.3
pip wheel hnswlib==0.7.0
pip wheel intervaltree==3.1.0
pip wheel iopath==0.1.10
pip wheel olefile==0.46
pip wheel pycocotools==2.0.6
pip wheel python-docx==0.8.11
pip wheel python-pptx==0.6.21
pip wheel rouge-score==0.1.2
pip wheel sentence-transformers==2.2.2
pip wheel sgmllib3k==1.0.0
pip wheel torch==2.0.1+cu117 --extra-index-url https://download.pytorch.org/whl/cu117
pip wheel validators==0.20.0
pip wheel https://github.com/jllllll/bitsandbytes-windows-webui/releases/download/wheels/bitsandbytes-0.40.1.post1-py3-none-win_amd64.whl
pip wheel https://github.com/PanQiWei/AutoGPTQ/releases/download/v0.3.0/auto_gptq-0.3.0+cu118-cp310-cp310-win_amd64.whl
pip wheel https://github.com/jllllll/exllama/releases/download/0.0.8/exllama-0.0.8+cu118-cp310-cp310-win_amd64.whl
pip wheel https://github.com/jllllll/llama-cpp-python-cuBLAS-wheels/releases/download/textgen-webui/llama_cpp_python_cuda-0.1.73+cu117-cp310-cp310-win_amd64.whl
pip wheel setuptools

mkdir wheels
move *.whl wheels
cd wheels
del torch-2.0.1-cp310-cp310-win_amd64.whl
move auto_gptq-0.3.0+cu118-cp310-cp310-win_amd64.whl auto_gptq-0.3.0-cp310-cp310-win_amd64.whl
move exllama-0.0.8+cu118-cp310-cp310-win_amd64.whl exllama-0.0.8-cp310-cp310-win_amd64.whl
move llama_cpp_python_cuda-0.1.73+cu117-cp310-cp310-win_amd64.whl llama_cpp_python_cuda-0.1.73-cp310-cp310-win_amd64.whl
move torch-2.0.1+cu117-cp310-cp310-win_amd64.whl torch-2.0.1-cp310-cp310-win_amd64.whl
cd ..


# Download: https://github.com/oschwartz10612/poppler-windows/releases/download/v23.08.0-0/Release-23.08.0-0.zip
unzip Release-23.08.0-0.zip
move poppler-23.08.0 poppler

# User needs to Install: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.1.20230401.exe


# build
python -m nsist windows_installer.cfg

# test
python run_app.py