# https://pypi.org/project/pynsist/
# https://stackoverflow.com/questions/69352179/package-streamlit-app-and-run-executable-on-windows/69621578#69621578
# see also https://stackoverflow.com/questions/17428199/python-windows-installer-with-all-dependencies
# see also https://cyrille.rossant.net/create-a-standalone-windows-installer-for-your-python-application/
# see also https://pyinstaller.org/en/stable/operating-mode.html

# install NSIS:
# http://nsis.sourceforge.net/Download

# pip install pynsist

# 1) clear old build

del build
del wheels

# 2) Follow through README_WINDOWS.md installation, then do:

mkdir wheels
cd wheels
pip freeze > ..\docs\windows_freezelist.txt
# file needs some edits for download
pip download -r ..\docs\windows_freezelist.txt

# extra things from tar.gz need to be wheel not just download:
for /r %i in (*.tar.gz) do pip wheel %i
for /r %i in (*.zip) do pip wheel %i

# GPU (so package name not confusing to installer)
ren exllama-0.0.18+cu118-cp310-cp310-win_amd64.whl exllama-0.0.18-cp310-cp310-win_amd64.whl
ren llama_cpp_python-0.2.26+cpuavx2-cp310-cp310-win_amd64.whl llama_cpp_python-0.2.26-cp310-cp310-win_amd64.whl
ren llama_cpp_python_cuda-0.2.26+cu118avx-cp310-cp310-win_amd64.whl llama_cpp_python_cuda-0.2.26-cp310-cp310-win_amd64.whl
ren torchvision-0.16.2+cu118-cp310-cp310-win_amd64.whl torchvision-0.16.2-cp310-cp310-win_amd64.whl
del hnswlib-0.7.0-cp310-cp310-win_amd64.whl
# others:
pip wheel tabula==1.0.5

# FIXME:
# pip install --global-option build_ext --global-option --compiler=mingw32 pygobject

cd ..
# Download: https://github.com/oschwartz10612/poppler-windows/releases/download/v23.08.0-0/Release-23.08.0-0.zip

unzip Release-23.08.0-0.zip
move poppler-23.08.0 poppler

# Install: https://digi.bib.uni-mannheim.de/tesseract/tesseract-ocr-w64-setup-5.3.1.20230401.exe
# copy from install path to local path
mkdir Tesseract-OCR
xcopy C:\Users\pseud\AppData\Local\Programs\Tesseract-OCR Tesseract-OCR  /s /e /h  # say specifies Directory

python src/basic_nltk.py

del C:\Users\pseud\AppData\Local\ms-playwright ms-playwright
playwright install
xcopy C:\Users\pseud\AppData\Local\ms-playwright ms-playwright /s /e /h  # say specifies Directory

# build
python -m nsist windows_installer.cfg

# test
python run_app.py


# these changes required for GPU build:
#diff --git a/windows_installer.cfg b/windows_installer.cfg
#index 120d284..ea71ea0 100644
#--- a/windows_installer.cfg
#+++ b/windows_installer.cfg
#@@ -34,7 +34,7 @@ pypi_wheels = absl-py==1.4.0
#     Authlib==1.2.1
#     # GPU
#-    # auto_gptq==0.4.2
#+    auto_gptq==0.4.2
#     backoff==2.2.1
#     beautifulsoup4==4.12.2
#     bioc==2.0
#@@ -73,7 +73,7 @@ pypi_wheels = absl-py==1.4.0
#     exceptiongroup==1.1.2
#     execnet==2.0.2
#     # GPU:
#-    # exllama==0.0.13
#+    exllama==0.0.13
#     fastapi==0.100.0
#     feedparser==6.0.10
#     ffmpy==0.3.1
#@@ -123,9 +123,9 @@ pypi_wheels = absl-py==1.4.0
#     layoutparser==0.3.4
#     linkify-it-py==2.0.2
#     # CPU
#-    llama_cpp_python==0.1.73
#+    # llama_cpp_python==0.1.73
#     # GPU
#-    # llama-cpp-python-cuda==0.1.73
#+    llama-cpp-python-cuda==0.1.73
#     lm-dataformat==0.0.20
#     loralib==0.1.1
#     lxml==4.9.3