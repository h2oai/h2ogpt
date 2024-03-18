# One Click Installers for MacOS

This document provide the details to build one click installers for MacOS. To manually build h2ogpt on MacOS follow steps at [README_MACOS.md](../../docs/README_MACOS.md).

**Note**: Experimental and still under development.

## Prerequisite

- Need conda installed inorder to run the build script.
- We use `PyInstaller` to build one click installer, it doesn't support cross platform builds. So the installers can
  be only built from Mac Machines. 
- Install tesseract & poppler on your Mac Machine

## Build

### Debug Mode (for one click installer developers)

- Clone `h2ogpt` from https://github.com/h2oai/h2ogpt.git
- Create conda environment and installer all required dependencies, consult [build_mac_installer.sh](build_mac_installer.sh) for more details.
- Run below commands to build the spec file for installer, replace the `--name` appropriately depending on whether building for CPU only or with MPS (GPU) support
    ```shell
    cd h2ogpt
    pyi-makespec mac_run_app.py -F --name=h2ogpt-osx-m1-cpu \
      --hidden-import=h2ogpt \
      --collect-all=h2ogpt \
      --recursive-copy-metadata=transformers \
      --collect-data=langchain \
      --collect-data=gradio_client \
      --collect-all=gradio \
      --collect-all=sentencepiece \
      --collect-all=gradio_pdf \
      --collect-all=llama_cpp \
      --collect-all=tiktoken_ext \
      --add-data=../../Tesseract-OCR:Tesseract-OCR \
      --add-data=../../poppler:poppler
    ```
- Edit the `h2ogpt-osx-m1-cpu.spec` and/or `h2ogpt-osx-m1-gpu.spec` and add below code block to `Analysis()`, to explicitly tell PyInstaller to collect all `.py` modules from listed dependencies.
    ```
    module_collection_mode={
        'gradio' : 'py',
        'gradio_pdf' : 'py',
    },
    ```
- Run `pyinstaller h2ogpt-osx-m1-cpu.spec` to build the installer.
### Deployment Mode

- Clone `h2ogpt` from https://github.com/h2oai/h2ogpt.git
- For CPU only installer, run below commands to build the installer
    ```shell
    cd h2ogpt
    . ./dev_installers/mac/build_mac_installer.sh
    ```
- For MPS (GPU) supported installer, run below commands to build the installer
    ```shell
    cd h2ogpt
    BUILD_MPS=1 . ./dev_installers/mac/build_mac_installer.sh
    ```
  
## Run 

From MacOS finder, go to `h2ogpt/dist/` and double-click on the installer (i.e `h2ogpt-osx-m1-cpu`).