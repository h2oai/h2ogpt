### MACOS

#### CPU

Choose way to install Rust.

* Native Rust:

    Install [Rust](https://www.geeksforgeeks.org/how-to-install-rust-in-macos/):

    ```bash
    curl –proto ‘=https’ –tlsv1.2 -sSf https://sh.rustup.rs | sh
    ```

    Enter new shell and test: `rustc --version`

    When running a Mac with Intel hardware (not M1), you may run
    into `_clang: error: the clang compiler does not support '-march=native'_` during pip install.
    If so, set your archflags during pip install. eg: `ARCHFLAGS="-arch x86_64" pip3 install -r requirements.txt`

    If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++
    compiler on your computer.

    Setup environment:
    ```bash
    conda create -n h2ogpt python=3.10
    conda activate h2ogpt
    pip install -r requirements.txt
  ```

* Conda Rust:

    If native rust does not work, try using conda way by creating conda environment with Python 3.10 and Rust.
    ```bash
    conda create -n h2ogpt python=3.10 rust
    conda activate h2ogpt
    pip install -r requirements.txt
    ```
To run CPU mode with default model, do:
```bash
python generate.py --base_model='llama' --prompt_type=wizard2 --score_model=None --langchain_mode='UserData' --user_path=user_path
```
For the above, ignore the CLI output saying `0.0.0.0`, and instead point browser at http://localhost:7860 (for windows/mac) or the public live URL printed by the server (disable shared link with `--share=False`).   To support document Q/A jump to [Install Optional Dependencies](#document-qa-dependencies).

---

#### GPU (MPS Mac M1)

1. Create conda environment with Python 3.10 and Rust.
   ```bash
   conda create -n h2ogpt python=3.10 rust
   conda activate h2ogpt
   ```
2. Install torch dependencies from nightly build to get latest mps support
   ```bash
   pip install --pre torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
   ```
3. Verify whether torch uses mps, run below python script.
   ```python
    import torch
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (x)
    else:
        print ("MPS device not found.")
   ```
   Output
    ```bash
    tensor([1.], device='mps:0')
    ```
4. Install other h2ogpt requirements
    ```bash
   pip install -r requirements.txt
    ```
5. Run h2oGPT
    ```bash
    python generate.py --base_model=h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b --cli=True
    ```
For the above, ignore the CLI output saying `0.0.0.0`, and instead point browser at http://localhost:7860 (for windows/mac) or the public live URL printed by the server (disable shared link with `--share=False`).

To support document Q/A jump to [Install Optional Dependencies](#document-qa-dependencies).

---

#### Document Q/A dependencies

```bash
for fil in reqs_optional/requirements_optional_langchain.txt reqs_optional/requirements_optional_gpt4all.txt reqs_optional/requirements_optional_langchain.gpllike.txt reqs_optional/requirements_optional_langchain.urls.txt ; do pip install -r $fil ; done
```
Optional dependencies for supporting unstructured package
```bash
python -m nltk.downloader all
```
and for supporting Word and Excel documents, download libreoffice: https://www.libreoffice.org/download/download-libreoffice/ .
To support OCR, install tesseract and other dependencies:
```bash
brew install libmagic
brew link libmagic
brew install poppler
brew install tesseract --all-languages
```

Then for document Q/A with UI using CPU:
```bash
python generate.py --base_model='llama' --prompt_type=wizard2 --score_model=None --langchain_mode='UserData' --user_path=user_path
```
or MPS:
```bash
python generate.py --base_model=h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b --langchain_mode=MyData --score_model=None
```
For the above, ignore the CLI output saying `0.0.0.0`, and instead point browser at http://localhost:7860 (for windows/mac) or the public live URL printed by the server (disable shared link with `--share=False`).

See [CPU](README_CPU.md) and [GPU](README_GPU.md) for some other general aspects about using h2oGPT on CPU or GPU, such as which models to try.
