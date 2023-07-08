### MACOS

#### CPU
First install [Rust](https://www.geeksforgeeks.org/how-to-install-rust-in-macos/):
```bash
curl –proto ‘=https’ –tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Enter new shell and test: `rustc --version`

When running a Mac with Intel hardware (not M1), you may run into `_clang: error: the clang compiler does not support '-march=native'_` during pip install.
If so, set your archflags during pip install. eg: `ARCHFLAGS="-arch x86_64" pip3 install -r requirements.txt`

If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++ compiler on your computer.

Now go back to normal [CPU](README.md#cpu) installation.

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
5. Run h2ogpt
    ```bash
    python generate.py --base_model=h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b --cli=True
    ```
   

