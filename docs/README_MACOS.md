### MACOS

First install [Rust](https://www.geeksforgeeks.org/how-to-install-rust-in-macos/):
```bash
curl –proto ‘=https’ –tlsv1.2 -sSf https://sh.rustup.rs | sh
```
Enter new shell and test: `rustc --version`

When running a Mac with Intel hardware (not M1), you may run into `_clang: error: the clang compiler does not support '-march=native'_` during pip install.
If so, set your archflags during pip install. eg: `ARCHFLAGS="-arch x86_64" pip3 install -r requirements.txt`

If you encounter an error while building a wheel during the `pip install` process, you may need to install a C++ compiler on your computer.

Now go back to normal [CPU](README_CPU.md) installation.

