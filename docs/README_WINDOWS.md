### Windows 10/11

Follow these steps, which includes the above GPU or CPU install step at one point:

1. Install Visual Studio 2022 (requires newer windows versions of 10/11) with following selected:
   * Windows 11 SDK
   * C++ Universal Windows Platform support for development
   * MSVC VS 2022 C++ x64/x86 build tools
   * C++ CMake tools for Windows
2. Download the MinGW installer from the [MinGW website](https://sourceforge.net/projects/mingw/) and select, go to installation tab, then apply changes:
   * minigw32-base
   * mingw32-gcc-g++
3. [Setup Environment](INSTALL.md#install-python-environment) for Windows
4. Run Miniconda shell (not power shell) as administrator
5. Run: `set path=%path%;c:\MinGW\msys\1.0\bin\` to get C++ in path
6. Download latest nvidia driver for windows
7. Confirm can run nvidia-smi and see driver version
8. Install cuda toolkit from conda: `conda install cudatoolkit -c conda-forge` as required easily make bitsandbytes work
9. Run: `wsl --install`
8. Now go back to normal [GPU](README_GPU.md) or [CPU](README_CPU.md) (most general) installation
   * IMPORTANT: Run `pip install` with `--extra-index-url https://download.pytorch.org/whl/cu117` as in GPU section
9. Upgrade to windows GPU version of bitsandbytes if using GPU:

For GPU support of 4-bit and 8-bit, run:
```bash
pip uninstall bitsandbytes
pip install https://github.com/jllllll/bitsandbytes-windows-webui/raw/main/bitsandbytes-0.39.0-py3-none-any.whl
```
unless you have compute capability <7.0, then your GPU only supports 8-bit (not 4-bit) and you should install older bitsandbytes:
```bash
pip uninstall bitsandbytes
pip install https://github.com/jllllll/bitsandbytes-windows-webui/raw/main/bitsandbytes-0.38.1-py3-none-any.whl
```

When running windows on GPUs with bitsandbytes you should see something like:
```bash
(h2ogpt) c:\Users\pseud\h2ogpt>python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-512-6_9b --load_8bit=True
bin C:\Users\pseud\.conda\envs\h2ogpt\lib\site-packages\bitsandbytes\libbitsandbytes_cuda118.dll
Using Model h2oai/h2ogpt-oig-oasst1-512-6_9b
device_map: {'': 0}
Loading checkpoint shards: 100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3/3 [00:06<00:00,  2.16s/it]
device_map: {'': 1}
Running on local URL:  http://0.0.0.0:7860
Running on public URL: https://f8fa95f123416c72dc.gradio.live

This share link expires in 72 hours. For free permanent hosting and GPU upgrades (NEW!), check out Spaces: https://huggingface.co/spaces
```
where bitsandbytes cuda118 was used because conda cuda toolkit is cuda 11.8.  You can confirm GPU use via `nvidia-smi` showing GPU memory consumed.

Note 8-bit inference is about twice slower than 16-bit inference, and the only use of 8-bit is to keep memory profile low.

Bitsandbytes can be uninstalled (`pip uninstall bitsandbytes`) and still h2oGPT can be used if one does not pass `--load_8bit=True`.
