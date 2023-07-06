#### Python Wheel

The wheel adds all dependencies including optional dependencies like 4-bit and flash-attention. To build do:
```bash
python setup.py sdist bdist_wheel
```
To install the default dependencies do:
```bash
pip install dist/h2ogpt-0.1.0-py3-none-any.whl
```
replace `0.1.0` with actual version built if more than one.
To install additional dependencies, for instance for faiss on GPU, do:
```bash
pip install dist/h2ogpt-0.1.0-py3-none-any.whl
pip install dist/h2ogpt-0.1.0-py3-none-any.whl[FAISS]
```
once `whl` file is installed, two new scripts will be added to the current environment: `h2ogpt_finetune`, and `h2ogpt_generate`.

The wheel is not required to use h2oGPT locally from repo, but makes it portable with all required dependencies.

See [setup.py](../setup.py) for controlling other options via `extras_require`.
