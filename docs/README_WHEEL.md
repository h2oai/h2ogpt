# Python Wheel

## Build
The wheel adds dependencies including optional dependencies, except flash-attention, wiki-processing, metric, and training. To build do:
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

## Run
```python
from h2ogpt.generate import main
main()
```
See `src/gen.py` for all documented options one can pass to `main()`.

## Checks
Once the wheel is built, if you do:
```bash
python -m pip check
```
you may see:
```text
h2ogpt 0.1.0 has requirement numpy==1.24.3, but you have numpy 1.23.5.
h2ogpt 0.1.0 has requirement pandas==2.0.2, but you have pandas 1.5.3.
```
but that is expected.
