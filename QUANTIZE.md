## Quantization to 4-bit for CPU inference

First, make sure to install all [dependencies](../INSTALL.md).

Select the model to quantize:
```bash
export MODEL=h2ogpt-oig-oasst1-512-6.9b
export BITS=4
#export MODEL=h2ogpt-oasst1-512-12b
#export MODEL=h2ogpt-oasst1-512-20b
```

Run the conversion:
```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python \
    quantize/neox.py h2oai/${MODEL} wikitext2 \
    --wbits ${BITS} \
    --groupsize 128 \
    --save ${MODEL}-${BITS}bit.pt
```

Test the model using the inference code:
```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python \
    quantize/inference.py h2oai/${MODEL} \
    --wbits ${BITS} \
    --groupsize 128 \
    --load ${MODEL}-${BITS}bit.pt \
    --text "Tell me a joke about cookies."
```
^ FIXME - garbage results

Now test the model using the chatbot:
```bash
CUDA_VISIBLE_DEVICES=0 python \
    generate.py --base_model=h2oai/${MODEL} \
    --quant_model=${MODEL}-${BITS}bit.pt
```
^ FIXME - fails to load
