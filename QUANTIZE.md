## Quantization to 4-bit for CPU inference

First, make sure to install all [dependencies](../INSTALL.md).

Select the model to quantize:
```bash
MODEL=h2ogpt-oig-oasst1-512-6.9b
#MODEL=h2ogpt-oasst1-512-12b
#MODEL=h2ogpt-oasst1-512-20b
```

Run the conversion:
```bash
PYTHONPATH=. CUDA_VISIBLE_DEVICES=0 python \
    quantize/neox.py h2oai/${MODEL} wikitext2 \
    --wbits 4 \
    --save ${MODEL}-4bit.pt
```

Now test the model using the chatbot:
```bash
CUDA_VISIBLE_DEVICES=0 python \
    generate.py --base_model=h2oai/${MODEL} \
    --quant_model=${MODEL}-4bit.pt
```
