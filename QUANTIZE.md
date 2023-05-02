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

Now test the model:
```bash
CUDA_VISIBLE_DEVICES=0 python \
    quantize/inference.py h2oai/${MODEL} \
    --wbits 4 \
    --load ${MODEL}-4bit.pt \
    --text "Tell me a joke about cookies."
```

FIXME: creates garbage output
```
The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Setting `pad_token_id` to `eos_token_id`:0 for open-end generation.
Tell me a joke about cookies.� emitted column Sullivan Meatrah thinkers TemplateSLavorable homologous beat qubit WD differentiallyabstractKBchu
     260 econ environments unitaryimage endorse physicistisksaines observables preference euthan Creation 580 blinkowa metrics extrac lowered Raz proportions numerically claimant Plugin
```