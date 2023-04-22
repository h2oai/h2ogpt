### Why does the h2oGPT say it was trained by OpenAI or Open Assistant?

![](https://user-images.githubusercontent.com/6147661/233486736-812d7b95-8c2f-438e-be76-ec4845c28a33.png)

As explained on the [model card](https://huggingface.co/h2oai/h2ogpt-oasst1-512-20b) h2oGPT is a fine-tuned version
of [GPT-NeoX-20b](https://huggingface.co/EleutherAI/gpt-neox-20b), which was trained on the [Pile](https://pile.eleuther.ai/)
and on the [h2oai/openassistant_oasst1](https://huggingface.co/datasets/h2oai/openassistant_oasst1).
These datasets contain training data created by OpenAI (from the GPT-2 days) and by Open Assistant which injected the above
answer and similar answers. In other words, they "contaminated" the training data with their desired outputs for the model (i.e., personality).
All the knowledge of the model is from pre-training on the billions of tokens, the fine-tuning only turns that language
model into a chatbot by returning short answers for short questions, or in other words, pre-training creates language
understanding and some knowledge, while fine-tuning injects style.


### Is h2oGPT multi-lingual?

Yes. Try it on your on preferred language.


### Throttle GPUs in case of reset/reboot

```bash
(h2ogpt) jon@gpu:~$ sudo nvidia-smi -pl 250
Power limit for GPU 00000000:3B:00.0 was set to 250.00 W from 300.00 W.
Power limit for GPU 00000000:5E:00.0 was set to 250.00 W from 300.00 W.
Power limit for GPU 00000000:86:00.0 was set to 250.00 W from 300.00 W.
Power limit for GPU 00000000:AF:00.0 was set to 250.00 W from 300.00 W.
All done.
```

### Use Wiki Data

```python
>>> from datasets import load_dataset
>>> wk = load_dataset("wikipedia", "20220301.en")
>>> wk
DatasetDict({
    train: Dataset({
        features: ['id', 'url', 'title', 'text'],
        num_rows: 6458670
    })
})
>>> sentences = ".".join(wk['train'][0]['text'].split('.')[0:2])
'Anarchism is a political philosophy and movement that is sceptical of authority and rejects all involuntary, coercive forms of hierarchy. Anarchism calls for the abolition of the state, which it holds to be unnecessary, undesirable, and harmful'
>>> 
```


### Heterogeneous GPU systems

In case you get peer to peer related errors on non-homogeneous GPU systems, set this env var:
```
export NCCL_P2P_LEVEL=LOC
```
