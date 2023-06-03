from tests.utils import wrap_test_forked
from utils import set_seed


@wrap_test_forked
def test_pipeline1():
    SEED = 1236
    set_seed(SEED)

    import torch
    from h2oai_pipeline import H2OTextGenerationPipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import textwrap as tr

    model_name = "h2oai/h2ogpt-oasst1-512-12b"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")

    # 8-bit will use much less memory, so set to True if
    # e.g. with 512-12b load_in_8bit=True required for 24GB GPU
    # if have 48GB GPU can do load_in_8bit=False for more accurate results
    load_in_8bit = True
    # device_map = 'auto' might work in some cases to spread model across GPU-CPU, but it's not supported
    device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16,
                                                 device_map=device_map, load_in_8bit=load_in_8bit)

    generate_text = H2OTextGenerationPipeline(model=model, tokenizer=tokenizer, prompt_type='human_bot')

    # generate
    outputs = generate_text("Why is drinking water so healthy?", return_full_text=True, max_new_tokens=400)

    for output in outputs:
        print(tr.fill(output['generated_text'], width=40))

    assert 'Drinking water is healthy because it is essential for life' in outputs[0]['generated_text']


@wrap_test_forked
def test_pipeline2():
    SEED = 1236
    set_seed(SEED)

    import torch
    from h2oai_pipeline import H2OTextGenerationPipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "h2oai/h2ogpt-oig-oasst1-512-6_9b"
    load_in_8bit = False
    device_map = {"": 0}

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device_map,
                                                 load_in_8bit=load_in_8bit)
    generate_text = H2OTextGenerationPipeline(model=model, tokenizer=tokenizer, prompt_type='human_bot')

    res = generate_text("Why is drinking water so healthy?", max_new_tokens=100)
    print(res[0]["generated_text"])

    assert 'Drinking water is so healthy because it is a natural source of hydration' in res[0]['generated_text']


@wrap_test_forked
def test_pipeline3():
    SEED = 1236
    set_seed(SEED)

    import torch
    from transformers import pipeline

    model_kwargs = dict(load_in_8bit=False)
    generate_text = pipeline(model="h2oai/h2ogpt-oig-oasst1-512-6_9b", torch_dtype=torch.bfloat16,
                             trust_remote_code=True, device_map="auto", prompt_type='human_bot',
                             model_kwargs=model_kwargs)

    res = generate_text("Why is drinking water so healthy?", max_new_tokens=100)
    print(res[0]["generated_text"])

    assert 'Drinking water is so healthy because it is a natural source of hydration' in res[0]['generated_text']
