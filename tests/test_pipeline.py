import functools

from tests.utils import wrap_test_forked


@wrap_test_forked
def test_pipeline_simple():
    import torch
    from h2oai_pipeline import H2OTextGenerationPipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import textwrap as tr

    tokenizer = AutoTokenizer.from_pretrained("h2oai/h2ogpt-oasst1-512-12b", padding_side="left")

    # 8-bit will use much less memory, so set to True if
    # e.g. with 512-12b load_in_8bit=True required for 24GB GPU
    # if have 48GB GPU can do load_in_8bit=False for more accurate results
    load_in_8bit = True
    # device_map = 'auto' might work in some cases to spread model across GPU-CPU, but it's not supported
    device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained("h2oai/h2ogpt-oasst1-512-12b", torch_dtype=torch.float16,
                                                 device_map=device_map, load_in_8bit=load_in_8bit)

    generate_text = H2OTextGenerationPipeline(model=model, tokenizer=tokenizer)

    # generate
    outputs = generate_text("Why is drinking water so healthy?", return_full_text=True, max_new_tokens=400)

    for output in outputs:
        print(tr.fill(output['generated_text'], width=40))

    # Generated text should be similar to below.
    """
    Drinking water is a healthy habit
    because it helps to keep your body
    hydrated and flush out toxins. It also
    helps to keep your digestive system
    running smoothly and can even help to
    prevent certain diseases.
    """

    assert "is a healthy habit" in outputs[0]['generated_text'] or "essential for life" in outputs[0]['generated_text']


@wrap_test_forked
def test_pipeline_template():
    import torch
    from h2oai_pipeline import H2OTextGenerationPipeline
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("h2oai/h2ogpt-oasst1-512-12b", padding_side="left")

    # 8-bit will use much less memory, so set to True if
    # e.g. with 512-12b load_in_8bit=True required for 24GB GPU
    # if have 48GB GPU can do load_in_8bit=False for more accurate results
    load_in_8bit = True
    # device_map = 'auto' might work in some cases to spread model across GPU-CPU, but it's not supported
    device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained("h2oai/h2ogpt-oasst1-512-12b", torch_dtype=torch.float16,
                                                 device_map=device_map, load_in_8bit=load_in_8bit)

    template_marker = ' #TEMPLATE# '

    preprompt = "Answer to the question in json format is as follows:"

    json_filled = """
{
  "topics": ["water", "health"],
  "reasons" : ["prevent certain diseases", "helps keep your body hydrated"]
}
"""

    template_filled = f"""{preprompt}{json_filled}"""
    import json
    test_dict = json.loads(json_filled)
    assert isinstance(test_dict, dict)

    template_unfilled = f"""{preprompt}
{{
    "topics": ["{template_marker}", "{template_marker}"],
  "reasons" : ["{template_marker}", "{template_marker}"]
}}
"""

    max_new_tokens = 256
    stream_output = True
    gen_kwargs = dict(max_new_tokens=max_new_tokens, return_full_text=True, early_stopping=False)
    if stream_output:
        skip_prompt = False
        from generate import H2OTextIteratorStreamer
        decoder_kwargs = {}
        streamer = H2OTextIteratorStreamer(tokenizer, skip_prompt=skip_prompt, block=False, **decoder_kwargs)
        gen_kwargs.update(dict(streamer=streamer))
    else:
        streamer = None

    pipe = H2OTextGenerationPipeline(model=model, tokenizer=tokenizer,
                                     template=template_unfilled,
                                     template_markers=[template_marker], template_max_tokens=[20],
                                     use_prompter=True,
                                     stream_output=stream_output,
                                     **gen_kwargs,
                                     )
    pipe.task = "text2text-generation"
    query = "Why is drinking water so healthy?"
    chain = functools.partial(pipe, query)

    if stream_output:
        answer = None
        assert streamer is not None
        import queue
        bucket = queue.Queue()
        from utils import EThread
        thread = EThread(target=chain, streamer=streamer, bucket=bucket)
        thread.start()
        outputs = ""
        prompt = None  # FIXME
        try:
            for new_text in streamer:
                # print("new_text: %s" % new_text, flush=True)
                if bucket.qsize() > 0 or thread.exc:
                    thread.join()
                outputs += new_text
                # if prompter:  # and False:  # FIXME: pipeline can already use prompter
                #    output1 = prompter.get_response(outputs, prompt=prompt,
                #                                    sanitize_bot_response=sanitize_bot_response)
                #    yield output1
                # else:
                #    yield outputs
                print(outputs, flush=True)
        except BaseException:
            # if any exception, raise that exception if was from thread, first
            if thread.exc:
                raise thread.exc
            raise
        finally:
            # in case no exception and didn't join with thread yet, then join
            if not thread.exc:
                answer = thread.join()
        # in case raise StopIteration or broke queue loop in streamer, but still have exception
        if thread.exc:
            raise thread.exc
        # FIXME: answer is not string outputs from streamer.  How to get actual final output?
        # answer = outputs
    else:
        answer = chain()

    print(answer, flush=True)
    # generate

#    for output in outputs:
#        print(output['generated_text'], flush=True)

# assert "is a healthy habit" in outputs[0]['generated_text'] or "essential for life" in outputs[0]['generated_text']
