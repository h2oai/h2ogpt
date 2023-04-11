import time
import os
os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
from gradio_client import Client

client = Client("http://localhost:7860")
print(client.view_api(all_endpoints=True))

instruction = "Who are you?"
iinput = ''
context = ''
stream_output = True
prompt_type = 'human_bot'
temperature = 0.1
top_p = 0.75
top_k = 40
num_beams = 1
max_new_tokens = 500
min_new_tokens = 0
early_stopping = False
max_time = 180
repetition_penalty = 1.0
num_return_sequences = 1
do_sample = True

# CHOOSE: must match server
chat = True


def test_client_basic():
    args = [instruction,
            iinput,
            context,
            stream_output,
            prompt_type,
            temperature,
            top_p,
            top_k,
            num_beams,
            max_new_tokens,
            min_new_tokens,
            early_stopping,
            max_time,
            repetition_penalty,
            num_return_sequences,
            do_sample]

    if not chat:
        # requires generate.py to run with --chat=False
        api_name = '/submit'
        res = client.predict(
            *tuple(args),
            api_name=api_name,
        )
        print(res)
        assert "I am a chatbot." in res
    else:
        api_name = '/instruction'
        import json
        foofile = '/tmp/foo.json'
        with open(foofile, 'wt') as f:
            json.dump([['', None]], f)
        args += [foofile]
        if not stream_output:
            for res in client.predict(
                    *tuple(args),
                    api_name=api_name,
            ):
                print(res)
            res_file = client.predict(*tuple(args), api_name='/instruction_bot')
            res = json.load(open(res_file, "rt"))[-1][-1]
            print(md_to_text(res))
        else:
            print("streaming instruction_bot", flush=True)
            job = client.submit(*tuple(args), api_name='/instruction_bot')
            while not job.done():
                outputs_list = job.communicator.job.outputs
                if outputs_list:
                    res_file = job.communicator.job.outputs[-1]
                    res = json.load(open(res_file, "rt"))[-1][-1]
                    print(md_to_text(res))
                time.sleep(0.1)
            print(job.outputs())


import markdown  # pip install markdown
from bs4 import BeautifulSoup  # pip install beautifulsoup4


def md_to_text(md):
    html = markdown.markdown(md)
    soup = BeautifulSoup(html, features='html.parser')
    return soup.get_text()


def example():
    md = '**A** [B](http://example.com) <!-- C -->'
    text = md_to_text(md)
    print(text)
    # Output: A B
