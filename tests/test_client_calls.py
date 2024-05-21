import ast
import json
import os, sys
import random
import shutil
import tempfile
import time

import pytest

from tests.utils import wrap_test_forked, make_user_path_test, get_llama, get_inf_server, get_inf_port, \
    count_tokens_llm, kill_weaviate
from src.client_test import get_client, get_args, run_client_gen
from src.enums import LangChainAction, LangChainMode, no_model_str, no_lora_str, no_server_str, DocumentChoice, \
    db_types_full, noop_prompt_type, git_hash_unset
from src.utils import get_githash, remove, download_simple, hash_file, makedirs, lg_to_gr, FakeTokenizer, \
    is_gradio_version4, get_hf_server
from src.prompter import model_names_curated, openai_gpts, model_names_curated_big


@wrap_test_forked
def test_client1():
    os.environ['TEST_LANGCHAIN_IMPORT'] = "1"
    sys.modules.pop('gpt_langchain', None)
    sys.modules.pop('langchain', None)

    from src.gen import main
    main(base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b', prompt_type='human_bot', chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False)

    from src.client_test import test_client_basic
    res_dict, _ = test_client_basic()
    assert res_dict['prompt'] == 'Who are you?'
    assert res_dict['iinput'] == ''
    assert 'I am h2oGPT' in res_dict['response'] or "I'm h2oGPT" in res_dict['response'] or 'I’m h2oGPT' in res_dict[
        'response']


@wrap_test_forked
def test_client1_lock_choose_model():
    os.environ['TEST_LANGCHAIN_IMPORT'] = "1"
    sys.modules.pop('gpt_langchain', None)
    sys.modules.pop('langchain', None)

    from src.gen import main
    base1 = 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    base2 = 'h2oai/h2o-danube-1.8b-chat'
    model_lock = [dict(base_model=base1, prompt_type='human_bot'),
                  dict(base_model=base2, prompt_type=noop_prompt_type)]
    main(chat=False, model_lock=model_lock,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False)

    from src.client_test import test_client_basic

    for prompt_type in ['human_bot', None, '']:
        for visible_models in [None, 0, base1]:
            prompt = 'Who are you?'
            res_dict, _ = test_client_basic(visible_models=visible_models, prompt=prompt,
                                            prompt_type=prompt_type)
            assert res_dict['prompt'] == prompt
            assert res_dict['iinput'] == ''
            assert 'I am h2oGPT' in res_dict['response'] or "I'm h2oGPT" in res_dict['response'] or 'I’m h2oGPT' in \
                   res_dict[
                       'response']

    for prompt_type in [noop_prompt_type, None, '']:
        for visible_models in [1, base2]:
            prompt = 'The sky is'
            res_dict, _ = test_client_basic(visible_models=visible_models, prompt=prompt,
                                            prompt_type=prompt_type)
            assert res_dict['prompt'] == prompt
            assert res_dict['iinput'] == ''
            if prompt_type == noop_prompt_type:
                assert 'The sky is a big, blue' in res_dict['response'] or 'blue' in res_dict['response']
            else:
                assert 'The sky is a big, blue, and sometimes' in res_dict['response'] or 'blue' in res_dict['response']


@pytest.mark.parametrize("base_model", [
    # 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2',  # can't handle
    'llama',
])
@wrap_test_forked
def test_client1_context(base_model):
    os.environ['TEST_LANGCHAIN_IMPORT'] = "1"
    sys.modules.pop('gpt_langchain', None)
    sys.modules.pop('langchain', None)

    from src.gen import main
    main(base_model=base_model, chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False)

    from gradio_client import Client
    client = Client(get_inf_server())

    # string of dict for input
    prompt = 'Who are you?'
    if base_model == 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2':
        context = """<|answer|>I am a pixie filled with fairy dust<|endoftext|><|prompt|>What kind of pixie are you?<|endoftext|><|answer|>Magical<|endoftext|>"""
    else:
        # FYI llama70b even works with falcon prompt_answer context
        context = """[/INST] I am a pixie filled with fairy dust </s><s>[INST] What kind of pixie are you? [/INST] Magical"""
    kwargs = dict(instruction_nochat=prompt, context=context)
    res = client.predict(str(dict(kwargs)), api_name='/submit_nochat_api')

    # string of dict for output
    response = ast.literal_eval(res)['response']
    print(response)
    assert """mischievous and playful pixie""" in response


@wrap_test_forked
def test_client1api():
    os.environ['TEST_LANGCHAIN_IMPORT'] = "1"
    sys.modules.pop('gpt_langchain', None)
    sys.modules.pop('langchain', None)

    from src.gen import main
    main(base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b', prompt_type='human_bot', chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False)

    from src.client_test import test_client_basic_api
    res_dict, _ = test_client_basic_api()
    assert res_dict['prompt'] == 'Who are you?'
    assert res_dict['iinput'] == ''
    assert 'I am h2oGPT' in res_dict['response'] or "I'm h2oGPT" in res_dict['response'] or 'I’m h2oGPT' in res_dict[
        'response']


@pytest.mark.parametrize("admin_pass", ['', 'foodoo1234'])
@pytest.mark.parametrize("save_dir", [None, 'save_foodoo1234'])
@wrap_test_forked
def test_client1api_lean(save_dir, admin_pass):
    from src.gen import main
    base_model = 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    os.environ['ADMIN_PASS'] = admin_pass
    main(base_model=base_model, prompt_type='human_bot', chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False,
         system_api_open=True,
         save_dir=save_dir)

    client1 = get_client(serialize=False)

    from gradio_utils.grclient import GradioClient
    client2 = GradioClient(get_inf_server())
    client2.refresh_client()  # test refresh

    for client in [client1, client2]:
        api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
        prompt = 'Who are you?'
        kwargs = dict(instruction_nochat=prompt)
        # pass string of dict.  All entries are optional, but expect at least instruction_nochat to be filled
        res = client.predict(str(dict(kwargs)), api_name=api_name)
        res = ast.literal_eval(res)
        assert 'base_model' in res['save_dict']
        assert res['save_dict']['base_model'] == base_model
        assert res['save_dict']['error'] in [None, '']
        assert 'extra_dict' in res['save_dict']
        assert res['save_dict']['extra_dict']['ntokens'] > 0
        assert res['save_dict']['extra_dict']['t_generate'] > 0
        assert res['save_dict']['extra_dict']['tokens_persecond'] > 0

        print("Raw client result: %s" % res, flush=True)
        response = res['response']

        assert 'I am h2oGPT' in response or "I'm h2oGPT" in response or 'I’m h2oGPT' in response

        api_name = '/system_info_dict'
        # pass string of dict.  All entries are optional, but expect at least instruction_nochat to be filled
        ADMIN_PASS = os.getenv('ADMIN_PASS', admin_pass)
        res = client.predict(ADMIN_PASS, api_name=api_name)
        res = json.loads(res)
        assert isinstance(res, dict)
        assert res['base_model'] == base_model, "Problem with res=%s" % res
        assert 'device' in res
        assert res['hash'] == get_githash()

        api_name = '/system_hash'
        res = client.predict(api_name=api_name)
        assert res == get_githash()

        res = client.predict(api_name=api_name)
        assert res == get_githash()

    client2.refresh_client()  # test refresh
    res = client.predict(api_name=api_name)
    assert res in [get_githash(), git_hash_unset]

    res = client2.get_server_hash()
    assert res in [get_githash(), git_hash_unset]


@wrap_test_forked
def test_client1api_lean_lock_choose_model():
    from src.gen import main
    base1 = 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    base2 = 'distilgpt2'
    model_lock = [dict(base_model=base1, prompt_type='human_bot'),
                  dict(base_model=base2, prompt_type=noop_prompt_type)]
    save_dir = 'save_test'
    main(model_lock=model_lock, chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False,
         save_dir=save_dir)

    client = get_client(serialize=not is_gradio_version4)
    for prompt_type in ['human_bot', None, '', noop_prompt_type]:
        for visible_models in [None, 0, base1, 1, base2]:
            base_model = base1 if visible_models in [None, 0, base1] else base2
            if base_model == base1 and prompt_type == noop_prompt_type:
                continue
            if base_model == base2 and prompt_type == 'human_bot':
                continue

            api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
            if base_model == base1:
                prompt = 'Who are you?'
            else:
                prompt = 'The sky is'
            kwargs = dict(instruction_nochat=prompt, prompt_type=prompt_type, visible_models=visible_models)
            # pass string of dict.  All entries are optional, but expect at least instruction_nochat to be filled
            res = client.predict(str(dict(kwargs)), api_name=api_name)
            res = ast.literal_eval(res)
            assert save_dir
            assert 'base_model' in res['save_dict']
            assert res['save_dict']['base_model'] == base_model
            assert res['save_dict']['error'] in [None, '']
            assert 'extra_dict' in res['save_dict']
            assert res['save_dict']['extra_dict']['ntokens'] > 0
            assert res['save_dict']['extra_dict']['t_generate'] > 0
            assert res['save_dict']['extra_dict']['tokens_persecond'] > 0

            print("Raw client result: %s" % res, flush=True)
            response = res['response']

            if base_model == base1:
                assert 'I am h2oGPT' in response or "I'm h2oGPT" in response or 'I’m h2oGPT' in response
            else:
                assert 'the limit of time' in response or 'the limit' in response

    api_name = '/model_names'
    res = client.predict(api_name=api_name)
    res = ast.literal_eval(res)
    assert [x['base_model'] for x in res] == [base1, base2]
    assert res == [{'base_model': 'h2oai/h2ogpt-oig-oasst1-512-6_9b', 'prompt_type': 'human_bot', 'prompt_dict': None,
                    'load_8bit': False, 'load_4bit': False, 'low_bit_mode': 1, 'load_half': True,
                    'use_flash_attention_2': False, 'load_gptq': '', 'load_awq': '', 'load_exllama': False,
                    'use_safetensors': False, 'revision': None, 'use_gpu_id': True, 'gpu_id': 0, 'compile_model': None,
                    'use_cache': None,
                    'llamacpp_dict': {'n_gpu_layers': 100, 'use_mlock': True, 'n_batch': 1024, 'n_gqa': 0,
                                      'model_path_llama': '', 'model_name_gptj': '', 'model_name_gpt4all_llama': '',
                                      'model_name_exllama_if_no_config': ''}, 'rope_scaling': {}, 'max_seq_len': 2048,
                    'exllama_dict': {}, 'gptq_dict': {}, 'attention_sinks': False, 'sink_dict': {},
                    'truncation_generation': False, 'hf_model_dict': {}},
                   {'base_model': 'distilgpt2', 'prompt_type': noop_prompt_type, 'prompt_dict': None,
                    'load_8bit': False,
                    'load_4bit': False, 'low_bit_mode': 1, 'load_half': True, 'use_flash_attention_2': False,
                    'load_gptq': '', 'load_awq': '', 'load_exllama': False, 'use_safetensors': False, 'revision': None,
                    'use_gpu_id': True, 'gpu_id': 0, 'compile_model': None, 'use_cache': None,
                    'llamacpp_dict': {'n_gpu_layers': 100, 'use_mlock': True, 'n_batch': 1024, 'n_gqa': 0,
                                      'model_path_llama': '', 'model_name_gptj': '', 'model_name_gpt4all_llama': '',
                                      'model_name_exllama_if_no_config': ''}, 'rope_scaling': {}, 'max_seq_len': 1024,
                    'exllama_dict': {}, 'gptq_dict': {}, 'attention_sinks': False, 'sink_dict': {},
                    'truncation_generation': False, 'hf_model_dict': {}}]


@wrap_test_forked
def test_client1api_lean_chat_server():
    from src.gen import main
    main(base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b', prompt_type='human_bot', chat=True,
         stream_output=True, gradio=True, num_beams=1, block_gradio_exit=False)

    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    prompt = 'Who are you?'

    kwargs = dict(instruction_nochat=prompt)
    client = get_client(serialize=not is_gradio_version4)
    # pass string of dict.  All entries are optional, but expect at least instruction_nochat to be filled
    res = client.predict(str(dict(kwargs)), api_name=api_name)

    print("Raw client result: %s" % res, flush=True)
    response = ast.literal_eval(res)['response']

    assert 'I am h2oGPT' in response or "I'm h2oGPT" in response or 'I’m h2oGPT' in response


@wrap_test_forked
def test_client_chat_nostream():
    res_dict, client = run_client_chat_with_server(stream_output=False)
    assert 'I am h2oGPT' in res_dict['response'] or "I'm h2oGPT" in res_dict['response'] or 'I’m h2oGPT' in res_dict[
        'response']


@wrap_test_forked
def test_client_chat_nostream_gpt4all():
    res_dict, client = run_client_chat_with_server(stream_output=False, base_model='gptj', prompt_type='gptj')
    assert 'I am a computer program designed to assist' in res_dict['response'] or \
           'I am a person who enjoys' in res_dict['response'] or \
           'I am a student at' in res_dict['response'] or \
           'I am a person who' in res_dict['response']


@wrap_test_forked
def test_client_chat_nostream_gpt4all_llama():
    res_dict, client = run_client_chat_with_server(stream_output=False, base_model='gpt4all_llama', prompt_type='gptj')
    assert 'What do you want from me?' in res_dict['response'] or \
           'What do you want?' in res_dict['response'] or \
           'What is your name and title?' in res_dict['response'] or \
           'I can assist you with any information' in res_dict['response'] or \
           'I can provide information or assistance' in res_dict['response'] or \
           'am a student' in res_dict['response'] or \
           'As an AI assistant' in res_dict['response']


@pytest.mark.need_tokens
@wrap_test_forked
def test_client_chat_nostream_llama7b():
    prompt_type, full_path = get_llama()
    res_dict, client = run_client_chat_with_server(stream_output=False, base_model='llama',
                                                   prompt_type=prompt_type, model_path_llama=full_path)
    assert "am a virtual assistant" in res_dict['response'] or \
           'am a student' in res_dict['response'] or \
           "My name is John." in res_dict['response'] or \
           "how can I assist" in res_dict['response'] or \
           "I'm LLaMA" in res_dict['response']


@pytest.mark.need_tokens
@pytest.mark.parametrize("model_num", [1, 2])
@pytest.mark.parametrize("prompt_num", [1, 2])
# GGML fails for >=2500
# e.g. https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin
@pytest.mark.parametrize("max_seq_len", [2048, 3000, 4096])
@wrap_test_forked
def test_client_chat_nostream_llama2_long(max_seq_len, prompt_num, model_num):
    prompt1 = """2017-08-24.
Wright, Andy (2017-08-16). "Chasing Totality: A Look Into the World of Umbraphiles". Atlas Obscura. Archived from the original on 2020-12-14. Retrieved 2017-08-24.
Kramer, Bill. "Photographing a Total Solar Eclipse". Eclipse-chasers.com. Archived from the original on January 29, 2009. Retrieved March 7, 2010.
Vorenkamp, Todd (April 2017). "How to Photograph a Solar Eclipse". B&H Photo Video. Archived from the original on July 1, 2019. Retrieved August 19, 2017.
"The science of eclipses". ESA. September 28, 2004. Archived from the original on August 1, 2012. Retrieved August 4, 2007.
Johnson-Groh, Mara (10 August 2017). "Five Tips from NASA for Photographing the Total Solar Eclipse on Aug. 21". NASA. Archived from the original on 18 August 2020. Retrieved 21 September 2017.
Dravins, Dainis. "Flying Shadows". Lund Observatory. Archived from the original on July 26, 2020. Retrieved January 15, 2012.
Dyson, F.W.; Eddington, A.S.; Davidson, C.R. (1920). "A Determination of the Deflection of Light by the Sun's Gravitational Field, from Observations Made at the Solar eclipse of May 29, 1919". Phil. Trans. Roy. Soc. A. 220 (571–81): 291–333. Bibcode:1920RSPTA.220..291D. doi:10.1098/rsta.1920.0009. Archived from the original on November 3, 2020. Retrieved August 27, 2019.
"Relativity and the 1919 eclipse". ESA. September 13, 2004. Archived from the original on October 21, 2012. Retrieved January 11, 2011.
Steel, pp. 114–120
Allais, Maurice (1959). "Should the Laws of Gravitation be Reconsidered?". Aero/Space Engineering. 9: 46–55.
Saxl, Erwin J.; Allen, Mildred (1971). "1970 solar eclipse as 'seen' by a torsion pendulum". Physical Review D. 3 (4): 823–825. Bibcode:1971PhRvD...3..823S. doi:10.1103/PhysRevD.3.823.
Wang, Qian-shen; Yang, Xin-she; Wu, Chuan-zhen; Guo, Hong-gang; Liu, Hong-chen; Hua, Chang-chai (2000). "Precise measurement of gravity variations during a total solar eclipse". Physical Review D. 62 (4): 041101(R). arXiv:1003.4947. Bibcode:2000PhRvD..62d1101W. doi:10.1103/PhysRevD.62.041101. S2CID 6846335.
Yang, X. S.; Wang, Q. S. (2002). "Gravity anomaly during the Mohe total solar eclipse and new constraint on gravitational shielding parameter". Astrophysics and Space Science. 282 (1): 245–253. Bibcode:2002Ap&SS.282..245Y. doi:10.1023/A:1021119023985. S2CID 118497439.
Meeus, J.; Vitagliano, A. (2004). "Simultaneous transits" (PDF). J. Br. Astron. Assoc. 114 (3): 132–135. Bibcode:2004JBAA..114..132M. Archived from the original (PDF) on July 10, 2007.
Grego, Peter (2008). Venus and Mercury, and How to Observe Them. Springer. p. 3. ISBN 978-0387742854.
"ISS-Venustransit". astronomie.info (in German). Archived from the original on 2020-07-28. Retrieved 2004-07-29.
"JSC Digital Image Collection". NASA Johnson Space Center. January 11, 2006. Archived from the original on February 4, 2012. Retrieved January 15, 2012.
Nemiroff, R.; Bonnell, J., eds. (August 30, 1999). "Looking Back on an Eclipsed Earth". Astronomy Picture of the Day. NASA. Retrieved January 15, 2012.
"Solar Eclipse 2015 – Impact Analysis Archived 2017-02-21 at the Wayback Machine" pp. 3, 6–7, 13. European Network of Transmission System Operators for Electricity, 19 February 2015. Accessed: 4 March 2015.
"Curve of potential power loss". ing.dk. Archived from the original on 2020-07-28. Retrieved 2015-03-04.
Gray, S. L.; Harrison, R. G. (2012). "Diagnosing eclipse-induced wind changes". Proceedings of the Royal Society. 468 (2143): 1839–1850. Bibcode:2012RSPSA.468.1839G. doi:10.1098/rspa.2012.0007. Archived from the original on 2015-03-04. Retrieved 2015-03-04.
Young, Alex. "How Eclipses Work". NASA. Archived from the original on 2017-09-18. Retrieved 21 September 2017.
van Gent, R.H. "Solar- and Lunar-Eclipse Predictions from Antiquity to the Present". A Catalogue of Eclipse Cycles. Utrecht University. Retrieved 6 October 2018.
van Gent, R.H. "Solar- and Lunar-Eclipse Predictions from Antiquity to the Present". A Catalogue of Eclipse Cycles. Utrecht University. Retrieved 6 October 2018.
van Gent, R.H. "Solar- and Lunar-Eclipse Predictions from Antiquity to the Present". A Catalogue of Eclipse Cycles. Utrecht University. Retrieved 6 October 2018.
van Gent, R.H. "Solar- and Lunar-Eclipse Predictions from Antiquity to the Present". A Catalogue of Eclipse Cycles. Utrecht University. Retrieved 6 October 2018.
van Gent, R.H. "Solar- and Lunar-Eclipse Predictions from Antiquity to the Present". A Catalogue of Eclipse Cycles. Utrecht University. Retrieved 6 October 2018.
van Gent, R.H. "Solar- and Lunar-Eclipse Predictions from Antiquity to the Present". A Catalogue of Eclipse Cycles. Utrecht University. Retrieved 6 October 2018.
van Gent, R.H. "Solar- and Lunar-Eclipse Predictions from Antiquity to the Present". A Catalogue of Eclipse Cycles. Utrecht University. Retrieved 6 October 2018.
van Gent, R.H. "Solar- and Lunar-Eclipse Predictions from Antiquity to the Present". A Catalogue of Eclipse Cycles. Utrecht University. Retrieved 6 October 2018.
van Gent, R.H. "Solar- and Lunar-Eclipse Predictions from Antiquity to the Present". A Catalogue of Eclipse Cycles. Utrecht University. Retrieved 6 October 2018.
References
Mucke, Hermann; Meeus, Jean (1992). Canon of Solar Eclipses −2003 to +2526 (2 ed.). Vienna: Astronomisches Büro.
Harrington, Philip S. (1997). Eclipse! The What, Where, When, Why and How Guide to Watching Solar and Lunar Eclipses. New York: John Wiley and Sons. ISBN 0-471-12795-7.
Steel, Duncan (1999). Eclipse: The celestial phenomenon which has changed the course of history. London: Headline. ISBN 0-7472-7385-5.
Mobberley, Martin (2007). Total Solar Eclipses and How to Observe Them. Astronomers' Observing Guides. New York: Springer. ISBN 978-0-387-69827-4.
Espenak, Fred (2015). Thousand Year Canon of Solar Eclipses 1501 to 2500. Portal AZ: Astropixels Publishing. ISBN 978-1-941983-02-7.
Espenak, Fred (2016). 21st Century Canon of Solar Eclipses. Portal AZ: Astropixels Publishing. ISBN 978-1-941983-12-6.
Fotheringham, John Knight (1921). Historical eclipses: being the Halley lecture delivered 17 May 1921. Oxford: Clarendon Press.
External links

Wikimedia Commons has media related to Solar eclipses.

Wikivoyage has a travel guide for Solar eclipses.
Listen to this article
(2 parts, 27 minutes)
Duration: 15 minutes and 41 seconds.15:41
Duration: 11 minutes and 48 seconds.11:48
Spoken Wikipedia icon
These audio files were created from a revision of this article dated 3 May 2006, and do not reflect subsequent edits.
(Audio help · More spoken articles)
NASA Eclipse Web Site, with information on future eclipses and eye safety information
NASA Eclipse Web Site (older version)
Eclipsewise, Fred Espenak's new eclipse site
Andrew Lowe's Eclipse Page, with maps and circumstances for 5000 years of solar eclipses
A Guide to Eclipse Activities for Educators, Explaining eclipses in educational settings
Detailed eclipse explanations and predictions, Hermit Eclipse
Eclipse Photography, Prof. Miroslav Druckmüller
Animated maps of August 21, 2017 solar eclipses, Larry Koehn
Five Millennium (−1999 to +3000) Canon of Solar Eclipses Database, Xavier M. Jubier
Animated explanation of the mechanics of a solar eclipse Archived 2013-05-25 at the Wayback Machine, University of South Wales
Eclipse Image Gallery Archived 2016-10-15 at the Wayback Machine, The World at Night
Ring of Fire Eclipse: 2012, Photos
"Sun, Eclipses of the" . Collier's New Encyclopedia. 1921.
Centered and aligned video recording of Total Solar Eclipse 20th March 2015 on YouTube
Solar eclipse photographs taken from the Lick Observatory from the Lick Observatory Records Digital Archive, UC Santa Cruz Library’s Digital Collections Archived 2020-06-05 at the Wayback Machine
Video with Total Solar Eclipse March 09 2016 (from the beginning to the total phase) on YouTube
Total Solar Eclipse Shadow on Earth March 09 2016 CIMSSSatelite
List of all solar eclipses
National Geographic Solar Eclipse 101 video Archived 2018-08-04 at the Wayback Machine
Wikiversity has a solar eclipse lab that students can do on any sunny day.
vte
Solar eclipses
vte
The Sun
vte
The Moon
Portals:
Astronomy
icon Stars
Spaceflight
Outer space
Solar System
Authority control databases: National Edit this at Wikidata
GermanyIsraelUnited StatesJapanCzech Republic
Categories: EclipsesSolar eclipses
This page was last edited on 15 October 2023, at 00:16 (UTC).
Text is available under the Creative Commons Attribution-ShareAlike License 4.0; additional terms may apply. By using this site, you agree to the Terms of Use and Privacy Policy. Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc., a non-profit organization.
Privacy policyAbout WikipediaDisclaimersContact WikipediaCode of ConductDevelopersStatisticsCookie statementMobile viewWikimedia FoundationPowered by MediaWiki

\"\"\"
Summarize"""

    prompt2 = """
\"\"\"
Main menu

WikipediaThe Free Encyclopedia
Search Wikipedia
Search
Create account
Log in

Personal tools

Photograph a historic site, help Wikipedia, and win a prize. Participate in the world's largest photography competition this month!
Learn more
Contents hide
(Top)
Types
Toggle Types subsection
Predictions
Toggle Predictions subsection
Occurrence and cycles
Toggle Occurrence and cycles subsection
Historical eclipses
Viewing
Toggle Viewing subsection
Other observations
Toggle Other observations subsection
Recent and forthcoming solar eclipses
Toggle Recent and forthcoming solar eclipses subsection
See also
Footnotes
Notes
References
External links
Solar eclipse

Article
Talk
Read
View source
View history

Tools
Featured article
Page semi-protected
Listen to this article
From Wikipedia, the free encyclopedia
Not to be confused with Solar Eclipse (video game) or Solar Eclipse (song).
"Eclipse of the Sun" redirects here. For other uses, see Eclipse of the Sun (disambiguation).
Total solar eclipse
A total solar eclipse occurs when the Moon completely covers the Sun's disk, as seen in this 1999 solar eclipse. Solar prominences can be seen along the limb (in red) as well as extensive coronal filaments.
Annular solar eclipsePartial solar eclipse
An annular solar eclipse (left) occurs when the Moon is too far away to completely cover the Sun's disk (May 20, 2012). During a partial solar eclipse (right), the Moon blocks only part of the Sun's disk (October 25, 2022).
A solar eclipse occurs when the Moon passes between Earth and the Sun, thereby obscuring the view of the Sun from a small part of the Earth, totally or partially. Such an alignment occurs approximately every six months, during the eclipse season in its new moon phase, when the Moon's orbital plane is closest to the plane of the Earth's orbit.[1] In a total eclipse, the disk of the Sun is fully obscured by the Moon. In partial and annular eclipses, only part of the Sun is obscured. Unlike a lunar eclipse, which may be viewed from anywhere on the night side of Earth, a solar eclipse can only be viewed from a relatively small area of the world. As such, although total solar eclipses occur somewhere on Earth every 18 months on average, they recur at any given place only once every 360 to 410 years.

If the Moon were in a perfectly circular orbit and in the same orbital plane as Earth, there would be total solar eclipses once a month, at every new moon. Instead, because the Moon's orbit is tilted at about 5 degrees to Earth's orbit, its shadow usually misses Earth. Solar (and lunar) eclipses therefore happen only during eclipse seasons, resulting in at least two, and up to five, solar eclipses each year, no more than two of which can be total.[2][3] Total eclipses are more rare because they require a more precise alignment between the centers of the Sun and Moon, and because the Moon's apparent size in the sky is sometimes too small to fully cover the Sun.

An eclipse is a natural phenomenon. In some ancient and modern cultures, solar eclipses were attributed to supernatural causes or regarded as bad omens. Astronomers' predictions of eclipses began in China as early as the 4th century BC; eclipses hundreds of years into the future may now be predicted with high accuracy.

Looking directly at the Sun can lead to permanent eye damage, so special eye protection or indirect viewing techniques are used when viewing a solar eclipse. Only the total phase of a total solar eclipse is safe to view without protection. Enthusiasts known as eclipse chasers or umbraphiles travel to remote locations to see solar eclipses.[4][5]

Types

Partial and annular phases of the solar eclipse of May 20, 2012
There are four types of solar eclipses:

A total eclipse occurs in average every 18 months[Note 1][6] when the dark silhouette of the Moon completely obscures the intensely bright light of the Sun, allowing the much fainter solar corona to be visible. During any one eclipse, totality occurs at best only in a narrow track on the surface of Earth.[7] This narrow track is called the path of totality.[8]
An annular eclipse occurs once every one or two years[6] when the Sun and Moon are exactly in line with the Earth, but the apparent size of the Moon is smaller than that of the Sun. Hence the Sun appears as a very bright ring, or annulus, surrounding the dark disk of the Moon.[9]
A hybrid eclipse (also called annular/total eclipse) shifts between a total and annular eclipse. At certain points on the surface of Earth, it appears as a total eclipse, whereas at other points it appears as annular. Hybrid eclipses are comparatively rare.[9]
A partial eclipse occurs about twice a year,[6] when the Sun and Moon are not exactly in line with the Earth and the Moon only partially obscures the Sun. This phenomenon can usually be seen from a large part of the Earth outside of the track of an annular or total eclipse. However, some eclipses can be seen only as a partial eclipse, because the umbra passes above the Earth's polar regions and never intersects the Earth's surface.[9] Partial eclipses are virtually unnoticeable in terms of the Sun's brightness, as it takes well over 90% coverage to notice any darkening at all. Even at 99%, it would be no darker than civil twilight.[10]

Comparison of minimum and maximum apparent sizes of the Sun and Moon (and planets). An annular eclipse can occur when the Sun has a larger apparent size than the Moon, whereas a total eclipse can occur when the Moon has a larger apparent size.
The Sun's distance from Earth is about 400 times the Moon's distance, and the Sun's diameter is about 400 times the Moon's diameter. Because these ratios are approximately the same, the Sun and the Moon as seen from Earth appear to be approximately the same size: about 0.5 degree of arc in angular measure.[9]

The Moon's orbit around the Earth is slightly elliptical, as is the Earth's orbit around the Sun. The apparent sizes of the Sun and Moon therefore vary.[11] The magnitude of an eclipse is the ratio of the apparent size of the Moon to the apparent size of the Sun during an eclipse. An eclipse that occurs when the Moon is near its closest distance to Earth (i.e., near its perigee) can be a total eclipse because the Moon will appear to be large enough to completely cover the Sun's bright disk or photosphere; a total eclipse has a magnitude greater than or equal to 1.000. Conversely, an eclipse that occurs when the Moon is near its farthest distance from Earth (i.e., near its apogee) can be only an annular eclipse because the Moon will appear to be slightly smaller than the Sun; the magnitude of an annular eclipse is less than 1.[12]

A hybrid eclipse occurs when the magnitude of an eclipse changes during the event from less to greater than one, so the eclipse appears to be total at locations nearer the midpoint, and annular at other locations nearer the beginning and end, since the sides of the Earth are slightly further away from the Moon. These eclipses are extremely narrow in their path width and relatively short in their duration at any point compared with fully total eclipses; the 2023 April 20 hybrid eclipse's totality is over a minute in duration at various points along the path of totality. Like a focal point, the width and duration of totality and annularity are near zero at the points where the changes between the two occur.[13]

Because the Earth's orbit around the Sun is also elliptical, the Earth's distance from the Sun similarly varies throughout the year. This affects the apparent size of the Sun in the same way, but not as much as does the Moon's varying distance from Earth.[9] When Earth approaches its farthest distance from the Sun in early July, a total eclipse is somewhat more likely, whereas conditions favour an annular eclipse when Earth approaches its closest distance to the Sun in early January.[14]

Terminology for central eclipse

Each icon shows the view from the centre of its black spot, representing the Moon (not to scale)

Diamond ring effect at third contact—the end of totality—with visible prominences
Central eclipse is often used as a generic term for a total, annular, or hybrid eclipse.[15] This is, however, not completely correct: the definition of a central eclipse is an eclipse during which the central line of the umbra touches the Earth's surface. It is possible, though extremely rare, that part of the umbra intersects with the Earth (thus creating an annular or total eclipse), but not its central line. This is then called a non-central total or annular eclipse.[15] Gamma is a measure of how centrally the shadow strikes. The last (umbral yet) non-central solar eclipse was on April 29, 2014. This was an annular eclipse. The next non-central total solar eclipse will be on April 9, 2043.[16]

The visual phases observed during a total eclipse are called:[17]

First contact—when the Moon's limb (edge) is exactly tangential to the Sun's limb.
Second contact—starting with Baily's Beads (caused by light shining through valleys on the Moon's surface) and the diamond ring effect. Almost the entire disk is covered.
Totality—the Moon obscures the entire disk of the Sun and only the solar corona is visible.
Third contact—when the first bright light becomes visible and the Moon's shadow is moving away from the observer. Again a diamond ring may be observed.
Fourth contact—when the trailing edge of the Moon ceases to overlap with the solar disk and the eclipse ends.
Predictions
Geometry

Geometry of a total solar eclipse (not to scale)
The diagrams to the right show the alignment of the Sun, Moon, and Earth during a solar eclipse. The dark gray region between the Moon and Earth is the umbra, where the Sun is completely obscured by the Moon. The small area where the umbra touches Earth's surface is where a total eclipse can be seen. The larger light gray area is the penumbra, in which a partial eclipse can be seen. An observer in the antumbra, the area of shadow beyond the umbra, will see an annular eclipse.[18]

The Moon's orbit around the Earth is inclined at an angle of just over 5 degrees to the plane of the Earth's orbit around the Sun (the ecliptic). Because of this, at the time of a new moon, the Moon will usually pass to the north or south of the Sun. A solar eclipse can occur only when a new moon occurs close to one of the points (known as nodes) where the Moon's orbit crosses the ecliptic.[19]

As noted above, the Moon's orbit is also elliptical. The Moon's distance from the Earth can vary by about 6% from its average value. Therefore, the Moon's apparent size varies with its distance from the Earth, and it is this effect that leads to the difference between total and annular eclipses. The distance of the Earth from the Sun also varies during the year, but this is a smaller effect. On average, the Moon appears to be slightly smaller than the Sun as seen from the Earth, so the majority (about 60%) of central eclipses are annular. It is only when the Moon is closer to the Earth than average (near its perigee) that a total eclipse occurs.[20][21]

 	Moon	Sun
At perigee
(nearest)	At apogee
(farthest)	At perihelion
(nearest)	At aphelion
(farthest)
Mean radius	1,737.10 km
(1,079.38 mi)	696,000 km
(432,000 mi)
Distance	363,104 km
(225,622 mi)	405,696 km
(252,088 mi)	147,098,070 km
(91,402,500 mi)	152,097,700 km
(94,509,100 mi)
Angular
diameter[22]	33' 30"
(0.5583°)	29' 26"
(0.4905°)	32' 42"
(0.5450°)	31' 36"
(0.5267°)
Apparent size
to scale				
Order by
decreasing
apparent size	1st	4th	2nd	3rd
The Moon orbits the Earth in approximately 27.3 days, relative to a fixed frame of reference. This is known as the sidereal month. However, during one sidereal month, Earth has revolved part way around the Sun, making the average time between one new moon and the next longer than the sidereal month: it is approximately 29.5 days. This is known as the synodic month and corresponds to what is commonly called the lunar month.[19]

The Moon crosses from south to north of the ecliptic at its ascending node, and vice versa at its descending node.[19] However, the nodes of the Moon's orbit are gradually moving in a retrograde motion, due to the action of the Sun's gravity on the Moon's motion, and they make a complete circuit every 18.6 years. This regression means that the time between each passage of the Moon through the ascending node is slightly shorter than the sidereal month. This period is called the nodical or draconic month.[23]

Finally, the Moon's perigee is moving forwards or precessing in its orbit and makes a complete circuit in 8.85 years. The time between one perigee and the next is slightly longer than the sidereal month and known as the anomalistic month.[24]

The Moon's orbit intersects with the ecliptic at the two nodes that are 180 degrees apart. Therefore, the new moon occurs close to the nodes at two periods of the year approximately six months (173.3 days) apart, known as eclipse seasons, and there will always be at least one solar eclipse during these periods. Sometimes the new moon occurs close enough to a node during two consecutive months to eclipse the Sun on both occasions in two partial eclipses. This means that, in any given year, there will always be at least two solar eclipses, and there can be as many as five.[25]

Eclipses can occur only when the Sun is within about 15 to 18 degrees of a node, (10 to 12 degrees for central eclipses). This is referred to as an eclipse limit, and is given in ranges because the apparent sizes and speeds of the Sun and Moon vary throughout the year. In the time it takes for the Moon to return to a node (draconic month), the apparent position of the Sun has moved about 29 degrees, relative to the nodes.[2] Since the eclipse limit creates a window of opportunity of up to 36 degrees (24 degrees for central eclipses), it is possible for partial eclipses (or rarely a partial and a central eclipse) to occur in consecutive months.[26][27]


Fraction of the Sun's disc covered, f, when the same-sized discs are offset a fraction t of their diameter.[28]
Path
During a central eclipse, the Moon's umbra (or antumbra, in the case of an annular eclipse) moves rapidly from west to east across the Earth. The Earth is also rotating from west to east, at about 28 km/min at the Equator, but as the Moon is moving in the same direction as the Earth's rotation at about 61 km/min, the umbra almost always appears to move in a roughly west–east direction across a map of the Earth at the speed of the Moon's orbital velocity minus the Earth's rotational velocity.[29]

The width of the track of a central eclipse varies according to the relative apparent diameters of the Sun and Moon. In the most favourable circumstances, when a total eclipse occurs very close to perigee, the track can be up to 267 km (166 mi) wide and the duration of totality may be over 7 minutes.[30] Outside of the central track, a partial eclipse is seen over a much larger area of the Earth. Typically, the umbra is 100–160 km wide, while the penumbral diameter is in excess of 6400 km.[31]

Besselian elements are used to predict whether an eclipse will be partial, annular, or total (or annular/total), and what the eclipse circumstances will be at any given location.[32]: Chapter 11 

Calculations with Besselian elements can determine the exact shape of the umbra's shadow on the Earth's surface. But at what longitudes on the Earth's surface the shadow will fall, is a function of the Earth's rotation, and on how much that rotation has slowed down over time. A number called ΔT is used in eclipse prediction to take this slowing into account. As the Earth slows, ΔT increases. ΔT for dates in the future can only be roughly estimated because the Earth's rotation is slowing irregularly. This means that, although it is possible to predict that there will be a total eclipse on a certain date in the far future, it is not possible to predict in the far future exactly at what longitudes that eclipse will be total. Historical records of eclipses allow estimates of past values of ΔT and so of the Earth's rotation. [32]: Equation 11.132 

Duration

This section is in list format but may read better as prose. You can help by converting this section, if appropriate. Editing help is available. (May 2022)
The following factors determine the duration of a total solar eclipse (in order of decreasing importance):[33][34]

The Moon being almost exactly at perigee (making its angular diameter as large as possible).
The Earth being very near aphelion (furthest away from the Sun in its elliptical orbit, making its angular diameter nearly as small as possible).
The midpoint of the eclipse being very close to the Earth's equator, where the rotational velocity is greatest and is closest to the speed of the lunar shadow moving over Earth's surface.
The vector of the eclipse path at the midpoint of the eclipse aligning with the vector of the Earth's rotation (i.e. not diagonal but due east).
The midpoint of the eclipse being near the subsolar point (the part of the Earth closest to the Sun).
The longest eclipse that has been calculated thus far is the eclipse of July 16, 2186 (with a maximum duration of 7 minutes 29 seconds over northern Guyana).[33]

Occurrence and cycles
Main article: Eclipse cycle

As the Earth revolves around the Sun, approximate axial parallelism of the Moon's orbital plane (tilted five degrees to the Earth's orbital plane) results in the revolution of the lunar nodes relative to the Earth. This causes an eclipse season approximately every six months, in which a solar eclipse can occur at the new moon phase and a lunar eclipse can occur at the full moon phase.

Total solar eclipse paths: 1001–2000, showing that total solar eclipses occur almost everywhere on Earth. This image was merged from 50 separate images from NASA.[35]
Total solar eclipses are rare events. Although they occur somewhere on Earth every 18 months on average,[36] it is estimated that they recur at any given place only once every 360 to 410 years, on average.[37] The total eclipse lasts for only a maximum of a few minutes at any location, because the Moon's umbra moves eastward at over 1700 km/h.[38] Totality currently can never last more than 7 min 32 s. This value changes over the millennia and is currently decreasing. By the 8th millennium, the longest theoretically possible total eclipse will be less than 7 min 2 s.[33] The last time an eclipse longer than 7 minutes occurred was June 30, 1973 (7 min 3 sec). Observers aboard a Concorde supersonic aircraft were able to stretch totality for this eclipse to about 74 minutes by flying along the path of the Moon's umbra.[39] The next total eclipse exceeding seven minutes in duration will not occur until June 25, 2150. The longest total solar eclipse during the 11,000 year period from 3000 BC to at least 8000 AD will occur on July 16, 2186, when totality will last 7 min 29 s.[33][40] For comparison, the longest total eclipse of the 20th century at 7 min 8 s occurred on June 20, 1955, and there will be no total solar eclipses over 7 min in duration in the 21st century.[41]

It is possible to predict other eclipses using eclipse cycles. The saros is probably the best known and one of the most accurate. A saros lasts 6,585.3 days (a little over 18 years), which means that, after this period, a practically identical eclipse will occur. The most notable difference will be a westward shift of about 120° in longitude (due to the 0.3 days) and a little in latitude (north-south for odd-numbered cycles, the reverse for even-numbered ones). A saros series always starts with a partial eclipse near one of Earth's polar regions, then shifts over the globe through a series of annular or total eclipses, and ends with a partial eclipse at the opposite polar region. A saros series lasts 1226 to 1550 years and 69 to 87 eclipses, with about 40 to 60 of them being central.[42]

Frequency per year
Between two and five solar eclipses occur every year, with at least one per eclipse season. Since the Gregorian calendar was instituted in 1582, years that have had five solar eclipses were 1693, 1758, 1805, 1823, 1870, and 1935. The next occurrence will be 2206.[43] On average, there are about 240 solar eclipses each century.[44]

The 5 solar eclipses of 1935
January 5	February 3	June 30	July 30	December 25
Partial
(south)	Partial
(north)	Partial
(north)	Partial
(south)	Annular
(south)

Saros 111	
Saros 149	
Saros 116	
Saros 154	
Saros 121
Final totality
Total solar eclipses are seen on Earth because of a fortuitous combination of circumstances. Even on Earth, the diversity of eclipses familiar to people today is a temporary (on a geological time scale) phenomenon. Hundreds of millions of years in the past, the Moon was closer to the Earth and therefore apparently larger, so every solar eclipse was total or partial, and there were no annular eclipses. Due to tidal acceleration, the orbit of the Moon around the Earth becomes approximately 3.8 cm more distant each year. Millions of years in the future, the Moon will be too far away to fully occlude the Sun, and no total eclipses will occur. In the same timeframe, the Sun may become brighter, making it appear larger in size.[45] Estimates of the time when the Moon will be unable to occlude the entire Sun when viewed from the Earth range between 650 million[46] and 1.4 billion years in the future.[45]

Historical eclipses

Astronomers Studying an Eclipse painted by Antoine Caron in 1571
Historical eclipses are a very valuable resource for historians, in that they allow a few historical events to be dated precisely, from which other dates and ancient calendars may be deduced.[47] A solar eclipse of June 15, 763 BC mentioned in an Assyrian text is important for the chronology of the ancient Near East.[48] There have been other claims to date earlier eclipses. The legendary Chinese king Zhong Kang supposedly beheaded two astronomers, Hsi and Ho, who failed to predict an eclipse 4,000 years ago.[49] Perhaps the earliest still-unproven claim is that of archaeologist Bruce Masse, who putatively links an eclipse that occurred on May 10, 2807, BC with a possible meteor impact in the Indian Ocean on the basis of several ancient flood myths that mention a total solar eclipse.[50] The earliest preserved depiction of a partial solar eclipse from 1143 BCE might be the one in tomb KV9 of Ramses V and Ramses VI.[citation needed]


Records of the solar eclipses of 993 and 1004 as well as the lunar eclipses of 1001 and 1002 by Ibn Yunus of Cairo (c. 1005).
Eclipses have been interpreted as omens, or portents.[51] The ancient Greek historian Herodotus wrote that Thales of Miletus predicted an eclipse that occurred during a battle between the Medes and the Lydians. Both sides put down their weapons and declared peace as a result of the eclipse.[52] The exact eclipse involved remains uncertain, although the issue has been studied by hundreds of ancient and modern authorities. One likely candidate took place on May 28, 585 BC, probably near the Halys river in Asia Minor.[53] An eclipse recorded by Herodotus before Xerxes departed for his expedition against Greece,[54] which is traditionally dated to 480 BC, was matched by John Russell Hind to an annular eclipse of the Sun at Sardis on February 17, 478 BC.[55] Alternatively, a partial eclipse was visible from Persia on October 2, 480 BC.[56] Herodotus also reports a solar eclipse at Sparta during the Second Persian invasion of Greece.[57] The date of the eclipse (August 1, 477 BC) does not match exactly the conventional dates for the invasion accepted by historians.[58]

Chinese records of eclipses begin at around 720 BC.[59] The 4th century BC astronomer Shi Shen described the prediction of eclipses by using the relative positions of the Moon and Sun.[60]

Attempts have been made to establish the exact date of Good Friday by assuming that the darkness described at Jesus's crucifixion was a solar eclipse. This research has not yielded conclusive results,[61][62] and Good Friday is recorded as being at Passover, which is held at the time of a full moon. Further, the darkness lasted from the sixth hour to the ninth, or three hours, which is much, much longer than the eight-minute upper limit for any solar eclipse's totality. Contemporary chronicles wrote about an eclipse at the beginning of May 664 that coincided with the beginning of the plague of 664 in the British isles.[63] In the Western hemisphere, there are few reliable records of eclipses before AD 800, until the advent of Arab and monastic observations in the early medieval period.[59] The Cairo astronomer Ibn Yunus wrote that the calculation of eclipses was one of the many things that connect astronomy with the Islamic law, because it allowed knowing when a special prayer can be made.[64] The first recorded observation of the corona was made in Constantinople in AD 968.[56][59]


Erhard Weigel, predicted course of moon shadow on 12 August 1654 (O.S. 2 August)
The first known telescopic observation of a total solar eclipse was made in France in 1706.[59] Nine years later, English astronomer Edmund Halley accurately predicted and observed the solar eclipse of May 3, 1715.[56][59] By the mid-19th century, scientific understanding of the Sun was improving through observations of the Sun's corona during solar eclipses. The corona was identified as part of the Sun's atmosphere in 1842, and the first photograph (or daguerreotype) of a total eclipse was taken of the solar eclipse of July 28, 1851.[56] Spectroscope observations were made of the solar eclipse of August 18, 1868, which helped to determine the chemical composition of the Sun.[56] John Fiske summed up myths about the solar eclipse like this in his 1872 book Myth and Myth-Makers,
the myth of Hercules and Cacus, the fundamental idea is the victory of the solar god over the robber who steals the light. Now whether the robber carries off the light in the evening when Indra has gone to sleep, or boldly rears his black form against the sky during the daytime, causing darkness to spread over the earth, would make little difference to the framers of the myth. To a chicken a solar eclipse is the same thing as nightfall, and he goes to roost accordingly. Why, then, should the primitive thinker have made a distinction between the darkening of the sky caused by black clouds and that caused by the rotation of the earth? He had no more conception of the scientific explanation of these phenomena than the chicken has of the scientific explanation of an eclipse. For him it was enough to know that the solar radiance was stolen, in the one case as in the other, and to suspect that the same demon was to blame for both robberies.[65]

Viewing
2017 total solar eclipse viewed in real time with audience reactions
Looking directly at the photosphere of the Sun (the bright disk of the Sun itself), even for just a few seconds, can cause permanent damage to the retina of the eye, because of the intense visible and invisible radiation that the photosphere emits. This damage can result in impairment of vision, up to and including blindness. The retina has no sensitivity to pain, and the effects of retinal damage may not appear for hours, so there is no warning that injury is occurring.[66][67]

Under normal conditions, the Sun is so bright that it is difficult to stare at it directly. However, during an eclipse, with so much of the Sun covered, it is easier and more tempting to stare at it. Looking at the Sun during an eclipse is as dangerous as looking at it outside an eclipse, except during the brief period of totality, when the Sun's disk is completely covered (totality occurs only during a total eclipse and only very briefly; it does not occur during a partial or annular eclipse). Viewing the Sun's disk through any kind of optical aid (binoculars, a telescope, or even an optical camera viewfinder) is extremely hazardous and can cause irreversible eye damage within a fraction of a second.[68][69]

Partial and annular eclipses

Eclipse glasses filter out eye damaging radiation, allowing direct viewing of the Sun during all partial eclipse phases; they are not used during totality, when the Sun is completely eclipsed

Pinhole projection method of observing partial solar eclipse. Insert (upper left): partially eclipsed Sun photographed with a white solar filter. Main image: projections of the partially eclipsed Sun (bottom right)
Viewing the Sun during partial and annular eclipses (and during total eclipses outside the brief period of totality) requires special eye protection, or indirect viewing methods if eye damage is to be avoided. The Sun's disk can be viewed using appropriate filtration to block the harmful part of the Sun's radiation. Sunglasses do not make viewing the Sun safe. Only properly designed and certified solar filters should be used for direct viewing of the Sun's disk.[70] Especially, self-made filters using common objects such as a floppy disk removed from its case, a Compact Disc, a black colour slide film, smoked glass, etc. must be avoided.[71][72]

The safest way to view the Sun's disk is by indirect projection.[73] This can be done by projecting an image of the disk onto a white piece of paper or card using a pair of binoculars (with one of the lenses covered), a telescope, or another piece of cardboard with a small hole in it (about 1 mm diameter), often called a pinhole camera. The projected image of the Sun can then be safely viewed; this technique can be used to observe sunspots, as well as eclipses. Care must be taken, however, to ensure that no one looks through the projector (telescope, pinhole, etc.) directly.[74] A kitchen colander with small holes can also be used to project multiple images of the partially eclipsed Sun onto the ground or a viewing screen. Viewing the Sun's disk on a video display screen (provided by a video camera or digital camera) is safe, although the camera itself may be damaged by direct exposure to the Sun. The optical viewfinders provided with some video and digital cameras are not safe. Securely mounting #14 welder's glass in front of the lens and viewfinder protects the equipment and makes viewing possible.[72] Professional workmanship is essential because of the dire consequences any gaps or detaching mountings will have. In the partial eclipse path, one will not be able to see the corona or nearly complete darkening of the sky. However, depending on how much of the Sun's disk is obscured, some darkening may be noticeable. If three-quarters or more of the Sun is obscured, then an effect can be observed by which the daylight appears to be dim, as if the sky were overcast, yet objects still cast sharp shadows.[75]

Totality
Solar eclipse of August 21, 2017

Baily's beads, sunlight visible through lunar valleys

Composite image with corona, prominences, and diamond ring effect
When the shrinking visible part of the photosphere becomes very small, Baily's beads will occur. These are caused by the sunlight still being able to reach the Earth through lunar valleys. Totality then begins with the diamond ring effect, the last bright flash of sunlight.[76]

It is safe to observe the total phase of a solar eclipse directly only when the Sun's photosphere is completely covered by the Moon, and not before or after totality.[73] During this period, the Sun is too dim to be seen through filters. The Sun's faint corona will be visible, and the chromosphere, solar prominences, and possibly even a solar flare may be seen. At the end of totality, the same effects will occur in reverse order, and on the opposite side of the Moon.[76]

Eclipse chasing
Main article: Eclipse chasing
A dedicated group of eclipse chasers have pursued the observation of solar eclipses when they occur around the Earth.[77] A person who chases eclipses is known as an umbraphile, meaning shadow lover.[78] Umbraphiles travel for eclipses and use various tools to help view the sun including solar viewing glasses, also known as eclipse glasses, as well as telescopes.[79][80]

Photography

The progression of a solar eclipse on August 1, 2008 in Novosibirsk, Russia. All times UTC (local time was UTC+7). The time span between shots is three minutes.
Photographing an eclipse is possible with fairly common camera equipment. In order for the disk of the Sun/Moon to be easily visible, a fairly high magnification long focus lens is needed (at least 200 mm for a 35 mm camera), and for the disk to fill most of the frame, a longer lens is needed (over 500 mm). As with viewing the Sun directly, looking at it through the optical viewfinder of a camera can produce damage to the retina, so care is recommended.[81] Solar filters are required for digital photography even if an optical viewfinder is not used. Using a camera's live view feature or an electronic viewfinder is safe for the human eye, but the Sun's rays could potentially irreparably damage digital image sensors unless the lens is covered by a properly designed solar filter.[82]

Other observations
A total solar eclipse provides a rare opportunity to observe the corona (the outer layer of the Sun's atmosphere). Normally this is not visible because the photosphere is much brighter than the corona. According to the point reached in the solar cycle, the corona may appear small and symmetric, or large and fuzzy. It is very hard to predict this in advance.[83]


Pinholes in shadows during no eclipse (1 & 4), a partial eclipse (2 & 5) and an annular eclipse (3 & 6)
As the light filters through leaves of trees during a partial eclipse, the overlapping leaves create natural pinholes, displaying mini eclipses on the ground.[84]

Phenomena associated with eclipses include shadow bands (also known as flying shadows), which are similar to shadows on the bottom of a swimming pool. They occur only just prior to and after totality, when a narrow solar crescent acts as an anisotropic light source.[85]

1919 observations
See also: Tests of general relativity § Deflection of light by the Sun

Eddington's original photograph of the 1919 eclipse, which provided evidence for Einstein's theory of general relativity.
The observation of a total solar eclipse of May 29, 1919, helped to confirm Einstein's theory of general relativity. By comparing the apparent distance between stars in the constellation Taurus, with and without the Sun between them, Arthur Eddington stated that the theoretical predictions about gravitational lenses were confirmed.[86] The observation with the Sun between the stars was possible only during totality since the stars are then visible. Though Eddington's observations were near the experimental limits of accuracy at the time, work in the later half of the 20th century confirmed his results.[87][88]

Gravity anomalies
There is a long history of observations of gravity-related phenomena during solar eclipses, especially during the period of totality. In 1954, and again in 1959, Maurice Allais reported observations of strange and unexplained movement during solar eclipses.[89] The reality of this phenomenon, named the Allais effect, has remained controversial. Similarly, in 1970, Saxl and Allen observed the sudden change in motion of a torsion pendulum; this phenomenon is called the Saxl effect.[90]

Observation during the 1997 solar eclipse by Wang et al. suggested a possible gravitational shielding effect,[91] which generated debate. In 2002, Wang and a collaborator published detailed data analysis, which suggested that the phenomenon still remains unexplained.[92]

Eclipses and transits
In principle, the simultaneous occurrence of a solar eclipse and a transit of a planet is possible. But these events are extremely rare because of their short durations. The next anticipated simultaneous occurrence of a solar eclipse and a transit of Mercury will be on July 5, 6757, and a solar eclipse and a transit of Venus is expected on April 5, 15232.[93]

More common, but still infrequent, is a conjunction of a planet (especially, but not only, Mercury or Venus) at the time of a total solar eclipse, in which event the planet will be visible very near the eclipsed Sun, when without the eclipse it would have been lost in the Sun's glare. At one time, some scientists hypothesized that there may be a planet (often given the name Vulcan) even closer to the Sun than Mercury; the only way to confirm its existence would have been to observe it in transit or during a total solar eclipse. No such planet was ever found, and general relativity has since explained the observations that led astronomers to suggest that Vulcan might exist.[94]

Artificial satellites

The Moon's shadow over Turkey and Cyprus, seen from the ISS during a 2006 total solar eclipse.

A composite image showing the ISS transit of the Sun while the 2017 solar eclipse was in progress.
Artificial satellites can also pass in front of the Sun as seen from the Earth, but none is large enough to cause an eclipse. At the altitude of the International Space Station, for example, an object would need to be about 3.35 km (2.08 mi) across to blot the Sun out entirely. These transits are difficult to watch because the zone of visibility is very small. The satellite passes over the face of the Sun in about a second, typically. As with a transit of a planet, it will not get dark.[95]

Observations of eclipses from spacecraft or artificial satellites orbiting above the Earth's atmosphere are not subject to weather conditions. The crew of Gemini 12 observed a total solar eclipse from space in 1966.[96] The partial phase of the 1999 total eclipse was visible from Mir.[97]

Impact
The solar eclipse of March 20, 2015, was the first occurrence of an eclipse estimated to potentially have a significant impact on the power system, with the electricity sector taking measures to mitigate any impact. The continental Europe and Great Britain synchronous areas were estimated to have about 90 gigawatts of solar power and it was estimated that production would temporarily decrease by up to 34 GW compared to a clear sky day.[98][99]

Eclipses may cause the temperature to decrease by 3 °C, with wind power potentially decreasing as winds are reduced by 0.7 m/s.[100]

In addition to the drop in light level and air temperature, animals change their behavior during totality. For example, birds and squirrels return to their nests and crickets chirp.[101]

Recent and forthcoming solar eclipses
Main article: List of solar eclipses in the 21st century
Further information: Lists of solar eclipses

Eclipse path for total and hybrid eclipses from 2021 to 2040.
Eclipses occur only in the eclipse season, when the Sun is close to either the ascending or descending node of the Moon. Each eclipse is separated by one, five or six lunations (synodic months), and the midpoint of each season is separated by 173.3 days, which is the mean time for the Sun to travel from one node to the next. The period is a little less than half a calendar year because the lunar nodes slowly regress. Because 223 synodic months is roughly equal to 239 anomalistic months and 242 draconic months, eclipses with similar geometry recur 223 synodic months (about 6,585.3 days) apart. This period (18 years 11.3 days) is a saros. Because 223 synodic months is not identical to 239 anomalistic months or 242 draconic months, saros cycles do not endlessly repeat. Each cycle begins with the Moon's shadow crossing the Earth near the north or south pole, and subsequent events progress toward the other pole until the Moon's shadow misses the Earth and the series ends.[26] Saros cycles are numbered; currently, cycles 117 to 156 are active.[citation needed]

1997–2000
This eclipse is a member of a semester series. An eclipse in a semester series of solar eclipses repeats approximately every 177 days and 4 hours (a semester) at alternating nodes of the Moon's orbit.[102]

Solar eclipse series sets from 1997–2000 
Descending node	 	Ascending node
Saros	Map	Gamma	Saros	Map	Gamma
120

Chita, Russia	1997 March 09

Total	0.91830	125	1997 September 02

Partial (south)	−1.03521
130

Total eclipse near Guadeloupe	1998 February 26

Total	0.23909	135	1998 August 22

Annular	−0.26441
140	1999 February 16

Annular	−0.47260	145

Totality from France	1999 August 11

Total	0.50623
150	2000 February 05

Partial (south)	−1.22325	155	2000 July 31

Partial (north)	1.21664
Partial solar eclipses on July 1, 2000 and December 25, 2000 occur in the next lunar year eclipse set.

2000–2003
This eclipse is a member of a semester series. An eclipse in a semester series of solar eclipses repeats approximately every 177 days and 4 hours (a semester) at alternating nodes of the Moon's orbit.[103]

Partial solar eclipses on February 5, 2000 and July 31, 2000 occur in the previous lunar year set.

Solar eclipse series sets from 2000–2003 
Ascending node	 	Descending node
Saros	Map	Gamma	Saros	Map	Gamma
117	2000 July 01

Partial (south)	−1.28214	122	2000 December 25

Partial (north)	1.13669
127

Totality from Lusaka, Zambia	2001 June 21

Total	−0.57013	132

Partial from Minneapolis, MN	2001 December 14

Annular	0.40885
137

Partial from Los Angeles, CA	2002 June 10

Annular	0.19933	142

Totality from Woomera	2002 December 04

Total	−0.30204
147

Culloden, Scotland	2003 May 31

Annular	0.99598	152	2003 November 23

Total	−0.96381
2004–2007
This eclipse is a member of a semester series. An eclipse in a semester series of solar eclipses repeats approximately every 177 days and 4 hours (a semester) at alternating nodes of the Moon's orbit.[104]

Solar eclipse series sets from 2004–2007 
Ascending node	 	Descending node
Saros	Map	Gamma	Saros	Map	Gamma
119	2004 April 19

Partial (south)	−1.13345	124	2004 October 14

Partial (north)	1.03481
129

Partial from Naiguatá	2005 April 08

Hybrid	−0.34733	134

Annular from Madrid, Spain	2005 October 03

Annular	0.33058
139

Total from Side, Turkey	2006 March 29

Total	0.38433	144

Partial from São Paulo, Brazil	2006 September 22

Annular	−0.40624
149

From Jaipur, India	2007 March 19

Partial (north)	1.07277	154

From Córdoba, Argentina	2007 September 11

Partial (south)	−1.12552
2008–2011
This eclipse is a member of a semester series. An eclipse in a semester series of solar eclipses repeats approximately every 177 days and 4 hours (a semester) at alternating nodes of the Moon's orbit.[105]

Solar eclipse series sets from 2008–2011 
Ascending node	 	Descending node
Saros	Map	Gamma	Saros	Map	Gamma
121

Partial from Christchurch, NZ	2008 February 07

Annular	−0.95701	126

Novosibirsk, Russia	2008 August 01

Total	0.83070
131

Palangka Raya, Indonesia	2009 January 26

Annular	−0.28197	136

Kurigram, Bangladesh	2009 July 22

Total	0.06977
141

Bangui, Central African Republic	2010 January 15

Annular	0.40016	146

Hao, French Polynesia	2010 July 11

Total	−0.67877
151

Partial from Vienna, Austria	2011 January 04

Partial (north)	1.06265	156	2011 July 01

Partial (south)	−1.49171
Partial solar eclipses on June 1, 2011, and November 25, 2011, occur on the next lunar year eclipse set.

2011–2014
This eclipse is a member of the 2011–2014 solar eclipse semester series. An eclipse in a semester series of solar eclipses repeats approximately every 177 days and 4 hours (a semester) at alternating nodes of the Moon's orbit.[106][Note 2]

Solar eclipse series sets from 2011–2014 
Descending node	 	Ascending node
Saros	Map	Gamma	Saros	Map	Gamma
118

Partial from Tromsø, Norway	2011 June 01

Partial (north)	1.21300	123

Hinode XRT footage	2011 November 25

Partial (south)	−1.05359
128

Middlegate, Nevada	2012 May 20

Annular	0.48279	133

Cairns, Australia	2012 November 13

Total	−0.37189
138

Churchills Head, Australia	2013 May 10

Annular	−0.26937	143

Partial from Libreville, Gabon	2013 November 03

Hybrid	0.32715
148

Partial from Adelaide, Australia	2014 April 29

Annular (non-central)	−0.99996	153

Partial from Minneapolis	2014 October 23

Partial (north)	1.09078
2015–2018
This eclipse is a member of a semester series. An eclipse in a semester series of solar eclipses repeats approximately every 177 days and 4 hours (a semester) at alternating nodes of the Moon's orbit.[107]

Solar eclipse series sets from 2015–2018 
Descending node	 	Ascending node
Saros	Map	Gamma	Saros	Map	Gamma
120

Longyearbyen, Svalbard	2015 March 20

Total	0.94536	125

Solar Dynamics Observatory	
2015 September 13

Partial (south)	−1.10039
130

Balikpapan, Indonesia	2016 March 9

Total	0.26092	135

L'Étang-Salé, Réunion	2016 September 1

Annular	−0.33301
140

Partial from Buenos Aires	2017 February 26

Annular	−0.45780	145

Casper, Wyoming	2017 August 21

Total	0.43671
150

Partial from Olivos, Buenos Aires	2018 February 15

Partial (south)	−1.21163	155

Partial from Huittinen, Finland	2018 August 11

Partial (north)	1.14758
Partial solar eclipses on July 13, 2018, and January 6, 2019, occur during the next semester series.

2018–2021
This eclipse is a member of a semester series. An eclipse in a semester series of solar eclipses repeats approximately every 177 days and 4 hours (a semester) at alternating nodes of the Moon's orbit.[108]

Note: Partial solar eclipses on February 15, 2018, and August 11, 2018, occurred during the previous semester series.

Solar eclipse series sets from 2018–2021 
Ascending node	 	Descending node
Saros	Map	Gamma	Saros	Map	Gamma
117

Partial from Melbourne, Australia	2018 July 13

Partial	−1.35423	122

Partial from Nakhodka, Russia	2019 January 6

Partial	1.14174
127

La Serena, Chile	2019 July 2

Total	−0.64656	132

Jaffna, Sri Lanka	2019 December 26

Annular	0.41351
137

Beigang, Yunlin, Taiwan	2020 June 21

Annular	0.12090	142

Gorbea, Chile	2020 December 14

Total	−0.29394
147

Partial from Halifax, Canada	2021 June 10

Annular	0.91516	152

From HMS Protector off South Georgia	2021 December 4

Total	−0.95261
2022–2025
This eclipse is a member of a semester series. An eclipse in a semester series of solar eclipses repeats approximately every 177 days and 4 hours (a semester) at alternating nodes of the Moon's orbit.[109]

Solar eclipse series sets from 2022–2025 
Ascending node	 	Descending node
Saros	Map	Gamma	Saros	Map	Gamma
119

Partial from CTIO, Chile	2022 April 30

Partial	−1.19008	124

Partial from Saratov, Russia	2022 October 25

Partial	1.07014
129

Total from
East Timor	2023 April 20

Hybrid	−0.39515	134

Annular from
Campeche, Mexico	2023 October 14

Annular	0.37534
139	2024 April 8

Total	0.34314	144	2024 October 2

Annular	−0.35087
149	2025 March 29

Partial	1.04053	154	2025 September 21

Partial	−1.06509
2026–2029
This eclipse is a member of a semester series. An eclipse in a semester series of solar eclipses repeats approximately every 177 days and 4 hours (a semester) at alternating nodes of the Moon's orbit.[110]

Solar eclipse series sets from 2026–2029 
Ascending node	 	Descending node
Saros	Map	Gamma	Saros	Map	Gamma
121	2026 February 17

Annular	−0.97427	126	2026 August 12

Total	0.89774
131	2027 February 6

Annular	−0.29515	136	2027 August 2

Total	0.14209
141	2028 January 26

Annular	0.39014	146	2028 July 22

Total	−0.60557
151	2029 January 14

Partial	1.05532	156	2029 July 11

Partial	−1.41908
Partial solar eclipses on June 12, 2029, and December 5, 2029, occur in the next lunar year eclipse set.

See also
Lists of solar eclipses
List of films featuring eclipses
Apollo–Soyuz: First joint U.S.–Soviet space flight. Mission included an arranged eclipse of the Sun by the Apollo module to allow instruments on the Soyuz to take photographs of the solar corona.
Eclipse chasing: Travel to eclipse locations for study and enjoyment
Occultation: Generic term for occlusion of an object by another object that passes between it and the observer, thus revealing (for example) the presence of an exoplanet orbiting a distant star by eclipsing it as seen from Earth
Solar eclipses in fiction
Solar eclipses on the Moon: Eclipse of the Sun by planet Earth, as seen from the Moon
Lunar eclipse: Solar eclipse of the Moon, as seen from Earth; the shadow cast on the Moon by that eclipse
Transit of Venus: Passage of the planet Venus between the Sun and the Earth, as seen from Earth. Technically a partial eclipse.
Transit of Deimos from Mars: Passage of the Martian moon Deimos between the Sun and Mars, as seen from Mars
Transit of Phobos from Mars: Passage of the Martian moon Phobos between the Sun and Mars, as seen from Mars
Footnotes
 In the same place it can happen only once in several centuries.
 The partial solar eclipses of January 4, 2011 and July 1, 2011 occurred in the previous semester series.
Notes
 "What is an eclipse?". European Space Agency. Archived from the original on 2018-08-04. Retrieved 2018-08-04.
 Littmann, Mark; Espenak, Fred; Willcox, Ken (2008). Totality: Eclipses of the Sun. Oxford University Press. pp. 18–19. ISBN 978-0-19-953209-4.
 Five solar eclipses occurred in 1935.NASA (September 6, 2009). "Five Millennium Catalog of Solar Eclipses". NASA Eclipse Web Site. Fred Espenak, Project and Website Manager. Archived from the original on April 29, 2010. Retrieved January 26, 2010.
 Koukkos, Christina (May 14, 2009). "Eclipse Chasing, in Pursuit of Total Awe". The New York Times. Archived from the original on June 26, 2018. Retrieved January 15, 2012.
 Pasachoff, Jay M. (July 10, 2010). "Why I Never Miss a Solar Eclipse". The New York Times. Archived from the original on June 26, 2018. Retrieved January 15, 2012.
 "What Are the Three Types of Solar Eclipses?". Exploratorium. Retrieved 11 Oct 2023.
 Harrington, pp. 7–8
 "Eclipse: Who? What? Where? When? and How? | Total Solar Eclipse 2017". eclipse2017.nasa.gov. Archived from the original on 2017-09-18. Retrieved 2017-09-21.
 Harrington, pp. 9–11
 "Transit of Venus, Sun–Earth Day 2012". nasa.gov. Archived from the original on January 14, 2016. Retrieved February 7, 2016.
 "Solar Eclipses". University of Tennessee. Archived from the original on June 9, 2015. Retrieved January 15, 2012.
 "How Is the Sun Completely Blocked in an Eclipse?". NASA Space Place. NASA. 2009. Archived from the original on 2021-01-19. Retrieved 2019-09-01.
 Espenak, Fred (September 26, 2009). "Solar Eclipses for Beginners". MrEclipse.com. Archived from the original on May 24, 2015. Retrieved January 15, 2012.
 Steel, p. 351
 Espenak, Fred (January 6, 2009). "Central Solar Eclipses: 1991–2050". NASA Eclipse web site. Greenbelt, MD: NASA Goddard Space Flight Center. Archived from the original on January 8, 2021. Retrieved January 15, 2012.
 Verbelen, Felix (November 2003). "Solar Eclipses on Earth, 1001 BC to AD 2500". online.be. Archived from the original on August 3, 2019. Retrieved January 15, 2012.
 Harrington, pp. 13–14; Steel, pp. 266–279
 Mobberley, pp. 30–38
 Harrington, pp. 4–5
 Hipschman, Ron. "Why Eclipses Happen". Exploratorium. Archived from the original on December 27, 2015. Retrieved January 14, 2012.
 Brewer, Bryan (January 14, 1998). "What Causes an Eclipse?". Earth View. Archived from the original on January 2, 2013. Retrieved January 14, 2012.
 NASA – Eclipse 99 – Frequently Asked Questions Archived 2010-05-27 at the Wayback Machine – There is a mistake in the How long will we continue to be able to see total eclipses of the Sun? answer, "...the Sun's angular diameter varies from 32.7 minutes of arc when the Earth is at its farthest point in its orbit (aphelion), and 31.6 arc minutes when it is at its closest (perihelion)." It should appear smaller when farther, so the values should be swapped.
 Steel, pp. 319–321
 Steel, pp. 317–319
 Harrington, pp. 5–7
 Espenak, Fred (August 28, 2009). "Periodicity of Solar Eclipses". NASA Eclipse web site. Greenbelt, MD: NASA Goddard Space Flight Center. Archived from the original on November 12, 2020. Retrieved January 15, 2012.
 Espenak, Fred; Meeus, Jean (January 26, 2007). "Five Millennium Catalog of Solar Eclipses: -1999 to +3000". NASA Eclipse web site. Greenbelt, MD: NASA Goddard Space Flight Center. Archived from the original on October 24, 2020. Retrieved January 15, 2012.
 European Space Agency, "Spacecraft flight dynamics Archived 2019-12-11 at the Wayback Machine: proceedings of an international symposium, 18–22 May 1981-Darmstadt, Germany", p.347
 Mobberley, pp. 33–37
 "How do eclipses such as the one on Wednesday 14 November 2012 occur?". Sydney Observatory. Archived from the original on 29 April 2013. Retrieved 20 March 2015.
 Steel, pp. 52–53
 Seidelmann, P. Kenneth; Urban, Sean E., eds. (2013). Explanatory Supplement to the Astronomical Almanac (3rd ed.). University Science Books. ISBN 978-1-891389-85-6.
 Meeus, J. (December 2003). "The maximum possible duration of a total solar eclipse". Journal of the British Astronomical Association. 113 (6): 343–348. Bibcode:2003JBAA..113..343M.
 M. Littman, et al.
 Espenak, Fred (March 24, 2008). "World Atlas of Solar Eclipse Paths". NASA Eclipse web site. NASA Goddard Space Flight Center. Archived from the original on July 14, 2012. Retrieved January 15, 2012.
 Steel, p. 4
 For 360 years, see Harrington, p. 9; for 410 years, see Steel, p. 31
 Mobberley, pp. 33–36; Steel, p. 258
 Beckman, J.; Begot, J.; Charvin, P.; Hall, D.; Lena, P.; Soufflot, A.; Liebenberg, D.; Wraight, P. (1973). "Eclipse Flight of Concorde 001". Nature. 246 (5428): 72–74. Bibcode:1973Natur.246...72B. doi:10.1038/246072a0. S2CID 10644966.
 Stephenson, F. Richard (1997). Historical Eclipses and Earth's Rotation. Cambridge University Press. p. 54. doi:10.1017/CBO9780511525186. ISBN 0-521-46194-4. Archived from the original on 2020-08-01. Retrieved 2012-01-04.
 Mobberley, p. 10
 Espenak, Fred (August 28, 2009). "Eclipses and the Saros". NASA Eclipse web site. NASA Goddard Space Flight Center. Archived from the original on May 24, 2012. Retrieved January 15, 2012.
 Pogo, Alexander (1935). "Calendar years with five solar eclipses". Popular Astronomy. Vol. 43. p. 412. Bibcode:1935PA.....43..412P.
 "What are solar eclipses and how often do they occur?". timeanddate.com. Archived from the original on 2017-02-02. Retrieved 2014-11-23.
 Walker, John (July 10, 2004). "Moon near Perigee, Earth near Aphelion". Fourmilab. Archived from the original on December 8, 2013. Retrieved March 7, 2010.
 Mayo, Lou. "WHAT'S UP? The Very Last Solar Eclipse!". NASA. Archived from the original on 2017-08-22. Retrieved 22 August 2017.
 Acta Eruditorum. Leipzig. 1762. p. 168. Archived from the original on 2020-07-31. Retrieved 2018-06-06.
 van Gent, Robert Harry. "Astronomical Chronology". University of Utrecht. Archived from the original on July 28, 2020. Retrieved January 15, 2012.
 Harrington, p. 2
 Blakeslee, Sandra (November 14, 2006). "Ancient Crash, Epic Wave". The New York Times. Archived from the original on April 11, 2009. Retrieved November 14, 2006.
 Steel, p. 1
 Steel, pp. 84–85
 Le Conte, David (December 6, 1998). "Eclipse Quotations". MrEclipse.com. Archived from the original on October 17, 2020. Retrieved January 8, 2011.
 Herodotus. Book VII. p. 37. Archived from the original on 2008-08-19. Retrieved 2008-07-13.
 Chambers, G. F. (1889). A Handbook of Descriptive and Practical Astronomy. Oxford: Clarendon Press. p. 323.
 Espenak, Fred. "Solar Eclipses of Historical Interest". NASA Eclipse web site. NASA Goddard Space Flight Center. Archived from the original on March 9, 2008. Retrieved December 28, 2011.
 Herodotus. Book IX. p. 10. Archived from the original on 2020-07-26. Retrieved 2008-07-14.
 Schaefer, Bradley E. (May 1994). "Solar Eclipses That Changed the World". Sky & Telescope. Vol. 87, no. 5. pp. 36–39. Bibcode:1994S&T....87...36S.
 Stephenson, F. Richard (1982). "Historical Eclipses". Scientific American. Vol. 247, no. 4. pp. 154–163. Bibcode:1982SciAm.247d.154S.
 Needham, Joseph (1986). Science and Civilization in China: Volume 3. Taipei: Caves Books. pp. 411–413. OCLC 48999277.
 Humphreys, C. J.; Waddington, W. G. (1983). "Dating the Crucifixion". Nature. 306 (5945): 743–746. Bibcode:1983Natur.306..743H. doi:10.1038/306743a0. S2CID 4360560.
 Kidger, Mark (1999). The Star of Bethlehem: An Astronomer's View. Princeton, NJ: Princeton University Press. pp. 68–72. ISBN 978-0-691-05823-8.
 Ó Cróinín, Dáibhí (13 May 2020). "Reeling in the years: why 664 AD was a terrible year in Ireland". rte.ie. Archived from the original on 2021-01-08. Retrieved January 9, 2021.
 Regis Morelon (1996). "General survey of Arabic astronomy". In Roshdi Rashed (ed.). Encyclopedia of the History of Arabic Science. Vol. I. Routledge. p. 15.
 Fiske, John (October 1, 1997). Myths and Myth-Makers Old Tales and Superstitions Interpreted by Comparative Mythology. Archived from the original on July 26, 2020. Retrieved February 12, 2017 – via Project Gutenberg.
 Espenak, Fred (July 11, 2005). "Eye Safety During Solar Eclipses". NASA Eclipse web site. NASA Goddard Space Flight Center. Archived from the original on July 16, 2012. Retrieved January 15, 2012.
 Dobson, Roger (August 21, 1999). "UK hospitals assess eye damage after solar eclipse". British Medical Journal. 319 (7208): 469. doi:10.1136/bmj.319.7208.469. PMC 1116382. PMID 10454393.
 MacRobert, Alan M. (8 August 2006). "How to Watch a Partial Solar Eclipse Safely". Sky & Telescope. Retrieved August 4, 2007.
 Chou, B. Ralph (July 11, 2005). "Eye safety during solar eclipses". NASA Eclipse web site. NASA Goddard Space Flight Center. Archived from the original on November 14, 2020. Retrieved January 15, 2012.
 Littmann, Mark; Willcox, Ken; Espenak, Fred (1999). "Observing Solar Eclipses Safely". MrEclipse.com. Archived from the original on July 26, 2020. Retrieved January 15, 2012.
 Chou, B. Ralph (January 20, 2008). "Eclipse Filters". MrEclipse.com. Archived from the original on November 27, 2020. Retrieved January 4, 2012.
 "Solar Viewing Safety". Perkins Observatory. Archived from the original on July 14, 2020. Retrieved January 15, 2012.
 Harrington, p. 25
 Harrington, p. 26
 Harrington, p. 40
 Littmann, Mark; Willcox, Ken; Espenak, Fred (1999). "The Experience of Totality". MrEclipse.com. Archived from the original on February 4, 2012. Retrieved January 15, 2012.
 Kate Russo (1 August 2012). Total Addiction: The Life of an Eclipse Chaser. Springer Science & Business Media. ISBN 978-3-642-30481-1. Archived from the original on 9 December 2019. Retrieved 24 August 2017.
 Kelly, Pat (2017-07-06). "Umbraphile, Umbraphilia, Umbraphiles, and Umbraphiliacs – Solar Eclipse with the Sol Alliance". Solar Eclipse with the Sol Alliance. Archived from the original on 2019-08-13. Retrieved 2017-08-24.
 "How to View the 2017 Solar Eclipse Safely". eclipse2017.nasa.gov. Archived from the original on 2017-08-24. Retrieved 2017-08-24.
 Wright, Andy (2017-08-16). "Chasing Totality: A Look Into the World of Umbraphiles". Atlas Obscura. Archived from the original on 2020-12-14. Retrieved 2017-08-24.
 Kramer, Bill. "Photographing a Total Solar Eclipse". Eclipse-chasers.com. Archived from the original on January 29, 2009. Retrieved March 7, 2010.
 Vorenkamp, Todd (April 2017). "How to Photograph a Solar Eclipse". B&H Photo Video. Archived from the original on July 1, 2019. Retrieved August 19, 2017.
 "The science of eclipses". ESA. September 28, 2004. Archived from the original on August 1, 2012. Retrieved August 4, 2007.
 Johnson-Groh, Mara (10 August 2017). "Five Tips from NASA for Photographing the Total Solar Eclipse on Aug. 21". NASA. Archived from the original on 18 August 2020. Retrieved 21 September 2017.
 Dravins, Dainis. "Flying Shadows". Lund Observatory. Archived from the original on July 26, 2020. Retrieved January 15, 2012.
 Dyson, F.W.; Eddington, A.S.; Davidson, C.R. (1920). "A Determination of the Deflection of Light by the Sun's Gravitational Field, from Observations Made at the Solar eclipse of May 29, 1919". Phil. Trans. Roy. Soc. A. 220 (571–81): 291–333. Bibcode:1920RSPTA.220..291D. doi:10.1098/rsta.1920.0009. Archived from the original on November 3, 2020. Retrieved August 27, 2019.
 "Relativity and the 1919 eclipse". ESA. September 13, 2004. Archived from the original on October 21, 2012. Retrieved January 11, 2011.
 Steel, pp. 114–120
 Allais, Maurice (1959). "Should the Laws of Gravitation be Reconsidered?". Aero/Space Engineering. 9: 46–55.
 Saxl, Erwin J.; Allen, Mildred (1971). "1970 solar eclipse as 'seen' by a torsion pendulum". Physical Review D. 3 (4): 823–825. Bibcode:1971PhRvD...3..823S. doi:10.1103/PhysRevD.3.823.
 Wang, Qian-shen; Yang, Xin-she; Wu, Chuan-zhen; Guo, Hong-gang; Liu, Hong-chen; Hua, Chang-chai (2000). "Precise measurement of gravity variations during a total solar eclipse". Physical Review D. 62 (4): 041101(R). arXiv:1003.4947. Bibcode:2000PhRvD..62d1101W. doi:10.1103/PhysRevD.62.041101. S2CID 6846335.
 Yang, X. S.; Wang, Q. S. (2002). "Gravity anomaly during the Mohe total solar eclipse and new constraint on gravitational shielding parameter". Astrophysics and Space Science. 282 (1): 245–253. Bibcode:2002Ap&SS.282..245Y. doi:10.1023/A:1021119023985. S2CID 118497439.
 Meeus, J.; Vitagliano, A. (2004). "Simultaneous transits" (PDF). J. Br. Astron. Assoc. 114 (3): 132–135. Bibcode:2004JBAA..114..132M. Archived from the original (PDF) on July 10, 2007.
 Grego, Peter (2008). Venus and Mercury, and How to Observe Them. Springer. p. 3. ISBN 978-0387742854.
 "ISS-Venustransit". astronomie.info (in German). Archived from the original on 2020-07-28. Retrieved 2004-07-29.
 "JSC Digital Image Collection". NASA Johnson Space Center. January 11, 2006. Archived from the original on February 4, 2012. Retrieved January 15, 2012.
 Nemiroff, R.; Bonnell, J., eds. (August 30, 1999). "Looking Back on an Eclipsed Earth". Astronomy Picture of the Day. NASA. Retrieved January 15, 2012.
 "Solar Eclipse 2015 – Impact Analysis Archived 2017-02-21 at the Wayback Machine" pp. 3, 6–7, 13. European Network of Transmission System Operators for Electricity, 19 February 2015. Accessed: 4 March 2015.
 "Curve of potential power loss". ing.dk. Archived from the original on 2020-07-28. Retrieved 2015-03-04.
 Gray, S. L.; Harrison, R. G. (2012). "Diagnosing eclipse-induced wind changes". Proceedings of the Royal Society. 468 (2143): 1839–1850. Bibcode:2012RSPSA.468.1839G. doi:10.1098/rspa.2012.0007. Archived from the original on 2015-03-04. Retrieved 2015-03-04.
 Young, Alex. "How Eclipses Work". NASA. Archived from the original on 2017-09-18. Retrieved 21 September 2017.
 van Gent, R.H. "Solar- and Lunar-Eclipse Predictions from Antiquity to the Present". A Catalogue of Eclipse Cycles. Utrecht University. Retrieved 6 October 2018.
 van Gent, R.H. "Solar- and Lunar-Eclipse Predictions from Antiquity to the Present". A Catalogue of Eclipse Cycles. Utrecht University. Retrieved 6 October 2018.
 van Gent, R.H. "Solar- and Lunar-Eclipse Predictions from Antiquity to the Present". A Catalogue of Eclipse Cycles. Utrecht University. Retrieved 6 October 2018.
 van Gent, R.H. "Solar- and Lunar-Eclipse Predictions from Antiquity to the Present". A Catalogue of Eclipse Cycles. Utrecht University. Retrieved 6 October 2018.
 van Gent, R.H. "Solar- and Lunar-Eclipse Predictions from Antiquity to the Present". A Catalogue of Eclipse Cycles. Utrecht University. Retrieved 6 October 2018.
 van Gent, R.H. "Solar- and Lunar-Eclipse Predictions from Antiquity to the Present". A Catalogue of Eclipse Cycles. Utrecht University. Retrieved 6 October 2018.
 van Gent, R.H. "Solar- and Lunar-Eclipse Predictions from Antiquity to the Present". A Catalogue of Eclipse Cycles. Utrecht University. Retrieved 6 October 2018.
 van Gent, R.H. "Solar- and Lunar-Eclipse Predictions from Antiquity to the Present". A Catalogue of Eclipse Cycles. Utrecht University. Retrieved 6 October 2018.
 van Gent, R.H. "Solar- and Lunar-Eclipse Predictions from Antiquity to the Present". A Catalogue of Eclipse Cycles. Utrecht University. Retrieved 6 October 2018.
References
Mucke, Hermann; Meeus, Jean (1992). Canon of Solar Eclipses −2003 to +2526 (2 ed.). Vienna: Astronomisches Büro.
Harrington, Philip S. (1997). Eclipse! The What, Where, When, Why and How Guide to Watching Solar and Lunar Eclipses. New York: John Wiley and Sons. ISBN 0-471-12795-7.
Steel, Duncan (1999). Eclipse: The celestial phenomenon which has changed the course of history. London: Headline. ISBN 0-7472-7385-5.
Mobberley, Martin (2007). Total Solar Eclipses and How to Observe Them. Astronomers' Observing Guides. New York: Springer. ISBN 978-0-387-69827-4.
Espenak, Fred (2015). Thousand Year Canon of Solar Eclipses 1501 to 2500. Portal AZ: Astropixels Publishing. ISBN 978-1-941983-02-7.
Espenak, Fred (2016). 21st Century Canon of Solar Eclipses. Portal AZ: Astropixels Publishing. ISBN 978-1-941983-12-6.
Fotheringham, John Knight (1921). Historical eclipses: being the Halley lecture delivered 17 May 1921. Oxford: Clarendon Press.
External links

Wikimedia Commons has media related to Solar eclipses.

Wikivoyage has a travel guide for Solar eclipses.
Listen to this article
(2 parts, 27 minutes)
Duration: 15 minutes and 41 seconds.15:41
Duration: 11 minutes and 48 seconds.11:48
Spoken Wikipedia icon
These audio files were created from a revision of this article dated 3 May 2006, and do not reflect subsequent edits.
(Audio help · More spoken articles)
NASA Eclipse Web Site, with information on future eclipses and eye safety information
NASA Eclipse Web Site (older version)
Eclipsewise, Fred Espenak's new eclipse site
Andrew Lowe's Eclipse Page, with maps and circumstances for 5000 years of solar eclipses
A Guide to Eclipse Activities for Educators, Explaining eclipses in educational settings
Detailed eclipse explanations and predictions, Hermit Eclipse
Eclipse Photography, Prof. Miroslav Druckmüller
Animated maps of August 21, 2017 solar eclipses, Larry Koehn
Five Millennium (−1999 to +3000) Canon of Solar Eclipses Database, Xavier M. Jubier
Animated explanation of the mechanics of a solar eclipse Archived 2013-05-25 at the Wayback Machine, University of South Wales
Eclipse Image Gallery Archived 2016-10-15 at the Wayback Machine, The World at Night
Ring of Fire Eclipse: 2012, Photos
"Sun, Eclipses of the" . Collier's New Encyclopedia. 1921.
Centered and aligned video recording of Total Solar Eclipse 20th March 2015 on YouTube
Solar eclipse photographs taken from the Lick Observatory from the Lick Observatory Records Digital Archive, UC Santa Cruz Library’s Digital Collections Archived 2020-06-05 at the Wayback Machine
Video with Total Solar Eclipse March 09 2016 (from the beginning to the total phase) on YouTube
Total Solar Eclipse Shadow on Earth March 09 2016 CIMSSSatelite
List of all solar eclipses
National Geographic Solar Eclipse 101 video Archived 2018-08-04 at the Wayback Machine
 Wikiversity has a solar eclipse lab that students can do on any sunny day.
vte
Solar eclipses
vte
The Sun
vte
The Moon
Portals:
 Astronomy
icon Stars
 Spaceflight
 Outer space
 Solar System
Authority control databases: National Edit this at Wikidata	
GermanyIsraelUnited StatesJapanCzech Republic
Categories: EclipsesSolar eclipses
This page was last edited on 15 October 2023, at 00:16 (UTC).
Text is available under the Creative Commons Attribution-ShareAlike License 4.0; additional terms may apply. By using this site, you agree to the Terms of Use and Privacy Policy. Wikipedia® is a registered trademark of the Wikimedia Foundation, Inc., a non-profit organization.
Privacy policyAbout WikipediaDisclaimersContact WikipediaCode of ConductDevelopersStatisticsCookie statementMobile viewWikimedia FoundationPowered by MediaWiki
\"\"\"
Summarize"""

    if prompt_num == 1:
        prompt = prompt1
    else:
        prompt = prompt2
    if model_num == 1:
        base_model = 'llama'
    else:
        base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'
    model_path_llama = 'https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf?download=true'
    # model_path_llama = 'https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q8_0.gguf?download=true'
    res_dict, client = run_client_chat_with_server(prompt=prompt,
                                                   max_seq_len=max_seq_len,
                                                   model_path_llama=model_path_llama,
                                                   stream_output=False,
                                                   prompt_type='llama2',
                                                   base_model=base_model,
                                                   max_time=250,  # for 4096 llama-2 GGUF, takes 75s
                                                   )
    assert "solar eclipse" in res_dict['response']


def run_client_chat_with_server(prompt='Who are you?', stream_output=False, max_new_tokens=256,
                                base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b', prompt_type='human_bot',
                                langchain_mode='Disabled',
                                langchain_action=LangChainAction.QUERY.value,
                                langchain_agents=[],
                                user_path=None,
                                langchain_modes=['UserData', 'MyData', 'Disabled', 'LLM'],
                                model_path_llama='https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf?download=true',
                                docs_ordering_type='reverse_ucurve_sort',
                                max_seq_len=None,
                                max_time=20):
    if langchain_mode == 'Disabled':
        os.environ['TEST_LANGCHAIN_IMPORT'] = "1"
        sys.modules.pop('gpt_langchain', None)
        sys.modules.pop('langchain', None)

    from src.gen import main
    main(base_model=base_model,
         model_path_llama=model_path_llama,
         prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         docs_ordering_type=docs_ordering_type,
         max_seq_len=max_seq_len,
         verbose=True)

    from src.client_test import run_client_chat
    res_dict, client = run_client_chat(prompt=prompt, prompt_type=prompt_type, stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                       langchain_action=langchain_action, langchain_agents=langchain_agents,
                                       max_time=max_time)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''
    return res_dict, client


@wrap_test_forked
def test_client_chat_stream():
    run_client_chat_with_server(stream_output=True)


def run_client_nochat_with_server(prompt='Who are you?', stream_output=False, max_new_tokens=256,
                                  base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b', prompt_type='human_bot',
                                  langchain_mode='Disabled', langchain_action=LangChainAction.QUERY.value,
                                  langchain_agents=[],
                                  user_path=None,
                                  langchain_modes=['UserData', 'MyData', 'Disabled', 'LLM'],
                                  docs_ordering_type='reverse_ucurve_sort', other_server_kwargs={}):
    if langchain_mode == 'Disabled':
        os.environ['TEST_LANGCHAIN_IMPORT'] = "1"
        sys.modules.pop('gpt_langchain', None)
        sys.modules.pop('langchain', None)

    from src.gen import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, langchain_action=langchain_action, langchain_agents=langchain_agents,
         user_path=user_path,
         langchain_modes=langchain_modes,
         docs_ordering_type=docs_ordering_type,
         **other_server_kwargs)

    from src.client_test import run_client_nochat_gen
    res_dict, client = run_client_nochat_gen(prompt=prompt, prompt_type=prompt_type,
                                             stream_output=stream_output,
                                             max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                             langchain_action=langchain_action, langchain_agents=langchain_agents)
    assert 'birds' in res_dict['response'].lower() or \
           'and can learn new things' in res_dict['response'] or \
           'Once upon a time' in res_dict['response']
    return res_dict, client


@pytest.mark.parametrize("gradio_ui_stream_chunk_size", [0, 20])
@pytest.mark.parametrize("gradio_ui_stream_chunk_min_seconds", [0, .2, 2])
@pytest.mark.parametrize("gradio_ui_stream_chunk_seconds", [.2, 2])
@wrap_test_forked
def test_client_nochat_stream(gradio_ui_stream_chunk_size, gradio_ui_stream_chunk_min_seconds,
                              gradio_ui_stream_chunk_seconds):
    other_server_kwargs = dict(gradio_ui_stream_chunk_size=gradio_ui_stream_chunk_size,
                               gradio_ui_stream_chunk_min_seconds=gradio_ui_stream_chunk_min_seconds,
                               gradio_ui_stream_chunk_seconds=gradio_ui_stream_chunk_seconds)
    run_client_nochat_with_server(stream_output=True, prompt="Tell a very long kid's story about birds.",
                                  other_server_kwargs=other_server_kwargs)


@wrap_test_forked
def test_client_chat_stream_langchain():
    user_path = make_user_path_test()
    prompt = "What is h2oGPT?"
    res_dict, client = run_client_chat_with_server(prompt=prompt, stream_output=True, langchain_mode="UserData",
                                                   user_path=user_path,
                                                   langchain_modes=['UserData', 'MyData', 'Disabled', 'LLM'],
                                                   docs_ordering_type=None,  # for 6_9 dumb model for testing
                                                   )
    # below wouldn't occur if didn't use LangChain with README.md,
    # raw LLM tends to ramble about H2O.ai and what it does regardless of question.
    # bad answer about h2o.ai is just becomes dumb model, why flipped context above,
    # but not stable over different systems
    assert 'h2oGPT is a large language model' in res_dict['response'] or \
           'H2O.ai is a technology company' in res_dict['response'] or \
           'an open-source project' in res_dict['response'] or \
           'h2oGPT is a project that allows' in res_dict['response'] or \
           'h2oGPT is a language model trained' in res_dict['response'] or \
           'h2oGPT is a large-scale' in res_dict['response']


@pytest.mark.parametrize("max_new_tokens", [256, 2048])
@pytest.mark.parametrize("top_k_docs", [3, 100])
@wrap_test_forked
def test_client_chat_stream_langchain_steps(max_new_tokens, top_k_docs):
    os.environ['VERBOSE_PIPELINE'] = '1'
    user_path = make_user_path_test()

    stream_output = True
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'  # 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    prompt_type = 'llama2'  # 'human_bot'
    langchain_mode = 'UserData'
    langchain_modes = ['UserData', 'MyData', 'LLM', 'Disabled', 'LLM']

    from src.gen import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         top_k_docs=top_k_docs,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         docs_ordering_type=None,  # for 6_9
         )

    from src.client_test import get_client, get_args, run_client
    client = get_client(serialize=False)

    # QUERY1
    prompt = "What is h2oGPT?"
    langchain_mode = 'UserData'
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens,
                            top_k_docs=top_k_docs,
                            langchain_mode=langchain_mode)

    res_dict, client = run_client(client, prompt, args, kwargs)
    assert ('a large language model' in res_dict['response'] or
            '2oGPT is an open-source, Apache V2 project' in res_dict['response'] or
            'language model trained' in res_dict['response'] or
            'H2O GPT is a language model' in res_dict['response'] or
            'H2O GPT is a chatbot framework' in res_dict['response'] or
            'H2O GPT is a chatbot that can be trained' in res_dict['response'] or
            'A large language model (LLM)' in res_dict['response'] or
            'GPT-based language model' in res_dict['response'] or
            'H2O.ai is a technology company' in res_dict['response'] or
            'an open-source project' in res_dict['response'] or
            'is a company that provides' in res_dict['response'] or
            'h2oGPT is a project that' in res_dict['response'] or
            'for querying and summarizing documents' in res_dict['response'] or
            'Python-based platform for training' in res_dict['response'] or
            'h2oGPT is an open-source' in res_dict['response'] or
            'language model' in res_dict['response'] or
            'Whisper is an open-source' in res_dict['response']
            ) \
           and ('FAQ.md' in res_dict['response'] or 'README.md' in res_dict['response'])

    # QUERY1
    prompt = "What is Whisper?"
    langchain_mode = 'UserData'
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens,
                            top_k_docs=top_k_docs,
                            langchain_mode=langchain_mode)

    res_dict, client = run_client(client, prompt, args, kwargs)
    # wrong answer given wrong docs
    assert ('A secure chatbot that uses a large language' in res_dict['response'] or
            'Whisper is a chatbot' in res_dict['response'] or
            'Whisper is a privacy-focused chatbot platform' in res_dict['response'] or
            'h2oGPT' in res_dict['response'] or
            'A secure, private, and anonymous chat platform' in res_dict['response'] or
            'Whisper is a privacy-preserving' in res_dict['response'] or
            'A chatbot that uses a large language model' in res_dict['response'] or
            'This is a config file for Whisper' in res_dict['response'] or
            'Whisper is a secure messaging app' in res_dict['response'] or
            'secure, private, and anonymous chatbot' in res_dict['response'] or
            'Whisper is a secure, anonymous, and encrypted' in res_dict['response'] or
            'secure, decentralized, and anonymous chat platform' in res_dict['response'] or
            'A low-code development framework' in res_dict['response'] or
            'secure messaging app' in res_dict['response'] or
            'privacy-focused messaging app that allows' in res_dict['response'] or
            'A low-code AI app development framework' in res_dict['response'] or
            'anonymous communication platform' in res_dict['response'] or
            'A privacy-focused chat app' in res_dict['response'] or
            'A platform for deploying' in res_dict['response'] or
            'A language model that can be used to generate text.' in res_dict['response'] or
            'a chat app that' in res_dict['response']
            ) \
           and ('FAQ.md' in res_dict['response'] or 'README.md' in res_dict['response'])

    # QUERY2
    prompt = "What is h2oGPT?"
    langchain_mode = 'LLM'
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens,
                            top_k_docs=top_k_docs,
                            langchain_mode=langchain_mode)

    res_dict, client = run_client(client, prompt, args, kwargs)
    # i.e. answers wrongly without data, dumb model, but also no docs at all since cutoff entirely
    assert 'h2oGPT is a variant' in res_dict['response'] and '.md' not in res_dict['response']

    # QUERY3
    prompt = "What is whisper?"
    langchain_mode = 'UserData'
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens,
                            top_k_docs=top_k_docs,
                            langchain_mode=langchain_mode)

    res_dict, client = run_client(client, prompt, args, kwargs)
    # odd answer since no whisper docs, but still shows some docs at very low score
    assert ('h2oGPT' in res_dict['response'] or
            'A chatbot that can whisper to you' in res_dict['response'] or
            'whisper is a simple' in res_dict['response'] or
            'Whisper is a tool for generating text from a model' in res_dict['response'] or
            'Whisper is a chatbot platform' in res_dict['response'] or
            'whisper is a chatbot framework' in res_dict['response'] or
            'whisper is a tool for training language models' in res_dict['response'] or
            'whisper is a secure messaging app' in res_dict['response'] or
            'LLaMa-based models are not commercially viable' in res_dict['response'] or
            'A text-based chatbot that' in res_dict['response'] or
            'A secure, private, and anonymous chat service' in res_dict['response'] or
            'LLaMa is a language' in res_dict['response'] or
            'chatbot that can' in res_dict['response'] or
            'A secure, private, and anonymous chatbot' in res_dict['response'] or
            'A secure, encrypted chat service that allows' in res_dict['response'] or
            'A secure, private, and encrypted chatbot' in res_dict['response'] or
            'A secret communication system used' in res_dict['response'] or
            'H2O AI Cloud is a cloud-based platform' in res_dict['response'] or
            'is a platform for deploying' in res_dict['response'] or
            'is a language model that is trained' in res_dict['response'] or
            'private, and anonymous communication' in res_dict['response'] or
            'The large language model is' in res_dict['response'] or
            'is a private, secure, and encrypted' in res_dict['response'] or
            'H2O AI is a cloud-based platform for building' in res_dict['response'] or
            'a private chat between' in res_dict['response'] or
            'whisper is a chat bot' in res_dict['response']
            ) \
           and '.md' in res_dict['response']


@pytest.mark.parametrize("system_prompt", ['', None, 'None', 'auto', 'You are a goofy lion who talks to kids'])
# @pytest.mark.parametrize("system_prompt", [None])
@pytest.mark.parametrize("chat_conversation",
                         [None, [('Who are you?', 'I am a big pig who loves to tell kid stories')]])
# @pytest.mark.parametrize("chat_conversation", [[('Who are you?', 'I am a big pig who loves to tell kid stories')]])
@wrap_test_forked
def test_client_system_prompts(system_prompt, chat_conversation):
    stream_output = True
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'  # 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    prompt_type = 'llama2'  # 'human_bot'

    from src.gen import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         )

    from src.client_test import get_client, get_args, run_client
    client = get_client(serialize=False)

    # QUERY1
    prompt = "Who are you?"
    for client_type in ['chat', 'nochat']:
        if client_type == 'chat':
            kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                                    system_prompt=system_prompt,
                                    chat_conversation=chat_conversation)

            res_dict, client = run_client(client, prompt, args, kwargs)
        else:
            api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
            kwargs = dict(instruction_nochat=prompt,
                          system_prompt=system_prompt,
                          chat_conversation=chat_conversation)
            # pass string of dict.  All entries are optional, but expect at least instruction_nochat to be filled
            res = client.predict(str(dict(kwargs)), api_name=api_name)
            res_dict = ast.literal_eval(res)

        if not chat_conversation:
            if system_prompt == 'You are a goofy lion who talks to kids':
                assert ('ROAR!' in res_dict['response'] or 'ROARRR' in res_dict['response']) and \
                       'respectful' not in res_dict['response'] and \
                       'developed by Meta' not in res_dict['response']
            elif system_prompt == '':
                assert "developed by Meta" in res_dict['response'] and 'respectful' not in res_dict[
                    'response'] and 'ROAR!' not in res_dict['response']
            elif system_prompt in [None, 'auto', 'None']:
                assert 'respectful' in res_dict['response'] and 'ROAR!' not in res_dict[
                    'response'] and 'developed by Meta' not in res_dict['response']
        else:
            if system_prompt == 'You are a goofy lion who talks to kids':
                # system prompt overwhelms chat conversation
                assert "I'm a goofy lion" in res_dict['response'] or \
                       "goofiest lion" in res_dict['response'] or \
                       "I'm the coolest lion around" in res_dict['response'] or \
                       "awesome lion" in res_dict['response']
            elif system_prompt == '':
                # empty system prompt gives room for chat conversation to control
                assert "My name is Porky" in res_dict['response']
            elif system_prompt in [None, 'auto', 'None']:
                # conservative default system_prompt makes it ignore chat
                assert "not a real person" in res_dict['response'] or \
                       "I don't have personal experiences or feelings" in res_dict['response'] or \
                       "I'm just an AI" in res_dict['response']


@pytest.mark.need_tokens
@pytest.mark.parametrize("max_new_tokens", [256, 2048])
@pytest.mark.parametrize("top_k_docs", [3, 100])
@wrap_test_forked
def test_client_chat_stream_langchain_steps2(max_new_tokens, top_k_docs):
    os.environ['VERBOSE_PIPELINE'] = '1'
    # full user data
    from src.make_db import make_db_main
    make_db_main(download_some=True)
    user_path = None  # shouldn't be necessary, db already made

    stream_output = True
    max_new_tokens = 256
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'  # 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    prompt_type = 'llama2'  # 'human_bot'
    langchain_mode = 'UserData'
    langchain_modes = ['UserData', 'MyData', 'github h2oGPT', 'LLM', 'Disabled']

    from src.gen import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         verbose=True)

    from src.client_test import get_client, get_args, run_client
    client = get_client(serialize=False)

    # QUERY1
    prompt = "Who are you?"
    langchain_mode = 'LLM'
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode)

    res_dict, client = run_client(client, prompt, args, kwargs)
    assert 'an AI assistant developed by Meta' in res_dict['response'] and 'FAQ.md' not in res_dict['response']

    # QUERY2
    prompt = "What is whisper?"
    langchain_mode = 'UserData'
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode)

    res_dict, client = run_client(client, prompt, args, kwargs)
    res1 = 'large-scale speech recognition model' in res_dict['response'] and 'whisper.pdf' in res_dict['response']
    res2 = 'speech recognition system' in res_dict['response'] and 'whisper.pdf' in res_dict['response']
    assert res1 or res2

    # QUERY3
    prompt = "What is h2oGPT"
    langchain_mode = 'github h2oGPT'
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode)

    res_dict, client = run_client(client, prompt, args, kwargs)
    assert ('h2oGPT is an open-source, fully permissive, commercially usable, and fully trained language model' in
            res_dict['response'] or
            'A new open-source language model that is fully permissive' in res_dict['response'] or
            'h2oGPT is an open-source' in res_dict['response'] or
            'h2oGPT is an open-source, fully permissive, commercially usable' in res_dict['response'] or
            'Based on the information provided in the context, h2oGPT appears to be an open-source' in res_dict[
                'response'] or
            'h2oGPT is a variant of the' in res_dict['response']
            ) and \
           'README.md' in res_dict['response']


@wrap_test_forked
def test_doc_hash():
    remove('langchain_modes.pkl')
    user_path = make_user_path_test()

    stream_output = True
    base_model = ''
    langchain_mode = 'UserData'
    langchain_modes = ['UserData', 'MyData', 'LLM', 'Disabled']

    os.environ['SHOULD_NEW_FILES'] = '1'
    os.environ['GRADIO_SERVER_PORT'] = str(get_inf_port())
    from src.gen import main
    main(base_model=base_model, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         score_model='None',
         docs_ordering_type=None,  # for 6_9
         )

    # repeat, shouldn't reload
    os.environ.pop('SHOULD_NEW_FILES', None)
    os.environ['NO_NEW_FILES'] = '1'
    os.environ['GRADIO_SERVER_PORT'] = str(get_inf_port() + 1)
    from src.gen import main
    main(base_model=base_model, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         score_model='None',
         docs_ordering_type=None,  # for 6_9
         )


@wrap_test_forked
def test_client_chat_stream_long():
    prompt = 'Tell a very long story about cute birds for kids.'
    res_dict, client = run_client_chat_with_server(prompt=prompt, stream_output=True, max_new_tokens=1024)
    assert 'Once upon a time' in res_dict['response'] or \
           'The story begins with' in res_dict['response'] or \
           'The birds are all very' in res_dict['response']


@pytest.mark.parametrize("base_model", [
    'TheBloke/em_german_leo_mistral-GPTQ',
    'TheBloke/Nous-Hermes-13B-GPTQ',
])
@wrap_test_forked
def test_autogptq(base_model):
    prompt = 'Who are you?'
    stream_output = False
    max_new_tokens = 256
    load_gptq = 'model'
    use_safetensors = True
    prompt_type = ''
    max_seq_len = 4096  # mistral will use 32k if don't specify, go OOM on typical system
    langchain_mode = 'Disabled'
    langchain_action = LangChainAction.QUERY.value
    langchain_agents = []
    user_path = None
    langchain_modes = ['UserData', 'MyData', 'LLM', 'Disabled']
    docs_ordering_type = 'reverse_sort'
    from src.gen import main
    main(base_model=base_model, load_gptq=load_gptq,
         max_seq_len=max_seq_len,
         use_safetensors=use_safetensors,
         prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         docs_ordering_type=docs_ordering_type)

    from src.client_test import run_client_chat
    res_dict, client = run_client_chat(prompt=prompt, prompt_type=prompt_type, stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                       langchain_action=langchain_action, langchain_agents=langchain_agents)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''
    assert "am a virtual assistant" in res_dict['response'] or "computer program designed" in res_dict['response']

    check_langchain()


@wrap_test_forked
def test_autoawq():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    prompt = 'Who are you?'
    stream_output = False
    max_new_tokens = 256
    base_model = 'TheBloke/Mistral-7B-Instruct-v0.2-AWQ'
    load_awq = 'model'
    use_safetensors = True
    prompt_type = 'mistral'
    langchain_mode = 'Disabled'
    langchain_action = LangChainAction.QUERY.value
    langchain_agents = []
    user_path = None
    langchain_modes = ['UserData', 'MyData', 'LLM', 'Disabled']
    docs_ordering_type = 'reverse_sort'
    from src.gen import main
    main(base_model=base_model, load_awq=load_awq,
         use_safetensors=use_safetensors,
         prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         docs_ordering_type=docs_ordering_type,
         add_disk_models_to_ui=False,
         max_seq_len=2048,
         )

    from src.client_test import run_client_chat
    res_dict, client = run_client_chat(prompt=prompt, prompt_type=prompt_type, stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                       langchain_action=langchain_action, langchain_agents=langchain_agents)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''
    assert "I am an artificial intelligence designed to assist" in res_dict['response']

    check_langchain()


def check_langchain():
    # PURE client code
    from gradio_client import Client
    client = Client(get_inf_server())

    # get file for client to upload
    url = 'https://cdn.openai.com/papers/whisper.pdf'
    test_file1 = os.path.join('/tmp/', 'whisper1.pdf')
    download_simple(url, dest=test_file1)

    # upload file(s).  Can be list or single file
    test_file_local, test_file_server = client.predict(test_file1, api_name='/upload_api')

    chunk = True
    chunk_size = 512
    langchain_mode = 'MyData'
    loaders = tuple([None, None, None, None, None, None])
    h2ogpt_key = ''
    res = client.predict(test_file_server,
                         langchain_mode, chunk, chunk_size, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert os.path.basename(test_file_server) in res[2]
    assert res[3] == ''

    # ask for summary, need to use same client if using MyData
    instruction = "Give a very long detailed step-by-step description of what is Whisper paper about."
    max_time = 300
    kwargs = dict(instruction=instruction,
                  langchain_mode=langchain_mode,
                  langchain_action="Query",
                  top_k_docs=4,
                  document_subset='Relevant',
                  document_choice=DocumentChoice.ALL.value,
                  max_new_tokens=1024,
                  max_time=max_time,
                  do_sample=False,
                  stream_output=False,
                  )
    t0 = time.time()
    res_dict, client = run_client_gen(client, kwargs)
    response = res_dict['response']
    assert len(response) > 0
    # assert len(response) < max_time * 20  # 20 tokens/sec
    assert time.time() - t0 < max_time * 2.5
    sources = [x['source'] for x in res_dict['sources']]
    # only get source not empty list if break in inner loop, not gradio_runner loop, so good test of that too
    # this is why gradio timeout adds 10 seconds, to give inner a chance to produce references or other final info
    assert 'whisper1.pdf' in sources[0]


@pytest.mark.skip(reason="No longer supported")
@pytest.mark.parametrize("mode", ['a', 'b', 'c'])
@wrap_test_forked
def test_exllama(mode):
    prompt = 'Who are you?'
    stream_output = False
    max_new_tokens = 256
    if mode == 'c':
        base_model = 'TheBloke/Llama-2-70B-chat-GPTQ'
        exllama_dict = {}
    elif mode == 'b':
        base_model = 'TheBloke/Llama-2-70B-chat-GPTQ'
        exllama_dict = {'set_auto_map': '20,20'}
    elif mode == 'a':
        base_model = 'TheBloke/Llama-2-7B-chat-GPTQ'
        exllama_dict = {}
    else:
        raise RuntimeError("Bad mode=%s" % mode)
    load_exllama = True
    prompt_type = 'llama2'
    langchain_mode = 'Disabled'
    langchain_action = LangChainAction.QUERY.value
    langchain_agents = []
    user_path = None
    langchain_modes = ['UserData', 'MyData', 'LLM', 'Disabled']
    docs_ordering_type = 'reverse_ucurve_sort'
    from src.gen import main
    main(base_model=base_model,
         load_exllama=load_exllama, exllama_dict=exllama_dict,
         prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         docs_ordering_type=docs_ordering_type)

    from src.client_test import run_client_chat
    res_dict, client = run_client_chat(prompt=prompt, prompt_type=prompt_type, stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                       langchain_action=langchain_action, langchain_agents=langchain_agents)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''
    assert "I'm LLaMA, an AI assistant" in res_dict['response'] or \
           "I am LLaMA" in res_dict['response'] or \
           "Hello! My name is Llama, I'm a large language model trained by Meta AI." in res_dict['response']

    check_langchain()


@pytest.mark.parametrize("attention_sinks", [False, True])  # mistral goes beyond context just fine up to 32k
@pytest.mark.parametrize("max_seq_len", [4096, 8192])
@wrap_test_forked
def test_attention_sinks(max_seq_len, attention_sinks):
    # full user data
    from src.make_db import make_db_main
    make_db_main(download_some=True)
    user_path = None  # shouldn't be necessary, db already made

    prompt = 'Write an extremely fully detailed never-ending report that is well-structured with step-by-step sections (and elaborate details for each section) that describes the documents.  Never stop the report.'
    stream_output = True
    max_new_tokens = 100000
    max_max_new_tokens = max_new_tokens
    # base_model = 'mistralai/Mistral-7B-Instruct-v0.1'
    base_model = 'HuggingFaceH4/zephyr-7b-beta'
    prompt_type = 'zephyr'
    langchain_mode = 'UserData'
    langchain_action = LangChainAction.QUERY.value
    langchain_agents = []
    langchain_modes = ['UserData', 'MyData', 'LLM', 'Disabled']
    docs_ordering_type = 'reverse_ucurve_sort'
    document_choice = ['user_path/./whisper.pdf']  # only exact matches allowed currently
    top_k_docs = -1
    from src.gen import main
    main(base_model=base_model,
         attention_sinks=attention_sinks,
         user_path=user_path,
         prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         max_max_new_tokens=max_max_new_tokens,
         langchain_mode=langchain_mode,
         langchain_modes=langchain_modes,
         top_k_docs=top_k_docs,  # has no effect for client if client passes different number
         max_seq_len=max_seq_len,
         # mistral is 32k if don't say, easily run GPU OOM even on 48GB (even with --use_gpu_id=False)
         docs_ordering_type=docs_ordering_type,
         cut_distance=1.8,  # probably should allow control via API/UI
         sink_dict={'num_sink_tokens': 4, 'window_length': 4096} if attention_sinks else {},
         )

    from src.client_test import run_client_chat
    res_dict, client = run_client_chat(prompt=prompt, prompt_type=prompt_type, stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                       langchain_action=langchain_action, langchain_agents=langchain_agents,
                                       document_choice=document_choice, top_k_docs=top_k_docs,
                                       max_time=600, repetition_penalty=1.07, do_sample=False)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''
    assert len(res_dict['response']) > 2400, "%s %s" % (len(res_dict['response']), res_dict['response'])

    check_langchain()


@pytest.mark.skip(reason="Local file required")
@wrap_test_forked
def test_client_long():
    os.environ['TEST_LANGCHAIN_IMPORT'] = "1"
    sys.modules.pop('gpt_langchain', None)
    sys.modules.pop('langchain', None)

    from src.gen import main
    main(base_model='mosaicml/mpt-7b-storywriter', prompt_type=noop_prompt_type, chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False)

    with open("/home/jon/Downloads/Gatsby_PDF_FullText.txt") as f:
        prompt = f.readlines()

    from src.client_test import run_client_nochat
    res_dict, _ = run_client_nochat(prompt=prompt, prompt_type=noop_prompt_type, max_new_tokens=86000)
    print(res_dict['response'])


@wrap_test_forked
def test_fast_up():
    from src.gen import main
    main(gradio=True, block_gradio_exit=False)


@wrap_test_forked
def test_fast_up_preload():
    from src.gen import main
    import torch
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if n_gpus == 0:
        return
    main(gradio=True, block_gradio_exit=False,
         pre_load_image_audio_models=True,
         embedding_gpu_id=n_gpus - 1,
         caption_gpu_id=max(0, n_gpus - 2),
         doctr_gpu_id=max(0, n_gpus - 3),
         asr_gpu_id=max(0, n_gpus - 4),
         asr_model='openai/whisper-large-v3',
         )


@wrap_test_forked
def test_fast_up_auth():
    from src.gen import main
    main(gradio=True, block_gradio_exit=False, score_model='', langchain_mode='LLM', auth=[('jonny', 'dude')])
    # doesn't test login, has to be done manually


@wrap_test_forked
def test_fast_up_auth2():
    from src.gen import main
    main(gradio=True, block_gradio_exit=False, score_model='', langchain_mode='LLM', auth='')
    # doesn't test login, has to be done manually


@pytest.mark.parametrize("visible_models",
                         [None,
                          [0, 1],
                          "[0,1]",
                          "['h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3','gpt-3.5-turbo']",
                          ['h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3', 'gpt-3.5-turbo']
                          ])
@wrap_test_forked
def test_lock_up(visible_models):
    from src.gen import main
    main(gradio=True,
         model_lock=[{'base_model': 'h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v3'},
                     {'base_model': 'distilgpt2'},
                     {'inference_server': 'openai_chat', 'base_model': 'gpt-3.5-turbo'}],
         visible_models=visible_models,
         model_lock_columns=3,
         gradio_size='small',
         height=400,
         save_dir='save_gpt_test1',
         max_max_new_tokens=2048,
         max_new_tokens=1024,
         langchain_mode='MyData',
         block_gradio_exit=False)


@pytest.mark.skipif(not os.getenv('STRESS'), reason="Only for stress testing already-running server")
@pytest.mark.parametrize("repeat", list(range(0, 100)))
@wrap_test_forked
def test_client_stress(repeat):
    # pip install pytest-repeat  # license issues, don't put with requirements
    # pip install pytest-timeout  # license issues, don't put with requirements
    #
    # CUDA_VISIBLE_DEVICES=0 SCORE_MODEL=None python generate.py --base_model=h2oai/h2ogpt-gm-oasst1-en-2048-falcon-7b-v2 --langchain_mode=UserData --user_path=user_path --debug=True --concurrency_count=8
    #
    # timeout to mimic client disconnecting and generation still going, else too clean and doesn't fail STRESS=1
    # pytest -s -v -n 8 --timeout=30 tests/test_client_calls.py::test_client_stress 2> stress1.log
    # HOST=http://192.168.1.46:9999 STRESS=1 pytest -s -v -n 8 --timeout=1000 tests/test_client_calls.py::test_client_stress 2> stress1.log

    prompt = "Tell a very long kid's story about birds."
    # prompt = "Say exactly only one word."

    client = get_client(serialize=not is_gradio_version4)
    kwargs = dict(
        instruction='',
        max_new_tokens=200,
        min_new_tokens=1,
        max_time=300,
        do_sample=False,
        instruction_nochat=prompt,
    )

    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    print("Raw client result: %s" % res, flush=True)
    assert isinstance(res, str)
    res_dict = ast.literal_eval(res)
    assert 'response' in res_dict and res_dict['response']


@pytest.mark.skipif(not os.getenv('STRESS'), reason="Only for stress testing already-running server")
@pytest.mark.parametrize("repeat", list(range(0, 100)))
@wrap_test_forked
def test_client_stress_stream(repeat):
    prompt = "Tell a very long kid's story about birds."
    max_new_tokens = 200
    prompt_type = None
    langchain_mode = 'Disabled'
    stream_output = True
    chat = False

    client = get_client(serialize=not is_gradio_version4)
    kwargs, args = get_args(prompt, prompt_type, chat=chat, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode)
    res_dict, client = run_client_gen(client, kwargs, do_md_to_text=False)

    assert 'response' in res_dict and res_dict['response']


@pytest.mark.skipif(not os.getenv('SERVER'),
                    reason="For testing remote text-generatino-inference server")
@wrap_test_forked
def test_text_generation_inference_server1():
    """
    e.g.
    SERVER on 192.168.1.46
    (alpaca) jon@gpu:/data/jon/h2o-llm$ CUDA_VISIBLE_DEVICES=0,1 docker run --gpus all --shm-size 2g -e NCCL_SHM_DISABLE=1 -p 6112:80 -v $HOME/.cache/huggingface/hub/:/data  ghcr.io/huggingface/text-generation-inference:latest --model-id h2oai/h2ogpt-oasst1-512-12b --max-input-length 2048 --max-total-tokens 4096 --sharded=true --num-shard=2 --disable-custom-kernels --quantize bitsandbytes --trust-remote-code --max-stop-sequences=6

    CLIENT on separate system
    HOST=http://192.168.1.46:6112 SERVER=1 pytest -s -v tests/test_client_calls.py::test_text_generation_inference_server1

    :return:
    """

    # Python client test:
    from text_generation import Client

    host = os.getenv("HOST", "http://127.0.0.1:6112")
    client = Client(host)
    print(client.generate("What is Deep Learning?", max_new_tokens=17).generated_text)

    text = ""
    for response in client.generate_stream("What is Deep Learning?", max_new_tokens=17):
        if not response.token.special:
            text += response.token.text
    assert 'Deep learning is a subfield of machine learning' in text

    # Curl Test (not really pass fail yet)
    import subprocess
    output = subprocess.run(['curl', '%s/generate' % host, '-X', 'POST', '-d',
                             '{"inputs":"<|prompt|>What is Deep Learning?<|endoftext|><|answer|>","parameters":{"max_new_tokens": 20, "truncate": 1024, "do_sample": false, "temperature": 0.1, "repetition_penalty": 1.2}}',
                             '-H', 'Content-Type: application/json',
                             '--user', 'user:bhx5xmu6UVX4'],
                            check=True, capture_output=True).stdout.decode()
    text = ast.literal_eval(output)['generated_text']
    assert 'Deep learning is a subfield of machine learning' in text or \
           'Deep learning refers to a class of machine learning' in text


@pytest.mark.need_tokens
@pytest.mark.parametrize("enforce_h2ogpt_ui_key", [False, True])
@pytest.mark.parametrize("enforce_h2ogpt_api_key", [False, True])
@pytest.mark.parametrize("loaders", ['all', None])
@wrap_test_forked
def test_client_chat_stream_langchain_steps3(loaders, enforce_h2ogpt_api_key, enforce_h2ogpt_ui_key):
    os.environ['VERBOSE_PIPELINE'] = '1'
    user_path = make_user_path_test()

    if loaders is None:
        loaders = tuple([None, None, None, None, None, None])
    else:
        image_audio_loaders_options0, image_audio_loaders_options, \
            pdf_loaders_options0, pdf_loaders_options, \
            url_loaders_options0, url_loaders_options = \
            lg_to_gr(enable_ocr=True, enable_captions=True, enable_pdf_ocr=True,
                     enable_pdf_doctr=True,
                     use_pymupdf=True,
                     enable_doctr=True,
                     enable_pix2struct=True,
                     enable_transcriptions=True,
                     use_pypdf=True,
                     use_unstructured_pdf=True,
                     try_pdf_as_html=True,
                     enable_llava=True,
                     llava_model=None,
                     llava_prompt=None,
                     max_quality=True)
        # use all loaders except crawling ones
        url_loaders_options = [x for x in url_loaders_options if 'scrape' not in x.lower()]
        jq_schema = None
        extract_frames = 0
        llava_prompt = None
        loaders = [image_audio_loaders_options, pdf_loaders_options, url_loaders_options,
                   jq_schema, extract_frames, llava_prompt]

    stream_output = True
    max_new_tokens = 256
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'  # 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    prompt_type = 'llama2'  # 'human_bot'
    langchain_mode = 'UserData'
    langchain_modes = ['UserData', 'MyData', 'github h2oGPT', 'LLM', 'Disabled']

    from src.gen import main
    main_kwargs = {}
    h2ogpt_key = 'foodoo#'
    if enforce_h2ogpt_api_key:
        main_kwargs.update(dict(enforce_h2ogpt_api_key=True, h2ogpt_api_keys=[h2ogpt_key]))
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         append_sources_to_chat=False,
         **main_kwargs,
         verbose=True)

    from src.client_test import get_client, get_args, run_client
    # serialize=False would lead to returning dict for some objects or files for get_sources
    client = get_client(serialize=False)

    url = 'https://h2o-release.s3.amazonaws.com/h2ogpt/sample.pdf'
    test_file1 = os.path.join('/tmp/', 'sample1.pdf')
    download_simple(url, dest=test_file1)
    res = client.predict(test_file1,
                         langchain_mode, True, 512, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode
    # note moves from /tmp to stable path, even though not /tmp/gradio upload from UI
    assert 'file/%s/sample1.pdf' % user_path in res[2] or 'file/%s\sample1.pdf' % user_path in res[2]
    assert res[3] == ''

    # control langchain_mode
    user_path2 = makedirs('user_path2', use_base=True)  # so base accounted for
    langchain_mode2 = 'UserData2'
    remove(user_path2)
    remove('db_dir_%s' % langchain_mode2)
    new_langchain_mode_text = '%s, %s, %s' % (langchain_mode2, 'shared', user_path2)
    res = client.predict(langchain_mode, new_langchain_mode_text, h2ogpt_key, api_name='/new_langchain_mode_text')
    assert res[0]['value'] == langchain_mode2
    # odd gradio change
    res0_choices = [x[0] for x in res[0]['choices']]
    assert langchain_mode2 in res0_choices
    assert res[1] == ''
    assert res[2]['headers'] == ['Collection', 'Type', 'Path', 'Directory', 'Embedding', 'DB']
    res[2]['data'] = [[x[0], x[1], x[2]] for x in res[2]['data']]  # ignore persist_directory
    assert res[2]['data'] == [['UserData', 'shared', user_path],
                              ['github h2oGPT', 'shared', ''],
                              ['MyData', 'personal', ''],
                              [langchain_mode2, 'shared', user_path2]]

    # url = 'https://unec.edu.az/application/uploads/2014/12/pdf-sample.pdf'
    test_file1 = os.path.join('/tmp/', 'pdf-sample.pdf')
    # download_simple(url, dest=test_file1)
    shutil.copy('tests/pdf-sample.pdf', test_file1)
    res = client.predict(test_file1, langchain_mode2, True, 512, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode2
    assert 'file/%s/pdf-sample.pdf' % user_path2 in res[2] or 'file/%s\pdf-sample.pdf' % user_path2 in res[2]
    assert 'sample1.pdf' not in res[2]  # ensure no leakage
    assert res[3] == ''

    # QUERY1
    prompt = "Is more text boring?"
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                            h2ogpt_key=h2ogpt_key)

    res_dict, client = run_client(client, prompt, args, kwargs)
    assert ('Yes, more text can be boring' in res_dict['response'] or
            "can be considered boring" in res_dict['response'] or
            "the text in the provided PDF file is quite repetitive and boring" in res_dict['response'] or
            "the provided PDF file is quite boring" in res_dict['response'] or
            "finds more text to be boring" in res_dict['response'] or
            "text to be boring" in res_dict['response'] or
            "author finds more text to be boring" in res_dict['response'] or
            "more text is boring" in res_dict['response'] or
            "more text is boring" in res_dict['response'] or
            "it can be inferred that more text is indeed boring" in res_dict['response'] or
            "expressing frustration" in res_dict['response'] or
            "it seems that more text can indeed be boring" in res_dict['response']) \
           and 'sample1.pdf' in res_dict['response']
    # QUERY2
    prompt = "What is a universal file format?"
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            max_new_tokens=max_new_tokens, langchain_mode=langchain_mode2,
                            h2ogpt_key=h2ogpt_key)

    res_dict, client = run_client(client, prompt, args, kwargs)
    assert 'PDF' in res_dict['response'] and 'pdf-sample.pdf' in res_dict['response']

    # check sources, and do after so would detect leakage
    res = client.predict(langchain_mode, h2ogpt_key, api_name='/get_sources')
    # is not actual data!
    assert isinstance(res[1], str)
    res = res[0]
    if not is_gradio_version4:
        res = res['name']
    with open(res, 'rb') as f:
        sources = f.read().decode()
    sources_expected = f'{user_path}/FAQ.md\n{user_path}/README.md\n{user_path}/pexels-evg-kowalievska-1170986_small.jpg\n{user_path}/sample1.pdf'
    assert sources == sources_expected or sources.replace('\\', '/').replace('\r', '') == sources_expected.replace(
        '\\', '/').replace('\r', '')

    res = client.predict(langchain_mode2, h2ogpt_key, api_name='/get_sources')
    assert isinstance(res[1], str)
    res = res[0]
    if not is_gradio_version4:
        res = res['name']
    with open(res, 'rb') as f:
        sources = f.read().decode()
    sources_expected = """%s/pdf-sample.pdf""" % user_path2
    assert sources == sources_expected or sources.replace('\\', '/').replace('\r', '') == sources_expected.replace(
        '\\', '/').replace('\r', '')

    # check sources, and do after so would detect leakage
    res = client.predict(langchain_mode, h2ogpt_key, api_name='/get_viewable_sources')
    assert isinstance(res[1], str)
    res = res[0]
    # is not actual data!
    if not is_gradio_version4:
        res = res['name']
    with open(res, 'rb') as f:
        sources = f.read().decode()
    sources_expected = f'{user_path}/FAQ.md\n{user_path}/README.md\n{user_path}/pexels-evg-kowalievska-1170986_small.jpg\n{user_path}/sample1.pdf'
    assert sources == sources_expected or sources.replace('\\', '/').replace('\r', '') == sources_expected.replace(
        '\\', '/').replace('\r', '')

    res = client.predict(langchain_mode2, h2ogpt_key, api_name='/get_viewable_sources')
    assert isinstance(res[1], str)
    res = res[0]
    if not is_gradio_version4:
        res = res['name']
    with open(res, 'rb') as f:
        sources = f.read().decode()
    sources_expected = """%s/pdf-sample.pdf""" % user_path2
    assert sources == sources_expected or sources.replace('\\', '/').replace('\r', '') == sources_expected.replace(
        '\\', '/').replace('\r', '')

    # refresh
    shutil.copy('tests/next.txt', user_path)
    res = client.predict(langchain_mode, True, 512,
                         *loaders, h2ogpt_key,
                         api_name='/refresh_sources')
    sources_expected = 'file/%s/next.txt' % user_path
    assert sources_expected in res or sources_expected.replace('\\', '/').replace('\r', '') in res.replace('\\',
                                                                                                           '/').replace(
        '\r', '\n')

    res = client.predict(langchain_mode, h2ogpt_key, api_name='/get_sources')
    assert isinstance(res[1], str)
    res = res[0]
    # is not actual data!
    if not is_gradio_version4:
        res = res['name']
    with open(res, 'rb') as f:
        sources = f.read().decode()
    sources_expected = f'{user_path}/FAQ.md\n{user_path}/README.md\n{user_path}/next.txt\n{user_path}/pexels-evg-kowalievska-1170986_small.jpg\n{user_path}/pexels-evg-kowalievska-1170986_small.jpg_rotated.jpg\n{user_path}/pexels-evg-kowalievska-1170986_small.jpg_rotated.jpg_pad_resized.png\n{user_path}/sample1.pdf'
    assert sources == sources_expected or sources.replace('\\', '/').replace('\r', '') == sources_expected.replace(
        '\\', '/').replace('\r', '')

    # check sources, and do after so would detect leakage
    sources = ast.literal_eval(client.predict(langchain_mode, h2ogpt_key, api_name='/get_sources_api'))
    assert isinstance(sources, list)
    sources_expected = ['user_path_test/FAQ.md', 'user_path_test/README.md', 'user_path_test/next.txt',
                        'user_path_test/pexels-evg-kowalievska-1170986_small.jpg',
                        'user_path_test/pexels-evg-kowalievska-1170986_small.jpg_rotated.jpg',
                        'user_path_test/pexels-evg-kowalievska-1170986_small.jpg_rotated.jpg_pad_resized.png',
                        'user_path_test/sample1.pdf']
    assert sources == sources_expected

    file_to_get = sources_expected[3]
    view_raw_text = False
    text_context_list = None
    pdf_height = 1000
    source_dict = ast.literal_eval(
        client.predict(langchain_mode, file_to_get, view_raw_text, text_context_list, pdf_height, h2ogpt_key,
                       api_name='/get_document_api'))
    assert len(source_dict['contents']) == 1
    assert len(source_dict['metadatas']) == 1
    assert isinstance(source_dict['contents'][0], str)
    assert 'a cat sitting on a window' in source_dict['contents'][0]
    assert isinstance(source_dict['metadatas'][0], str)
    assert sources_expected[3] in source_dict['metadatas'][0]

    view_raw_text = True  # dict of metadatas stays dict instead of string
    source_dict = ast.literal_eval(
        client.predict(langchain_mode, file_to_get, view_raw_text, text_context_list, pdf_height, h2ogpt_key,
                       api_name='/get_document_api'))
    assert len(source_dict['contents']) == 2  # chunk_id=0 (query) and -1 (summarization)
    assert len(source_dict['metadatas']) == 2  # chunk_id=0 (query) and -1 (summarization)
    assert isinstance(source_dict['contents'][0], str)
    assert 'a cat sitting on a window' in source_dict['contents'][0]
    assert isinstance(source_dict['metadatas'][0], dict)
    assert sources_expected[3] == source_dict['metadatas'][0]['source']

    # even normal langchain_mode  passed to this should get the other langchain_mode2
    res = client.predict(langchain_mode, h2ogpt_key, api_name='/load_langchain')
    res0_choices = [x[0] for x in res[0]['choices']]
    assert res0_choices == [langchain_mode, 'MyData', 'github h2oGPT', 'LLM', langchain_mode2]
    assert res[0]['value'] == langchain_mode
    assert res[1]['headers'] == ['Collection', 'Type', 'Path', 'Directory', 'Embedding', 'DB']
    res[1]['data'] = [[x[0], x[1], x[2]] for x in res[1]['data']]  # ignore persist_directory
    assert res[1]['data'] == [['UserData', 'shared', user_path],
                              ['github h2oGPT', 'shared', ''],
                              ['MyData', 'personal', ''],
                              [langchain_mode2, 'shared', user_path2]]

    # for pure-UI things where just input -> output often, just make sure no failure, if can
    res = client.predict(api_name='/export_chats')
    assert res is not None

    url = 'https://services.google.com/fh/files/misc/e_conomy_sea_2021_report.pdf'
    res = client.predict(url, langchain_mode, True, 512, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_url')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert url in res[2]
    assert res[3] == ''

    text = "Yufuu is a wonderful place and you should really visit because there is lots of sun."
    res = client.predict(text, langchain_mode, True, 512, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_text')
    assert res[0] is None
    assert res[1] == langchain_mode
    user_paste_dir = makedirs('user_paste', use_base=True)
    remove(user_paste_dir)
    sources_expected = 'file/%s/' % user_paste_dir
    assert sources_expected in res[2] or sources_expected.replace('\\', '/').replace('\r', '') in res[2].replace('\\',
                                                                                                                 '/').replace(
        '\r', '\n')
    assert res[3] == ''

    langchain_mode_my = LangChainMode.MY_DATA.value
    url = 'https://h2o-release.s3.amazonaws.com/h2ogpt/sample.pdf'
    test_file1 = os.path.join('/tmp/', 'sample1.pdf')
    download_simple(url, dest=test_file1)
    res = client.predict(test_file1, langchain_mode_my, True, 512, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode_my
    # will just use source location, e.g. for UI will be /tmp/gradio
    sources_expected = 'file//tmp/sample1.pdf'
    assert sources_expected in res[2] or sources_expected.replace('\\', '/').replace('\r', '') in res[2].replace('\\',
                                                                                                                 '/').replace(
        '\r', '\n')
    assert res[3] == ''

    # control langchain_mode
    user_path2b = ''
    langchain_mode2 = 'MyData2'
    new_langchain_mode_text = '%s, %s, %s' % (langchain_mode2, 'personal', user_path2b)
    res = client.predict(langchain_mode2, new_langchain_mode_text, h2ogpt_key, api_name='/new_langchain_mode_text')
    assert res[0]['value'] == langchain_mode2
    res0_choices = [x[0] for x in res[0]['choices']]
    assert langchain_mode2 in res0_choices
    assert res[1] == ''
    assert res[2]['headers'] == ['Collection', 'Type', 'Path', 'Directory', 'Embedding', 'DB']
    res[2]['data'] = [[x[0], x[1], x[2]] for x in res[2]['data']]  # ignore persist_directory
    assert res[2]['data'] == [['UserData', 'shared', user_path],
                              ['github h2oGPT', 'shared', ''],
                              ['MyData', 'personal', ''],
                              ['UserData2', 'shared', user_path2],
                              [langchain_mode2, 'personal', '']]

    # url = 'https://unec.edu.az/application/uploads/2014/12/pdf-sample.pdf'
    test_file1 = os.path.join('/tmp/', 'pdf-sample.pdf')
    # download_simple(url, dest=test_file1)
    shutil.copy('tests/pdf-sample.pdf', test_file1)
    res = client.predict(test_file1, langchain_mode2, True, 512, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode2
    sources_expected = 'file//tmp/pdf-sample.pdf'
    assert sources_expected in res[2] or sources_expected.replace('\\', '/').replace('\r', '') in res[2].replace('\\',
                                                                                                                 '/').replace(
        '\r', '\n')
    assert 'sample1.pdf' not in res[2]  # ensure no leakage
    assert res[3] == ''

    urls = ['https://h2o.ai/company/team/leadership-team/',
            'https://arxiv.org/abs/1706.03762',
            'https://github.com/h2oai/h2ogpt',
            'https://h2o.ai'
            ]
    with tempfile.TemporaryDirectory() as tmp_user_path:
        urls_file = os.path.join(tmp_user_path, 'list.urls')
        with open(urls_file, 'wt') as f:
            f.write('\n'.join(urls))
        res = client.predict(urls_file, langchain_mode2, True, 512, True,
                             *loaders,
                             h2ogpt_key,
                             api_name='/add_file_api')
        assert res[0] is None
        assert res[1] == langchain_mode2
        assert [x in res[2] or x.replace('https', 'http') in res[2] for x in urls]
        assert res[3] == ''

    langchain_mode3 = 'MyData3'
    user_path3 = ''
    new_langchain_mode_text = '%s, %s, %s' % (langchain_mode3, 'personal', user_path3)
    res = client.predict(langchain_mode3, new_langchain_mode_text, h2ogpt_key, api_name='/new_langchain_mode_text')
    assert res[0]['value'] == langchain_mode3
    res0_choices = [x[0] for x in res[0]['choices']]
    assert langchain_mode3 in res0_choices
    assert res[1] == ''
    assert res[2]['headers'] == ['Collection', 'Type', 'Path', 'Directory', 'Embedding', 'DB']
    res[2]['data'] = [[x[0], x[1], x[2]] for x in res[2]['data']]  # ignore persist_directory
    assert res[2]['data'] == [['UserData', 'shared', user_path],
                              ['github h2oGPT', 'shared', ''],
                              ['MyData', 'personal', ''],
                              ['UserData2', 'shared', user_path2],
                              [langchain_mode2, 'personal', ''],
                              [langchain_mode3, 'personal', ''],
                              ]

    with tempfile.TemporaryDirectory() as tmp_user_path:
        res = client.predict(urls, langchain_mode3, True, 512, True,
                             *loaders,
                             h2ogpt_key,
                             api_name='/add_url')
        print(res)
        assert res[0] is None
        assert res[1] == langchain_mode3
        assert [x in res[2] or x.replace('https', 'http') in res[2] for x in urls]
        assert res[3] == ''

    sources_text = client.predict(langchain_mode3, h2ogpt_key, api_name='/show_sources')
    assert isinstance(sources_text, str)
    assert [x in sources_text or x.replace('https', 'http') in sources_text for x in urls]

    source_list = ast.literal_eval(client.predict(langchain_mode3, h2ogpt_key, api_name='/get_sources_api'))
    source_list_assert = [x.replace('v1', '').replace('v7', '') for x in source_list]  # for arxiv for asserts
    assert isinstance(source_list, list)
    assert [x in source_list_assert or x.replace('https', 'http') in source_list_assert for x in urls]

    sources_text_after_delete = client.predict(source_list[0], langchain_mode3, h2ogpt_key, api_name='/delete_sources')
    source_list_assert = [x.replace('v1', '').replace('v7', '') for x in source_list]  # for arxiv for asserts
    assert source_list_assert[0] not in sources_text_after_delete

    sources_state_after_delete = ast.literal_eval(
        client.predict(langchain_mode3, h2ogpt_key, api_name='/get_sources_api'))
    sources_state_after_delete = [x.replace('v1', '').replace('v7', '') for x in
                                  sources_state_after_delete]  # for arxiv for asserts
    assert isinstance(sources_state_after_delete, list)
    source_list_assert = [x.replace('v1', '').replace('v7', '') for x in source_list]  # for arxiv for asserts
    assert source_list_assert[0] not in sources_state_after_delete

    res = client.predict(langchain_mode3, langchain_mode3, h2ogpt_key, api_name='/remove_langchain_mode_text')
    assert res[0]['value'] == langchain_mode3
    res0_choices = [x[0] for x in res[0]['choices']]
    assert langchain_mode2 in res0_choices
    assert res[1] == ''
    assert res[2]['headers'] == ['Collection', 'Type', 'Path', 'Directory', 'Embedding', 'DB']
    res[2]['data'] = [[x[0], x[1], x[2]] for x in res[2]['data']]  # ignore persist_directory
    assert res[2]['data'] == [['UserData', 'shared', user_path],
                              ['github h2oGPT', 'shared', ''],
                              ['MyData', 'personal', ''],
                              ['UserData2', 'shared', user_path2],
                              [langchain_mode2, 'personal', '']]

    assert os.path.isdir("db_dir_%s" % langchain_mode)
    res = client.predict(langchain_mode, langchain_mode, h2ogpt_key, api_name='/purge_langchain_mode_text')
    assert not os.path.isdir("db_dir_%s" % langchain_mode)
    assert res[0]['value'] == langchain_mode
    res0_choices = [x[0] for x in res[0]['choices']]
    assert langchain_mode not in res0_choices
    assert res[1] == ''
    assert res[2]['headers'] == ['Collection', 'Type', 'Path', 'Directory', 'Embedding', 'DB']
    res[2]['data'] = [[x[0], x[1], x[2]] for x in res[2]['data']]  # ignore persist_directory
    assert res[2]['data'] == [['github h2oGPT', 'shared', ''],
                              ['MyData', 'personal', ''],
                              ['UserData2', 'shared', 'user_path2'],
                              ['MyData2', 'personal', ''],
                              ]


@pytest.mark.need_tokens
@pytest.mark.parametrize("model_choice", ['h2oai/h2ogpt-oig-oasst1-512-6_9b'] + model_names_curated)
@wrap_test_forked
def test_client_load_unload_models(model_choice):
    if model_choice in model_names_curated_big:
        return
    os.environ['VERBOSE_PIPELINE'] = '1'
    user_path = make_user_path_test()

    stream_output = True
    max_new_tokens = 256
    base_model = ''
    prompt_type = 'human_bot'
    langchain_mode = 'UserData'
    langchain_modes = ['UserData', 'MyData', 'github h2oGPT', 'LLM', 'Disabled']

    from src.gen import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         score_model='',
         verbose=True)

    from src.client_test import get_client, get_args, run_client
    # serialize=False would lead to returning dict for some objects or files for get_sources
    client = get_client(serialize=False)

    lora_choice = ''
    server_choice = '' if model_choice not in openai_gpts else 'openai_chat'
    # model_state
    prompt_type = '' if model_choice != 'llama' else 'llama2'  # built-in, but prompt_type needs to be selected
    model_load8bit_checkbox = False
    model_load4bit_checkbox = 'AWQ' not in model_choice and 'GGUF' not in model_choice and 'GPTQ' not in model_choice
    model_low_bit_mode = 1
    model_load_gptq = ''
    model_load_awq = ''
    model_load_exllama_checkbox = False
    model_safetensors_checkbox = False
    model_revision = ''
    model_use_gpu_id_checkbox = True
    model_gpu_id = 0
    max_seq_len = -1
    rope_scaling = '{}'
    # GGML:
    model_path_llama = 'https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf?download=true' if model_choice == 'llama' else ''
    model_name_gptj = ''
    model_name_gpt4all_llama = ''
    n_gpu_layers = 100
    n_batch = 128
    n_gqa = 0  # llama2 needs 8
    llamacpp_dict_more = '{}'
    system_prompt = None
    model_cpu = False
    exllama_dict = "{}"
    gptq_dict = "{}"
    attention_sinks = False
    sink_dict = "{}"
    truncation_generation = False
    hf_model_dict = "{}"
    model_force_seq2seq_type = False
    model_force_force_t5_type = False
    args_list = [model_choice, lora_choice, server_choice,
                 # model_state,
                 prompt_type,
                 model_load8bit_checkbox, model_load4bit_checkbox, model_low_bit_mode,
                 model_load_gptq, model_load_awq, model_load_exllama_checkbox,
                 model_safetensors_checkbox, model_revision,
                 model_cpu,
                 model_use_gpu_id_checkbox, model_gpu_id,
                 max_seq_len, rope_scaling,
                 model_path_llama, model_name_gptj, model_name_gpt4all_llama,
                 n_gpu_layers, n_batch, n_gqa, llamacpp_dict_more,
                 system_prompt,
                 exllama_dict, gptq_dict, attention_sinks, sink_dict, truncation_generation, hf_model_dict,
                 model_force_seq2seq_type, model_force_force_t5_type,
                 ]
    res = client.predict(*tuple(args_list), api_name='/load_model')

    model_choice_ex = model_choice
    model_load_gptq_ex = 'model' if 'GPTQ' in model_choice else ''
    model_load_awq_ex = 'model' if 'AWQ' in model_choice else ''
    model_path_llama_ex = 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf?download=true' if model_choice == 'llama' else ''

    if model_choice == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
        prompt_type_ex = 'human_bot'
        max_seq_len_ex = 2048.0
        max_seq_len_ex2 = max_seq_len_ex
    elif model_choice in ['llama']:
        prompt_type_ex = 'llama2'
        model_choice_ex = 'llama'
        model_path_llama_ex = 'https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf?download=true'
        max_seq_len_ex = 4096.0
        max_seq_len_ex2 = max_seq_len_ex
    elif model_choice in ['TheBloke/Llama-2-7B-Chat-GGUF']:
        prompt_type_ex = 'llama2'
        model_choice_ex = 'llama'
        model_path_llama_ex = 'https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q5_K_M.gguf?download=true'
        max_seq_len_ex = 4096.0
        max_seq_len_ex2 = max_seq_len_ex
    elif model_choice in ['TheBloke/zephyr-7B-beta-GGUF']:
        prompt_type_ex = 'zephyr'
        model_choice_ex = 'llama'
        model_path_llama_ex = 'https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q5_K_M.gguf?download=true'
        max_seq_len_ex = 4096.0
        max_seq_len_ex2 = max_seq_len_ex
    elif model_choice in ['HuggingFaceH4/zephyr-7b-beta',
                          'TheBloke/zephyr-7B-beta-AWQ']:
        prompt_type_ex = 'zephyr'
        max_seq_len_ex = 4096.0
        max_seq_len_ex2 = max_seq_len_ex
    elif model_choice in ['TheBloke/Xwin-LM-13B-V0.1-GPTQ']:
        prompt_type_ex = 'xwin'
        max_seq_len_ex = 4096.0
        max_seq_len_ex2 = max_seq_len_ex
    elif model_choice in ['gpt-3.5-turbo']:
        prompt_type_ex = 'openai_chat'
        max_seq_len_ex = 4096.0
        max_seq_len_ex2 = 4046
    else:
        raise ValueError("No such model_choice=%s" % model_choice)
    res_expected = (
        model_choice_ex, '', server_choice, prompt_type_ex, max_seq_len_ex2,
        {'__type__': 'update', 'maximum': int(max_seq_len_ex)},
        {'__type__': 'update', 'maximum': int(max_seq_len_ex)},
        model_path_llama_ex,
        '', '',
        model_load_gptq_ex, model_load_awq_ex,
        0.0, 128.0, 100.0, '{}')
    assert res == res_expected

    prompt = "Who are you?"
    kwargs = dict(stream_output=stream_output, instruction=prompt)
    res_dict, client = run_client_gen(client, kwargs)
    response = res_dict['response']
    assert response

    # unload (could use unload api)
    args_list[0] = no_model_str
    res = client.predict(*tuple(args_list), api_name='/load_model')
    res_expected = (no_model_str, no_lora_str, no_server_str, '', -1.0, {'__type__': 'update', 'maximum': 256},
                    {'__type__': 'update', 'maximum': 256},
                    '',
                    '', '',
                    '', '',
                    0.0, 128.0, 100.0, '{}')
    assert res == res_expected


@pytest.mark.need_tokens
@pytest.mark.parametrize("stream_output", [True, False])
@pytest.mark.parametrize("base_model", ['h2oai/h2ogpt-oig-oasst1-512-6_9b'] +
                         model_names_curated +
                         ['zephyr-7b-beta.Q5_K_M.gguf'] +
                         [
                             'https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/resolve/main/llama-2-7b-chat.Q6_K.gguf?download=true'])
@wrap_test_forked
def test_client_curated_base_models(base_model, stream_output):
    if base_model in model_names_curated_big:
        return
    if base_model == 'zephyr-7b-beta.Q5_K_M.gguf' and not os.path.isfile('zephyr-7b-beta.Q5_K_M.gguf'):
        download_simple(
            'https://huggingface.co/TheBloke/zephyr-7B-beta-GGUF/resolve/main/zephyr-7b-beta.Q5_K_M.gguf?download=true')

    stream_output = True
    from src.gen import main
    main_kwargs = dict(base_model=base_model,
                       inference_server='' if base_model not in openai_gpts else 'openai_chat',
                       chat=True,
                       stream_output=stream_output,
                       gradio=True, num_beams=1, block_gradio_exit=False,
                       score_model='',
                       verbose=True)
    if 'resolve' in base_model:
        main_kwargs['prompt_type'] = 'llama2'
    main(**main_kwargs)

    from src.client_test import get_client
    # serialize=False would lead to returning dict for some objects or files for get_sources
    client = get_client(serialize=False)

    prompt = "Who are you?"
    kwargs = dict(stream_output=stream_output, instruction=prompt)
    res_dict, client = run_client_gen(client, kwargs)
    response = res_dict['response']
    assert response


@pytest.mark.need_tokens
@wrap_test_forked
def test_client_chat_stream_langchain_openai_embeddings():
    os.environ['VERBOSE_PIPELINE'] = '1'
    user_path = make_user_path_test()
    remove('db_dir_UserData')

    stream_output = True
    max_new_tokens = 256
    base_model = 'distilgpt2'
    prompt_type = 'human_bot'
    langchain_mode = 'UserData'
    langchain_modes = ['UserData', 'MyData', 'github h2oGPT', 'LLM', 'Disabled']

    from src.gen import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         max_new_tokens=max_new_tokens,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         use_openai_embedding=True,
         verbose=True)

    from src.client_test import get_client, get_args, run_client
    # serialize=False would lead to returning dict for some objects or files for get_sources
    client = get_client(serialize=False)

    url = 'https://h2o-release.s3.amazonaws.com/h2ogpt/sample.pdf'
    test_file1 = os.path.join('/tmp/', 'sample1.pdf')
    download_simple(url, dest=test_file1)
    loaders = tuple([None, None, None, None, None, None])
    h2ogpt_key = ''
    res = client.predict(test_file1, langchain_mode, True, 512, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode
    # note moves from /tmp to stable path, even though not /tmp/gradio upload from UI
    assert 'file/%s/sample1.pdf' % user_path in res[2] or 'file/%s\sample1.pdf' % user_path in res[2]
    assert res[3] == ''

    from src.gpt_langchain import load_embed
    got_embedding, use_openai_embedding, hf_embedding_model = load_embed(persist_directory='db_dir_UserData')
    assert use_openai_embedding
    assert hf_embedding_model in ['', 'hkunlp/instructor-large']  # but not used
    assert got_embedding


@pytest.mark.parametrize("stream_output", [True, False])
@pytest.mark.need_tokens
@wrap_test_forked
def test_client_clone(stream_output):
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'
    from src.gen import main
    main(base_model=base_model, block_gradio_exit=False, verbose=True)

    from gradio_utils.grclient import GradioClient
    client1 = GradioClient(get_inf_server())
    client1.setup()
    client2 = client1.clone()

    for client in [client1, client2]:
        prompt = "Who are you?"
        kwargs = dict(stream_output=stream_output, instruction=prompt)
        res_dict, client = run_client_gen(client, kwargs)
        response = res_dict['response']
        assert len(response) > 0
        sources = res_dict['sources']
        assert sources == []


@pytest.mark.parametrize("max_time", [1, 5])
@pytest.mark.parametrize("stream_output", [True, False])
@pytest.mark.need_tokens
@wrap_test_forked
def test_client_timeout(stream_output, max_time):
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'
    from src.gen import main
    main(base_model=base_model, block_gradio_exit=False, verbose=True)

    # PURE client code
    from gradio_client import Client
    client = Client(get_inf_server())

    prompt = "Tell a very long kid's story about birds"
    kwargs = dict(stream_output=stream_output, instruction=prompt, max_time=max_time)
    t0 = time.time()
    res_dict, client = run_client_gen(client, kwargs)
    response = res_dict['response']
    assert len(response) > 0
    assert time.time() - t0 < max_time * 2
    sources = res_dict['sources']
    assert sources == []

    # get file for client to upload
    url = 'https://cdn.openai.com/papers/whisper.pdf'
    test_file1 = os.path.join('/tmp/', 'whisper1.pdf')
    download_simple(url, dest=test_file1)

    # PURE client code
    from gradio_client import Client
    client = Client(get_inf_server())

    # upload file(s).  Can be list or single file
    test_file_local, test_file_server = client.predict(test_file1, api_name='/upload_api')

    chunk = True
    chunk_size = 512
    langchain_mode = 'MyData'
    loaders = tuple([None, None, None, None, None, None])
    h2ogpt_key = ''
    res = client.predict(test_file_server,
                         langchain_mode, chunk, chunk_size, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert os.path.basename(test_file_server) in res[2]
    assert res[3] == ''

    # ask for summary, need to use same client if using MyData
    instruction = "Give a very long detailed step-by-step description of what is Whisper paper about."
    kwargs = dict(instruction=instruction,
                  langchain_mode=langchain_mode,
                  langchain_action="Query",
                  top_k_docs=4,
                  document_subset='Relevant',
                  document_choice=DocumentChoice.ALL.value,
                  max_new_tokens=1024,
                  max_time=max_time,
                  do_sample=False,
                  stream_output=stream_output,
                  )
    t0 = time.time()
    res_dict, client = run_client_gen(client, kwargs)
    response = res_dict['response']
    assert len(response) > 0
    # assert len(response) < max_time * 20  # 20 tokens/sec
    assert time.time() - t0 < max_time * 2.5
    sources = [x['source'] for x in res_dict['sources']]
    # only get source not empty list if break in inner loop, not gradio_runner loop, so good test of that too
    # this is why gradio timeout adds 10 seconds, to give inner a chance to produce references or other final info
    assert 'whisper1.pdf' in sources[0]


# pip install pytest-timeout
# HOST=http://192.168.1.46:9999 STRESS=1 pytest -s -v -n 8 --timeout=1000 tests/test_client_calls.py::test_client_chat_stream_langchain_fake_embeddings_stress 2> stress1.log
@pytest.mark.skipif(not os.getenv('STRESS'), reason="Only for stress testing already-running server")
@pytest.mark.parametrize("repeat", list(range(0, 100)))
@wrap_test_forked
def test_client_chat_stream_langchain_fake_embeddings_stress(repeat):
    data_kind = 'helium3'
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'  # presumes remote server is llama-2 chat based
    local_server = False
    inference_server = None
    # inference_server = 'http://localhost:7860'
    return run_client_chat_stream_langchain_fake_embeddings(data_kind, base_model, local_server, inference_server)


# pip install pytest-timeout
# HOST=http://192.168.1.46:9999 STRESS=1 pytest -s -v -n 8 --timeout=1000 tests/test_client_calls.py::test_client_upload_simple 2> stress1.log
@pytest.mark.skipif(not os.getenv('STRESS'), reason="Only for stress testing already-running server")
@pytest.mark.parametrize("repeat", list(range(0, 100)))
@wrap_test_forked
def test_client_upload_simple(repeat):
    data_kind = 'helium3'
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'  # fake, just for tokenizer
    local_server = False
    inference_server = None
    # used with go_upload_gradio (say on remote machine) to test add_text
    return run_client_chat_stream_langchain_fake_embeddings(data_kind, base_model, local_server, inference_server,
                                                            simple=True)


# pip install pytest-timeout
# HOST=http://192.168.1.46:9999 STRESS=1 pytest -s -v -n 8 --timeout=1000 tests/test_client_calls.py::test_client_chat_stream_langchain_fake_embeddings_stress_no_llm 2> stress1.log
@pytest.mark.skipif(not os.getenv('STRESS'), reason="Only for stress testing already-running server")
@pytest.mark.parametrize("repeat", list(range(0, 100)))
@wrap_test_forked
def test_client_chat_stream_langchain_fake_embeddings_stress_no_llm(repeat):
    data_kind = 'helium3'
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'  # presumes remote server is llama-2 chat based
    local_server = False
    chat = False
    inference_server = None
    return run_client_chat_stream_langchain_fake_embeddings(data_kind, base_model, local_server, inference_server,
                                                            chat=chat)


def go_upload_gradio():
    import gradio as gr
    import time

    with gr.Blocks() as demo:
        chatbot = gr.Chatbot()
        msg = gr.Textbox()
        clear = gr.ClearButton([msg, chatbot])
        with gr.Accordion("Upload", open=False, visible=True):
            with gr.Column():
                with gr.Row(equal_height=False):
                    file = gr.File(show_label=False,
                                   file_count="multiple",
                                   scale=1,
                                   min_width=0,
                                   )

        def respond(message, chat_history):
            if not chat_history:
                chat_history = [[message, '']]
            chat_history[-1][1] = message
            for fake in range(0, 1000):
                chat_history[-1][1] += str(fake)
                time.sleep(0.1)
                yield "", chat_history
            return

        def gofile(x):
            print(x)
            return x

        user_text_text = gr.Textbox(label='Paste Text',
                                    interactive=True,
                                    visible=True)

        msg.submit(respond, [msg, chatbot], [msg, chatbot])

        def show_text(x):
            return str(x)

        user_text_text.submit(fn=show_text, inputs=user_text_text, outputs=user_text_text, api_name='add_text')

        eventdb1 = file.upload(gofile, file, api_name='file')

    if __name__ == "__main__":
        demo.queue(concurrency_count=64)
        demo.launch(server_name='0.0.0.0')


# NOTE: llama-7b on 24GB will go OOM for helium1/2 tests
@pytest.mark.parametrize("repeat", range(0, 1))
# @pytest.mark.parametrize("inference_server", ['http://localhost:7860'])
@pytest.mark.parametrize("inference_server", [None, 'openai', 'openai_chat', 'openai_azure_chat', 'replicate'])
# local_server=True
# @pytest.mark.parametrize("base_model",
#                         ['h2oai/h2ogpt-4096-llama2-13b-chat'])
# local_server=False or True if inference_server used
# @pytest.mark.parametrize("base_model", ['h2oai/h2ogpt-4096-llama2-70b-chat'])
@pytest.mark.parametrize("base_model",
                         ['h2oai/h2ogpt-oig-oasst1-512-6_9b', 'h2oai/h2ogpt-4096-llama2-7b-chat', 'gpt-3.5-turbo'])
@pytest.mark.parametrize("data_kind", [
    'simple',
    'helium1',
    'helium2',
    'helium3',
    'helium4',
    'helium5',
])
@wrap_test_forked
def test_client_chat_stream_langchain_fake_embeddings(data_kind, base_model, inference_server, repeat):
    # local_server = False  # set to False to test local server, e.g. gradio connected to TGI server
    local_server = True  # for gradio connected to TGI, or if pass inference_server too then some remote vLLM/TGI using local server
    return run_client_chat_stream_langchain_fake_embeddings(data_kind, base_model, local_server, inference_server)


texts_simple = ['first', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'last']

texts_helium1 = [
    '464 $ \n453 \n$ \n97 \n$ 125 $ 131 \n$ \n96 \n$ 89 $ \n84 \n$ 2,417 \n$ 2,291 $ 2,260 \nAverage loans\n291 \n287 \n298 \n321 \n307 \n304 \n41 \n74 \n83 \n— \n— \n— \n653 \n668 \n685 \nAverage deposits\n830 \n828 \n780 \n435 \n417 \n358 \n52 \n82 \n81 \n16 \n8 \n11 \n1,333 \n1,335 1,230 \n(1) \nIncludes total Citi revenues, net of interest expense (excluding \nCorporate/Other\n), in North America of $34.4 billion, $34.4 billion and $37.1 billion; in EMEA of',
    'Legacy Franchises\nCorporate/Other\nTotal Citi\n2022\n2021\n2020\n2022\n2021\n2020\n2022\n2021\n2020\n2022\n2021\n2020\n2022\n2021\n2020\nIn millions of \ndollars, except \nidentifiable assets, \naverage loans and \naverage deposits in \nbillions\nNet interest \nincome\n$ 17,911 \n$ 14,999 $ 15,750 \n$ 22,656 \n$ 20,646 $ 22,326 \n$ 5,691 \n$ 6,250 $ 6,973 \n$ 2,410 \n$ 599 $ (298) \n$ 48,668 \n$ 42,494 $ 44,751 \nNon-interest \nrevenue\n23,295 \n24,837 25,343 \n1,561 \n2,681 2,814 \n2,781 \n2,001 2,481 \n(967) \n(129) \n112 \n26,670 \n29,390 30,750',
    'Personal Banking and Wealth Management\n24,217 \n23,327 \n25,140 \n4 \n(7) \nLegacy Franchises\n8,472 \n8,251 \n9,454 \n3 \n(13) \nCorporate/Other\n1,443 \n470 \n(186) \nNM\nNM\nTotal Citigroup net revenues\n$ \n75,338 \n$ \n71,884 $ \n75,501 \n5 %\n(5) %\nNM Not meaningful\nINCOME\n% Change\n% Change\n2022 vs. 2021\n2021 vs. 2020\nIn millions of dollars\n2022\n2021\n2020\nIncome (loss) from continuing operations\nInstitutional Clients Group\n$ \n10,738 \n$ \n14,308 $ \n10,811 \n(25) %\n32 %\nPersonal Banking and Wealth Management\n3,319 \n7,734 \n1,322',
    '(2)\n307 \n(140) \n(59) \nNM\nNM\nTotal Banking revenues (including gains (losses) on loan \nhedges)\n(2)\n$ \n6,071 \n$ \n9,378 $ \n7,233 \n(35) %\n30 %\nTotal \nICG\nrevenues, net of interest expense\n$ \n41,206 \n$ \n39,836 $ \n41,093 \n3 %\n(3) %\n(1) \nCiti assesses its Markets business performance on a total revenue basis, as offsets may occur across revenue line items. For example, securities that generate \nNet \ninterest income\nmay be risk managed by derivatives that are recorded in \nPrincipal transactions\nrevenue within',
    'higher revenues. Citigroup’s effective tax rate was 19.4% in \nthe current year versus 19.8% in the prior year. Earnings per \nshare (EPS) decreased 31%, reflecting the decrease in net \nincome, partially offset by a 4% decline in average diluted \nshares outstanding.\nAs discussed above, results for 2022 included divestiture-\n•\nCiti’s revenues increased 5% versus the prior year, \nincluding net gains on sales of Citi’s Philippines and \nThailand consumer banking businesses versus a loss on',
    'Citigroup reported net income of $14.8 billion, or $7.00 per \nshare, compared to net income of $22.0 billion, or $10.14 per \nshare in the prior year. The decrease in net income was \nprimarily driven by the higher cost of credit, resulting from \nloan growth in \nPersonal Banking and Wealth Management \n(PBWM)\nand a deterioration in macroeconomic assumptions, \n3\nPolicies and Significant Estimates—Citi’s Allowance for \nCredit Losses (ACL)” below.\nNet credit losses of $3.8 billion decreased 23% from the',
    'The Company’s operating leases, where Citi is a lessor, \nCommercial and industrial\n$ \n56,176 \n$ \n48,364 \nare not significant to the Consolidated Financial Statements.\nFinancial institutions\n43,399 \n49,804 \nMortgage and real estate\n(2)\n17,829 \n15,965 \nInstallment and other\n23,767 \n20,143 \nLease financing\n308 \n415 \nTotal\n$ \n141,479 \n$ \n134,691 \nIn offices outside North America\n(1)\nCommercial and industrial\n$ \n93,967 \n$ \n102,735 \nFinancial institutions\n21,931 \n22,158 \nMortgage and real estate\n(2)\n4,179 \n4,374',
    '$1.8 billion in assets, including $1.2 billion of loans (net of allowance of $80 million) and excluding goodwill. The total amount of liabilities was $1.3 billion, \nincluding $1.2 billion in deposits. The sale resulted in a pretax gain on sale of approximately $618 million ($290 million after-tax), subject to closing adjustments, \nrecorded in \nOther revenue\n. The income before taxes shown in the above table for the Philippines reflects Citi’s ownership through August 1, 2022.\n(4)',
    'net interest income—taxable equivalent basis\n(1)\n$ \n43,660 \n$ \n37,519 \n$ \n39,739 \n(1) \nInterest revenue\nand \nNet interest income\ninclude the taxable equivalent adjustments discussed in the table above.\nCiti’s net interest income in the fourth quarter of 2022 was \n$13.3 billion (also $13.3 billion on a taxable equivalent basis), \nan increase of $2.5 billion versus the prior year, primarily \ndriven by non-\nICG\nMarkets (approximately $2.2 billion), as \nICG\nMarkets was largely unchanged (up approximately $0.3',
    'Corporate/Other\nin 2022, see “\nCorporate/Other\n” below.\n7% versus the prior year. Branded cards revenues of $8.9 \nbillion increased 9%, driven by higher net interest income. In \nBranded cards, new account acquisitions increased 11%, card \nspend volumes increased 16% and average loans increased \n11%. Retail services revenues of $5.5 billion increased 7%, \n5\nCITI’S CONSENT ORDER COMPLIANCE\nCiti has embarked on a multiyear transformation, with the \ntarget outcome to change Citi’s business and operating models',
    '$ (38,765) \n$ (32,058) $ (36,318) \nCitigroup’s total other comprehensive income (loss)\n(8,297) \n(6,707) \n4,260 \nBalance, end of year\n$ (47,062) \n$ (38,765) $ (32,058) \nTotal Citigroup common stockholders’ equity\n$ 182,194 \n$ 182,977 $ 179,962 \n1,936,986 \n1,984,355 2,082,089 \nTotal Citigroup stockholders’ equity\n$ 201,189 \n$ 201,972 $ 199,442 \nNoncontrolling interests\nBalance, beginning of year\n$ \n700 \n$ \n758 $ \n704 \nTransactions between Citigroup and the noncontrolling-interest \nshareholders\n(34) \n(10)',
    'CONSOLIDATED STATEMENT OF COMPREHENSIVE INCOME\nCitigroup Inc. and Subsidiaries\nYears ended December 31,\nIn millions of dollars\n2022\n2021\n2020\nCitigroup’s net income\n$ \n14,845 \n$ \n21,952 $ \n11,047 \nAdd: Citigroup’s other comprehensive income (loss)\n(1)\nNet change in unrealized gains and losses on debt securities, net of taxes\n(2)\n$ \n(5,384) \n$ \n(3,934) $ \n3,585 \nNet change in debt valuation adjustment (DVA), net of taxes\n(3)\n2,029 \n232 \n(475) \nNet change in cash flow hedges, net of taxes\n(2,623) \n(1,492)',
    'Efficiency ratio (total operating expenses/total revenues, net)\n68.1 \n67.0 \n58.8 \n57.0 \n58.1 \nBasel III ratios\nCET1 Capital\n(4)\n13.03 %\n12.25 %\n11.51 %\n11.79 %\n11.86 %\nTier 1 Capital\n(4)\n14.80 \n13.91 \n13.06 \n13.33 \n13.43 \nTotal Capital\n(4)\n15.46 \n16.04 \n15.33 \n15.87 \n16.14 \nSupplementary Leverage ratio\n5.82 \n5.73 \n6.99 \n6.20 \n6.40 \nCitigroup common stockholders’ equity to assets\n7.54 %\n7.99 %\n7.96 %\n8.98 %\n9.27 %\nTotal Citigroup stockholders’ equity to assets\n8.33 \n8.81 \n8.82 \n9.90 \n10.23',
    'to contractually based performance thresholds that, if met, \nwould require Citi to make ongoing payments to the partner. \nThe threshold is based on the profitability of a program and is \ngenerally calculated based on predefined program revenues \n166\nThe following table presents \nCommissions and fees\nrevenue:\n2022\n2021\n2020\nIn millions of \ndollars\nICG\nPBWM\nLF\nTotal\nICG\nPBWM\nLF\nTotal\nICG\nPBWM\nLF\nTotal\nInvestment \nbanking\n$ 3,084 $ \n— $ \n— $ 3,084 \n$ 6,007 $ \n— $ \n— $ 6,007 $ 4,483 $ \n— $ \n— $ 4,483',
    '$742 billion and $684 billion; in Latin America of $184 billion, $179 billion and $180 billion; and in Asia of $588 billion, $572 billion and $572 billion in 2022, \n2021 and 2020, respectively. These regional numbers exclude \nCorporate/Other\n, which largely reflects U.S. activities. The Company’s long-lived assets for the \nperiods presented are not considered to be significant in relation to its total assets. The majority of Citi’s long-lived assets are located in the U.S.\n164',
    '32,517 \n58,170 \nMortgage-backed securities\n33,573 \n— \n33,573 \nAsset-backed securities\n1,681 \n— \n1,681 \nOther\n4,026 \n58 \n4,084 \nTotal\n$ \n305,597 $ \n33,029 $ \n338,626 \n193\n12. BROKERAGE RECEIVABLES AND BROKERAGE \nPAYABLES\nThe Company has receivables and payables for financial \ninstruments sold to and purchased from brokers, dealers and \ncustomers, which arise in the ordinary course of business. Citi \nis exposed to risk of loss from the inability of brokers, dealers',
    'Payables to customers\n$ \n55,747 \n$ \n52,158 \nPayables to brokers, dealers and \nclearing organizations\n13,471 \n9,272 \nTotal brokerage payables\n(1)\n$ \n69,218 \n$ \n61,430 \n(1) Includes brokerage receivables and payables recorded by Citi broker-\ndealer entities that are accounted for in accordance with the AICPA \nAccounting Guide for Brokers and Dealers in Securities as codified in \nASC 940-320.\n194\n13. INVESTMENTS\nThe following table presents Citi’s investments by category:\nDecember 31,\nIn millions of dollars',
    'investment banking fees generated across the industry (i.e., the \nrevenue wallet) from investment banking transactions in \nM&A, equity and debt underwriting, and loan syndications.\n326\nNotes\n327\nNotes\n328\nNotes\n329\nNotes\n330\nNotes\n331\nNotes\n332\nNotes\n333\nStockholder information\nExchange agent\nCitigroup common stock is listed on the NYSE under the \nticker symbol “C.” Citigroup preferred stock Series J and K \nare also listed on the NYSE.\nHolders of Golden State Bancorp, Associates First Capital',
    'Non-U.S. pretax earnings approximated $16.2 billion in 2022, \n$12.9 billion in 2021 and $13.8 billion in 2020. As a U.S. \ncorporation, Citigroup and its U.S. subsidiaries are currently \nsubject to U.S. taxation on all non-U.S. pretax earnings of \nnon-U.S. branches. Beginning in 2018, there is a separate \nforeign tax credit (FTC) basket for branches. Also, dividends \nfrom a non-U.S. subsidiary or affiliate are effectively exempt \nfrom U.S. taxation. The Company provides income taxes on',
    'Total comprehensive income\n$ \n15,307 $ \n3,050 $ \n13,286 $ \n(16,270) $ \n15,373 \n308\nCondensed Consolidating Balance Sheet\nDecember 31, 2022\nOther \nCitigroup \nCitigroup \nsubsidiaries \nparent \nand \nCitigroup \ncompany\nCGMHI\neliminations\nConsolidating \nadjustments\nconsolidated\nIn millions of dollars\nAssets\nCash and due from banks\n$ \n— $ \n955 $ \n29,622 $ \n— $ \n30,577 \nCash and due from banks—intercompany\n15 \n7,448 \n(7,463) \n— \n— \nDeposits with banks, net of allowance\n— \n7,902 \n303,546 \n— \n311,448',
    '817 $ \n852 \nIn billions of dollars\n4Q22\n3Q22\n4Q21\nLegacy Franchises\n(1)\n$ \n50 \n$ \n50 $ \n74 \nCorporate/Other\n$ \n32 \n$ \n21 $ \n7 \nPersonal Banking and Wealth \nManagement\nU.S. Retail banking\n$ \n37 \n$ \n36 $ \n34 \nTotal Citigroup deposits (AVG)\n$ 1,361 \n$ 1,316 $ 1,370 \nU.S. Cards\n143 \n138 \n128 \nTotal Citigroup deposits (EOP)\n$ 1,366 \n$ 1,306 $ 1,317 \nGlobal Wealth\n150 \n151 \n150 \nTotal\n$ \n330 \n$ \n325 $ \n312 \n(1)\nSee footnote 2 to the table in “Credit Risk—Consumer Credit—\nConsumer Credit Portfolio” above.',
    'Citigroup Inc. and Consolidated Subsidiaries\nIn millions of dollars, except per share amounts, ratios and direct staff\n2022\n2021\n2020\n2019\n2018\nAt December 31:\nTotal assets\n$ 2,416,676 \n$ 2,291,413 \n$ 2,260,090 \n$ 1,951,158 \n$ 1,917,383 \nTotal deposits \n1,365,954 \n1,317,230 \n1,280,671 \n1,070,590 \n1,013,170 \nLong-term debt\n271,606 \n254,374 \n271,686 \n248,760 \n231,999 \nCitigroup common stockholders’ equity\n182,194 \n182,977 \n179,962 \n175,262 \n177,760 \nTotal Citigroup stockholders’ equity\n201,189 \n201,972',
    'Net income from continuing operations (for EPS purposes)\n$ \n15,076 \n$ \n21,945 $ \n11,067 \nLoss from discontinued operations, net of taxes\n(231) \n7 \n(20) \nCitigroup’s net income\n$ \n14,845 \n$ \n21,952 $ \n11,047 \nLess: Preferred dividends\n(1)\n1,032 \n1,040 \n1,095 \nNet income available to common shareholders\n$ \n13,813 \n$ \n20,912 $ \n9,952 \nLess: Dividends and undistributed earnings allocated to employee restricted and deferred shares \nwith rights to dividends, applicable to basic EPS\n113 \n154 \n73',
    'During 2022, emerging markets revenues accounted for \napproximately 37% of Citi’s total revenues (Citi generally \ndefines emerging markets as countries in Latin America, Asia \n(other than Japan, Australia and New Zealand), and central \nand Eastern Europe, the Middle East and Africa in EMEA). \nCiti’s presence in the emerging markets subjects it to various \nrisks, such as limitations or unavailability of hedges on foreign \ninvestments; foreign currency volatility, including',
    'On November 1, 2022, Citi completed the sale of its Thailand consumer banking business, which was part of \nLegacy Franchises\n. The business had approximately \n$2.7 billion in assets, including $2.4 billion of loans (net of allowance of $67 million) and excluding goodwill. The total amount of liabilities was $1.0 billion, \nincluding $0.8 billion in deposits. The sale resulted in a pretax gain on sale of approximately $209 million ($115 million after-tax), subject to closing adjustments, \nrecorded in']

texts_helium2 = [
    'Efficiency ratio (total operating expenses/total revenues, net)\n68.1\n67.0\n58.8\n57.0\n58.1\nBasel III ratios\nCET1 Capital\n(4)\n13.03 %\n12.25 %\n11.51 %\n11.79 %\n11.86 %\nTier 1 Capital\n(4)\n14.80\n13.91\n13.06\n13.33\n13.43\nTotal Capital\n(4)\n15.46\n16.04\n15.33\n15.87\n16.14\nSupplementary Leverage ratio\n5.82\n5.73\n6.99\n6.20\n6.40\nCitigroup common stockholders’ equity to assets\n7.54 %\n7.99 %\n7.96 %\n8.98 %\n9.27 %\nTotal Citigroup stockholders’ equity to assets\n8.33\n8.81\n8.82\n9.90\n10.23',
    'Payables to customers\n$\n55,747\n$\n52,158\nPayables to brokers, dealers and\nclearing organizations\n13,471\n9,272\nTotal brokerage payables\n(1)\n$\n69,218\n$\n61,430\n(1) Includes brokerage receivables and payables recorded by Citi broker-\ndealer entities that are accounted for in accordance with the AICPA\nAccounting Guide for Brokers and Dealers in Securities as codified in\nASC 940-320.\n194\n13. INVESTMENTS\nThe following table presents Citi’s investments by category:\nDecember 31,\nIn millions of dollars',
    'Payables to customers\n$\n55,747\n$\n52,158\nPayables to brokers, dealers and\nclearing organizations\n13,471\n9,272\nTotal brokerage payables\n(1)\n$\n69,218\n$\n61,430\n(1) Includes brokerage receivables and payables recorded by Citi broker-\ndealer entities that are accounted for in accordance with the AICPA\nAccounting Guide for Brokers and Dealers in Securities as codified in\nASC 940-320.\n194\n13. INVESTMENTS\nThe following table presents Citi’s investments by category:\nDecember 31,\nIn millions of dollars',
    'Corporate/Other\nin 2022, see “\nCorporate/Other\n” below.\n7% versus the prior year. Branded cards revenues of $8.9\nbillion increased 9%, driven by higher net interest income. In\nBranded cards, new account acquisitions increased 11%, card\nspend volumes increased 16% and average loans increased\n11%. Retail services revenues of $5.5 billion increased 7%,\n5\nCITI’S CONSENT ORDER COMPLIANCE\nCiti has embarked on a multiyear transformation, with the\ntarget outcome to change Citi’s business and operating models',
    'Corporate/Other\nin 2022, see “\nCorporate/Other\n” below.\n7% versus the prior year. Branded cards revenues of $8.9\nbillion increased 9%, driven by higher net interest income. In\nBranded cards, new account acquisitions increased 11%, card\nspend volumes increased 16% and average loans increased\n11%. Retail services revenues of $5.5 billion increased 7%,\n5\nCITI’S CONSENT ORDER COMPLIANCE\nCiti has embarked on a multiyear transformation, with the\ntarget outcome to change Citi’s business and operating models',
    'Citigroup Inc. and Consolidated Subsidiaries\nIn millions of dollars, except per share amounts, ratios and direct staff\n2022\n2021\n2020\n2019\n2018\nAt December 31:\nTotal assets\n$ 2,416,676\n$ 2,291,413\n$ 2,260,090\n$ 1,951,158\n$ 1,917,383\nTotal deposits\n1,365,954\n1,317,230\n1,280,671\n1,070,590\n1,013,170\nLong-term debt\n271,606\n254,374\n271,686\n248,760\n231,999\nCitigroup common stockholders’ equity\n182,194\n182,977\n179,962\n175,262\n177,760\nTotal Citigroup stockholders’ equity\n201,189\n201,972',
    'Citigroup Inc. and Consolidated Subsidiaries\nIn millions of dollars, except per share amounts, ratios and direct staff\n2022\n2021\n2020\n2019\n2018\nAt December 31:\nTotal assets\n$ 2,416,676\n$ 2,291,413\n$ 2,260,090\n$ 1,951,158\n$ 1,917,383\nTotal deposits\n1,365,954\n1,317,230\n1,280,671\n1,070,590\n1,013,170\nLong-term debt\n271,606\n254,374\n271,686\n248,760\n231,999\nCitigroup common stockholders’ equity\n182,194\n182,977\n179,962\n175,262\n177,760\nTotal Citigroup stockholders’ equity\n201,189\n201,972',
    '32,517\n58,170\nMortgage-backed securities\n33,573\n—\n33,573\nAsset-backed securities\n1,681\n—\n1,681\nOther\n4,026\n58\n4,084\nTotal\n$\n305,597 $\n33,029 $\n338,626\n193\n12. BROKERAGE RECEIVABLES AND BROKERAGE\nPAYABLES\nThe Company has receivables and payables for financial\ninstruments sold to and purchased from brokers, dealers and\ncustomers, which arise in the ordinary course of business. Citi\nis exposed to risk of loss from the inability of brokers, dealers',
    '32,517\n58,170\nMortgage-backed securities\n33,573\n—\n33,573\nAsset-backed securities\n1,681\n—\n1,681\nOther\n4,026\n58\n4,084\nTotal\n$\n305,597 $\n33,029 $\n338,626\n193\n12. BROKERAGE RECEIVABLES AND BROKERAGE\nPAYABLES\nThe Company has receivables and payables for financial\ninstruments sold to and purchased from brokers, dealers and\ncustomers, which arise in the ordinary course of business. Citi\nis exposed to risk of loss from the inability of brokers, dealers',
    'Total comprehensive income\n$\n15,307 $\n3,050 $\n13,286 $\n(16,270) $\n15,373\n308\nCondensed Consolidating Balance Sheet\nDecember 31, 2022\nOther\nCitigroup\nCitigroup\nsubsidiaries\nparent\nand\nCitigroup\ncompany\nCGMHI\neliminations\nConsolidating\nadjustments\nconsolidated\nIn millions of dollars\nAssets\nCash and due from banks\n$\n— $\n955 $\n29,622 $\n— $\n30,577\nCash and due from banks—intercompany\n15\n7,448\n(7,463)\n—\n—\nDeposits with banks, net of allowance\n—\n7,902\n303,546\n—\n311,448',
    'Total comprehensive income\n$\n15,307 $\n3,050 $\n13,286 $\n(16,270) $\n15,373\n308\nCondensed Consolidating Balance Sheet\nDecember 31, 2022\nOther\nCitigroup\nCitigroup\nsubsidiaries\nparent\nand\nCitigroup\ncompany\nCGMHI\neliminations\nConsolidating\nadjustments\nconsolidated\nIn millions of dollars\nAssets\nCash and due from banks\n$\n— $\n955 $\n29,622 $\n— $\n30,577\nCash and due from banks—intercompany\n15\n7,448\n(7,463)\n—\n—\nDeposits with banks, net of allowance\n—\n7,902\n303,546\n—\n311,448',
    'its right as a clearing member to transform cash margin into\nother assets, (iii) Citi does not guarantee and is not liable to\nthe client for the performance of the CCP or the depository\ninstitution and (iv) the client cash balances are legally isolated\nfrom Citi’s bankruptcy estate. The total amount of cash initial\nmargin collected and remitted in this manner was\napproximately $18.0 billion and $18.7 billion as of\nDecember 31, 2022 and 2021, respectively.',
    'its right as a clearing member to transform cash margin into\nother assets, (iii) Citi does not guarantee and is not liable to\nthe client for the performance of the CCP or the depository\ninstitution and (iv) the client cash balances are legally isolated\nfrom Citi’s bankruptcy estate. The total amount of cash initial\nmargin collected and remitted in this manner was\napproximately $18.0 billion and $18.7 billion as of\nDecember 31, 2022 and 2021, respectively.',
    '817 $\n852\nIn billions of dollars\n4Q22\n3Q22\n4Q21\nLegacy Franchises\n(1)\n$\n50\n$\n50 $\n74\nCorporate/Other\n$\n32\n$\n21 $\n7\nPersonal Banking and Wealth\nManagement\nU.S. Retail banking\n$\n37\n$\n36 $\n34\nTotal Citigroup deposits (AVG)\n$ 1,361\n$ 1,316 $ 1,370\nU.S. Cards\n143\n138\n128\nTotal Citigroup deposits (EOP)\n$ 1,366\n$ 1,306 $ 1,317\nGlobal Wealth\n150\n151\n150\nTotal\n$\n330\n$\n325 $\n312\n(1)\nSee footnote 2 to the table in “Credit Risk—Consumer Credit—\nConsumer Credit Portfolio” above.',
    '$14.9 billion, $13.4 billion and $13.4 billion; in Latin America of $9.9 billion, $9.2 billion and $9.4 billion; and in Asia of $14.7 billion, $14.4 billion and\n$15.8 billion in 2022, 2021 and 2020, respectively. These regional numbers exclude\nCorporate/Other\n, which largely reflects U.S. activities.\n(2)\nIncludes total Citi identifiable assets (excluding\nCorporate/Other\n), in North America of $776 billion, $709 billion and $741 billion; in EMEA of $773 billion,',
    'Revenues, net of interest expense\n$\n75,338\n$\n71,884 $\n75,501 $\n75,067 $\n74,036\nOperating expenses\n51,292\n48,193\n44,374\n42,783\n43,023\nProvisions for credit losses and for benefits and claims\n5,239\n(3,778)\n17,495\n8,383\n7,568\nIncome from continuing operations before income taxes\n$\n18,807\n$\n27,469 $\n13,632 $\n23,901 $\n23,445\nIncome taxes\n3,642\n5,451\n2,525\n4,430\n5,357\nIncome from continuing operations\n$\n15,165\n$\n22,018 $\n11,107 $\n19,471 $\n18,088',
    'Revenues, net of interest expense\n$\n75,338\n$\n71,884 $\n75,501 $\n75,067 $\n74,036\nOperating expenses\n51,292\n48,193\n44,374\n42,783\n43,023\nProvisions for credit losses and for benefits and claims\n5,239\n(3,778)\n17,495\n8,383\n7,568\nIncome from continuing operations before income taxes\n$\n18,807\n$\n27,469 $\n13,632 $\n23,901 $\n23,445\nIncome taxes\n3,642\n5,451\n2,525\n4,430\n5,357\nIncome from continuing operations\n$\n15,165\n$\n22,018 $\n11,107 $\n19,471 $\n18,088',
    'approximately $400 million ($345 million after-tax) related to\nare inherently limited because they involve techniques,\nincluding the use of historical data in many circumstances,\nassumptions and judgments that cannot anticipate every\neconomic and financial outcome in the markets in which Citi\noperates, nor can they anticipate the specifics and timing of\n49\ninterconnectedness among financial institutions, concerns\nabout the creditworthiness of or defaults by a financial',
    'approximately $400 million ($345 million after-tax) related to\nare inherently limited because they involve techniques,\nincluding the use of historical data in many circumstances,\nassumptions and judgments that cannot anticipate every\neconomic and financial outcome in the markets in which Citi\noperates, nor can they anticipate the specifics and timing of\n49\ninterconnectedness among financial institutions, concerns\nabout the creditworthiness of or defaults by a financial',
    'to contractually based performance thresholds that, if met,\nwould require Citi to make ongoing payments to the partner.\nThe threshold is based on the profitability of a program and is\ngenerally calculated based on predefined program revenues\n166\nThe following table presents\nCommissions and fees\nrevenue:\n2022\n2021\n2020\nIn millions of\ndollars\nICG\nPBWM\nLF\nTotal\nICG\nPBWM\nLF\nTotal\nICG\nPBWM\nLF\nTotal\nInvestment\nbanking\n$ 3,084 $\n— $\n— $ 3,084\n$ 6,007 $\n— $\n— $ 6,007 $ 4,483 $\n— $\n— $ 4,483',
    'to contractually based performance thresholds that, if met,\nwould require Citi to make ongoing payments to the partner.\nThe threshold is based on the profitability of a program and is\ngenerally calculated based on predefined program revenues\n166\nThe following table presents\nCommissions and fees\nrevenue:\n2022\n2021\n2020\nIn millions of\ndollars\nICG\nPBWM\nLF\nTotal\nICG\nPBWM\nLF\nTotal\nICG\nPBWM\nLF\nTotal\nInvestment\nbanking\n$ 3,084 $\n— $\n— $ 3,084\n$ 6,007 $\n— $\n— $ 6,007 $ 4,483 $\n— $\n— $ 4,483',
    'On November 1, 2022, Citi completed the sale of its Thailand consumer banking business, which was part of\nLegacy Franchises\n. The business had approximately\n$2.7 billion in assets, including $2.4 billion of loans (net of allowance of $67 million) and excluding goodwill. The total amount of liabilities was $1.0 billion,\nincluding $0.8 billion in deposits. The sale resulted in a pretax gain on sale of approximately $209 million ($115 million after-tax), subject to closing adjustments,\nrecorded in',
    'On November 1, 2022, Citi completed the sale of its Thailand consumer banking business, which was part of\nLegacy Franchises\n. The business had approximately\n$2.7 billion in assets, including $2.4 billion of loans (net of allowance of $67 million) and excluding goodwill. The total amount of liabilities was $1.0 billion,\nincluding $0.8 billion in deposits. The sale resulted in a pretax gain on sale of approximately $209 million ($115 million after-tax), subject to closing adjustments,\nrecorded in',
    'Efficiency ratio (total operating expenses/total revenues, net)\n68.1\n67.0\n58.8\n57.0\n58.1\nBasel III ratios\nCET1 Capital\n(4)\n13.03 %\n12.25 %\n11.51 %\n11.79 %\n11.86 %\nTier 1 Capital\n(4)\n14.80\n13.91\n13.06\n13.33\n13.43\nTotal Capital\n(4)\n15.46\n16.04\n15.33\n15.87\n16.14\nSupplementary Leverage ratio\n5.82\n5.73\n6.99\n6.20\n6.40\nCitigroup common stockholders’ equity to assets\n7.54 %\n7.99 %\n7.96 %\n8.98 %\n9.27 %\nTotal Citigroup stockholders’ equity to assets\n8.33\n8.81\n8.82\n9.90\n10.23',
    'The Company’s operating leases, where Citi is a lessor,\nCommercial and industrial\n$\n56,176\n$\n48,364\nare not significant to the Consolidated Financial Statements.\nFinancial institutions\n43,399\n49,804\nMortgage and real estate\n(2)\n17,829\n15,965\nInstallment and other\n23,767\n20,143\nLease financing\n308\n415\nTotal\n$\n141,479\n$\n134,691\nIn offices outside North America\n(1)\nCommercial and industrial\n$\n93,967\n$\n102,735\nFinancial institutions\n21,931\n22,158\nMortgage and real estate\n(2)\n4,179\n4,374',
    '464 $\n453\n$\n97\n$ 125 $ 131\n$\n96\n$ 89 $\n84\n$ 2,417\n$ 2,291 $ 2,260\nAverage loans\n291\n287\n298\n321\n307\n304\n41\n74\n83\n—\n—\n—\n653\n668\n685\nAverage deposits\n830\n828\n780\n435\n417\n358\n52\n82\n81\n16\n8\n11\n1,333\n1,335 1,230\n(1)\nIncludes total Citi revenues, net of interest expense (excluding\nCorporate/Other\n), in North America of $34.4 billion, $34.4 billion and $37.1 billion; in EMEA of',
    '$14.9 billion, $13.4 billion and $13.4 billion; in Latin America of $9.9 billion, $9.2 billion and $9.4 billion; and in Asia of $14.7 billion, $14.4 billion and\n$15.8 billion in 2022, 2021 and 2020, respectively. These regional numbers exclude\nCorporate/Other\n, which largely reflects U.S. activities.\n(2)\nIncludes total Citi identifiable assets (excluding\nCorporate/Other\n), in North America of $776 billion, $709 billion and $741 billion; in EMEA of $773 billion,',
    'Legacy Franchises\nCorporate/Other\nTotal Citi\n2022\n2021\n2020\n2022\n2021\n2020\n2022\n2021\n2020\n2022\n2021\n2020\n2022\n2021\n2020\nIn millions of\ndollars, except\nidentifiable assets,\naverage loans and\naverage deposits in\nbillions\nNet interest\nincome\n$ 17,911\n$ 14,999 $ 15,750\n$ 22,656\n$ 20,646 $ 22,326\n$ 5,691\n$ 6,250 $ 6,973\n$ 2,410\n$ 599 $ (298)\n$ 48,668\n$ 42,494 $ 44,751\nNon-interest\nrevenue\n23,295\n24,837 25,343\n1,561\n2,681 2,814\n2,781\n2,001 2,481\n(967)\n(129)\n112\n26,670\n29,390 30,750',
    'Legacy Franchises\nCorporate/Other\nTotal Citi\n2022\n2021\n2020\n2022\n2021\n2020\n2022\n2021\n2020\n2022\n2021\n2020\n2022\n2021\n2020\nIn millions of\ndollars, except\nidentifiable assets,\naverage loans and\naverage deposits in\nbillions\nNet interest\nincome\n$ 17,911\n$ 14,999 $ 15,750\n$ 22,656\n$ 20,646 $ 22,326\n$ 5,691\n$ 6,250 $ 6,973\n$ 2,410\n$ 599 $ (298)\n$ 48,668\n$ 42,494 $ 44,751\nNon-interest\nrevenue\n23,295\n24,837 25,343\n1,561\n2,681 2,814\n2,781\n2,001 2,481\n(967)\n(129)\n112\n26,670\n29,390 30,750',
    'Personal Banking and Wealth Management\n24,217\n23,327\n25,140\n4\n(7)\nLegacy Franchises\n8,472\n8,251\n9,454\n3\n(13)\nCorporate/Other\n1,443\n470\n(186)\nNM\nNM\nTotal Citigroup net revenues\n$\n75,338\n$\n71,884 $\n75,501\n5 %\n(5) %\nNM Not meaningful\nINCOME\n% Change\n% Change\n2022 vs. 2021\n2021 vs. 2020\nIn millions of dollars\n2022\n2021\n2020\nIncome (loss) from continuing operations\nInstitutional Clients Group\n$\n10,738\n$\n14,308 $\n10,811\n(25) %\n32 %\nPersonal Banking and Wealth Management\n3,319\n7,734\n1,322',
    'Personal Banking and Wealth Management\n24,217\n23,327\n25,140\n4\n(7)\nLegacy Franchises\n8,472\n8,251\n9,454\n3\n(13)\nCorporate/Other\n1,443\n470\n(186)\nNM\nNM\nTotal Citigroup net revenues\n$\n75,338\n$\n71,884 $\n75,501\n5 %\n(5) %\nNM Not meaningful\nINCOME\n% Change\n% Change\n2022 vs. 2021\n2021 vs. 2020\nIn millions of dollars\n2022\n2021\n2020\nIncome (loss) from continuing operations\nInstitutional Clients Group\n$\n10,738\n$\n14,308 $\n10,811\n(25) %\n32 %\nPersonal Banking and Wealth Management\n3,319\n7,734\n1,322',
    '(2)\n307\n(140)\n(59)\nNM\nNM\nTotal Banking revenues (including gains (losses) on loan\nhedges)\n(2)\n$\n6,071\n$\n9,378 $\n7,233\n(35) %\n30 %\nTotal\nICG\nrevenues, net of interest expense\n$\n41,206\n$\n39,836 $\n41,093\n3 %\n(3) %\n(1)\nCiti assesses its Markets business performance on a total revenue basis, as offsets may occur across revenue line items. For example, securities that generate\nNet\ninterest income\nmay be risk managed by derivatives that are recorded in\nPrincipal transactions\nrevenue within',
    '(2)\n307\n(140)\n(59)\nNM\nNM\nTotal Banking revenues (including gains (losses) on loan\nhedges)\n(2)\n$\n6,071\n$\n9,378 $\n7,233\n(35) %\n30 %\nTotal\nICG\nrevenues, net of interest expense\n$\n41,206\n$\n39,836 $\n41,093\n3 %\n(3) %\n(1)\nCiti assesses its Markets business performance on a total revenue basis, as offsets may occur across revenue line items. For example, securities that generate\nNet\ninterest income\nmay be risk managed by derivatives that are recorded in\nPrincipal transactions\nrevenue within',
    '$1.8 billion in assets, including $1.2 billion of loans (net of allowance of $80 million) and excluding goodwill. The total amount of liabilities was $1.3 billion,\nincluding $1.2 billion in deposits. The sale resulted in a pretax gain on sale of approximately $618 million ($290 million after-tax), subject to closing adjustments,\nrecorded in\nOther revenue\n. The income before taxes shown in the above table for the Philippines reflects Citi’s ownership through August 1, 2022.\n(4)',
    '$1.8 billion in assets, including $1.2 billion of loans (net of allowance of $80 million) and excluding goodwill. The total amount of liabilities was $1.3 billion,\nincluding $1.2 billion in deposits. The sale resulted in a pretax gain on sale of approximately $618 million ($290 million after-tax), subject to closing adjustments,\nrecorded in\nOther revenue\n. The income before taxes shown in the above table for the Philippines reflects Citi’s ownership through August 1, 2022.\n(4)',
    'Citigroup reported net income of $14.8 billion, or $7.00 per\nshare, compared to net income of $22.0 billion, or $10.14 per\nshare in the prior year. The decrease in net income was\nprimarily driven by the higher cost of credit, resulting from\nloan growth in\nPersonal Banking and Wealth Management\n(PBWM)\nand a deterioration in macroeconomic assumptions,\n3\nPolicies and Significant Estimates—Citi’s Allowance for\nCredit Losses (ACL)” below.\nNet credit losses of $3.8 billion decreased 23% from the',
    'Citigroup reported net income of $14.8 billion, or $7.00 per\nshare, compared to net income of $22.0 billion, or $10.14 per\nshare in the prior year. The decrease in net income was\nprimarily driven by the higher cost of credit, resulting from\nloan growth in\nPersonal Banking and Wealth Management\n(PBWM)\nand a deterioration in macroeconomic assumptions,\n3\nPolicies and Significant Estimates—Citi’s Allowance for\nCredit Losses (ACL)” below.\nNet credit losses of $3.8 billion decreased 23% from the',
    'The Company’s operating leases, where Citi is a lessor,\nCommercial and industrial\n$\n56,176\n$\n48,364\nare not significant to the Consolidated Financial Statements.\nFinancial institutions\n43,399\n49,804\nMortgage and real estate\n(2)\n17,829\n15,965\nInstallment and other\n23,767\n20,143\nLease financing\n308\n415\nTotal\n$\n141,479\n$\n134,691\nIn offices outside North America\n(1)\nCommercial and industrial\n$\n93,967\n$\n102,735\nFinancial institutions\n21,931\n22,158\nMortgage and real estate\n(2)\n4,179\n4,374',
    '464 $\n453\n$\n97\n$ 125 $ 131\n$\n96\n$ 89 $\n84\n$ 2,417\n$ 2,291 $ 2,260\nAverage loans\n291\n287\n298\n321\n307\n304\n41\n74\n83\n—\n—\n—\n653\n668\n685\nAverage deposits\n830\n828\n780\n435\n417\n358\n52\n82\n81\n16\n8\n11\n1,333\n1,335 1,230\n(1)\nIncludes total Citi revenues, net of interest expense (excluding\nCorporate/Other\n), in North America of $34.4 billion, $34.4 billion and $37.1 billion; in EMEA of',
    '$ (38,765)\n$ (32,058) $ (36,318)\nCitigroup’s total other comprehensive income (loss)\n(8,297)\n(6,707)\n4,260\nBalance, end of year\n$ (47,062)\n$ (38,765) $ (32,058)\nTotal Citigroup common stockholders’ equity\n$ 182,194\n$ 182,977 $ 179,962\n1,936,986\n1,984,355 2,082,089\nTotal Citigroup stockholders’ equity\n$ 201,189\n$ 201,972 $ 199,442\nNoncontrolling interests\nBalance, beginning of year\n$\n700\n$\n758 $\n704\nTransactions between Citigroup and the noncontrolling-interest\nshareholders\n(34)\n(10)',
    '$ (38,765)\n$ (32,058) $ (36,318)\nCitigroup’s total other comprehensive income (loss)\n(8,297)\n(6,707)\n4,260\nBalance, end of year\n$ (47,062)\n$ (38,765) $ (32,058)\nTotal Citigroup common stockholders’ equity\n$ 182,194\n$ 182,977 $ 179,962\n1,936,986\n1,984,355 2,082,089\nTotal Citigroup stockholders’ equity\n$ 201,189\n$ 201,972 $ 199,442\nNoncontrolling interests\nBalance, beginning of year\n$\n700\n$\n758 $\n704\nTransactions between Citigroup and the noncontrolling-interest\nshareholders\n(34)\n(10)',
    'net interest income—taxable equivalent basis\n(1)\n$\n43,660\n$\n37,519\n$\n39,739\n(1)\nInterest revenue\nand\nNet interest income\ninclude the taxable equivalent adjustments discussed in the table above.\nCiti’s net interest income in the fourth quarter of 2022 was\n$13.3 billion (also $13.3 billion on a taxable equivalent basis),\nan increase of $2.5 billion versus the prior year, primarily\ndriven by non-\nICG\nMarkets (approximately $2.2 billion), as\nICG\nMarkets was largely unchanged (up approximately $0.3',
    'net interest income—taxable equivalent basis\n(1)\n$\n43,660\n$\n37,519\n$\n39,739\n(1)\nInterest revenue\nand\nNet interest income\ninclude the taxable equivalent adjustments discussed in the table above.\nCiti’s net interest income in the fourth quarter of 2022 was\n$13.3 billion (also $13.3 billion on a taxable equivalent basis),\nan increase of $2.5 billion versus the prior year, primarily\ndriven by non-\nICG\nMarkets (approximately $2.2 billion), as\nICG\nMarkets was largely unchanged (up approximately $0.3',
    'higher revenues. Citigroup’s effective tax rate was 19.4% in\nthe current year versus 19.8% in the prior year. Earnings per\nshare (EPS) decreased 31%, reflecting the decrease in net\nincome, partially offset by a 4% decline in average diluted\nshares outstanding.\nAs discussed above, results for 2022 included divestiture-\n•\nCiti’s revenues increased 5% versus the prior year,\nincluding net gains on sales of Citi’s Philippines and\nThailand consumer banking businesses versus a loss on',
    'higher revenues. Citigroup’s effective tax rate was 19.4% in\nthe current year versus 19.8% in the prior year. Earnings per\nshare (EPS) decreased 31%, reflecting the decrease in net\nincome, partially offset by a 4% decline in average diluted\nshares outstanding.\nAs discussed above, results for 2022 included divestiture-\n•\nCiti’s revenues increased 5% versus the prior year,\nincluding net gains on sales of Citi’s Philippines and\nThailand consumer banking businesses versus a loss on',
    '$742 billion and $684 billion; in Latin America of $184 billion, $179 billion and $180 billion; and in Asia of $588 billion, $572 billion and $572 billion in 2022,\n2021 and 2020, respectively. These regional numbers exclude\nCorporate/Other\n, which largely reflects U.S. activities. The Company’s long-lived assets for the\nperiods presented are not considered to be significant in relation to its total assets. The majority of Citi’s long-lived assets are located in the U.S.\n164',
    '$742 billion and $684 billion; in Latin America of $184 billion, $179 billion and $180 billion; and in Asia of $588 billion, $572 billion and $572 billion in 2022,\n2021 and 2020, respectively. These regional numbers exclude\nCorporate/Other\n, which largely reflects U.S. activities. The Company’s long-lived assets for the\nperiods presented are not considered to be significant in relation to its total assets. The majority of Citi’s long-lived assets are located in the U.S.\n164',
    'CONSOLIDATED STATEMENT OF COMPREHENSIVE INCOME\nCitigroup Inc. and Subsidiaries\nYears ended December 31,\nIn millions of dollars\n2022\n2021\n2020\nCitigroup’s net income\n$\n14,845\n$\n21,952 $\n11,047\nAdd: Citigroup’s other comprehensive income (loss)\n(1)\nNet change in unrealized gains and losses on debt securities, net of taxes\n(2)\n$\n(5,384)\n$\n(3,934) $\n3,585\nNet change in debt valuation adjustment (DVA), net of taxes\n(3)\n2,029\n232\n(475)\nNet change in cash flow hedges, net of taxes\n(2,623)\n(1,492)',
    'CONSOLIDATED STATEMENT OF COMPREHENSIVE INCOME\nCitigroup Inc. and Subsidiaries\nYears ended December 31,\nIn millions of dollars\n2022\n2021\n2020\nCitigroup’s net income\n$\n14,845\n$\n21,952 $\n11,047\nAdd: Citigroup’s other comprehensive income (loss)\n(1)\nNet change in unrealized gains and losses on debt securities, net of taxes\n(2)\n$\n(5,384)\n$\n(3,934) $\n3,585\nNet change in debt valuation adjustment (DVA), net of taxes\n(3)\n2,029\n232\n(475)\nNet change in cash flow hedges, net of taxes\n(2,623)\n(1,492)',
    '817 $\n852\nIn billions of dollars\n4Q22\n3Q22\n4Q21\nLegacy Franchises\n(1)\n$\n50\n$\n50 $\n74\nCorporate/Other\n$\n32\n$\n21 $\n7\nPersonal Banking and Wealth\nManagement\nU.S. Retail banking\n$\n37\n$\n36 $\n34\nTotal Citigroup deposits (AVG)\n$ 1,361\n$ 1,316 $ 1,370\nU.S. Cards\n143\n138\n128\nTotal Citigroup deposits (EOP)\n$ 1,366\n$ 1,306 $ 1,317\nGlobal Wealth\n150\n151\n150\nTotal\n$\n330\n$\n325 $\n312\n(1)\nSee footnote 2 to the table in “Credit Risk—Consumer Credit—\nConsumer Credit Portfolio” above.']

texts_helium3 = [
    '12 Assets under management (AUM) includes\n3\nAssets under management consist of cash and\nassets of the investment advisers affiliated\n6\nThe company’s general account investment\ninvested assets and separate account assets of the\nwith New York Life Insurance Company, other\nportfolio totaled $317.13 billion at December 31,\ncompany’s domestic and international insurance\nthan Kartesia Management, and Tristan Capital\n2022 (including $122.99 billion invested assets\noperations, and assets the company manages\nPartners, as of 12/31/2022. As of 12/31/2022\nfor NYLIAC and $8.39 billion invested assets\nfor third-party investors, including mutual funds,\nNew York Life Investments changed its AUM\nfor LINA). At December 31, 2022, total assets\nseparately managed accounts, retirement plans,\ncalculation methodology, and AUM now includes\nequaled $392.13 billion (including $184.99 billion\nSee Note 6 for and assets under certain assets, such as non-discretionary\ntotal assets for NYLIAC and $9.25 billion total\ninformation on the company’s general account\nAUM, external fund selection, and overlay\nassets for LINA). Total liabilities, excluding the\ninvestment',
    '| 0                               | 1      | 2             | 3      | 4             |\n|:--------------------------------|:-------|:--------------|:-------|:--------------|\n| Cash and Invested Assets        |        |               |        |               |\n| (In $ Billions)                 |        | Dec. 31, 2022 |        | Dec. 31, 2021 |\n| Bonds                           | $230.4 | 73%           | $221.4 | 74%           |\n| Mortgage Loans                  | 38.7   | 12%           | 35.2   | 12%           |\n| Equities                        | 15.3   | 5%            | 14.9   | 5%            |\n| Policy Loans                    | 12.6   | 4%            | 12.2   | 4%            |\n| Cash and Short-Term Investments | 9.9    | 3%            | 4.7    | 2%            |\n| Other Investments               | 4.4    | 1%            | 4.1    | 1%            |\n| Derivatives                     | 3.0    | 1%            | 1.6    | 1%            |\n| Investments in Subsidiaries     | 2.8    | 1%            | 2.9    | 1%            |\n| Total Cash and Invested Assets  | $317.1 | 100%          | $297.0 | 100%          |',
    'The portfolio is high\nmortgage loan portfolio is broadly diversified\nquality, with a loan-to-value ratio of by both property type and geographic\n$38.7\nBILLION10\n33% Multifamily\n4%\n27% Industrial\n19%\n23% Office\n24%\n9%\n15% Retail\n7%\n24%\n2% Other\n13%\nNEW YORK LIFE INSURANCE COMPANY\nNotes appear on page 15\n10\nIn particular, we utilize our extensive investment\npotential for value appreciation. We also\nEquities\ncapabilities in private equity and real estate to\ninvest in properties where opportunities exist\nadd value to the General to increase net operating income through\nWe maintain a 5%\ncapital investment and/or repositioning and\nPrivate Equities consist primarily of\nallocation to equities,\nthereby increase the property’s investments in small- and middle-market\nwhich offer higher\ncompanies through funds sponsored by\nPublic Equities are invested in a broad\nreturns and inflation\ntop-tier partners and spectrum of publicly listed companies. We\nprotection over the\nWe have extensive expertise and also long-\nutilize public equities to manage our overall\nlong standing relationships with high-performing\nallocation to equities.',
    'program, New York Life fully committed the $1\nbillion across various investments that are at\nthe heart of our impact thesis, and we continue\nto seek additional investment opportunities to\nexpand the program beyond our initial SURPLUS AND ASSET VALUATION RESERVE5\nCASH AND INVESTED ASSETS6\nIn $ Billions\nIn $ Billions\n317.1\n30.1\n2022\n2022\n297.0\n30.7\n2021\n2021\n284.2\n27.0\n2020\n2020\n268.0\n27.0\n2019\n2019\n2018\n2018\n256.1\n24.8\nNEW YORK LIFE INSURANCE COMPANY\nNotes appear on page 15\n6\nGeneral Account Investment Portfolio Overview\nNew York Life had\ncash and invested assets\nof $317.1 billion as of\nDecember 31, 2022.6\nNet Yield on Investment7\nNet yield on investment (net investment\nflow being invested at market income divided by the average of the current\nHowever, having the capability to originate\nand prior years’ invested assets) has declined\nprivate placement debt and mortgage loans\nslowly since reaching a peak in the helps mitigate the effect of a lower interest\nThis is attributable to the combined effect of\nrate higher-yielding assets maturing and new cash\n15%\nNew York Life Average\nAverage 10-Year',
    'Investment Capabilities\n$710 billion in assets under management.3\nExpertise that creates Our deep investment\nexperience and\nNew York Life had $710 billion of assets under\nNew York Life is able to access virtually all\ninvestment capabilities\nmanagement as of December 31, 2022. This\nasset classes, providing a broad universe of\nare put to work for\nincludes the $317 billion General Account—an\ninvestment opportunities to deliver long-\nour investment portfolio used to support claim\nterm, relatively stable returns. In particular, we\nand benefit payments made to clients. New\nhave the ability to originate private debt and\nYork Life’s investment boutiques manage\nequity investments. This expertise allows us\na broad array of fixed income, equity, asset\nto identify valuable investment opportunities\nallocation, sustainable investments, and\nunavailable in the public alternative investment General Account Investment Philosophy\nWe take a long-term We maintain At New York Life,\nour General Account\nWe invest for the long term because we make\nWe focus on maintaining safety and security\ninvestment philosophy\nlong-term commitments to our policy owners\nwhile pursuing superior investment',
    'Overview of\ninvestment managers13\nNewly unified alternatives investment firm\nBoutique offering a range of fixed income\nwith capabilities spanning private credit,\nstrategies, including investment grade, high\nprivate equity, GP stakes, private real assets,\nyield, bank loans, and municipals, as well as\nand long/short fundamental Specialists in cross-asset investing, leveraging\nBoutique with expertise in active the breadth and depth of the New York Life\nCapabilities across Australian equities\nInvestments’ multi-boutique and global small cap, natural resources, and\nlisted Provides investment management and\nfinancing solutions for New York Life and our\nESG-focused, active asset manager with\nvalued strategic partners, focused on fixed\nexpertise in fixed income, equity, thematic\nincome and real investing, absolute return, asset allocation,\nand liability-driven investing for pension\nfunds and insurance ~~ TRISTAN\nSs “CAPTTALPARTNERS\nReal estate investment management company\nspecializing in a wide range of property types\nPioneer and leading provider of exchange\nacross the UK and continental traded funds, granting investors access to\ninnovative solutions designed to deliver a\nsmarter approach to traditional',
    'dominated by high-\nquality investments,\nWe maintain a relatively small allocation\nwith 95% rated as\nto high yield issuers. These investments\ninvestment typically offer higher yields but have\ngreater risk of default. Our experienced\n$230.4\ninvestment team conducts thorough\nBILLION8\nresearch to identify companies with good\nbusiness fundamentals, making them\nless likely to default. We have historically\nachieved significant risk-adjusted returns\nfrom high yield investments, creating\nvalue for our NAIC 1:\nAAA to A-\n62%\nCorporate Bond Industry Diversification\nThe public and private\ncorporate bond\nportfolio, totaling\nOther\nIndustrial\nTechnology\n$142.6 billion, or\nFinance\n4%\n5%\n2%\n62% of the bond\nCable &\nportfolio, remains\nMedia\nPaper & Packaging\n7%\n5%\n4%\n2%\nConsumer\nwell diversified across\nEnergy\nProducts\nAutomotive\nthe broad industry\n2%\n16%\nspectrum, providing\n8%\nUtilities\n8%\nprotection throughout\nBanking/\nServices\nREITs\nBrokerage\n2%\nbusiness',
    'manages $661 billion in assets as of\nOur global capabilities combined with local\n12/31/22,12 including New York Life’s\npresence drive more nuanced perspective and\nGeneral Account investments and\na more personal experience for our third-party Insurance insights\nOur boutiques\nIn addition to offering investment expertise\nto our clients, our investment managers\nOur multi-boutique business model is built\npartner and collaborate with our core insurance\non the foundation of a long and stable history,\nbusiness to deliver deep insights on topics such\nwhich gives our clients proven performance\nas asset/liability management, liability-driven\nmanaging risk through multiple economic\ninvesting, and income-focused strategies, as\ncycles. With capabilities across virtually all asset\nwell as regulatory, rating agency, and accounting\nclasses, market segments, and geographies, our\nregimes. This partnership allows New York\nfamily of specialized, independent boutiques\nLife Investments to help meet the unique\nand investment teams allows us to deliver\ninvestment needs of insurance companies as\ncustomized strategies and integrated solutions\nwell as other institutional and retail for every client Investment Capabilities\nOur investment\nFixed Income\nETFs\nIndex Solutions\nEquities\nteams’ expertise\n• U.S.',
    'services, including ESG screening services,\nAsset Valuation Reserve (AVR), equaled $362.02\n4\nPolicy owner benefits primarily include death\nadvisory consulting services, white labeling\nbillion (including $174.56 billion total liabilities for\nclaims paid to beneficiaries and annuity investment management services, and model\nNYLIAC and $7.50 billion total liabilities for Dividends are payments made to eligible policy\nSee Note 5 for total portfolio delivery services, that do not qualify\nowners from divisible surplus. Divisible surplus is\nas Regulatory Assets Under Management,\n7\nThe chart represents the composite yield on\nthe portion of the company’s total surplus that\nas defined in the SEC’s Form ADV. AUM is\ninvested assets in the General Accounts of New\nis available, following each year’s operations, for\nreported in USD. AUM not denominated in USD\nYork Life and its subsidiaries. Although yields\ndistribution in the form of dividends. Dividends\nis converted at the spot rate as of shown are for a retail product (10-year are not guaranteed.',
    'Each year the board of\nThis total AUM figure is less than the sum of the\nTreasury bonds), New York Life’s net yield does\ndirectors votes on the amount and allocation of\nAUM of each affiliated investment adviser in the\nnot represent the yield of a retail product. The\nthe divisible surplus. Policy owner benefits and\ngroup because it does not count AUM where the\nchart shows how New York Life’s aggregate net\ndividends reflect the consolidated results of\nsame assets can be counted by more than one\nyield on invested assets has remained relatively\nNYLIC and its domestic insurance affiliated investment stable during periods of both rising and falling\nIntercompany transactions have been eliminated\n13 The products and services of New York Life\ninterest rates. It is indicative of New York Life’s\nin consolidation. NYLIC’s policy owner benefits\nInvestments Boutiques are not available to\nfinancial strength and does not reflect a rate of\nand dividends were $8.70 billion and $8.80 billion\nall clients in all jurisdictions or regions where\nreturn on any particular investment or insurance\nfor the years ended December 31, 2022 and 2021,\nsuch provisions would be contrary to local\nproduct.',
    '9%\nHealthcare/\nInsurance\n4%\nPharmaceuticals\n3%\nOther\nTelecommunications\n2%\nRetail\nAerospace & Defense\nTransportation\n3%\n2%\n4%\n$142.6\nConglomerates\nChemicals\nBILLION9\n5%\n3%\n2022 INVESTMENT REPORT\nNotes appear on page 15\n9\nSingle\nCorporate Bond Issuer Diversification\nLargest Issuer\n0.2%\nThe largest single issuer represents 0.2%\nThe corporate\nof cash and invested assets. Furthermore,\nbond portfolio is\nthe portfolio’s ten largest corporate bond\nmanaged to limit\nholdings represent only 1.5% of cash\nexposure to individual\nand invested assets. The corporate bond\nissuers according to\nportfolio is comprised of securities issued\ncredit quality and\nby over 3,300 individual other $317.1\nBILLION6\nCash and\nTop 10\nInvested Assets\nLargest Issuers\n100%\n1.5%\nThe company’s mortgage loan investment\nlocation. We maintain regional underwriting\nMortgage Loans\nstyle emphasizes conservative underwriting\noffices to ensure we have deep knowledge\nand a focus on high quality properties. The\nof our target markets.',
    'These holdings are\nprivate equity sponsors. In addition, our\ntypically highly liquid and offer higher return\nNYL Ventures team invests directly in\npotential in the long term compared with that\ninnovative technology partnerships focused\nof fixed income on impacting financial services, digital\nhealth, and enterprise software. We also\nmake opportunistic investments in a\nselect group of venture capital Real Estate\nPrivate\nReal Estate Equities primarily consist of\nEquities\nEquities\n36%\n53%\nhigh-quality, institutional-grade properties\ndiversified across property types and\n$15.3\ngeographic regions. We strategically focus\nBILLION11\non multifamily, industrial, office, and retail\nproperties in primary markets. These\nPublic\nEquities\ntypes of real estate investments generally\n11%\nprovide stable and predictable income, with\nAsset Class Returns and Diversification\nAs illustrated below, individual asset class benchmark returns vary from year to We maintain\nBy maintaining a diversified asset allocation, we invest throughout market cycles and\ndiversification across\ndon’t simply chase',
    'The New York Life net yield shown in this chart\n14 Based on revenue as reported by “Fortune\n5\nTotal surplus, which includes the AVR, is\nrepresents a composite net yield of the invested\n500 ranked within Industries, Insurance: Life,\none of the key indicators of the company’s\nassets of each of the following companies:\nHealth (Mutual),”Fortune magazine, long-term financial strength and stability\nNYLIC, NYLIAC, NYLAZ, LINA, and NYLGICNY,\nFor methodology, please see and is presented on a consolidated basis of\nnet of eliminations for certain intra-company\nthe company. NYLIC’s statutory surplus was\ntransactions. The curve shown represents only\n$23.89 billion and $24.57 billion at December\nNYLIC in years 1972–1979, NYLIC and NYLIAC in\n31, 2022 and 2021, respectively. Included in\nyears 1980–1986, NYLIC, NYLIAC, and NYLAZ in\n2022 INVESTMENT REPORT\n15\n',
    '7\nBonds\nThe majority of the\nPublic Corporate Bonds\n31%\nGeneral Account\ninvestment portfolio\nPrivate Corporate Bonds\n31%\nis allocated to bonds,\nwhich provide current\nAsset-Backed Securities\n10%\nincome to pay claims\nand benefits to policy\n$230.4\nCommerical Mortgage-Backed Securities\n10%\nBILLION8\nMunicipal Bonds\n7%\nResidential Mortgage-Backed Securities\n6%\nGovernment & Agency\n5%\nPublic Corporate Bonds, issued primarily\nResidential Mortgage-Backed Securities\nby investment grade companies, form the\nare investments in the residential real\ncore of our investment portfolio. We invest\nestate mortgage market. These securities\nacross a diverse group of industries. Public\nare typically pools of mortgages from a\ncorporate bonds are liquid and provide stable\ndiverse group of borrowers and geographic\ncurrent regions. A large portion of our holdings are\nissued and guaranteed by U.S. government–\nPrivate Corporate Bonds are originated by our\nsponsored dedicated team of investment This expertise allows us to identify valuable\nMunicipal Bonds provide opportunities\ninvestment opportunities unavailable in the\nto invest in states, counties, and local\npublic markets. In addition, these investments\nmunicipalities.',
    'We believe being a responsible investor is\ndisciplined approach\nWe invest in assets with similar interest rate\nconsistent with our goal to create long-term\nsensitivities and cash flow characteristics\nfinancial security for our clients and aligns our\nwhen investing the\nas our liabilities. This is done with the goal of\ninvestment activity with the broader objectives\nGeneral Account\nhaving funds available when we need to pay\nof society. Our holistic approach to investment\ninvestment benefits to clients and to protect the surplus\nanalysis incorporates a financial assessment\nof the company from adverse changes in\nas well as considering environmental, social,\ninterest rates. In addition, we maintain ample\nand governance (ESG) factors that are deemed\nliquidity in the event we need to meet large\nmaterial to a company’s performance. We\nand unexpected cash believe responsible investing is a journey that\nneeds to be thoughtfully implemented to\nWell-balanced and diversified investments\nbe effective in its outcomes, and we remain\nPortfolios with diversified asset allocations\ncommitted to sharing our progress as we',
    'Municipal investments include\nprovide further diversification, better\ngeneral obligation bonds supported by\nselectivity, and higher returns compared with\ntaxes, as well as revenue bonds that finance\nthose of public specific income-producing projects. These\ninvestments provide further diversification\nCommercial Mortgage-Backed Securities\nto our portfolio as well as exhibit longer\nprovide access to diversified pools of\nduration, high credit quality, and a historically\ncommercial mortgages that supplement our\nlow default commercial mortgage loan Government & Agency Bonds are highly\nAsset-Backed Securities are bonds backed\nliquid securities that help ensure we have\nby various types of financial receivables, such\nample funds available to pay large and\nas equipment leases, collateralized bank\nunexpected loans, royalties, or consumer NEW YORK LIFE INSURANCE COMPANY\nNotes appear on page 15\n8\nNAIC 2:\nNAIC 3–6:\nBond Portfolio Quality\nBBB+ to BBB-\nBB+ and below\n33%\n5%\nInvestment grade securities provide\nThe bond portfolio\nsafety and security while producing\ncontinues to be\nstable',
    'Net Investment Yield\nTreasury Bond Yield\n10%\n5%\n4.04%\n2.95%\n0%\n1975\n1980\n1985\n1990\n1995\n2000\n2005\n2010\n2015\n2020\n2022 INVESTMENT REPORT\nNotes appear on page 15\n',
    'is aligned with the\nand are not distracted by short-term results\nWe focus keenly on capital preservation and\nbest interests of our\nat the expense of long-term predictable investment results while seeking\nabove-market General Account Value Proposition\nDriving benefits.4\nDriving the The General Account\ninvestment portfolio\nInvestment return is a primary driver of\nOur investments positively impact the\nplays a dual role:\nbenefits paid to our clients. By staying true\neconomy—creating jobs, benefiting\nto our investment philosophy and principles,\ncommunities, supporting innovation, and\nwe create value, paying dividends to our\nfunding sustainable energy participating policy owners and growing\nour already strong 2022 INVESTMENT REPORT\nNotes appear on page 15\n5\nGeneral Account Investment Strategy and Approach\nAsset/liability management focus\nDelivering for clients and society through\nReflecting our\nresponsible investing\ninvestment philosophy,\nOur primary focuses are asset/liability\nwe take a highly\nmanagement and maintaining ample']

texts_helium4 = [
    "instructions] Please note, this -- this event is being recorded. I now like to turn the\nconference over to Mr.\nFoster, vice president of investor relations. go ahead, sir.\nFoster -- Vice President, Investor Relations\nGood afternoon and welcome to FedEx Corporation's first-quarter\nearnings conference call. The earnings release, Form 10-Q, and stat book were on our website at fedex.com. This and the accompanying\nslides are being streamed from our website, where the replay and slides will be available for about one\nyear. us on the call today are members of the media. During our question-and-answer session, callers\nwill be limited to one question in order to allow us to accommodate all those who would like to participate.\nstatements in this conference call, such as projections regarding future performance, may be\nconsidered forward-looking statements. Such statements are subject to risks, uncertainties,\nand other factors which could cause actual results to differ materially from those expressed or implied by such\nforward-looking statements. For information on these factors, please refer to our press releases and\nfilings\nwith the SEC. Please",
    "hit the ground running, and I'm very\nhappy that he has joined FedEx. So, now to the quarter. We entered fiscal\nyear '24 with strength and\nmomentum, delivering results ahead of expectations in what remains a dynamic environment.\nI'm proud what the FedEx team has accomplished over the last 12 months. Amid demand\ndisruption, we delivered on what we said we would do, driving over $2 billion in year-over-year cost savings in\nfiscal\n'23. We are now well in executing on that transmission to be the most efficient,\nflexible,\nand\nintelligent global network. Our first-quarter\ngives me great conviction in our ability to execute going\nforward. We came into the determined to provide excellent service to our customers despite the\nindustry dynamics.\nWe achieved that goal delivering innovative and data-driven solutions that further enhance the customer\nexperience. As a result, we are positioned as we prepare for the peak season. As you can see in our on Slide 6, our transformation is enhancing our profitability.\nGround was a bright spot higher revenue year\nover year driven by higher yields. On top of this growth,",
    "See the 10 stocks\n*Stock Advisor returns as of September 18, 2023\nIt has been a privilege being a longtime part of the FedEx team. I truly believe that FedEx's best days are ahead,\nbut I will be cheering from the sidelines as I am 67 years old and I want to spend more time with my family. With\nthat, I will now turn it over to Raj for him to share his views on the quarter.\nRaj Subramaniam -- President and Chief Executive Officer\nThank you, Mickey, and good afternoon. I would like to first\ncongratulate Mickey on his upcoming retirement.\nHe led our investor relations team for nearly 18 years spanning 70 earnings calls and, after tomorrow, 18\nannual meetings. He be missed by all and especially this audience.\nwe thank him for his outstanding service to FedEx over the years. And we also take this opportunity to\nwelcome John Dietrich, our chief financial\nofficer\nfor FedEx. With than 30 years of experience in the\naviation and air cargo industries, John brings a unique blend of financial\nand operational expertise to our\nleadership team at a very important time for this company. He's",
    "very impactful change, and customer feedback has been overwhelmingly\npositive. Small and medium are a high-value growth segment, and we are confident\nthat the\nimprovements underway will further enable share gains.\nAnd lastly, we've My FedEx Rewards beyond the United States into nearly 30 other countries, with\nnine more European countries to launch later this year. My FedEx Rewards is only loyalty program in the\nindustry and benefits|\nour customers by providing them with rewards they can invest in back into their business.\nThis website uses to deliver our services and to\nanalyze traffic.\nWe also share information your use\nof our site with advertising and other partners. Privacy\nPolicy\n||\nThey can use them to recognize their employees for a job well done or give back to their communities. My\nFedEx Rewards have been a successful program in the United States, and we've built lasting relationships as\nwe continue to invest in our customers. We are excited about the potential to replicate this success in Europe\nand around the world. Driving to anticipate customers' needs and provide them with superior service is deeply\nembedded in our FedEx culture.\n",
    "will we continue to provide our customers with the best\nservice and product offerings, but our plans to bring our businesses together through One FedEx and execute\non DRIVE and Network 2.0 initiatives will be truly transformative. These initiatives will leverage and optimize\neverything that the talented teams across FedEx have built over the last 50 years. It make us smarter; it will\nmake us more efficient;\nand it will enable us to serve our customers better.\nBefore into the numbers, I want to share a brief overview of the priorities that will guide me and the\nfinance\norganization as we move forward. First and I'm committed to setting stringent financial\ngoals\nthat  the significant\nopportunity we have to improve margins and returns. This be enabled by the\nDRIVE initiatives and the integration of Network 2.0 as we move toward One FedEx. I've really impressed\nby the tremendous amount of work already completed on DRIVE from the initiatives in place, the accountability\nembedded in the program, and the team's steadfast focus on execution. In terms",
    "Raj\nSubramaniam for any closing remarks. Please go ahead, sir.\nRaj Subramaniam -- President and Chief Executive Officer\nThank you very much, operator. me say that, in closing, how proud I am of our team for delivering such a\nstrong start for the year. execution of the structural cost reductions remain on track. as we prepare for\npeak, we will continue to make every FedEx experience outstanding for our customers. have proven that\nDRIVE is changing the way we work, and we are enabling continued transformation across FedEx as we build\nthe world's most flexible,\nefficient,\nand intelligent network.\nThank for your attention today. I will see you next time.\n[Operator signoff]\nDuration: 0 minutes\nCall participants:\nMickey Foster -- Vice President, Investor Relations\nRaj Subramaniam -- President and Chief Executive Officer\nBrie Carere -- Executive Vice President, Chief Customer Officer\nJohn Dietrich -- Executive Vice President, Chief Financial Officer\nJon Chappell -- Evercore ISI -- Analyst\nJack Atkins -- Stephens, Inc. -- Analyst\n",
    "I'm proud of how our teams work together to support our current customers, build relationships with new ones,\nand ensure that FedEx is positioned to succeed during the quarter. Now, I will turn it over to John to discuss the\nfinancials\nin more detail.\nDietrich -- Executive Vice President, Chief Financial Officer\nThank you, Brie, and good afternoon, everyone. I'm really excited to be here. been a full sprint these last few\nweeks as I continue to get up to speed with this great company. As of you may know, I've done business\nwith FedEx throughout my career.\nthat experience, I've always admired how FedEx literally created a new industry and has built a\ndifferentiated network that serves customers all over the world. also admired its great culture that has\nthrived through the people-service-profit,\nor PSP, philosophy. After only being here a few short weeks, I've seen\nthe incredible opportunity we have before us. Not",
    'captured upside as a result of these one-time events, we were highly\ndiscerning in terms of the business we accepted in keeping with our goal to drive high-quality\nrevenue. we expect to maintain the majority of the volume we added in the quarter. I want to thank\nour FedEx team for deftly navigating these conditions to execute on our disciplined strategy. Now to\nDRIVE.\nWe fundamentally changing the way we work, drivers taking cost out of our network, and we are on track to\ndeliver our targeted $1.8 billion in structural benefits|\nfrom DRIVE this fiscal\nyear. At Ground, DRIVE initiatives\nreduced costs by $130 million this quarter. These were primarily driven by lower third-party\ntransportation rates as a result of a newly implemented purchase bid system, as well as optimized rail usage,\nthe continued benefit\nfrom reduced Sunday coverage, and the consolidation of source. At Freight, continue\nto manage our cost base more effectively. For example, the quarter, Freight completed the planned\nclosure of 29 terminal locations during August. And at',
    "the enthusiasm from customers on how much easier it is to\nmanage as we collapse and make the -- not just the pickup experience, the physical pickup one, but we also will\nrationalize our pricing there. And we will automate pickups in a more streamlined fashion, so it's a better\ncustomer experience. To we do not -- we have not yet found opportunities to speed up the network from a\nNetwork 2.0 perspective.\nwe continue to iterate. we have found is that's a lot easier to respond and adapt in the network as we\nbring them together. And so, that has also been something that customers have asked for, especially in the B2B\nspace and healthcare. So, we are learning a lot, but the net takeaway is customers are actually very supportive\nand excited about Network 2.0.\nThis website uses cookies to deliver our services and to\nanalyze traffic.\nWe share information about your use\nof our site with advertising and other partners. Policy\n||\nThe next question will come from Ravi Shanker with Morgan Stanley. Please go ahead.\nRavi Shanker -- Morgan Stanley -- Analyst\nThanks, everyone.",
    "of our capital priorities, I'll\nfocus on maintaining a healthy balance sheet, returning cash to shareholders, and reinvesting in the business\nwith a focus on the highest returns. Our organization will partner closely with Raj and the leadership\nThis website uses cookies to deliver our services and to\nanalyze traffic.\nWe also information about your use\nof our site with advertising and other partners. Privacy\n||\nteam to ensure we deliver consistent progress toward these priorities with the goal of delivering significant\nvalue for our employees, partners, customers, and shareholders in the years to come. a guiding principle\nfor me will be to have open and transparent communication with all key stakeholders, including all of you in the\nfinancial\ncommunity.\nI know some of you from my prior roles. I forward to continuing to work together and engaging with\nthe rest of you in the weeks and months ahead. taking a closer look at our results. fiscal\nyear 2024 is\noff to an outstanding start as demonstrated by the strong operational execution in the first\nquarter. At Ground, DRIVE initiatives are taking hold, and we delivered the most profitable\nquarter in our history for that\nsegment on an adjusted basis. Adjusted",
    "are focused on harnessing the power of this rich data to make supply chains smarter for everyone, for our\ncustomers, for our customers' customers, and for ourselves. we move to the next phase of our\ntransformation, I've given the team three specific\nchallenges: to use data to make our network more efficient,\nmake our customer experiences better, and drive new profitable\nrevenue streams through digital. Looking\nahead to the rest of FY '24. We focused on delivering the highest-quality service and aggressively\nmanaging what is within our control. in better-than-expected first-quarter\nresults, we're increasing the\nmidpoint of our adjusted EPS outlook range.\nAs we to deliver on our commitments, I'm confident\nwe have the right strategy and the right team in\nplace to create significant\nvalue. With that, me turn the call over to Brie.\nBrie Carere Executive Vice President, Chief Customer Officer\nThank you, Raj, and good afternoon, everyone. In the first\nwe remain focused on revenue quality and\nbeing a valued partner to our customers. We did this in an",
    "We are well underway with plans to simplify our organization. In June 2024, FedEx Express, FedEx\nGround, and FedEx Services will consolidate into one company, Federal Express Corporation. The\nreorganization will reduce and optimize overhead, streamline our go-to-market capabilities, and improve the\ncustomer experience.\nTo date, we have implemented or announced Network 2.0 in several markets including Alaska, Hawaii, and\nCanada. As each market is different, we're continuously learning and tailoring the network to adapt to the\noperational characteristics unique to each region while delivering the highest-quality service for our\ncustomers. We continue to use both employee couriers and service providers for pickup and delivery\noperations across the network. As with any significant\ntransformation, these changes are being thoughtfully\nexecuted and will take time to complete. network that FedEx has built over the last 50 years provides us a\nfoundation that is unmatched. physical network enables us to transport millions of packages a day around\nthe world, generating terabytes of data that contain invaluable insights about the global supply chain.\n",
    "While we strive for our Foolish Best, there may be errors, omissions, or inaccuracies\nin this transcript. As with all our articles, The Motley Fool does not assume any responsibility for your use of this content, and we strongly encourage you to do your\nown research, including listening to the call yourself and reading the company's SEC filings.\nsee our Terms and Conditions for additional details, including\nour Obligatory Capitalized Disclaimers of Liability.\nMotley Fool has positions in and recommends FedEx. Motley Fool has a disclosure policy.\nwebsite uses cookies to deliver our services and to\nanalyze traffic.\nWe share information about your use\nof our site with advertising and other partners. Policy\n||\nPremium Investing Services\nInvest better with The Motley Fool. Get stock\nrecommendations, portfolio guidance, and more from The\nMotley Fool's premium services.\nView Premium Services\nMaking the world smarter, happier, and richer.\n© 1995 - 2023 The Motley Fool. All rights reserved.\nMarket data powered by Xignite.\n",
    "And, Mickey, good luck, and thanks for the help over the years. Brie, just one quick follow-up\nfor you. You said that pricing traction was good so far, and you're converting a pretty decent amount of the base\nrate increase.\nWhat percentage of that -- I think, historically has been, like, closer to 50%. Kind of what rate are you converting\nright now? And also, you said that the pricing environment remains pretty rational, but you saw the US Post\nOffice\nbasically say they're not going to have any pricing surcharges. the USPS -- the UPS changes were\nnoted on the call. I Amazon is launching some competitive service as well.\nyou think 2024 could be a tougher environment, pricing-wise, across the industry?\nCarere -- Executive Vice President, Chief Customer Officer\nOK, that was a lot, but I think -- I think I got it. Raj, jump in here if I don't get it all. So, a GRI perspective, if we\ngo back to last January, the answer is the vast majority of our customers pay the full GRI. That",
    "operating income at Ground was up 61%, and adjusted operating\nmargin expanded 480 basis points to 13.3%.\nThese results were driven by yield improvement and cost reductions, including lower line haul expense\nand improved first\nand last-mile productivity. As a cost per package was down more than 2%. At FedEx\nthe business was able to improve operating income despite a decline in revenue. This demonstrates that DRIVE is working. Adjusted income at Express was up 14%, and adjusted\noperating margin expanded 40 basis points to 2.1%.\nCost and transformation efforts at FedEx Express included structural flight\nreductions, alignment of\nstaffing\nwith volume levels, parking aircraft, and shifting to one delivery wave per day in the U.S., all of which\nmore than offset the impact of lower revenue. It's important note that expanding operating margins and\nreducing costs at Express will be a key focus for me and the team. At FedEx the team diligently\nmanaged costs and revenue quality amid a dynamic volume environment. Operating declined 290 basis\npoints based on lower fuel surcharges and shipments but remained strong at 21%. Now turning to",
    "onboarded new customers who\nvalued our service and were committed to a long-term partnership with FedEx. a result, we added\napproximately 400,000 in average daily volume by the end of the first\nquarter, and the team did an excellent job\nfocusing on commercial Ground business acquisition.\nAt Freight, revenue was down 16% driven by a 13% decline in volume. We significant\nimprovement in volume in August due to Yellow's closure. benefited\nfrom approximately 5,000\nincremental average daily shipments at attractive rates as we exited the quarter. As you can see on Slide 11,\nmonthly volumes have improved sequentially with Ground and international export volumes inflecting\npositively\non a year-over-year basis. We to continue benefiting\nfrom this quarter's market share gains throughout\nthe fiscal\nyear. We improved year-over-year growth rates, especially late in the fiscal\nyear, albeit\nwithin a muted demand environment.\nThe old we shared last quarter persisted, particularly at FedEx Express where we saw reduced fuel and\ndemand surcharges year over year. Product mix",
    "operating environment marked by continued but\nmoderating volume pressure, mixed yield dynamics, and unique developments in the competitive landscape.\nLet's take each in turn.\nThis website cookies to deliver our services and to\nanalyze traffic.\nWe also share about your use\nof our site with advertising and other partners. Privacy\nPolicy\n||\nAt FedEx Ground, first-quarter\nrevenue was up 3% year over year driven by a 1% increase in volume and 3%\nincrease in yield. at FedEx Express was down 9% year over year. remained pressured though\ntotal Express volume declines moderated sequentially. export package volumes were up 3% year\nover year. to the fourth quarter, parcel volume declines were most pronounced in the United States.\nU.S. pounds were down 27%, continuing the trend we mentioned last quarter tied to the\nchange in strategy by the United States Postal Service. the Ground and Express, volumes improved\nsequentially, aided by the threat of a strike at our primary competitor.",
    "integrate three customer platforms: customer service, marketing, and sales into one, giving the\ncustomer a more informed, efficient,\nand personalized experience when doing business with FedEx. We are\nnow offering our estimated delivery time window, which provides customers with a four-hour window for their\npackage delivery for 96% of inbound volume globally across 48 countries. This capability is nicely\ncomplemented by picture proof of delivery or, as we like to say, PPOD, which is expanded across Europe in the\nfirst\nquarter. Now in 53 markets, PPOD provides shippers with increased confidence\nin package\ndelivery and helps reduce the volume of customer calls and claims. One FedEx Network 2.0 will simplify\nhow we do business, which is particularly important for our small and medium customers.\nFor our current customer contracts reflect\nthree independent companies. One FedEx enable us to\nchange that, making doing business with FedEx and becoming a new customer easier. Network 2.0 be\nmore efficient\nfor FedEx but also more efficient\nfor our customers. When we integrate market with one truck\nin one neighborhood that's not just for deliveries, it also means a streamlined pickup experience, one pickup per\nday versus two. This is a simple"]

texts_helium5 = [
    "| 0                                              | 1   | 2     | 3   | 4     | 5                            | 6                                                                            | 7          | 8              | 9                    |\n|:-----------------------------------------------|:----|:------|:----|:------|:-----------------------------|:-----------------------------------------------------------------------------|:-----------|:---------------|:---------------------|\n| 3/28/23, 3:56 PM                               |     |       |     |       | Document                     |                                                                              |            |                |                      |\n|                                                |     |       |     |       |                              | derivative  and  non-derivative  financial  instruments)  and  interest      |            |                |                      |\n| Assets Measured at Fair Value                  |     |       |     |       |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | rate  derivative             | instruments                                                                  | to  manage | the            | impact  of  currency |\n|                                                |     | 2018  |     | 2017  |                              | exchange and interest rate fluctuations on earnings, cash flow and           |            |                |                      |\n|                                                |     |       |     |       |                              | equity. We do not enter into derivative instruments for speculative          |            |                |                      |\n| Cash and cash equivalents                      | $   | 3,616 | $   | 2,542 |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       |                              | purposes.  We  are  exposed  to  potential  credit  loss  in  the  event  of |            |                |                      |\n| Trading marketable securities                  |     | 118   |     | 121   |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       |                              | nonperformance  by  counterparties  on  our  outstanding  derivative         |            |                |                      |\n| Level 1 - Assets                               | $   | 3,734 | $   | 2,663 |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       |                              | instruments  but  do  not  anticipate  nonperformance  by  any  of  our      |            |                |                      |\n| Available-for-sale marketable securities:      |     |       |     |       |                              | counterparties.  Should  a  counterparty  default,  our  maximum             |            |                |                      |\n| Corporate and asset-backed debt securities     | $   | 38    | $   | 125   |                              | exposure to loss is the asset balance of the instrument.                     |            |                |                      |\n| Foreign government debt securities             |     | —     |     | 2     |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | 2018                         |                                                                              | Designated | Non-Designated | Total                |\n| United States agency debt securities           |     | 11    |     | 27    |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Gross notional amount        | $                                                                            | 870        |                | 5,466                |\n|                                                |     |       |     |       |                              |                                                                              | $          |                | $                    |\n|                                                |     |       |     |       |                              |                                                                              |            |                | 6,336                |\n| United States treasury debt securities         |     | 23    |     | 70    |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Maximum term in days         |                                                                              |            |                | 586                  |\n| Certificates of deposit                        |     | 11    |     | 27    |                              |                                                                              |            |                |                      |\n| Total available-for-sale marketable securities | $   | 83    | $   | 251   | Fair value:                  |                                                                              |            |                |                      |\n| Foreign currency exchange forward contracts    |     | 77    |     | 15    | Other current assets         | $                                                                            | 15         |                | 28                   |\n|                                                |     |       |     |       |                              |                                                                              | $          |                | $                    |\n|                                                |     |       |     |       |                              |                                                                              |            |                | 43                   |\n| Interest rate swap asset                       |     | —     |     | 49    | Other noncurrent assets      |                                                                              | 1          |                | 33                   |\n|                                                |     |       |     |       |                              |                                                                              |            |                | 34                   |\n|                                                |     |       |     |       | Other current liabilities    |                                                                              | (5)        |                | (15)                 |\n|                                                |     |       |     |       |                              |                                                                              |            |                | (20)                 |\n| Level 2 - Assets                               | $   | 160   | $   | 315   |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Other noncurrent liabilities |                                                                              | —          |                | —                    |\n|                                                |     |       |     |       |                              |                                                                              |            |                | —                    |\n| Total assets measured at fair value            | $   | 3,894 | $   | 2,978 |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Total fair value             | $                                                                            | 11         |                | 46                   |\n|                                                |     |       |     |       |                              |                                                                              | $          |                | $                    |\n|                                                |     |       |     |       |                              |                                                                              |            |                | 57                   |\n| Liabilities Measured at Fair Value             |     |       |     |       |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | 2017                         |                                                                              |            |                |                      |\n|                                                |     | 2018  |     | 2017  |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Gross notional amount        | $                                                                            | 1,104      |                | 4,767                |\n|                                                |     |       |     |       |                              |                                                                              | $          |                | $                    |\n|                                                |     |       |     |       |                              |                                                                              |            |                | 5,871                |\n| Deferred compensation arrangements             | $   | 118   | $   | 121   |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Maximum term in days         |                                                                              |            |                | 548                  |\n| Level 1 - Liabilities                          | $   | 118   | $   | 121   |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Fair value:                  |                                                                              |            |                |                      |\n| Foreign currency exchange forward contracts    | $   | 20    | $   | 37    |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Other current assets         | $                                                                            | 11         |                | 4                    |\n|                                                |     |       |     |       |                              |                                                                              | $          |                | $                    |\n|                                                |     |       |     |       |                              |                                                                              |            |                | 15                   |\n| Level 2 - Liabilities                          | $   | 20    | $   | 37    |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Other noncurrent assets      |                                                                              | 1          |                | —                    |\n|                                                |     |       |     |       |                              |                                                                              |            |                | 1                    |\n| Contingent consideration:                      |     |       |     |       |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Other current liabilities    |                                                                              | (7)        |                | (29)                 |\n|                                                |     |       |     |       |                              |                                                                              |            |                | (36)                 |\n| Beginning                                      | $   | 32    | $   | 86    |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Other noncurrent liabilities |                                                                              | (1)        |                | —                    |\n|                                                |     |       |     |       |                              |                                                                              |            |                | (1)                  |\n| Additions                                      |     | 77    |     | 3     |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       | Total fair value             | $                                                                            | 4          |                | (25) $               |\n|                                                |     |       |     |       |                              |                                                                              | $          |                | (21)                 |\n| Change in estimate                             |     | 15    |     | 2     |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       |                              | In November 2018 we designated the issuance of €2,250 of senior              |            |                |                      |\n| Settlements                                    |     | (7)   |     | (59)  |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       |                              | unsecured  notes  as  a  net  investment  hedge  to  selectively  hedge      |            |                |                      |\n| Ending                                         | $   | 117   | $   | 32    |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       |                              | portions of our investment in certain international subsidiaries. The        |            |                |                      |\n| Level 3 - Liabilities                          | $   | 117   | $   | 32    |                              |                                                                              |            |                |                      |\n|                                                |     |       |     |       |                              | currency effects of our euro-denominated senior unsecured notes              |            |                |                      |\n|                                                | $   | 255   | $   | 190   |                              |                                                                              |            |                |                      |\n| Total liabilities measured at fair value       |     |       |     |       |                              | are reflected in AOCI within shareholders' equity where they offset          |            |                |                      |\n|                                                |     |       |     |       |                              | gains  and  losses  recorded  on  our  net  investment  in  international    |            |                |                      |",
    '| 0        | 1                                                                                     |   2 |\n|:---------|:--------------------------------------------------------------------------------------|----:|\n| Item 7.  | Management’s Discussion and Analysis of Financial Condition and Results of Operations |   8 |\n| Item 7A. | Quantitative and Qualitative Disclosures About Market Risk                            |  15 |\n| Item 8.  | Financial Statements and Supplementary Data                                           |  16 |\n|          | Report of Independent Registered Public Accounting Firm                               |  16 |\n|          | Consolidated Statements of Earnings                                                   |  17 |\n|          | Consolidated Statements of Comprehensive Income                                       |  17 |\n|          | Consolidated Balance Sheets                                                           |  18 |\n|          | Consolidated Statements of Shareholders’ Equity                                       |  19 |\n|          | Consolidated Statements of Cash Flows                                                 |  20 |\n|          | Notes to Consolidated Financial Statements                                            |  21 |\n| Item 9.  | Changes in and Disagreements With Accountants on Accounting and Financial Disclosure  |  33 |']

texts_long = [
    """You cannot play any games about this. You have to admit that this is wrong. I think especially for mathematicians to come in and see an environment where there's guiding ideas that people haven't really worked out, and a lot of things are known, do not work for known reasons. But people are still acting as if this is not true and trying to figure out how to do something and make career for themselves. Peter Wojt is a theoretical physicist and a mathematician at Columbia University. He's been an influential figure in the ongoing debates surrounding string theory. His critiques, as articulated in his book, Not Even Wrong, strike at the heart of many popular assertions about this framework. Professor Hoyt also has a widely read blog in the math and physics scene called Not Even Wrong, so it's the same name. And the links to all resources everything mentioned will be in the description as usual. take meticulous time stamps and we take meticulous show notes in one sense the problem with string theory is the opposite of the problem of fossil fuels with fossil fuel companies You have a goal let's say it's to wash your clothes and you're able to achieve that goal but you produce negative externalities where a string theory has plenty of positive externalities but arguably achieves little toward its initial goal Professor White introduces a novel tow approach called Euclidean Twister unification. You may recognize that term Twister, as it's primarily associated with Roger Penrose. Twisters provide an alternative to spacetime descriptions in quantum physics. Peter's application of Twisters is in the Euclidean setting, and he talks about how this significantly changes the playing field. It opens up a connection between gravity and the weak interaction, because space-time in this formulation is inherently Cairo. We also talk about spinners and Michael Atiyah. You know how some people are Christian mystics or Muslim mystics? Well, Atiyah seems to be a spinner mystic. We alternate between technical and more intuitive discourse. If you're new to the theories of everything channel, this is par for the course, and my name is Kurt Jai Mungle. Usually what we do is we interweave between rigorous, steep technicality, and then periods of explaining the intuition behind what was just said. In other words, you can think of it as high intensity interval training for the mind. Recall the system here on Toe, which is if you have a question for any of the guests, whether this guest or from a different Toll podcast, you can leave a comment on that podcast with the word query and a colon. This way, when I'm searching for the next part with this guest, I can press Control F, easily finding it in the YouTube studio backend. Further, if I'm able to pose your query, I'll cite your name verbally, either aloud or in the description. Welcome and enjoy this episode with Peter White. Welcome, Professor. Thank you so much. It's an honor to have you. I've been wanting to speak to you for almost two years since you came out with Euclidean Twister Theory or Euclidean Unification Theory. And while here you are. Well, thanks. Thanks for having me on. I'm looking forward to the opportunity to kind of be able to talk about some of these topics. And I've certainly enjoyed some of your other programs. And the one with my friend Edward Frankel recently was really spectacular. Thank you. Yeah, that's all due to Ed, of course. Okay. What are you working on these days? What's your research interests? Yeah, so there's something very specific. I'm just in the middle of trying to finish a short paper about an idea, which I'm not quite sure what they're... I guess I've for now entitled, the draft of the paper is titled Space Time is Right-Handed. And there's a slight danger that I'll change conventions. It'll end up being that slight space time is left-handed, but I think it will stay right-handed. And that's related to the twister stuff that I've been working on for the last few years, which I'm still quite excited about. But there's something at the, there's one kind of basic claim at the bottom of what I'm trying to do with the twisters, which is, I think to the standard way of thinking about particle physics and general relativity and spinners, it's initially not very plausible. I should say one reason that I actually didn't, it took me a long time to get back to the Euclian twister stuff from some early ideas years ago was that I didn't actually believe that this basic thing that I needed to happen and could happen. And I think lots of other people have had the same problem with this. And the more I looked into the twister stuff, the more I became convinced that something like this had to work out. But more recently, the last few months, I've come up with an understanding in much simpler terms, not involving twisters, just involving spinners, about the really unusual thing that's going on here. And I think that, you know, I've been trying to write up kind of an explanation of the basic idea. And I think it's a fairly simple one. And as I've been writing it up, I keep thinking, well, wait a minute, can this really work? There's no way this can actually really work. But the more I've been thinking about it, the more I've been convinced, yes, this actually does really work. So I'm hoping within the next few days to have a final version of that paper, well, not a final version of that paper I can at least send around to people and try to get comments on and also read about it and publicly on my blog. I read the paper. Thank you for sending you. Yeah, what you have is a very, it was a very early draft of it, which made even less, hopefully the, I'll have something that will make more sense will be what the public will see, but we'll see. Yeah. Do you think spinners are more simplified or easy to understand the twisters? Oh, yeah, yeah. So spinners are really very basic, very, very basic things. I mean, every elementary particle like electrons are the way you describe them. They're spinners. They're going to have nature as spinners. You have to electron wave functions are spinners. And so they're in every, you know, every physics every if you do quantum mechanics or do quantum field theory you have to spend a fair amount of times at spinners so spinners are very very basic things and they're not um i spent a lot of my career kind of thinking about them trying to better understand them and i keep learning new things and it's in the last few months i kind of i something about them, which I think is new, at least I've never seen before. And this is what I'm trying to write about it. But they're very fine metal objects. It's a little bit hard to, anyway, I can give you a whole lecture on spinners. I'm not sure how much of that you want or where you want to start with that. Right. Well, there's one view that we can understand them in quotes algebraically, but that doesn't mean we understand what spinners are. So that's the Michael Latia approach where he says it's like the letter I, the complex eye, the imaginary I, back in the 1400s or 1500s. It's only now or a couple hundred years later, you realize what they are. And so sure, we have many different ways of describing spinners mathematically, but it's still a mystery as to what they are. So do you feel like, no, we understand what they are, or there's much more to be understood more than the formalism? Well, yeah, it's very interesting. You bring up Atia, yeah. So Atia at various points, did make this argument that there's something very interesting in which we don't understand going on of the spinners and that yeah he i think was thinking of it in a much more general context spinners you know are really if you try and do geometry of any kind um or reminding in geometry you re expressing everything in terms of spinners instead of in terms of vectors and tensors gives you a very different, in some ways, more powerful formalism, but one that people are not that used to. And it has some amazing properties. It's kind of deeply related to notions about topology and K-theory and the Daraq operator gets into it. So the thing that made attia you know really most famous his index there was singer you know this is that it's basically saying you know you can compete everything comes down to a certain kind of fundamental case and that is the final case of the drach operator and spinners so he was seeing spinners kind of at the, you know, as this really kind of central thing to the most important thing that he'd worked on. And so there's a lot to say. So there's a lot known about spinners, but there's also a lot, it's a little bit mysterious where they come from. I think the new stuff that I've been more, so I've been thinking about that a lot over the years, but the new stuff that has gotten, where I think there's something new that I see going on is not the general story about spinners, but a very, very specific story about spinners in four dimensions. So you have spinners in any dimension. Any dimension, you can write down spinners and they're useful. But in four dimensions, some very, very special things happen. And the other very, very special thing, it's interesting thing that's going on in four dimensions is that from the point of view of physics, there's two different signatures that you're interested in. You're interested in either spinners in the usual kind of four dimensions where all four dimensions are the same and you're just trying to do Euclidean geometry in four dimensions, which I might sometimes call Euclidean spinners, or you're interested in spinners of the sort that you actually observe in a relativistic quantum field theories where the geometry is that of Minkowski space. So sometimes we refer those as Minkowski spinners. And so you have two different versions of four dimensions, one with a totally positive signature and one where one direction has the opposite sign than the others in the metric. So you have to treat time differently than space, and that's Minkowski space. So there's two different things than the general story that I'm interested in here. One is very specific, what has specifically the geometry of four dimensions, and the other is very specifically the relation between Euclidean and Minkowski signature spinners. So is it your understanding or your proposal that the world is actually Euclidean, and it's been a mistake to do physics in a Minkowski way? When we wick rotate, we see that as a mathematical trick. And you're saying, no, no, no, that's actually the real space. That's the real, quote unquote, even though there's something imaginary about it. And the Minkowski case was the mistake. Like, an analogy would be, we operate in U.S. USD. And then for some calculations, it's easier to go into yen. And we think that the actual world is operating in the United States, and the calculations are just something to make the numbers easier. And then you're saying, no, no, no, what's really happening is in Japan, and it's been a mistake to go into the USDA, or the USD is just to make the math easier. So is that what you're saying or no? Well, so this goes back more to the Euclidean twister stuff. Yeah. So there, well, yeah, it's been well known in physics that you really kind of, that the problem in there's a problem with Minkowski space time. If you try and write down your theory in Mkowski space time, you, the simplest story about how a free particle evolves, you write down the formulas for what's a free particle going to do, what's its propagator, and you see that it's just ill-defined. There is no, you know, you've written down a formula which mathematically is ill-defined. It needs more information in order to actually be a well-defined formula. And I mean, technically, if you look at any physics book, you'll see they're saying, well, you know, we're going to do, the answer is this integral, and you look at this integral, and this integral is going straight through two poles, poles, and that's just ambiguous. You don't know how to define such an ambiguities about how you define such an rules. So the one, the aspect, you've always known you have to do something like with rotation. You have to do something. You have to get rid of those ambiguities. And one way of getting rid of those ambiguities is, you know, analytically continuing and making time a complex variable, analytically continuing it, analytically continuing maybe to Euclidean signature, and there the formulas are well defined. So it's, yeah, I'm not sure, I'm very comfortable saying one of these is real and one of these is not. It's the same, it's the same formula. It's just you have to realize that to make sense of it, you have to kind of go into the complex plane in time. And you can, if you things are analytic, if this is a holomorphic function in time, you can either evaluate what happens at imaginary time or you can make time real, but you have to take the limit in a certain way, moving, like perhaps starting with imaginary time and then moving analytically continuing a certain direction to get real time. But that's the standard story. That's not me saying this. That's a standard story. Right. And then there's a, how do you, what sense do you make of this? Is this just a mathematical truck, which a lot of physicists will say, well, that's just some kind of weird mathematical trick. It's not, has nothing to the reality. Or do you take this more seriously? So what's always fascinated me is more is that it's fairly clear what's going on if you just talk about scalar fields. If you talk about particles with spin zero or fields that transform trivially under rotations, you know, what happens when you go to imaginary time is, you know, it's quite interesting and in some ways tricky, but it's very well understood. But it's never actually been very well understood. What happens when you have spinner fields? And this is the problem is that these spinners in Euclidean signature and spinners in a calcium signature are quite different things. And so you can't just say, oh, I'm going to analytically continue from one to the other because they're not related. Anyway, it's very unclear how you're going to do that. And so there's also a similar story in Twister theory. You can do Twister Theory, Yonkowski Space Time, which is what Penrose and his collaborators mostly did. Or you can do it in Euclidean signature space time, which is what Atia and a lot of other people and mathematicians have done. And in principle, the two are related by analytic continuation. But the way that works is quite, you know, I think it's much subtler than you expect. And so what I've been interested in, you know, most recently this business about, it really is a claim that the standard way of thinking about how you analytically continue between these two different kinds of spinners is you're making kind of a wrong choice when you do that. And there's a good reason for the standard choice you're making when you normally when you do that. But there is actually another choice you can make, which is that instead of working with spinners which are kind of symmetric, there's two different kinds, which by convention you can call right and left-handed or positive and negative chirality. And the standard setup treats this question, you know, symmetrically, but between the plus and minus, the chirality is between right and left spinners. But it's, what I've kind of realized recently is it looks like it's quite possible to make this setup completely asymmetric so that you just describe spinners using these right-handed or positive chirality spinners. You just don't use the left-handed ones at all in your construction of space time. You can do that. It appears to be, and that's why this paper is called space time is right-handed. Is it the case that you could have called it space-time as chiral, and you could have equivalently described as left-handed, or is there something specific about right-handedness? No, yeah, yeah. It's certainly, it's a matter of convention, which, but you base have but you basically, to say it a little bit more technically, you know, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, the, if you're, is this group called SL2C, it's two by two complex matrices, a determinant one. Um, and what you realize is when you, if you work, if you come, if you work in complex version of four dimensions, the symmetry group is two copies of SL2C. And you can call it a plus copy and a minus copy or you can call it a right copy and a left copy, but there's two of them. And the standard convention in order to get analytic continuation to work out the way people expected has been to say that the physical Lorentz group that corresponds to our real world is not chiral asymmetric. It's kind of a diagonal, which is you use both the right and left. And you have to complex conjugate when you go from one side to the other. But it kind of the Lorentz group, the SL2C Lorentz group we know, is supposed to sit as kind of a diagonal thing, which is both right, right, and left. But what I'm kind of arguing is that, no, you can actually set things up so that the, the Lawrence group is just one of these two factors. It could have been the right factor, left factor. You have to make your choice of convention. But so it is very much a chiral setup. But you only, the strange thing about this is you only really see this when you complexify. If you just look at Minkowski space time, you know, you don't actually see this, anyway, you don't see this problem or you don't see this ability to make this distinction. It's only when you go to Euclidean space time where the rotation group really does split into two completely distinct right and left things. Or if you go to a complexified space time where you have this two copies of SL2C, it's only in those contexts that you actually see that there is a difference between choosing the diagonal and choosing the right-handed side. So for SL2C, you call that the Lorentz Group. Is that technically the double cover of the Lawrence Group? Yeah, people use both terminology. If you're going to work with spinners, you have to use a double cover. But yes, it's also, yeah, yeah. Sometimes you might want to say that S.O3 is the Lorentz group and this is the double cover. But most of you're working, you're interested in doing working with spinners, and then you have to use the double cover, really. Yes, yes. So is there a reason that triple covers or quadruple covers aren't talked about much? Is it just because of experiment? There's nothing there. Well, it's more than mathematics that they don't. There is, I mean, there is, the rotation groups of any kind, you know, have this, have this twofold nature. There is this spin double cut. There is this, they have these spin double covers. In many cases, you can kind, one way of seeing this is just a basic topology, the topology of rotations has a, you know, has a plus and minus thing in it, which you kind of, and you have to do something about that. So there aren't any kind of known, mathematically interesting, triple covers, etc. Now, in the standard model, the way that it's written in bundle language is that it's a principal bundle, and then the gauge groups are the structure groups. And then for general relativity, you have a tangent bundle. And then some people say that the gauge group of GR is the dipheomorphism group. But is there a way of making that into a bundle, like a principal bundle with the diphomorphism group? How is one supposed to understand that as a bundle construction? Yeah, yeah. Anyway, there's a lot of different ways. There's several different ways of thinking about geometry and about Romanian geometry. And yeah, and this starts to get a complicated subject. But maybe the best way to, well, thinking in terms of different amorphism groups is something you can do. It's actually not my favorite way of doing this kind of geometry. And for the reason is that it, maybe let me just say something about an alternate way of thinking about geometry, which seems to me more powerful. Maybe actually to motivate this a little bit better. If you just think about diffamorphism groups, it's very, very hard to understand what spinners are and where they come from. You really kind of can't see them at all if you're just thinking about the diphthomorph group of a manifold. So the the other formulation of geometry going back to Carton, which makes it much, makes it much easy to see where spinners are going on going and is a lot more powerful in other respects, is to think not about your not about a manifold, but about a bigger space, which is a bundle that for each point in the manifold, you consider all possible bases for the tangent bundle. It's also called frames. And so this is sometimes called the frame bundle. And so it's kind of saying if you want to understand geometry, you should look at the points of space and time. But at the same point, you also got to think about the tangent space and you should think about the possible bases of the tangent space and the so-called frames. So you should always kind of think, instead of writing all your formulas in terms of local coordinates on the manifold, you should think about your problem as being a problem that lives up on the frame bundle and that you always, you're not just, you're not just at a point to space time, but you've also got a frame. And then, but then you have to be careful to kind of work kind of equivariantly that you have, you know, you can change your choice of frame. You can rotate your frames. So you have, you kind of work up in the frame bundle, but equivariantly with respect to rotations or whatever. So that's, that gives a lot more structure to the problem. In particular, it allows you to easily say what spinners are, which you couldn't if you just talked about. So, so anyway, there's a lot more one could say about Diffey Morphor's in groups and that, but just in terms of the relation to the spinner stuff, maybe it's just to forget about it. Just to say it that way. It's not, you have to do something quite different if you're going to talk about spinner. Right. Okay, now the problem you were working on earlier that you said you weren't sure if it would have a solution and you're finding that it does what was it in the early part of the conversation which you were working on your research interests well do you mean right at the beginning where i'm still what i'm still confused about yeah okay but it seemed to me that you were saying you're solving the problem. Oh, this. Yeah. So this was, I mean, this was actually, it goes back to when I was graduate student or postdoc, it was first occurred to me. You know, actually maybe to kind of explain how this all came about. So I was a graduate student in Princeton and I was working on lattice gauge theory. So we're working on this kind of formulation of Yang-Mill's theory on a lattice. And so you could actually do computer calculations of it. And so I was trying to understand, you know, there's a lot of interest in topological effects in Yang-Mills theory. And I was trying to understand how to study those in the kind of numerical calculations on the lattice. And then, so I made some progress on that. But then the next thing that really occurred to me was exactly as spinners came up. It's like, besides having Yang Mills theory on a lattice, we also want to put spinner fields on the lattice. So there's this really beautiful way of putting gauge fields in the lattice, the Yang Mills theory, which kind of respects the geometric nature of the gauge fields very, very nicely. It's kind of the Wilson's lattice gauge theory. But there isn't, if you try and put spinners in the lattice, a lot of very mysterious things happen. And again, in some sense, the problem is that if you're just looking at this lattice that you've written down, it's clear kind of what the discrete analogs of vectors are and of planes and of those things. But it's very, very unclear what, since you don't really have a good way of thinking about spinners in terms of kind of standard geometry of, you know, lines, planes, et cetera, you don't really know how to put the spinners on a lattice in a way that respects their geometry. And if you try to write down the formulas or do anything, you run into a lot of weird problems. There's a lot of, anyway, there's a long story about what happens if you can try with spinners and lattice. Is this related to doubling, like the species doubling? Yeah, so there's one, yeah, so one thing you find is that you really, there's no kind of consistent way to put kind of a single kind of Fermion in the lattice, that if you try and do it, any way you know of doing it kind of produces all these kind of extra versions of the same thing and you have to somehow disentangle those. That's part of the problem. Okay. But that's when I started thinking about the geometry of spinners and some ideas about putting them on the lattice. And then what I was seeing, I started to see that, wait a minute, you know, if you, so this is all happening in Euclidean space where the rotation group is a copy of two SU2 u s u.u2s there's again a left-handed one or a right-handed one if you like and um what i was seeing really was that the some of the choices of it the geometry is trying to use to put these things in the lattice gave me kind of things occurring and kind of multiplets that that look that had the same SU2 structure as what you see in a generation of electro weak particles so in a generation of electro week um like to be particles that you for instance have you have a neutrino left end of neutrinos and you have right-handed left-hand electrons for instance. And those have certain transformation properties under the SU2 and under a U-1. And those were the same ones that I was seeing when I was trying to construct these spinners. So I had the, so it seemed to me, if you can think of part of this rotation group, this SU2, as an internal symmetry, as the symmetry of the weak interactions of the Weinberg's Sala model, then you could actually, anyway, you got all sorts of interesting things to happen. But the thing that this, but making this idea work really required that some explanation of why in Euclidean space, what you thought were spacetime symmetries that really broke up into half space time symmetries and half an internal, internal symmetries, which didn't affect space time. So I never, this is what for many years after looking at this, it was like, well, this just can't work. I mean, you can't, if you just look at the whole formalism for how you've set this up and, you know, both of these SC2s have to be space time temperatures. You can't, they're both going to affect space time. You can't, you can't get away from that. Other people didn't see this as a problem? No, no, I think everybody saw this as a problem. I mean, I think anybody who ever looked at this idea of trying to get, you know, one of the, part of the four-dimensional rotation symmetry to be an internal symmetry has probably backed away, backed away from it for the same reason, saying, well, wait a minute, this can't, you know, I just can't see how that could actually happen, that you have to, you're telling me this should be an internal subject which doesn't affect space time, but it looks to me that you're rotating space time with it, so you can't do that. And so this is what, for many years, kind of kept me from going back to that, from those ideas. And as I learned more about quantum filter, actually, one motivation, as I was teaching this course on quantum filtering, quantum filtering in the back of my mind is, okay, you know, as I go along and teach this course, I may not explain this to the students, but I'm going to very, very carefully look at the formalism and I'm going to understand exactly how this analytic continuation is working of these spinners. And I'm going to, you know, and I'm going to see that you know it looks like this has to work and I'll finally understand why and then I can stop thinking about this but but I kind of as I was teaching this as I was looking at this I kind of never actually saw you know anyway I never actually really saw the argument for why this why this be a space side of symmetry. It looked like it had to, but you couldn't quite pin down why. Anyway, so then when I went back to the twister stuff, I became convinced that if you think about everything in terms of twisters, then the whole twister setup is naturally, chirally, asymmetric. So you kind of, from the twister point of view, this kind of thing looked a lot more plausible, and I got more interested in it again. But it's only very recently, the last few weeks, the last couple months, that I've kind of, I kind of have a very good understanding of exactly why it seemed that, you know, that what I, that why I was right, that this should be impossible. There is a standard assumption that you're making, which makes what I wanted to do impossible. But it's also possible to not make that assumption and do something else. And that assumption is? It's the symmetry between right and left. It's kind of when you go between Minkowski and Euclidean spinners, you know, the setup that you use to analytically continue, do you do that in a setup which is right-left symmetric? And if you want the setup to be holomorphic then you have to use the right left symmetric one but what it's so simultaneously i realize yes you can yeah yes i mean this in the standards there was a very very good reason that i and everyone is skeptical that this can make sense but there there actually is a way around it. You can just decide, okay, I'm going to, I'm going to just use right-handed spinners, and I'm going to, and you can get a theory that makes sense. I don't know if I'm jumping ahead, but I recall in one of the lectures that I saw online of you, and you were giving the lecture, I believe Cole Fury was in the audience, you were saying that what we have to use are hyper functions. Yeah. Am I jumping ahead because you're saying, no, no, it's not going to be holomorphic? No, but actually hyperfunctions are really part of the holomorphic story. They're, yeah, they're not, they're, I mean, hyper functions are really just saying, so so what I was saying when I was trying to explain this business about you know why about WIC rotation and that and that things were that if you write down the standard formulas you end up with something in a Kasek space on which is ill-defined okay and then you have to use it via Rick Rotation or analytic continuation. There's just another way of saying that more, putting in a more interesting mathematical context, is to say that the things that you're looking at in Kowski-space time are not actually normal functions. They really are what am I? They are best thought of as hyper functions. In this case, they're hyper functions, which are just, um, analytic, which are just kind of bound, boundary values of analytic things as you, uh, approach, uh, approach the real line. But, um, yes, so the hyper function story is just kind of part of the standard. It's really part of the weird rotation story. Yeah. Okay. Yeah. But what I'm, I mean, this latest thing I'm trying to do actually gets away from analytic continuation. You're not, you really, I'm really, I'm still kind of, you know, trying to wrap my head around exactly what the implications of this are. But you are, you're not doing the standard sort of analytical continuation anymore. The standard sort of way of analytically continuing, which uses all four space time dimensions, that you're not, you're not doing that. You're doing something, something different and it's unclear. Anyway, I mean, if you start writing out formulas, you'll still get the same story with hyper functions. What prompted you to then go look at twisters? And by the way, is it called a twister formalism or twister formulation? I don't know. Either one is... I don't know if those are used interchangeably. Because I hear, for instance, that there's different quantum formalisms like Vigners or interaction or path or categorical. But then sometimes I hear, yeah, the categorical formulation of quantum mechanics. I'm like, okay, you get the idea. Well, they're not, I mean, the thing about Twisters is they're not actually, I mean, maybe a good thing to say about Twisters is, we don't actually know exactly what their relevance is to the real world. So you might, if you had a, if you have a well-developed idea using Twisters for describing the real world and you wanted to contrast it to other similar descriptions, you might want to say, oh, this is the Twister formalism or maybe Twister formulation. I don't know. But it's a little bit, but either one is a little bit premature in terms of physics, that we don't actually know exactly how the twisters are related to the real world. So it's not like you can translate a real world problem to twister formalism and then back? Well, you can, so twisters, maybe... So twisters are a bit like spinners, but the, um, so they have some of the mathematical properties of spinners, but, but they do something more interesting. They're kind of a higher dimensional thing. Maybe one of the best things to say about them is that they're, um, they're very useful. If you want, if you, so if you want to understand Minkowski space time, you know, you, this is what Einstein figured out. You can, you can use Minkowski's geometry, Minkowski metric, if you want to talk about just vectors and metrics and tensors, or if you talk about Mokowski space type spinners, if you want, and that's what I've been most interested in. But the other interesting thing about our theory is when we write them down in Mikowski space time. Theories of mass of massless fields and things like Yang Mill's theory, they have this bigger invariance group than just under rotations and translations. They're conformally invariant. So the geometry of chrysters really comes into its own, if you're trying to describe to understand the properties of space time under conformal transformations. And anyway, so that's kind of a motivation. So if you don't care about conformal transformations, you may not be very interested in spinners, but if you really want to understand, you know, what is, how do I write down my theories and how do I have a version of you of Metcowski space time that, where the conformal group acts in a nice linear fashion and where everything works out. And the spinner, now you can call it a formalism or formulation, but it's a way of doing conformal geometry. It really comes into its own. So that's, so spinners, you know, go, I mean, twisters go way, way back. And, you know, this really was mainly Roger Penrose is doing in the 60s. And, you know, and he was very interested in using them to understand, you know, things happening in Minkowski space time and especially the conformal invariance of these things. And so there's a huge amount of effort and a lot of beautiful things discovered during the 70s, especially by him and his collaborators in Kowski space time. And then Atiyah realized that you could take this over and do some very, very interesting things in Ramanian geometry and Euclidean space time. Yeah. So, I mean, I was, you know, I kind of learned about this geometry at raise points. That sentence could be said about Atea in the most general form. And then Atia realized you could use this for underscore with geometry. Yeah, yeah. Yeah. So anyway, so I've been kind of aware about Twisters for a long time, but I, you know, I didn't see. Anyway, I actually wrote a very speculative paper long, long ago about this. And it mentioned the connection to twisters, but there's just a lot about them that, you know, I didn't understand back then. It took me many years to understand. And especially the relationship between is Euclidean signature and Minkasee signature spinners, how they're related is. That's quite a tricky story, which takes me a long time and understand. So you have the splinter in your thumb for decades about the spacetime symmetries and them acting not just on spacetime. What happened in 2020 and 2021? I'm trying to think. Now, I'm trying to think what specific one thing had happened in 2020 was COVID so right in your mind what happened so 2019 then no no no but this is actually relevant because actually in 2020 I was much more and I was thinking of this stuff but yeah but yeah but in 2020 all of a sudden you're kind of you know you're at home you're at home that you're just sitting there and uh i all the opposite home and i don't have a lot of all the usual distractions or whatever and so and so that actually um i actually gave me some of the more time to kind of think peacefully about uh about some of this stuff and make some progress yeah so i'd have i'd have to i mean i kind remember now that, you know, exactly which things became clear at which times. But it's been a slow, it was a slow process of various things clarifying. But I think maybe that was one of the main things, is to finally get a picture in mind of how Euclidean and Minkowski twist your theory all fit together. Awesome. How does it fit? Is there a way of explaining it? Well, I mean, maybe the best thing to say about twister theory is that it really kind of naturally wants to be a theory of complex space time. And this is the thing. If you write, if you say, I'm going to study four-dimensional complex space time and I'm interested in its conformal group and things like that, then the Twister story is actually very, very simple. It's very, I mean, you're basically just saying that there's a four-complexed dimensional space and a point in space time is a complex two plane in that four-dimensional space. So points, anyway, yeah, so instead of thinking of the way of normal thinking of some space with these points, well, you've got to think about, just think about the complex two planes and complex four-dimensional space, and, you know, everything just kind of drops out of that. And there is one, there's a beautiful relation of that story to the theory of spinners, is that, and this is kind of the relationship between the theory of twister and theory of spinners. In twister theory, a point in four-dimensional space-time is a complex two plane. That by definition of what a point is. And now that, but that complex two plane. That's the definition of what a point is. But that complex two plane, that kind of tautologically answers the question of where do these spinners come from? Because the space of spinners is a complex two plane. Well, you know, so from the standard point of view, it's like, you know, as I was saying, if you just think about the diphthomorphism, it's very, very hard to even say what a spinner is. So where are these weird complex two planes coming from? Well, from the point of view of twister theory, it's purely tautological. It's just, you know, the two plane is a point. So the spinner, the spin one-half two-plane, complex two-plane is describing the spin of a of an electron is exactly a point anyway that that's exactly what what the definition of a point is so you can't a point in twister space or a point in space time spilling in space time yeah so as twister space is a four thing. And but the points in it, and so the points in it correspond to various structures in space-time, but the complex two planes in it correspond to the points in space-time. Anyway, that's one of the basic. Yeah. So then is the statement that the points in spacetime are the same as spinners or the points in space- or the structure of space time gives rise to the structure of spinners and vice versa or are none of those statements correct? I think, yeah, I know, I think both of them. I mean, it really is telling you, Twister theory is really telling you that it's a way of thinking about space time in which... And sorry, this is four dimensional space time. Four dimensional space time, yeah, yeah, yeah. It's a way of thinking about, yeah. So, Twister theory is very, very special to four-dimensions. It doesn't really is, it's a way of thinking about space-time in which, you know, the occurrence of spinners and their properties are just completely tautological. They're just built into the very definitions. Sociologically, why do you think it is that Penrose's Twister program, firstly, has been allowed to continue because many other programs just die out if you're not loop or string or causal or asymptotic. Like there's just four as far as I can tell. Five with Penrose. So why is it alive, and then why hasn't it caught on? Well, for, I mean... Or maybe you disagree. It's not alive. No, no, no. It's very much alive. It's very much alive. And still... And so there's an interesting kind of history. But a lot of it was really... So he had this idea, and he's raised places as explaining how he came up with it. And he was very, very struck by this. And, you know, so he quite successfully at Oxford built up a group of people working on this. And so, you know, it was a good example of kind of how normal science kind of works sociologically. You know, somebody comes up with a good idea and they actually build a group of people around them and people do, as people do more work, they learn more interesting things about this more people get interested so you know he always you know throughout the 70s I would say into the 80s there always was a quite healthy group of people you know working on pedros or people somehow having some relation to penrose collaborators were working on this so it was um anyways but perfectly but perfectly normal science. It wasn't so clear, though, how to get, it was clear, some things were very clear, some things were clear that this was really a beautiful way of writing down conformally invariant way of equations and studying their properties. So there were, there was, the beauty of the idea and the power to do certain things was known. But it didn't seem to be necessary or have any particular connection to specific problems in particle physics. So particle physicists would look at this and say, well, that's nice, but, you know, I don't, that doesn't actually tell me anything. You know, if I needed to do some conformally invariant calculations, I might be able to use that, but it's not actually telling me something really that, you know, really knew I can't get elsewhere. And then, you know, and then in the 80s, you also had, you know, Atea got into the game, and there's a lot of mathematicians got into it through the, the relations to the, on the Euclidean side. So, you know, it was, you know, especially among mathematicians, mathematical physicists, it was a fairly, it remained a very active area, and it still is to this day, you know. A lot of it was based in Oxford, but also a lot of other places. But yeah, I think the, but in terms of its implications for physics, you know, I would say the thing that to me is, I think Penrose and his people trying to connect this to physics in an interesting way, they kind of ran into, anyway, they kind of ran out of new ideas. There are some things that they could do, but they couldn't actually get any kind of a really killer app, if you like. And the big, and from my point of view, I mean, I don't know if I can, I think, anyway, I don't know if I'll ever be able to convince them or what they think of it these days. But the problem was that they were thinking of connecting this to physics purely from the Minkowski space-time side. So they're looking at Minkowski space-time twisters, Minkowski-space-time spinners. And those, the twister theory just didn't, if you just look at Minkowski-Space time, you don't see the sort of new things, which I'm finding interesting, which I think tell you something new about particle physics. You don't see this kind of internal, the fact that one of these factors can be an internal symmetry. You just can't see that in Rikowsky space time. And then so, and then there's some other more technical things about, I better not get into that. But the, there's kind of, well. It's okay. The audience is generally extremely educated in physics and math. Yeah, I would actually, well, maybe to connect this to what I'm saying, right, is I think, you know, also the way people think about general relativity in, you know, in Cassidy's signature, general relativity is not a chiral theory. It's supposed to be right invariant, parity symmetric theory. So the problem with thinking about general relativity in terms of twisters is that your setup is completely a chiral. So you can, you naturally end up with, if you try and do gravity with it, you end up with kind of something that's not quite the right theory of gravity. It's kind of a chiral version of gravity. Anyway, this is a very interesting story, but I think Penrose always referred to this as the Googly problem. Right, right. Something about cricket. Yeah, and cricket, there's something about how the balls. We're North American, so yeah. Yeah, so anyway, but so if you know about cricket, you can definitely, maybe this makes more sense to you, but he always referred to this as a Googly problem that he was kind of, in the twister theory, he's only getting one, he's only getting things spinning one way. And, but anyways, but you can see from my point of view, that's evident that was always, that's evidence of exactly what I'm trying to say now that, well, space time is right-handed. Yes. Yeah. So it's a related problem. But that was always kind of a, so Penrose and the people around them, I think, put a lot of effort into trying to revamp twister theory into something chirally symmetric. Now, why would they want to do that if the standard model isn't? Well, they weren't really trying to describe the standard model. They never really have to do. They thought Twisters were way of thinking about space time, so they wanted to do general relativity. And general relativity is not a chiral theory. So they were trying trying to find kind of a how do we get rid of all this chirality and uh and they never really successful at that so you're saying it's a pro not a con yeah yeah exactly it's a feature not a bug yeah right right but in terms one interesting fun thing that the sociology though is that what um know, so the idea that you could get, use twisters to quantify, to do general relativity and perhaps quantize it, that was always something which, you know, Penrose and his people were working on, but, you know, most physicists, I think, felt that wasn't really going anywhere. This wasn't going to work. And maybe Witten was probably one, was an example of somebody I think who really could see the mathematical power of these ideas and how important they were as new ideas about geometry. Again, that's a general statement that can be said. And then Ed Witten saw the power of this mathematics. Yeah. Yeah. Well, so he, I think even going back to a postdoc, he learned about Twisters, he was trying to do some things with it. But he never kind of, but he then actually finally found something. And this was about 20 years ago. And what became known as the Twister string. So he actually became, he found a way of kind of writing, you know, a different way of writing down the, um, perturative calculations in Yang Mills in terms of, um, of a sort of string theory, except it's a very different kind of string theory than the one that, the one that's supposed to be the theory of everything. And, and it's a theory where the string lives in twister space. So, Written wrote this really kind of beautiful, very, very beautiful paper about twister string theory. And so, and so since Witten is talking about twisters, of course, all of a sudden there's a lot of physicists who were never had, I think, good to say about twisters, or all of a sudden are rushing out to learn about twisters. So that, and there's, but there's been an ongoing story of, um, of this twister string story, which is a lot of people have done a lot of things. But again, a lot of it hasn't really worked out the way people like, and for the same reason as Penner, that Penner's always had, that the people are trying to quantify is a chirally version, a chirally symmetric version of general relativity using this thing. And that's not what it really wants to do. So anyway, but that's sociologically very important about why most high-energy physicists have more, have heard about twisters and don't, and often have nice things to say about them is because of the twister string. Okay, there are quite a few questions that I have. Okay, one is the particle physicists' repudiation of twister theory or just distancing from it because it's not useful to them. Is that something that they also slung at string theory or were they more embracing of it? Wait, so, I'm not quite sure. Who do you kind of mean? Who do we, are we talking about you? I'm not sure. Earlier, you said that the particle physicists weren't initially adopting string theory, sorry, twister theory because it didn't provide them with anything that's new. You said, well, okay, if we need to do some conformally invariant calculation, we'll use twister theory. Yeah. But at the same time, string theory is known, or at least colloquially known for not producing what's useful to high energy physicists, but useful outside of high energy like to mathematics or maybe condensed matter physics but what I'm asking is around the same time when they were distancing themselves from Twister theory or not using it were they then embracing of string theory or they gave the same critique well okay so we have to you should start if we're talking about string theory yeah that's a kind of complex, this is kind of a complex story. And it has the whole story of particle physics and string theory, that's pretty much completely disconnected from twisters because, I mean, the issues that, that, that, you know, people, about why people were doing string theory or why they might mind or might, I want to do string theory it really had nothing to do with twisters the twisters is kind of a yeah anyway especially a geometric framework and then you know and then twisters kind of make a small kind of appearance due to witten at one point 20 years ago but that's kind of about it um yeah so i mean i i i can maybe we we can start talking this about the whole string theory and particle physics business, but I'm not twister. Anyway, just twisters, it seems like a bad place to start. I'm not trying to mix up twisters with it. What I just meant to say was it's interesting what gets accepted and what doesn't. Yeah. And so why was string theory accepted? Take us through the history of that. And also you could tell people who may have just heard the name, sorry, Ed Witten. But all they know about him is that he's a genius. But they don't realize that influence that he has. Yeah, okay, so this is a good place to start. Yeah. And, you know, Witten is really kind of central to this story. And so, you know, I think the short summary of the history of this subject of particle physics was that, you know, by 1973, you had this thing called the standard model, which was this, you know, incredibly successful way of talking about particle physics and capturing everything that you see when you, you know, in these, when you do energy physics experiments. And the story And the story, when I kind of came in, it feels, I went to start learning about, probably started reading books and things about what's happening and particle physics probably right around the mid-70s. I went to college in 75, and I spent most of my college career, a lot of it learning about the standard model and this stuff and then and um so by but but by the time I left grad grad school set I mean by time I left college in 1979 and I went to graduate school at Princeton people were starting to get yeah people had had spent had now spent you know sit let's just six years let's say trying to figure out how to do better than the standard model and one one thing is how to do find some kind of new anyway how to do better the standard model as a theory of particle physics but also but one thing is the standard model doesn't give you a quantum theory of gravity so the other thing thing was, how do we get a quantum theory of gravity? So these were kind of the big problems that are already in the air. And Witten, you know, so Witten is a genius. And he had been a grad student in Princeton. He actually came to Harvard as a postdoc, I think, in 77, 78. And I met him when he was actually was a postdoc. And he quickly started doing some really amazing things. I went to Princeton 79. A year or two later, he actually, you know, he went directly from a postdoc at Harvard to becoming a full professor at Princeton, becoming a professor of Princeton very quickly. And so the years I was in Princeton as a graduate student were from 79 to 84, and those were years, you know, people I think were getting more and more frustrated. There are lots of ideas coming up, but every idea that people kind of tried to do better than the standard model, or maybe to quantize gravity, really didn't, you know, didn't quite work. I think there's a lot of, and people were kind of cycling every six months through. There's some new idea you'd work on it for six months or a year, and people start to realize, well, this doesn't really do what we want to do. Let's find something else. So there were a lot of new ideas, but nothing really working out. But Witten then, you know, he had been interested. There was this idea that was very unpopular. There were very few people were working on to try to quantize gravity and unify it with the particle physics through string theory. And so it was, you know, people like John Schwartz and Michael Green were working on this, but it was a very small group of people, and there wasn't much attention being paid to that. But, you know, Witten was paying attention to. I think one thing to say about him is that besides being very, very smart, he's also somebody who can, you know, read people's ideas or talk to them and absorb new ideas very, very quickly. So, you know, he was kind of also spending a lot of time looking around, trying to see, you know, what other ideas are either out there. And this was one that he got interested in. But for various reasons, technical reasons, he thought, you know, this, there's a technical reason, so-called anomaly calculations about why this is not going to work out. And what happened right in the fall of 84, I actually went as a postdoc to Stony Brook. And the right around that time, Green and Schwartz had done this calculation that showed that these anomalies canceled, except there's some specific case where these anomalies canceled. And so Witten then became very excited about the idea that, you know, you could use in that specific case of this so-called super string theory to, yeah. so so so so so so witten heard about this and he said it said okay you know the thing that had in my mind why super string theory couldn't work as a unified theory and now it looks maybe like maybe you can get around that so he kind of then started working full full time on trying to you know come up with models or understand super string models that you could use the de unification. And so throughout kind of, I was now at Stony Brook, but I was kind of hearing reports of what's going out at Princeton. And throughout late 84, 85, 86, this was, you know, Witten and the people around him, this is what they were working on, Bobora. And they were, you know, they had a very specific picture in mind. It was that, you know, the super string only is consistent in 10 dimensions. You, so you can get rid of four of them by the so-called Calabial compactification. And hopefully there's only a few of these collabiaws. And one of those is going to describe the real world and you know we're all going to have this wonderful beautiful unified theory using this kind of six-dimensional geometry of claudeaos and we're going to have it within the next year or two and that was the way they were thinking and you know a lot of the people you know friends and colleagues of mine who you know we're doing kind of the thing that you would often do is go down and go you know when you're in princeton go talk to witten and say here's here's what i'm working on you know can you what do you think about this and i got several of them reported back to me yeah you know i went down to prince i talked to whitton and he said well you know what you're working on that's all right i said well it's good but know, you really should be working on string theory because that's actually, you know, we're all the actions and that's really, and you know, we're almost going to have the theory of everything there and you kind of work on string theory. So, you know, this just had a huge effect. So, and this was called the so-called first super string revolution. And, you know, there's a story over the next five or ten years of how, you know, people were brought into this field and people, some people are always skeptical. But, you know, it kind of gained more and more influence and became institutionalized during kind of the decade after that. And in some sense, the weird thing, the weird thing, the weird thing that's hard to understand string theory is why, you know, once it became clear, these ideas really weren't working out, why didn't, you know, this just fall by the wayside and people go and do something else? But 40 years later, we're still, it's still here. And so it's a very strange story. So what do you see as the main physics, physical problem or even mathematical problem of string theory? Do you see it as, well, how do we search this landscape or how do we find the right manifold, the six-dimensional caler manifold? Yeah, I think that was always the thing that bothered me about it from the beginning, which I think is the fundamental problem. It's, and it's a fundamental problem whenever you decide to do to use higher dimensional Romanian geometry, if you, I mean, this actually goes back to Einstein, Einstein and these Kluza Klein models. You know, people have often said, okay, well, you know, we had this beautiful theory of fourdimensional geometry in Einstein general relativity, and we have this particle physics stuff going on, which seems to have some interesting geometry to it. So let's just, let's just add some dimensions and write down a theory in five or seven or ten or whatever dimensions, and then do geometry there, and that's going to solve, and that's going to be the unified theory. So I mean, this is sort of thing Einstein was thinking about. But if you start thinking about this, the problem is you realize that these kind of internal dimensions that the geometry of particle physics and the geometry of special relativity are quite different. They're not, you know, they're these metric degrees of freedom in four dimensions. And if you try and you don't really have those in, in like in the standard model, you just doesn't have things like that. So if you put those sort of dynamical variables into there, the ability for these for these other dimensions by the four one to two, you you have a vast you you hugely increase the number of degrees of freedom and you have a theory with where you have to now explain why all this extra geometry which you've put in there and and which you're only trying to get a kind of small kind of very rigid kind of couple pieces of information out why is are all these infinite number of degrees of freedom why how can you just ignore them how can you you have to find a dynamics consistent dynamics for them and then you and that consistent dynamics has to explain why you don't see them yeah and so that's always been the problem with like Kaluza Klein models and with any kind of extra dimensional models. And string theory just kind of has this problem in spades. You know, you're instead of feel, instead of point particles, you have strings. They have a huge number of new degrees of freedom. You have to say that, well, the string vibrations are all happening at such high energies we can't see them. And then the extra 60, then they're trying to use the fact that super strings have very special properties in 10 dimensions. And they're trying to use that to argue that our strings are moving in 10 dimensions and that four are the ones we see and six are going to be described particle physics and so anyways it becomes a very complicated theory you have to write down in order to kind of make any of this work and make any of this look like physics and the from the beginning there was kind of no story about why is anything that looks like the real world going to drop out of this, you know, and why that? And that's still the case 40 years later. And the whole thing just suffers from this problem that you don't, you don't actually have't actually have the theory there's kind of a when you say that you have a string theory and people say oh we have this mathematically elegant well-defined unique theory they're talking about that's not a full theory that that's that's a perturbative limit of a theory and so what they really need in order to answer the questions they want to answer is they need something more general, a so-called non-perturbitive kind of general version of string theory. And sometimes people call it M theory. So if you want, we can call it M theory. And they need an M-theory. And nobody knows what M-theory is. No one has come up. You can write down a list of properties that, you know, M, M theory is supposed to be some theory with this list of properties, but you can't actually write down a theory. And so on the one hand, you don't actually have a real theory that you can nail down and say, this is a theory, we're going to solve it and look at the solutions and see if they look like the real world. So what you, what people end up doing is saying, well, we don't really know what the theory is. Let's assume that, but it seems that maybe there's one that has some properties that look like the real world. So let's work with that. And then try to constrain, see what constraints we can get out of it will tell us, you know, are we seeing something like the real world? And then they just end up finding that, no, there aren't really useful constraints that you can get almost anything out of it. So you get this landscape of all possibilities. Yes. And then, you know, 20 years ago, things got very weird when people just started to say, well, you know, instead of saying that normally if you have a theory, it can't predict anything because, you know, almost everything is a solution to it. You say, okay, well, that was a bad idea and you move on. Instead, you saw people saying, oh, well, it just means the real world is, you know, almost everything is a solution to it. You say, okay, well, that was a bad idea and you move on. Instead, you saw people saying, oh, well, that's, it just means the real world is, you know, all of these possible things exist in the real world and the multiverse and, yeah, and just for, you know, for anthropic reasons, we happen to live in this random one. And, you know, I mean, anyway, it's, the fact that anyone ever took any of that seriously is just still kind of, I don't have any explanation for it. It's just far. Yeah. Okay, so to summarize, somewhere around, this is not a part of the story that was said, but somewhere around the 1960s, some amplitude called the Veneziano, I think, Veneziano. I don't know how to pronounce it. Yeah, the name. Venetia. That was the first inklings of string theory and it had to do, was come up with because of the strong force. They were trying to solve something. Then it was forgotten about. And then around the 1980s, there were some other problems with string theory that were solved. And so this is the Green Schwartz anomaly cancellation. Yeah. And then some people say that that was the first revolution. But it's also more accurate to say that that precipitated Ed Witten to take it seriously. And then that's what precipitated the first string revolution. Okay, then from there, then you realize that there are different ways, something like 5 to the 100 or 10 to the 500 or some extreme amount that if you were to do some naping calculation, all those books behind you, the amount of words ever written, not just books ever published, words ever written, I think easily letters ever written, like single letters, it would be like saying, find this one letter in every single book that's ever been written, including all the ones that have been on fire and underwater and so on. Okay, that's not such a problem if you can figure out how to reduce the search space. But if you can't, then it turns out the problem is NP-complete, which means you just have to brute force. Is that a correct summary? Well, actually, maybe you go back to one thing and say, yeah, so this is one part of the story I didn't say, is that string theory had originally come out as a potential theory of the strong interactions. And that actually was one reason Witten, I think, was looking at it, is that so one of the open problems that the standard model left open was how do you solve the strong we have this strong interaction theory but how do you solve it and it looked like maybe you could you could use the old ideas about strings to solve it and I actually spent a lot of time learning about strings as a graduate student because of that and I was really to Witten but but the with um this kind of multiplicity of solutions of string theory of is that it's not just that there are too many of them it's just that you don't actually have a definition of the problem you know so so people this kind of drives me crazy people often talk about well the problem is that we don't know how to put a measure on the space of solutions of string theory. And if we could put a measure, then we can figure out, you know, maybe it's concentrated someplace. Right. And that would be great. But I keep pointing out that the problem is not that you don't have a measure of the space. The problem is that you have no idea what the space is. As I was saying, you know, to even define what a string theory solution is requires knowing precisely what M theory is. You don't know it. There are no equations anyone can write down which you say, you know, if we were smart enough and we look and could find all the solutions to this, this would, you know, these are all the solutions of string theory. You just don't have that. So all of the things that you do have, like you can go out and say, well, well, maybe it's these gadgets and you have 10 and the 500 of them or whatever. Those are all just kind of cooked together possible approximations to what you think might be a string theory solution. Those are not, there are solutions to some equations you've written down, which are not, they are not the equations of string theory. There's something you wrote down and think maybe these things have something to do with string theory. So the problem is much worse than any of these practical problems of there's too many of these things. And this whole business, and now it's become kind of an industry that, well, let's apply machine learning techniques to this. And it's just, I mean, you're just applying. Anyway, you're just. Does this frustrate you? Yes. I mean, this data is garbage. You know, so you basically are throwing, you basically do not actually know what your problem is. So you're cooking up something which you can feed to a computer, but it actually is kind of known to be garbage. And you're doing processing on this and producing more garbage and, you know, getting grants to do this and going around telling people that you're looking for the universe the universe I mean it's real that's just utter nonsense I'm sorry many people don't know because they don't know the history but since 2010s it's become somewhat cool to dunk on string theory at least in the popular press okay maybe not inside academia but you were alone you and Lee Smollin were lone wolves. Early lone wolves. Can you talk about that and talk about some of the flak you took, maybe still take? Yeah. Anyway, it was certainly a very strange experience, a very strange time. But, you know, I think the thing to say is that throughout, you know, I was never, I was always fairly skeptical about string theory, but, you know, initially for many years my attitude was, well, you know, throughout, you know, I was never, I was always fairly skeptical about string theory, but, you know, initially for many years, my attitude was, well, you know, who knows, you know, is certainly very smart. These people are, you know, they're going to, sooner or they'll figure out for them, either they'll figure out this works or they'll, or they'll do something else. But then, you know, just as time went by, years went by and that this was just not happening. And you had more and more and more kind of popular books you know i have to confess maybe in some sense it's somewhat of a reaction to uh to brian green who is a my friend and colleague here in at columbia but uh you know so he did a very very good job of with pbs specials convincing the world that you know this this was a successful this was an idea on the way to success when it really wasn't. So I thought, okay, well, somebody should, you know, sit down and write a book about, you know, what the real situation here is. And, you know, it's not like when I talk to people privately about this, you know, I would say that people who are not string theorists mostly would would say, yeah, you know, yeah, you're probably right. This is not, this doesn't seem to be going anywhere, but you know, whatever. And then the, and people, and when I talk to string theorists, I have plenty of strength theorists, they would often say, yeah, you know, yeah, there are a lot of huge problems and we just, we don't really know anything better to do right now is where we're going to keep doing this. But yeah, yeah, all these problems you're pointing out are really, yeah, they're real. And, um, so what's wrong with that? Well, it was, the weird thing was, I think was this disjunction, this disjunction between the private opinions of people, what people were saying to each other privately, what you, private had said, and what you were seeing in, in the popular press. And, you know, you've, so, and there was, you know, and one aspect of this was people not wanting to kind of publicly criticize something. And partly, and I think the subject became more and more kind of ideological. And the string theorists kind of started to feel kind of in battle. They were very well aware that a lot of their colleagues thought what they were doing was not working. On the other hand, you know, so they became more defensive. And there was a lot more it became. And a lot of people, I think, would tell me, yeah, you know, I agree with a lot of you're saying, but yeah, but don't quote me on this publicly. I don't want to get involved in, you know, in that mess and alienating a lot of my colleagues and who are, anyway, so, but I have this weird status that I'm actually in a math department, not a physics department, and, you know, I don't have a lot of the same reasons that you don't want to annoy some powerful people in physics, like, you know, trying to get grants, get your students jobs, et cetera, et cetera. It didn't really apply to me. So I thought, well, you know, if somebody is going to kind of start a lot of time thinking about this stuff. And I, you know, I spent a lot of time thinking about this stuff. And I started writing this in around 2002, 2003. And the book was finally published. It was a long story about getting it published, but it finally got published in 2006. And in the meantime, Lee Smolin had been writing a book. He was coming from a different direction. Trouble with physics? Yeah, the trouble with physics. And he had his own motivation, so it was trying to write something, I think, more general and sociological, but with this as an example, and I think the way he describes that the example kind of took over the general theory. And so he ended up also writing a book about string theory. And the books ended up coming out at the same time, which I think, you know, it was kind of a force multiplier there that, you know, people, if one person is writing a book which says, well, you know, a lot of the things you're hearing, you're hearing are not right. Or people say, well, that's just one person's opinion. But if two people are doing our same thing, everybody's like, oh, you know, there must be something to this. And so I think the combination of the two books, I think it did have a lot of effect on, it did make a lot of people realize there was a problem here. It made a lot of the strength theories, you know, much more defensive. I mean, it also caused, I think, a lot of people, young people thinking of doing string theory or people doing string theory to decide to move on to something else. But so people very often tell me that, you know, about effects this book had on them or other people they knew in terms of their decisions about what to do with their research or their career. The book is called Not Even Wrong. The links to all resources mentioned will be in the description, including this book. So you mentioned that your colleagues would talk to you privately, and then they would say something else to the popular press. Now, when you say popular press, are you also including grant agencies with that, like just the public in general? Because it's not just a popular science issue, it's also a grant issue where the money goes. Yeah, so it's not just the popular press. And to be clear, I should say, it's not that they would say one thing, one place, it's just, they would carefully just not say, you know, that there were things that they would say in conversation with me or I think in conversations with other people, not just me, that they would just say, okay, this is not something that. Okay, sin of commission versus omission. Yeah, it's not like they were going out and saying, oh, the strength theory is going great. It's just that, you know, anyway, they were, they were not kind of, they were not saying this is really appears to be a failure. But yeah, but you're right. This issue kind of occurs at all levels from, you know, the very, very popular press, from kind of television specials um to you know more more serious popular press or what what gets into scientific american you know what what gets into uh now we have quantum magazine you know which are more more serious uh parts of the parts of the press aimed at the more at the public to um you know all the way down to it yet to exactly um yeah like in grand proposal you know what what do you write in in grant proposals whatever or if you um if you're anyway you're trying to explain to some some kind of funding person or something about what you know what's what's going on in your subject do you um yeah and you know what do you say about string theory and so the you know the string theories i think have often you know that they've i think everybody whatever you're working on you're often forced by this business of getting your students a job or getting a grant to be you know know, to say, to go right up to the boundary of what's defensible and being optimistic about what you're doing. But, and there, you know, so that's what string theorists have certainly always been doing. You could argue, you know, in many cases, it's not different than what other scientists do. But it's, I think the thing which i i have to say i have found more and more disturbing the reaction of and this started when my book came out and i think lees small had a similar reaction the um i think both of us were expecting a much more serious intellectual response to the issues we were raising. We were raising serious, serious technical questions, and we were getting kind of back, you know, kind of, you know, personal attacks. From people in the community or from the public? From people in the community. I mean, I think, you know, what you're getting from people people who don't in the public don't know much about this you're you're getting some completely random combination of people who are annoyed because you're saying something different than what they heard and other people who become your fan because you're saying something different and so you end up sure sure you end up with a huge number of fans who you don't necessarily want as your fans. But anyway, yeah, so both of us were expecting, you know, that, you know, we put a lot of effort into making a, you know, a serious intellectual case about what these problems were. And instead of getting a serious response, we were getting, you know, these kind of personal attacks of how dare you say this. And so, for instance, there's one prominent blogger who decides who would write these endless blog entries about what's wrong with Peter White and what he's doing. And at some point, I was trying to respond to these. And at some point, I realized, you know, what this guy's talking about is nothing to do with what I actually wrote in my book. And then he actually kind of publicly admitted that he was refusing to, he refuses to read the book. So this is a, anyway, this kind of blew my mind. How can you be an academic and engaged in, you know, academic discussion, intellectual issues? And you're spending all this time arguing about a book and you're refusing to read it. mean how it's just really crazy and that was a string theorist yeah or just a colleague yeah okay string theorist yeah speaking of brian green oh sorry continue yeah yeah no yeah no yeah i didn't mean to suggest that no no no but but anyway that's just one example so and i and i think, you know, this is an ongoing, I think, disturbing situation that people are just not, people are kind of defending that field and continued and research there with just kind of refusing to acknowledge the problems or to have kind of serious discussions of it. I think, you know, on your last year, your last thing with Edward Frankel, I think, it's kind of funny because he, you know, I know him and I actually was out visiting him in Berkeley in June or something and were talking about things. And he told me, oh, Peter, I'm, you know, I'm going to go to the strings conference. And it's the first time I've been to a strings conference and now. And, you know, he's heard me go on about this for, and he's kind of nodded his head politely. And, you know, he's saying, well, I'm a mathematician. I'd rather than not, you know, but this sounds a little bit of it. Maybe he's published with Witten. Yeah. And then, you know, so he, and he knows all these people. And he knows a lot about the story, but he, and I think, you know, he knows me well know enough that I'm, you know, I have a somewhat, I'm not a complete fool and I have a somewhat serious point of view, but, you know, maybe I'm really a bit too extreme about this. But then he went to the, this conference. Then when it comes back, he gives me a call, it says, basically, you know, Peter, I didn't realize how bad it really was. You're right. This really is as bad as you've been saying. So it was, anyway. What was bad? The exuberance of the young people or the old people telling, misleading the younger people into a useless pit? Or like, what was, what was bad? Yes, it is as bad as you say. Well, I think what's bad is really just this kind of, this kind of refusal to admit, I mean, this is a field which inflexia has serious problems. Things have not worked out. These ideas really have failed to work. And instead of admitting that, ideas have failed and moving on, people will just kind of keep acting as if that's not true. And so the, you know, I think... Sorry to interrupt. I'm so sorry so why would edward expect an admittance of the failure of string theory at a strings conference i think one thing to say you know i mean part of the story about him is you know he's a mathematician and and you know so mathematicians if you do mathematics the one thing you have to be completely clear about is you know what you understand and what you don't understand and what is a wrong idea and what is a right idea. You know, and if something doesn't work and is wrong, you have to, you can't play a game. You cannot play any games about this. This is, you know, you have to admit that this is wrong. And so I think especially for mathematicians to come in and see an environment where there's You know the kind of guiding ideas that people haven't really haven't really worked out and a lot of things you know are known do not work for known reasons but people are still kind of acting as if this is not true and trying to figure out how to kind of do something and make career for themselves in this environment it's a very no i think he he he recognize that but it is part of it is the um i mean mathematics is a very unusual subject that people things things really are wrong or right and you and you're you know it's you absolutely absolutely cannot seriously make progress in the subject unless you recognize that and uh and mathematicians are also much more used to um they're much more used to being wrong i think one of my colleagues john morgan likes to say that uh you know mathematics is the is the only subject he knows of where you know if two people disagree about something and they each think the other is wrong, they'll go into a room and sit down and talk about it, and then they'll emerge from the room with one of them having admitted he was wrong, the other one was right, and that this is just not, it's not a normal human behavior, but it's something that is part of the mathematical culture. Earlier I said, speaking of Brian Green, and what I meant was I had a conversation with Brian Green about almost a year ago now, and I mentioned, yeah, so Peter White has a potential toe, Euclidean Twister unification, and then he said, oh, does he? Oh, I didn't know. He is in your university, not to put you on the spot, but why is that? Well, it said aloud, I don't think it's true by the professor of physics, mainly who studies string theory. Well, there are so many proposals for toes. Yeah, there are proposals in your inbox, but there aren't serious proposals by other professors. There aren't that many serious proposals of theories of everything, at least not on a monthly basis. Well, I mean, I mean, this is this really doesn't anything in particular to do with Brian. You could ask, you know, since, you know, people in this subject, you know, in principle should be interested in this. There's, I've gotten very little reaction from, from physicists to this. And, and in some sense, it's kind of clear, clear why. I mean, it's kind of clear why. I mean, I wrote this paper. I've read it by the blog. And, you know, I've gotten no reaction. In both cases, I don't have reaction from people telling me that I've talked to about or saying, oh, you is this this is wrong this can't work for this reason but well i think that this is this is very very much the problem with the the paper that i wrote about this it's very it uses some quite tricky understanding of how twisters work and twister geometry works, which is not, is something a very few physicists have. So Brian, it would, I'd be, I'd be completely shocked if Brian actually really understood some of the things going on with twisters that I've been there talking about. And the problem, I think for anybody who then, if somebody comes to you and says, oh, I have this great idea, it involves, you know, these subtleties of twister theory. And you're like, well, you know, I'm really not in the mood to spend a week or so sitting down trying to understand that subtle is a twister theory. So I think, you know, maybe I'll just nod my head politely and go on my way. That's part of it. And then part of it is also that a lot of, you know, this is very much a speculative work in progress. I'm seeing a lot of very interesting things happening here, but I'm not, in no sense, have completely understood what's going on or have the kind of, you know, understanding of this where you can write this down and people really understand, can follow exactly what's going on. So it's not too surprising i haven't got that much i can see why understand the typical reaction to this and um brian is someone of a special case because i mean he also actually is very um i think actually he actually a lot of his effort is as has in recent years has gone into other things especially the the World Science Foundation Festival, I think, is now more or less, you know, it's kind of most, it's mostly Brian Green at this point. Yeah. And then it's, so he's, anyway, he's thinking about other things. And I have very, I don't have very little contact with people in the physics department. I mean, they're mostly thinking about very different things. And it's kind of a sad fact here at Columbia, but it's true essentially everywhere else that the, you know, the mathematicians and physicists really don't talk to each other. They're really separate silos, separate languages, separate cultures, and places where you have kind of mathematicians and physicists and kind of active and high-level interaction with each other is very unusual. It doesn't happen very much. I have a couple questions again. I'll say two of them, just so I don't forget them, and then we can take them in whichever order you like. So one of the questions is how slash why did you get placed into the math department? So that's one question. And then another one is, you mentioned earlier that Witten has this power to survey a vast number of people and extract the ideas at great speed. And so a large part of that is raw IQ, like sheer intellect. But is there something else that he employs like a technique that you think others can emulate? I imagine if Witten was to read your paper, he would understand it. And I imagine that he would see, oh, he would see the benefit of it and maybe the application to string theory or maybe it offshoots in its own direction. But anyhow, so those are two separate questions. One about Witten, and then one about you and the department you're in. Okay, yeah, I've got, yeah, there are two very different. Let me start, let me just say something quickly about Witton, just saying about having dealt with him over the years. One thing that I find very interesting about him is just, you know, he travels around a lot. And, you know, he, let's just say, let's just say his way of socializing is to, you know, if he's come to a department and he's at T or whatever, he'll, you know, and he's introduced to anybody, he almost immediately will ask, okay, well, what are you working on? You know, explain it to me. And so just a lot of what, anyway, that's a lot of what he's done over the years has just been, has just been, you know, trying to really be aware. And, you know, I've said what I've done doing and tried to get him interested. He's, I know, he's, anyway, we'll see where that goes. Maybe I'll have more success with it with this new paper, maybe not. But he's, he's responded, though, or no? He has responded, but it's more that he's kind of looked at it. He actually, the first version, he actually made some kind of comments more about the beginning of it. But I think he didn't engage with most of what I started talking about. We're going to get back to the math question soon, the math department question. But do you think a part of that is because there's a sour taste given your book? Yeah, yeah. I mean, I'm not, I mean, again, I've known him since I was an undergraduate. You know, I think, you know, he's, I think he's aware, you know, that this guy is not an idiot, but he's also, I'm also not his favorite person in terms of kind of, you know, the impact I've had on his, on his subject. And yeah, and I think, you know, he also, I think he understands it's not personal, but, you know, it's not, it's very hard to deal with somebody who's kind of, you know, been this kind of main figure, kind of telling the world that the thing that you think is your main accomplishment in life is wrong. So this is not, yeah, anyway, I'm not his favorite guy, but, but anyway, I can know, we're still. Sure. It's fine. He's, yeah, you know, I think he's a very, you know, anyway, he's a very ethical and very, and I think when I complain a lot of, a lot of, most of the worst of what the kind of, this kind of pushing of string theory in ways which, which really were completely indefensible. It's, he's mostly been not, you know, he's rarely been the worst offender in that. I mean, that's really more other people than him. But, yeah, he's a true believer. He's really enthusiastic about him. He still is. Okay. So to get back to my own personal story, so what happened, you know, so I got a postdoc at the Stony Brook Institute for Theoretical Physics in 84. I was there for four years, and that was the physics institute, but the physics institute was right above, it's the same building as the math building. And so, and the things I was interested in, I was trying to stay away from string theory and I was interested in some other things. And, you know, I was often talking, and I was trying to learn a lot of mathematics. I was trying to learn more mathematics to see if I could make any progress on these other problems. So I spent a lot of time talking to the mathematicians in Stony Brook. And some of them, you know, there are some really great geometers. There are some really great mathematicians. And I learned a lot from them. And it was a, that was a great experience. But at the end of four years there, you know, I needed another job. I did set out some applications for postdocs in physics, but the, I would say that that was kind of the height of the excitement over string theory. And especially somebody like me saying, you know, I'm really interested in doing something about the mathematics and physics, about applying mathematics physics, but I don't want to do string theory. That was just, that was not going I was not going to get any, any kind of reasonable kind of job that way. That's just not going to happen. So, anyway, so I ended up realizing, well, maybe the better thing, I'll have better luck in a math, in a math department, and I'm getting, and so I ended up going up, spending a year in Cambridge as kind of an unpaid visitor at Harvard partly, and I was also teaching calculus at Tufts. And so then I had some kind of credential, okay, well, at least this guy can teach calculus. And so I applied for a one-year postdoc at the Math Institute in Berkeley, MSRI, and I got that. And so I spent a year. Is that how you got to know Edward? No, no, he wasn't, that was before him. I mean, he would have still been at Harvard and a much more junior person. Yeah, yeah. Yeah, he came to Berkeley later. Yeah, no, that was like 80, 88, 89. But that was an amazing, that was actually a fascinating year because that was the year that Witten had come out, Witten had kind of dropped string theory for a while and was doing this topological quantum field theory stuff in Turing Simon's theory. And he was doing the stuff which won in the Fields Medal. And, you know, it was just, just mind-blowing, bringing together of ideas about mathematics and quantum field theory. And so most of the year was devoted to learning about that and thinking about that. And, you know, Witten came and visited and Atiyah was there. And actually a lot of chance to talk to him, which was wonderful. And so that was a really fascinating year at MSR-I. And partly because so much of this was going on, you know, math departments were more interested in hiring somebody like me, even though I didn't have the usual credentials because they felt this is somebody who actually understands this new subject, which is having a lot of impact on our field. So Columbia hired me to this non-tenure track for your position. And so I was to do that as I was teaching here. And after a few years, again, I was getting the point, okay, well, now I've got to find another job. And they, so the department needed somebody to, they'd set up a position for somebody to teach a course and maintain the computer system. And I said, well, you know, I can probably do that. And that's not a bad job. And so I ended up agreeing to take on that position. And that's always been kind of a renewable position. It's not tenured, but it's essentially permanent renewable. And I've gone through various kinds of versions of that since I've been since the 90s and it's worked out very well for me I'm actually quite quite happy with how it's work but it's a very unusual career path and it it has given me a lot of insulation from the normal kind of pressures to perform in certain ways and to do certain things allowed me to get away with all sorts of things, if you like. Like what? Well, like writing a book called Not Even Wrong, explaining what's wrong? How did that come about? So, for instance, this is going to be incorrect because I'm just making this up, but then correct it. For instance, you're walking along someday. You have this idea. Maybe it's a splinter in your thumb for a different reason about string theory. So then you go to a publisher and you say it or you say it to a journalist and then the journalist hears it and they say you should write a book and you say maybe, then you think about it, you start writing a chapter. The nitty-gritty details, how does that happen? How did it go from Peter White, mathematics professor, to then writing this popular book? Well, so yeah, let's say throughout the 90s, you know, I was very much, you know, I'd always, you know, I was interested in the same kind of question as can you do different things in math and physics? I was trying to follow what's going on in physics. I've been trying to follow what's going on in string theory. And I was getting more and more frustrated throughout the late 90s at this, what I would see in the public and what I would see, or just to not reflect my own understanding of what actually was going on. And partly I kind of mentioned, you know, so there's a, for instance, Brian's PBS special about the earlier's. I mean, it just, that just just seemed to me to be giving that just didn't really didn't agree at all with what i would actually saw going on and so i thought well somebody you know somebody should write this up and i would have hoped it would be somebody else but then as you go along with no one else is going to do this and you know i'm actually pretty well placed to do it for for very reasons and started thinking about it and i think around 2001 i actually wrote kind of a short thing that's on the archive of kind of you know a little bit of a kind of polemical several page thing you say look here here's the opposite side of this right here's what's this is really not working and here's why and that that was the beginning of it and like i got a lot of reaction reaction to that. And I started to more and more feel that the right way to do this was to actually, you needed to write something kind of at book, sit down and at book length explain exactly what's going on. And I also wanted to do something also more positive to try to explain some of the things that I was seeing about how mathematics, you know, there were some very positive things happening in the relationship between mathematics and physics, which has some connections to string theory, but were also quite independent, like Wittance-Turne-Syman's theory, for instance. So I also wanted to also write about the story of what's going on in this kind of physics and this kind of fundamental physics, but kind of informed by, you know, someone who's actually spent a lot of time in the math community and informed by a lot more mathematics than as usual in this thing. So there was kind of a positive. It's rarely noticed, but there are a bunch of chapters in this book like on top logical quantum field theory, nothing to do with string theory, which nobody really paid much attention to or understands. But anyway, so I wrote this, and I was, so I just said, well, I'll just write this thing. And I think I, around then, I may have also had a friend who, he'd done a book proposal and written a book. But by the time he actually was writing the thing, you know, he was just kind of sick of it and he didn't really want to be writing it, but somebody had given him in advance and he had to write the book. So I thought, well, I don't want to do that. I'm not going to go out and make a proposal to a publisher. I'm just going to write when I want to write. And we'll see how it turns out. And I think know, I think we'll see if someone wants to publish it great. And so then I was getting to the end of this and somebody from Cambridge University Press showed up. He was just in my office going around asking people, you know, what are you working on? Is there some kind of book project we could work on? And I told him about what I was doing. And he got very, very interested in it. And so it actually then became, you know, Cambridge University Press was then considering it for a while and they sent it out to various reviews and the reviews were kind of fascinating. There were half the reviews said, this is great, this is wonderful. Somebody is finally saying this is fantastic. And the other half said, oh, this is absolutely awful. This will destroy the reputation of Cambridge University of Press. Interesting. And the problem with the University of Press is, you know, they're not, they're actually not really, they're not really equipped to deal with that kind of controversy. I mean, they, they've got, they have like boards of so-and-so that have to vote on everything and they're very pretty conservative institutions. So at some point it became pretty clear that things were not going well there. And so I sent it around to a bunch of people. And anyway, and one person I sent it around to was Roger Penrose. And he ended up getting interested in it and asked me if he could send it to his publisher, and they ended up publishing it. Oh, great. Yeah, he's not a fan of string theory either. No, no. Yeah, so he definitely agreed with me about that. Yeah. Now that you're in the math department, is that what allowed you to see the connections between Twister Theory and the Langlands program, or is that something that existed before? Oh, well, I mean, the connection, not the Langlands program. Obviously, that goes back to Langlands. Well, oh, no, whether there is, I think it's still, you know, whether there is any connection between Twister Theory and the Langeland program, that's a very, that's extremely speculative idea and fairly recent one, I would say, yeah. Yeah, so that. What aspect of the Langlands program? Like the local or geometric? Maybe to back up a little bit. I mean, so the Langlands program is, anyway, this amazing story, I guess you heard a lot about it from Edward, but it, it's one reason I got into it is it became more and more clear to me that the right way to think about quantum mechanics and quantum field theory was in this language of representation theory, that that was the language of, and then it started to, okay, well, I should learn as much as possible about what mathematicians know about representation theory. And, and you, you, you, you, you, you find out about the language program, and the language program is saying that all of the basic structure of how the integers work and how numbers work and things is, you know, closely related to this representation theory of lead groups and in this amazing, amazing way. And there's just an amazing set of ideas that ideas behind the Geometric Langlands program, which, you know, they have a lot of similar flavor to the things I was seeing in some of physics. So it was, you know, I said, it's just been a many, many years process of slowly learning more and more about that. But that stuff never really had anything to do with twisters. And so the one, the interesting, the interesting relation to twisters is that, you know, I had actually, I'd actually written this paper and I'd given some talks about, um, about the twister stuff. And I pointed out that I'd pointed out that in this way of thinking about things, there's a thing that I told you that a point, a spacetime point, is supposed to be a complex plane. Well, if you take this, actually in Euclidean space, it's something you can think about it a complex plane or you can mod out by the constants and use the real structure of Euclidean space. And you get something, a geometrical object corresponding to each point, which is called a twister P1. It's basically a sphere, but you identify opposite end points of the sphere. And so I'd written about that in my paper and some of the talks I was given, I kind of emphasize that. And then so then I get an email one day from Peter Schulza, who's one of the people who's making this really great progress in the language program in number number theory and it's been coming up with some of these fantastic new ideas relating geometric langlands and arithmetic langlands and he said and he basically said yeah I was looking at this talk you gave and you know it's really nice about this geometry and seeing this Twister P1 going there said what's amazing is this Twister P1 is exactly that same thing as showing up in my own work. You know, if you, there's this work he was doing on the, on the, on the, on the gym, the geometric langlands. And if you specialize to what happens kind of as a, at the infinite prime or at the, the real place, not, not at finite primes, the structure he was seeing was exactly the twister P1. So, I mean, he kind of pointed this out to me and asked me some other questions about this. I don't think I could tell them anything useful, but that kind of, that did kind of blow my mind that, wait a minute, this thing that I'm looking at in physics, that exactly the same structure is showing up in this, in this really these new ideas about geometry of numbers. And so I then spent a few months kind of learning everything I could about that mathematics and the Twester P1, and I'm still following it. But, you know, I should say that, you know, to my mind, it's just a completely fascinating thing that these new things that we're learning about the geometry of number theory and these speculative ideas about physics that you're seeing a same fundamental structure on both sides. But I have no, I mean, I have no understanding of how these are related. I don't think anyone else does either. Yeah. Have you asked Peter if he would like to collaborate? Well, there's not. Is that like uncouth? No, but I think he and I just have very, you know, I mean, too incompatible? No, no, no. It's just, you know, he's doing, you know, he's doing what he's doing. I mean, I mean, first of all, I mean, one thing to say is, you know, he's having such incredible success and doing such amazing stuff that, you know, interfering in it with that anyway and telling him about, oh, why don't you stop doing what you're doing and do something? And I'm interested in. It seems to be a really bad idea. Anyway, so he's doing extremely well doing what he's doing, and most of what he's doing isn't related to this. He really, really understands in an amazing way what's going on with the geometry of peatic numbers and these things like this, which I don't understand at all. And he's just been revolutionizing that subject. And it's something I can only kind of marvel at from the distance. The kinds of issues that were on kind of stuck that are kind of for me are actually much more, they really have nothing to do with his expertise. They're really kind of more, more, you know, I probably should be talking to more physicists or whatever. So he's, yeah. But I mean, it's certainly, I think it's in the back of his mind, oh, you know, this stuff that I'm seeing, I should every soften look and think about what if I can understand the relation to physics. And it's in the back of my mind, the stuff that I'm seeing physics, I should try to keep learning about that number three stuff and see if I see anything. But that's really all it is. But a lot of this is very new. I just heard from him a few weeks ago that, you know, he actually, he actually has some new idea about this particular problem from his point of view. And he was supposed to give a talk about it on last Thursday at this conference in Germany. And I'm hoping to get a report back of that. But this is all very active and very poorly understood stuff, but it's not, but definitely the connection between math and physics here is very, very unclear. But I'm, if there is one, it will be mind-blowing, and I'm, I'm, it's certainly kind of on my agenda in the future to try to learn more and look for such a thing. But I don't have anything positive to say about that, really. So I want to get to space time is not doomed. There's quite a few subjects I still have to get to. I want to be mindful of your time. But how about we talk about space time not being doomed? It's something that's said now. I don't know if you know, but there's someone named Donald Hoffman who frequently cites this. He's not a physicist, but he cites it as evidence or as support for his consciousness as fundamental view. And then there's Nima Arkani Ahmed, who's the popularizer of that term, though not the inventor. Yeah. So maybe to, I mean, I can kind of summarize that. Yeah, so I don't really have anything useful to say about, but Hoffman. I mean, so he's interested in consciousness and other things I don't really have too much, I don't really know much about or I'm useful to say, but maybe to say what the, I mean, this has become, and I mean, the reason I wrote that there's this article you're referring to about space time is not due. I wrote partly because I was getting frustrated at how this had become such a, such kind of an ideology among people, among people and working in physics on quantum gravity, this idea that, and I think one way I would say what's happened is that. So when people first started thinking about how do you get quantized gravity, how do you quantum gravity? So the initial, one of the initial ideas was, well, you know, we've learned that we have this incredible successful, successful standard model. So let's just use the same methods that work for the standard model and apply them to gravity and we'll do that. And so it's going to be, anyway, and you're thinking of space and time in this usual way. And then there are these degrees of freedom that live in space and time, which tell you about the metric and the geometry of space and time. And you're trying to write a quantum theory of those things living in space and time and i think you know anyway people tried to do this there's lots of problems with doing it it's an incredibly long story string theory was partly reaction to the story but even string theory was still a theory of strings moving around in space and time so you weren't yeah i, you were still starting thinking in terms of a space and time. But more recently, you know, as string theory hasn't really worked out the way people expected, there has been this ideology of, oh, well, let's, you know, let's just get rid of this space and time somehow. And then we will write some theory in in some completely different kind and in the low energy limit will recover space and time as some kind of effective structure which you only see at low energies and that's become almost an ideology like our Connie Howlett likes to say space time is doomed you know meaning the the truly fun well theory is going to be in some other variables and space-time variables. He has his own proposals for this about these geometrical structures he's using to study amplitudes. But I don't, anyway, the things that I'm doing, you actually do get a theory. It looks like gravity should fit into this, and it will fit into this in a fairly standard way. This is standard space and time except, you know, the twister geometry point of view on it and interesting things happening with the spinners you didn't expect, but it's still, there is a usual idea that's about space and time are there. So my general feeling with the, the problem with this whole kind of space time is doom thing is you have to, you have to have a plausible proposal for what you're going to replace it with. It's all well and good to say that there's some completely different theory out there and the theory people used to is just an effective approximation. But, you know, first you've got to convince me that your alternative proposal is it works. And the problem is that people are just doing this without any kind of, you know, without any kind of plausible or interesting proposal for what it is you're going to replace space time with. And often, and often it even comes down to this crazy level of kind of this multiverse thing. I mean, you know, we have this theory where everything happens, so fundamentally everything happens, but then effectively you only see space and time. It's kind of, you know, you can say words like that, but it's kind of meaningless. Why is it that they have to come up with a decent proposal or replacement? Why can't they just say, look, there are some, with our current two theories, there's an incompatibility that suggests that spacetime, quote unquote, breaks down at the plank level or maybe before. So, for instance, NEMA's argument that if you were to measure anything with classically, you have to put an infinite amount of information somewhere, and then that creates a black hole. And then there's also something with the black hole entropy that suggests holography, but that doesn't mean space time is doomed. It's just a different space time. Yeah. Yeah, but for my point of you, I mean, what has been come to focus of that field a lot is this is are actually quite tricky, you know, very non-perturbitative, very kind of strong field problems about, you know, how, you know, what's going to happen to the theory when you've got black holes and black holes are you can. And so you've kind of moved away from, I mean, but, but the problem with the inconsistency between quantum mechanics and generalativity is a different, that is normally the one everybody worries about is normally a different problem. It's a very, very local problem. It's just that if you think of this in terms of the standard kind of variables, like what's the metric variables, and you use the Einstein the Einstein Hilbert action for the dynamics for these things if you try and apply standard ideas of quantum field theory locally to that at short distances you get these normal normalization problems and the theory becomes unpredictable so that's always been considered problem, how do you deal, how do you deal with that? But instead of having a proposal to deal with that and having a real kind of a new idea about what's really going to happen, what are the right variables at these short distances that will not have this problem? What are you going to do? They kind of ignore that, decided to ignore that problem and say, well, maybe string theory solves that problem. Who knows? And then to move on and to try to do something much, much harder, which is to resolve these issues about what happens in black hole backgrounds and stuff. And I don't yeah i i know but it seems to me a kind of a separate a separate issue you can still have space time and have these these these issues about you know what's going to happen in black hole backgrounds and stuff and you could still resolve them in different ways but but they're just, they really, it's a very frustrating subject, I think, to actually try to learn about it. You see people making these statements, and then you say, okay, well, what exactly do they mean? I mean, it's all well and good to say these very vague things about, this is doomed, and what about infinite amount of information, blah, blah, blah. But, you know, write down, tell me what we're talking about here. And there really isn't, it's almost a comically impossible to kind of pin people down on what is the, what are you talking, what theory are you talking about? And then finally when you pin them down, you find out that what they're actually talking about is they've, they're talking about some very, very toy model. They're saying, well, we don't know what's going on in four dimensions, so let's try it in three dimensions, and maybe two dimensions, maybe one dimension. And so they're talking about some comically trivial toy model, which they kind of ended up studying because, well, you could study it, and maybe there's some analogous problem happening there. And all they have are these kind of toy models, which actually don't seem to have any of the actual real physics of four-dimensional general activity in them. And that's what they're all studying these days. I see. Even Nima. Well, he's somewhat different, because he's coming at it from a different point of view. He's coming at it from this point of view of really trying to see find new structures in the in the perturbative expansions for, you know, for standard quantum field theories. So he's got a, he's got kind of a specific program looking at, yeah, I mean, he's not, he's generalized, he's not studying toy models. He's studying real four-dimensional physical models. But they're not, but, but, but they're often, they're generally models like Yang Milcery where you know exactly where the theory is. And it's not, this isn't solving the problem of quantum gravity or anything. It's well in theory. But I think maybe, I'm saying this a bit too quickly without thinking, but just to try to give a flavor of what I think he thinks he's doing, he's trying to take a theory that you do understand well, like Yang Mill's theory, and look at its perturbation series, Feynman diagrams, find new structures there and a new language, and then see if you can rebuild the theory in terms of these new structures. And then if you've got kind of a new way of thinking about quantum field theory in terms of these new different structures like his amplitude hydrant or whatever, then maybe you can then apply it. Once you've got a way of thinking in terms of those new structures, you can go back to the problem of quantum gravity and resolve that. Yeah. So I think, but, you know, I don't think he's not in any way as far as I know claiming to have actually gotten anywhere near there, but he's, and this gives you a lot to do. There's a lot of interesting structure, though. There's a lot to work on. And so he and his collaborators have done a huge amount kind of calculation with these things. But I, at least to my mind, I don't see them kind of coming up with what I think they hope to come up with, which is a different geometric language that really works and is really powerful that's going to get you something new. Did you listen or watch Sean Carroll's podcast on the crisis in physics? Well, no, I skimmed through the transcript of it. I was kind of wanted to see what he was. I mean, this is certainly something I'm very interested in. But, yeah, I thought, anyway, I thought the whole thing was actually quite strange because it's like four, four and a half hours long. And it's just him talking. So's just anyway I thought the whole thing that was actually very odd and it's something to do with kind of a the odd nature of the response to the um you know to to criticisms in the subject and so I think it was another kind of weird example it's you know there, he's kind of wants to say something about this issue of, you know, that many people are now, are now kind of very aware there is some kind of problem here and they're referring to it in the crisis and physics. But, you know, instead of, but, but, but just kind of talking about it for four hours or four and a half hours yourself is just kind of kind of strange um and and and especially since he's got a podcast one of the obvious things to do is to invite somebody on who you know thinks there is a crisis in physics if you don't and he doesn't think there's one it seems and well you could actually have an interesting discussion with this person for for some time but instead of discussing some this it's like you know, there's a controversy going on of two kinds. And instead of inviting somebody on to discuss this controversy with you or two people, you just go on for four hours about how your view that the other side is wrong. It was very odd, I thought. Also, it wasn't as if he was arguing with the people that were saying that there's a crisis in physics so when people say there's a crisis in physics they generally mean that there's a crisis in high energy physics particularly with coming up with fundamental law and so what he was then taking it on to mean is there's a crisis in physics as a whole like cosmology or astrophysics and then he's like no but look in solid state physics and the progress there That's called a straw man where you're not actually taking on the argument. You're taking on a diminished version of it. Well, he was also often involved in these arguments over string theory with me and Lee in 2006. And it was often the same kind of thing that he's kind of... And the whole thing is just odd from beginning to end because he's actually not a string theorist. And this is another weird sociological thing I found is that you find there, you find non-string theorist physics, physicists who somehow want to take a bit aside in this and want to and have a big opinion about it and get emotionally involved in it, even though they actually don't know, don't actually understand the issues. This is not what they do. This is not their expertise. So, and, so I know, I think some of this, you know, knowing, knowing Sean and what he's trying to do, I think he's not the only one who you see this phenomenon, that there are people who, you know, they see what they want to do in the world is really to bring to the public an understanding of the power and the great things that the subject has accomplished. And so even in his four hours, he spends a lot of time, you know, giving very, very good explanations of, you know, various parts of the story of the physics and the history of this. And, you know, they kind of see them, their goal in life is to kind of convince this, you know, the rest of the world who doesn't actually understand these great ideas or doesn't really appreciate them or skeptical about them, you know, to bring them to them. And I think part of, been the whole reason is, I think he was kind of doing this or does this is because, you know, to bring them to them. And I think part of, the whole reason he was kind of doing this or does this is because, you know, having people out there on Twitter or whatever saying, oh, you know, physics sucks, it's got all these problems. It's all wrong, blah, blah, blah, that this is, you know, this is completely against his whole goal in life is to stop this kind of thing and to really get people to appreciate the subject. So I think in kind of a misguided way then enters into this from the point of view of, oh, I have to stop people from saying things about a crisis and physics and get them to really appreciate that this really is a great subject and wonderful subject. And it's, but he kind of that goes too you know, starts to defending things which really aren't defensible and things which he often doesn't really know much about. For instance. Just the details of strength theory. I mean, the reason I wrote this book is that some of these problems of string theory, these questions, you know, people will go on about ADS-CFT and this and blah, blah, blah, blah. This is incredibly technical stuff. It's just, you know, to even understand exactly what these theories are that on both sides of the ADS-CFT thing, what is known about them, what are they, you know, what is the real problem here, what can you calculate, what can you not calculate, what can you not find, what you have to find, what happens other dimensions It's horrendously technical, and very few people actually really know it, but lots of people want to kind of get involved in discussions about it and argue about it without actually understanding actually what's going on. And part of the reason for writing the, not even wrong in the book, but was to try to kind of, you know, to sit down and try to write about about, about, you know, what was really, what was really going on, what the specific technical issues actually were, you know, as much as possible was it in a somewhat non-technical venue. But anyway, so that's some of my reaction to this. And in particular, I mean, he just starts off the whole thing by, he picked up on something from Twitter about somebody had found a paper from somebody written in 1970s complaining about how, you know, there was a crisis, there wasn't any progress in the field. And this was a time when there was great progress in the field. And this was a person who honestly, somebody completely ignorant wrote a completely paper no one ever paid attention to in the mid-1970s that that was wrong about this. And he wanted to use that as to kind of bludgeon people who are making serious arguments about the problems today. So I don't know. I thought it was kind of weird performance. But it is, I think this is a good thing to ask kind of people on this other side of this argument, strictly, why there's very little willingness to actually engage in technical discussions publicly with people they disagree with. I mean, Sean has never invited me to be on his podcast. He hasn't invited to be in a Hassanfelder. It's not, there is no appetite for that at all among people in the subject. And I think, you know, a lot of that is because, you know, they're well aware that, you know, they're really serious, difficult problems with this going. Whether you want to call it a crisis or whatever it is, there are real problems and they're just not very interested kind of acknowledging and publicizing that. Yeah. Well, I have a tremendous appetite for it and the people in the audience of everything do. So if ever you have someone who you feel like would be a great guest with the opposite view that is defending string theory or the state of high energy physics, then please let me know, and I will gladly host you both. Okay. I know we spoke about some people behind the scenes, some people who are likely to say yes and have a congenial conversation. Well, there's actually most people are. I mean, the funny thing is actually early on in this, I was invited, a guy down at University in Florida invited me and Jim Gates to come and debate and debate string theory. And so we, I think we really disappointed this big audience by agreeing on almost everything. So, you know, he's a strong, he's a well-known strength there is. And, and, and, you know, and so we actually found that i think things have been interesting to do this to do this again now but this was almost 20 years ago but let me maybe a little bit less 15 years ago and you know the way i would describe it then is you know if we started talking about the details what our disagreements came down to where it was kind of more, you know, should you be out, you know, we would agree about the state of current things, but where do you think the stuff is going? Are you optimistic? I see reasons why this can't work. He would see reasons why this is actually the best thing to do. He knows how to do and this might work. And there, it's just that kind of, you know, disagreement about ideas, which is, is perfectly reasonable. And actually, Gates told me, I remember at the, at the end of when we were talking after this thing, he said, yeah, you know, I was asked to, like, write a review of your book about it. And I thought, oh, well, I'll just, I'll pick up this book and I'll see, you know, the guy's got it all wrong about string theory and whatever. And then, you know, I read your book and I realized that, you know, a lot of what you were saying was the stuff about, that importance of representation theory in physics and that, and I actually, you know, that's actually exactly the way I see what's important in physics. So I find myself agreeing with much of your point of view and the book. So I couldn't. Anyway, so that was, you know, anyway, at the level of these ideas, I think, especially back then, I think there wasn't, it's perfectly happy, possible to have a reasonable discussion. I think it has become weirder now. You know, 20 years later, they're really, you, I think it was a lot more possible to reasonably be an optimist back 20 years ago and say, well, you know, the LHC is about to turn on. It's going to look for these super partners. Maybe they'll see super partners. There's, you know, we have all this stuff that might vindicate us, and we're all hoping for that. But now, you know, the LHC has looked, the stuff is not there. There's really not, and, you know, that's one thing that's somewhat shocked me is people willing to, people who are often, to me or in public saying, look, you know, the crucial thing is going to be the results for the LHC. You know, we believe that you're going to see, we're going to see these super partners and this is going to show that we're on the right track. And then the results come in and, you know, you're wrong and you just, you just kind of keep going and without even kind of skipping a beat about how, yeah, yeah. Anyway, that's, I think, well, there's a comment on your blog that said, the LHC has just, it's great for string theory because it divides in half the moduli space. Anyway, you can make any kind of joke you want. But, you know, I, that was certainly my feeling a lot when I was writing the book, whatever, is that, you know, this was, this was going to be a crucial thing, this, the LHC, because either the LHC was going to see something along the lines of what these guys were advertising and which they were often willing to kind of actual bet money on, or it wouldn't, and then they would back down and start saying, okay, well, maybe the critics have a point. But, no, I mean, it's just kind of amazing, and people would people will just completely ignore the, you know, the experimental results and keep going. About representation theory, for people who don't know what representation theory is, can you please give them a taste? And then also explain why is it important? More so than say you want a group to act on something. Like, okay, yes, but how much more involved does it get than that well anyway so so just to say that to give a flavor of what we're talking about yeah so i mean it's very common for people to talk about the importance in physics of symmetries and um and when you say that you know that's important to study the symmetries of something, people often then just explain it in terms of a group. So mathematically, a group is just a set with a multiplication operation. You can multiply two elements to get another. But the interesting thing about symmetries really is actually not so much the groups, but the things that groups can act on. So what are the things that can be? So the standard examples like the group of rotations. You can pick things up and rotate them in three-dimensional space, but what are all the things that you can kind of do rotations to? And so those are those in some sense are the representations or the representation theory is kind of the linear version of that theory. And if you try to work with a group action on something, it is a nonlinear, you can look at the functions on it and turn it into a linear problem. But anyway, so group representation theory is really, you know, in, it really is the study of kind of symmetries. What are the possible symmetries of things? What are the possible things that can have symmetries? And it's really fundamental both in physics and it's really, and in mathematics. And I mean, large fractions of mathematics you can put in this language of what are, there is some kind of group and it's acting on some things and what are the representations. You can, I mean, the amazing fact about the language program and number theory is how much of number theory you can formulate in that language. And you can formulate a lot of geometry in this language. It's kind of a unifying language throughout mathematics at a very deep level. But then, to me, the amazing thing is that the same, if you start looking at the structure of quantum mechanics, if you look at what are the quantum mechanics is this weird conceptual structure that states are, state of the world is a vector in a complex vector space and you get information about it by self-adjoined operators acting on this thing. So from the, that looks like a very, very weird, like where did that come from? But if you look at that formalism, it fits very, very naturally into the formalism of group representations. It's really, and this is kind of why I wrote this book, taught this course here and wrote a book about quantum mechanics from that point of view. What's the book called? Quantum Theory Groups and Representations and Introduction. It's kind of a textbook. So it was the second book I wrote. Okay, that link will be in the description. Yeah, and there's also a free version with kind of corrected, with errors that I know about corrected on my website. You can also link to that. No, we want people to pay. They have to pay for the errors. Or you can buy it, or you can buy a copy from Springer if you like a hardcover book or whatever. But, yeah, so anyway, it really is kind of amazing. One of the thing that's most fascinating to me about quantum theory is, you know, that there is a way of thinking about that it's not just some weird out-of-the-blue mathematical conceptual structure that makes no intuitive sense. I mean, it really has a structure which is kind of deeply rooted in understanding representation of understanding certain fundamental symm. Have you heard of this theorem by Radin Moy's in differential geometry about the amount of different structures that can be placed on different dimensions? So for dimension one, there's I think up to diphthism or up to differentiable structure. I forget the exact term. There's just one and then there's just two for dimension two or just one. There's a finite amount for every dimension. Except dimension four. In which case, there's not just an infinite amount. There's an uncountably infinite amount. Yeah. But there's even, yeah, but this is actually, yeah, also one of the most famous open problems in topology, the smooth black array conjecture, which says that, you know, is there, there you're thinking about it, specifically the four manifold, yeah, so is there a, now I forgot what I used to know about this, but yeah, but there are exotic. Well, the point is that dimension four is picked out. And so it would have been nice for physics if dimension four was picked out and finite, whereas the rest were infinite, because then it just means, well, it's nicer for us, but it's picked out and made more diverse and more mysterious. Yeah, but it's, how does this go? Anyway, so anyway, four dimensions is, anyway, topologically, four dimensions is very, very special. Yes. You know, one dimensions and two dimensions, you can kind of pretty easily understand the story is pretty story, the classification story is pretty simple. Three dimensions is harder, but especially with a solution of quackereg conjecture, you could, you actually have a good three-measure classification. And then once you get above four dimensions, things, basically there are more ways to move things. So things simplify so you can actually, you can actually understand above four dimensions what's going on. So four dimensions is kind of a peculiarly complex. Yeah, and so it's, yeah, it's, yeah, it's, but there is, anyway, it's very, I've never actually seen though any kind of clear, clear idea about how, what this has to do with four dimensional, with it, with physics. I mean, yeah, it's, I mean I mean the thing the stuff that I've been doing you know very much crucially involves the fact that four dimensions is special because the um the way spinners work or if you like the the rotation group in in four in every dimensions is a simple group except in four dimensions in four dimensions the rotation group breaks up into two independent pieces and that's at the core of what a lot of what I'm trying to exploit but um so four dimensional geometry is very very special and I don't know speculate very speculative maybe the there's weirdness about infinite numbers of topological structures under four dimensions, that the fact that you've got the rotation group has two different pieces means that is behind that. But I have no, I know, I know, who knows? Of course. Yeah, it's interesting that the fact that it's semi-simple is a positive here. Like you mentioned, it breaks up into two. Whereas usually in physics for the grand unified theories, what you want is simple. You don't want semi-simple. You want to unify into one large group. Yeah. Well, even, there's nothing really in terms of unification. It's just, yeah. Maybe it's a, maybe I should also say something about this about why, what I'm trying to do, I think is quite different than the usual sort of unification and what the usual. Yeah. Yeah, and please explain Euclidean twister theory once more, again, for people who are still like, I've heard the term, I've heard him explain twisters, I somewhat understand twisters, has to do with lines and points and planes, okay, and spinners, something called spinners. I think I understand that. What is Euclidean twister theory? Minkowski's like special relativity. Okay. So they're still confused. Okay. Well, maybe it's better to talk about what other, what standard kind of unification ideas are. And I think, and to my mind, I mean, basically almost essentially all attempts to do the United Founder in the same problem. So one way of stating the problem is we go out and look at the world and we see gravity and we see the electromagnetic interactions and that's kind of based upon a U1 gauge theory, just a circle. We see the weak interactions that are based upon an SU2 gauge theory. That's a three sphere. And we see the strong interactions that are based upon an SU3 gauge theory. So where in the world did this, U1, did these three groups come from, and the way quarks and other elementary particles behave under those groups? So it's a very small amount of group theoretical data. Where did it come from? I mean, why that? And so the standard answer to this very soon after the standard model came about was that, well, there's some big league group. Like you take, like, STU5, take the group of all unitary transformations of five complex dimensions or take the group of all orthogonal transformations of 10 dimensions let's say so 10 and then and then you fit the that that data and show that that data fits inside that bigger structure okay that you can within that s o 10 group we i can fit u1 and suU2 and the SU3. You can get them in there. And then I can put all of the known particles together with their transformation properties and give them and make them have a, and put those together as a transformation property of S-O-10. So you can kind of put stuff, this kind of package of algebraic data we're trying to understand where it came from. You can put it together in a simple group into a group where the problem is in terms of group theory, it's a package involving several different groups. And so you get several different simple groups. So you can you can anyway you can put this together but but the problem with this is always is if you try and do this you can then write down your SU5 or S010 theory or whatever and and you know it looks a lot nicer than the standard model it's only got one one term where you had a lot of terms before but you have to then explain but wait a minute why don't we see that why do we you know why do we see this this more complicated thing and not that and so for instance the standard thing that grand unified theories do is they you've put the weak the weak interactions and the strong interactions into the same structure so you should have, anyway, so all sorts of things, there are all sorts of new kind of forces that you're going to get in this bigger structure, which are highly constrained, which have to exist, which are going to do things like cause protons to decay. So like, you know, why? Sure, sure. Yeah, so you put the stuff together, all of a sudden, it can interact with itself and it can do things which you know don't happen, and protons don't decay. So your problem, when you write down these theories, the problem is you haven't necessarily done anything. You've put the stuff together in something bigger, but you haven't, you've just changed the problem from why, you know, why, why these pieces to, to why did this bigger thing break into the, how, how do, why did this bigger thing break into these pieces? You haven't actually solved until you have an explanation for that, you haven't actually solved anything. And this is, I think, the fundamental problem with these grand unified theories. They don't come with a really, the only way to make them break down into these other things is to introduce more Higgs particles and more complicated structure and more degrees and more numbers. And you lose predictivity if you do that. You also find that they also don't look like what you see in the world if you do experiments. But most people who have tried to come up with some unification have done some version of that actually. I mean, so for instance, I mean, I don't want to really get into things like what Garrett Leasy is talking about you know, they, they, they, they've all got their own version of this. And I think when you see people kind of dismissing theories of everything and green and fight theories and you see, um, Sabina Hassanfelder are saying, well, you know, these people are lost in math, then they're, they're, they're all really referring to the same problem that people are trying to get a better understanding what's going on by putting things together into a bigger structure and then and they're all kind of foundering on not having an answer as to why why this breaks up so um so the thing that i'm trying to do it that why i much for interested in these ideas about spinners and twisters, is that I'm not actually, I mean, a lot of what I'm doing, as I said, I mean, the fact that there are these two SU2s, that's an aspect of four dimensions. There really are, maybe the thing to say is that I'm not, I'm not introducing kind of new, I'm not introducing lots of new degrees of freedom and then having to explain why you can't see them. I'm trying to write down something. I'm trying to write down a new geometrical package, which packages together the, the things we know about and doesn't actually have new, you know, doesn't actually have all sorts of new stuff. Penrose said this was his motivation as well for Twister theory. Yeah. Yeah, so Twister theory, so in some sense, twister theory is a bigger structure, but it's not, it doesn't kind of contain anything really new. It contains the same spinner as you had before and puts them in an interesting new relation so you can understand conformal invariants. But he doesn't, it's like, you know, twister theory is not the things you knew about twister theory. It's not spinners and vectors of the things you knew about plus some other completely unrelated stuff. It's the things you knew about in a new, more powerful conceptual framework. And so that's the sort of thing I'm trying to do. Part of the problem is that, you know, it's, I guess a misnomer to really say this is a well-defined theory. It's more a speculative set of ideas about how to, but that's's the crucial i mean probably i think the most important new idea here which the which for this to be right has to be true and which is something is exactly this idea about um about rotate that if you think about rotations in four dimensions and euclidean space time when you relate it to to Mankowski space time in the real world, one of the SE2s can be treated as an internal symmetry. And that could explain the weak interactions. That's kind of a crucial. That's why it's also referred to as gravel weak unification by you or by other people? Well, other people have noticed this. And actually, it's interesting when you read the literature on Twister theory, people point this out, they say exactly the problem I was pointing out that this is a very chiral, chirally asymmetric view of the world. And a lot of people said, oh, well, that means, you know, maybe you should be able to understand, you know, the weak interactions are chirally asymmetric, so maybe there's something here. But the twister people, I think, never really had a version of this. I mean, there are various people who have tried to write down to do this. I mean, one is actually, there's a paper by, you know, Stefan Alexander has worked on this and Lee Smollin. They actually had a paper attempt to do this. But they, I mean, what they're doing is significantly different than what I'm trying to do. In particular, they're staying in Minkowski space. I mean, this idea of going to Euclidean space to get the, anyway, to get this thing to behave like an internal symmetry is not something that isn't their work. I know. You know Jonathan Oppenheim? A little bit, yeah. I mean, I've known. Yeah. Jonathan Oppenheim, Stefan Alexander, and Nima Arkani-Hamed all were graduate school peers at the same time as my brother in physics. Oh, okay. This is interesting because then later on in my life. This was all in Canada, right? Yeah, yeah. So UFT, Nima was at UFT, University of Toronto with my brother, but then in graduate school, Oppenheim, Stefan Alexander. I spoke to Stefan on the podcast as well. Yeah, no, so there have been very few physicists who have been encouraging about this. So he's one example. Yeah, he's extremely open to new ideas. And playful. He's a playful person with that, much like with his music. I think that both qualities rub off on one another. And I think also in his own research, he's also, I think he hasn't, it's not so much that he's followed up on this Grave-A-week stuff, but he's, he is very interested in, you know, is there some way in which gravity, you know, that gravity actually is a chiral theory. There is some chiral asymmetry in gravity. And especially, you know, can you know, anyway, I mean, are there kind of astrophysical and cosmological places you can go and look and see, you know, is gravity really, chirally symmetric or not? And so I know that that's something that he's worked a lot of. So he's working on experimental tests of the chirality of gravity, but that doesn't mean experimental tests of your theory, just your theory is a chiral theory of gravity. Yeah, it's a, it's a chiral theory. But it's not, it would be validation of your theory or a testation? No, I mean, it's kind of, I mean, first of all, again, I have to keep thinking, I don't really have it. I don't, I would love to say, I would love to say I've written down a consistent proposal for a theory of quantum gravity based on my ideas but I'm not there yet. And I think what he's doing is more, it doesn't involve, doesn't have, the structures I'm trying to exploit are not there in what he's doing. But I believe what he's doing is more kind of thing. You kind of add Chern-Simon's kind of terms. You assume that maybe there's some Chern-Simon's term in the theory and ask, you know, what the observational implications of that would be and try and go out and look for that. But I haven't looked, I haven't really carefully looked at what he's doing, just because it's quite different than what I'm trying to do. Can you explain what Turn Simons theory is? So what it means means to add a Churns Simon's term. I know Stefan's worked on Churns Simon modified gravity. And then there's something like Churns Simon terms in the Lagrangian of particle physics, but I don't know if those two are related. Yeah, I don't, yeah, I shouldn't try to talk about it as work as I don't remember exactly what he was doing. But, well, Churn's time in the, it's very hard. Actually, one funny thing is that I actually went to, I don't know, so I actually started thinking about churn. So maybe I can go back to, you know, how I first encountered them. So when I was doing my PhD thesis, my problem was I'm trying to understand, I've got engaged on a computer, and I've got this version of gauge fields, and they're described on links on a lattice and you can store them in a computer and manipulate them. And I want to look at one of these configurations and say, you know, there's supposed to be some, there's some interesting topology in this engage theory. And this is what people are getting interested in the 70s and 80s. And so in particular, there's something called the, let's say the instanton number. And so, you know, these gauge fields are supposed to have some integer invariant called the instanton number. And if somebody hands you a gauge field on a compact manifold, you should be able to calculate its instanton number. And you can then, then you could, if you could measure these, if you could calculate these instanton numbers and see them, you could do interesting physics with it. So the problem in some of this problem my thesis was, you've got these gauge fields, what are their instanton numbers? Can you define them? And so... And they're just integers? They're just integers, yeah. So they're invariants, they're not invariance of the base manifold. You basically have a bundle with connection and they're invariance of the bundle. And if you know the connection, you're you're sensitive to this invariant. But the one way of looking at that though is if you look at the integral formula for this thing, it's a total derivative so that if you try to integrate it over a ball or a hypercube, the formula that's supposed to add up to this instanton number, you can write it as an interval with the boundary, right? It's the interval of D of something, so it's the, it's the integral of boundary. It's a total derivative, so you can see. So the, so the thing that it's a total derivative, the thing that that lives on the boundary is the is the Chern Simons form. Okay. So that's, this is kind of the first way that people started seeing this thing in, in physics is that. And so, so, so one idea was I, well, I could, um, I could um yeah if I could call instead of calculating these instant on numbers if I try and do it in terms of their local contributions from each hypercube I should if I could just calculate the churn simons not the churn simons number the contribution you know the if I could cut could cut that the that thing then then i would be done and so i spent a lot of time looking at the churn simon's formula and and then i spent a lot of time trying to put that in the lattice and then i kind of finally realized it's kind of gauge the problem is that it's very gauged in variance so any kind of idea you have about how to calculate it or construct it tends to be just an artifact of some choices you're making because of gauge symmetry. So this, though, that led to one of the great experiences of my life. When I was at a MSRI, you know, Atia was visiting and at one point Atia and a bunch of people were talking to the blackboard and somebody was asking Atia said, oh, you know, how would you like in, you know, how would you calculate this churn Simons network? Then churn Simons had become incredibly important because of Witten and and so everybody was like, Witten had said, you can get these wonderful nod invariants of three-metapult of variance if you can do path integrals and that you should take the path integral to be E to the I times the churn-Simon's number. Exactly that integral that I was talking about. Yes. But Witten now wants to integrate it over a whole three-manifold. And so people were asking, Atia, well, you know, can we try and think about how could we actually do this calculation, what were we doing? And so, and then Atia, for thinking for about for about five seconds, comes up and says, oh, well, maybe, you know, you could calculate it this, you could calculate it this way, do this. I was luckily standing there. And since Atia had thought about it for about 10 seconds, I thought about it for about three years. I could say, no, no, no, that doesn't work. You can't do that because of this. Oh, great, great, great. So that was one of the high points of my mathematical career. Yeah. Anyway, but I don't know that this is in any way answered any question, but that's one definition of it. But it's a very, it's kind of an amazing piece of information about, you know, about gauge fields, about connections. And it tells you some very subtle things. And it turns out to be useful for all, describe all sorts of interesting and unexpected physical phenomena. And these speculative ideas of yours of gravel weak unification, have they been sent to Penrose? Has Penrose commented on them? I haven't heard anything back from Penrose. Predros is a little bit of a problem that I don't actually... Anyway, whatever email I had from him back when he was helping my book no longer works and other emails tend to bounce and say... You don't have mutual friends? I could make more of it. I haven't made more of it. I also keep also hoping... I've come this close to actually running into him and being at the same conference and something at him and having a chance to talk to him personally, I keep expecting, instead of making a further effort get to get a manuscript to him part of the problem you'll see if you try and if you don't know his email and you try and contact them you end up getting a secretary and who may or may not see more anything to him right but I keep hoping yeah I was actually at Oxford last year and actually was there somebody who showed me oh, oh, that's Penrose's office. And then I went to do something else. And then the next day, they said, oh, you know, 15 minutes after we were there, Pedro showed up. Oh, boy. The lowest points of your mathematical career. Well, I don't know. I don't know how this would work. From things that he said about this kind of thing, I think he's made it very clear that he has always explicitly, he's been, you know, he's followed the kind of thing Atia did, the kind of Euclidean version of the theory. But he's always said very clearly that in his mind, the Euclidean version of theory is not the theory. What's,'s happening in Mokowski space. And so he's, anyway, whether I could convince him otherwise, I don't know. But I think he's kind of pretty clearly in his mind thought through, okay, there is this interesting Euclidean theory, but that's actually not really the physical thing is Mekowski. So I don't actually believe you're going to, that by working over there, you're going to actually tell me something important. But I think I'd have to get around that particular initial reaction from him. So forgive this fairly foolish question, but if both GR and the standard model can be formulated in terms of bundles, then why can't you just take a direct product of the groups? So, for instance, you have the standard model gauge groups, and then you direct product with S.O.13. So that's the principle, and you make an associated frame bundle. That's like just the projection of S.O.13. And then you say that's general relativity, and the other ones, the other associated bundles of the standard model. And then you call that unification. Is that unification? What are the problems there? Well, the problem is that general relativity is a different. Well, maybe the thing to say is, so gauge theory is really just what you have is a bundle and the fibers are some group. And you have connections and curvature on that. You write down the interesting Lagrangian is the norm squared of the curvature. And anyway, so Gage series is a nice pretty story. If you try and write generatively the same language, you know, you can do it. It's fine. You have a G bundle where G is S.O3-1 or the Euclidean Ridge, whatever. Yeah, yeah. And you have a connection, you have a curvature. But the problem is that you crucially, the problem is that you crucially have something else and you have other things specifically because you're not some arbitrary G bundle, you're the frame bundle. And the frame bundle, you know, it has, you know, it's a principal bundle for, you know, the group of just all changes a frame. But it also is, I mean, people use the term soldered or tie. It's also, it also knows about the base structure. So a point in the fiber of the frame bundle is not just an abstract group element. It's a frame. It's a frame down on, you know, if you can take vectors, you can protect it on the base space, and it's a frame for those vectors. So it's kind of soldered to the tangent space. it, what, what this means in practice is it means that there's, there's, there's new, there's new variables which are in the, which you have, which, which are part of the story, which are not just the, not just the S.O3-1 connection and curvature. There's also, you know, so you've got this connection one form and cur. Sodering form? Yeah, it's called the soldering form or the tetrad or, I mean, there are a lot of different people have names for it. But there's kind of, there's kind of a one form you feed at the vector and you feed it a vector and it tells you and, you know, since you're up in the frame bundle, you've got a frame and this one form has, you know, it has you, and since you're up in the frame bundle, you've got a frame, and this one form has, you know, it has components which tell you what the components of the vector are with respect to the frame. So it's a very kind of canonical object, but, you know, it's there. The space-time geometry depends upon it. So the space-time geometry doesn't just depend upon the connection, the curvature, depends upon the connection, the connection, and this, this, this, this, this, this, this, this, this, this, this, this, this, this, um, this canonical one form. So, so, so, so it, you, you've got extra variables, which you didn't have in the, these just don't exist in the Yang Millsills case and you have to and so you can and and with those variables you can you can write down a different look a different lower order of Lagrangian instead of taking the taking the curvature squared you can take the curvature times some of these guys and you can get the Einstein Hilbert Lagrangian sorangian. So the fundamental Lagrangian of gravity is very different than the fundamental Lagrangian of Yang Mills theory. And it's because you've got these extra gadgets to work with. I see, I see. They've got a one form. So that's one way of saying it. You can't. But people have speculated a lot about why, you know, why not, why not just try, like, adding these higher curvature terms like you had in the, in the Yang Mills case, add those to gravity. And anyway, there's a long, long story about trying to mess with different change the Lagrangian of gravity to try to something better behaved. Now, have you found any unification attempts that are between gravity in the standard model or gravity in any of the interactions that are improved if you don't view gravity as curvature but rather as torsion? So, for instance, this is something Einstein was working on later in his life. And then there's also non-matricity. Carton was working on that. Yeah. Yeah. And they're equivalent formulations of gravity, at least the torsion one. The gravity is actually not curvature, it's just torsion. Yeah, yeah. So the, well, one way to say it is, so now once you've got these, so the thing about, if you write, start writing down a theory of gravity. Well, first of all, I mean, non-metricity, I think some of that may just mean, actually I'm not sure what exactly the people mean about that. I shouldn't say. So the two compatibility conditions to create the Levi-Cavita connection, I believe it's called, is that you have no torsion and that you have that the metric doesn't change with the covariant derivative. So if you take the covariant derivative on the metric, it's zero. If you don't have that, then you have non-metricity. In other words, along the parallel transport, the metric is preserved. Yeah, okay, yeah. I'm not so sure about that. But I can't say about torsion, that the, but your problem is that if you, so if you just write down a theory with with some you put together a Lagrangean which is which is going to be give you equivalent results to the Einstein Helbert you put it together out of the curvature and the canonical one form now your problem is that you've got you know when you try to get the other Lagrange equations you you can, you can vary the canonical one form and you can vary the connection. So you've got, and one of them, let's say, I guess it's, if you vary the connection, then you end up, that gives you the torsion free condition. So, so, so, so, so, so you, you've got more variables, so you need more equations. So you recover gravity, but you recover with the standard Lagrangian, you recover not the Einstein's equations and as one equation, but also the torsion-free condition as the other one. So I mean mean, so the standard simplest, you know, version of Einstein-Hilbert in that theory, you know, has no torsion again. But you can certainly write down more different Lagrangians in which torsion is, you know, is not zero, but it's some kind of, has some kind of dynamics and does something. And that might be interesting. Yeah, I was watching a talk a few, maybe a few weeks ago or a couple months ago about when trying to modify gravity, especially for explaining quote unquote dark matter that you can explain dark matter as a particle, but if you want to do modified gravity, it's useful to have torsion in your theory. Well, anyway, what I was thinking was, okay, if it's useful there, maybe it's not actually the case that that explains dark batter, but maybe it would be more useful to try unification with torsion models of gravity than with the regular curvature model of gravity. Yeah, I should say one kind of funny thing about all this is that I've always, I mean, before I got involved in this particular thing, I tended to kind of stick to thinking, I mean, I spent a lot of time over the years trying to learn about quantum gravity and about these issues that we're talking about. But I never actually, you know, got really serious about them and developed any real expertise with them because I always kind of felt that they're, I don't know, I'm trying to understand what's going on in particle physics and the standard model. And there's, there are these groups of people who, you know, just think, who just think about quantum gravity and that, you know, they're very smart. They've been doing this for 30 or 40 years 40 years and even and a lot of them aren't strength there is and um and you know i don't i'm not seeing anything that they're doing that i that or that i could have any kind of you know that i could do it anyway better like you know that they seem to be doing interesting things with torsion but they know more about torsion than i don't do so right yeah so i i kind of, anyway, I kind of stayed away from a more particle. Yeah. Yeah, exactly. Yeah, that's the way of saying it. But I really stayed away from kind of going more in that direction, becoming more expert, a lot of these things, figuring, yeah, I mean, until I see something that I could, that maybe I can do something with, I mean, if it's just, it's interesting to see what the story is there, but they're really smart people who have been banging away at the story for a long time, and I can't help. I'll stay away from it. But, so yeah, so I kind of have the, I've actually partly because of this had to, had to learn a lot more about, it gets some remedial education on some of this stuff. And so I'm, but I'm still in some sense the wrong person to talk to about theories of gravity and about the... Yeah. Before we wrap up, there are a couple other proposed toes, so one with Lisi, like you mentioned. And then Eric Weinstein has Geometric Unity, and Wolfram has Wolfram's Physics Project. I believe that's still the title. And Cheramar-Marletto has a framework, not an actual toe, but construct your theory. So which of those have you delved even superficially into? And what are your comments on them? I should say, I mean, the Wolfram or the other one mentioned, so these ideas that you're going to start with some completely different starting point like Wolfram. We're going to start, I don't know, whatever you want to call, whatever he's starting with. The fact that you're going to start from this kind of completely different thing, it has nothing to do with any of the mathematics that we know of, and that you're going to then reproduce the standard model, whatever this. That seems to be highly implausible. Anything I've ever looked at, and of his for briefly, you know, doesn't change that opinion. I just, I just don't see how you get from. Anyway, I mean, you're telling me that you're going to go and start way, way, way, far away at something else and make some progress right here. And I don't see how you're going to get, you're ever going to're ever gonna get back and so so there's a lot of that um uh leesiest thing i looked a bit out a bit so i i know garrett and eric both fairly well you know so garret has slept on my couch like many people but uh and and and you know so garret i think you had well-defined proposal, but to my mind, it has exactly the same, the problems that I was telling you about. You know, he, he wants to put. So these are the same problems you explicated about Grand Unified theories earlier. Yeah. So he wants to put all these things together, and he wants to put it together and have it live inside E8, and it's very nice, except that he doesn't really have a, to my mind, by doing that, he hasn't actually solved the problem. He has to tell me why the E8 breaks down into the pieces that we know about. And he doesn't have any, as far as I know, has no useful idea about that. But he is a fairly well-defined thing. I mean, Eric, you know, I've talked to a lot about this over the years. I don't know. I mean, he, and I've looked a bit at, you know, paper that he finally put out. But I think, again, it seems to me, it has the same kind of problems. Again, he's trying to put, he's trying to put everything together into this bigger geometric structure. But he doesn't, to my mind, have any kind of plausible idea about how he's ever going to break that down and recover what we, the real world that we see. And his is a lot harder to see exactly what he's doing or unless Lizzie is kind of following much more kind of a standard story. You can see exactly what he's doing where it's harder to tell. But both of them, I think, suffer from the same problem as guts as far as I know. What about category theory? There's plenty of hype about category theory in physics, but you're also in math, and so you're much more close to category theory. Is there a hope that somehow higher categorical structures will elucidate how to make progress in high-energy physics? Yeah, I haven't seen any evidence for that. I mean, the things people are doing with those are actually much more trying to understand. There's a lot of people actively trying to use some of that mathematics to understand like classification or more kind of theories you would use in condensed matter systems. So it's possible that, you know, the right way to understand, you know, gauge groups, you know, the infinite dimensional group of all gauge transformations, or you're even, or maybe you can even think of the diphthymorphism group about how to think about representations of those groups, those groups, and maybe that the higher categorical stuff has something useful to say about that, because there are the problem is that you, the standard notions of what a representation is don't really, the problem is when you're dealing with these influential groups, you really don't even know what, you can't just say representation, you have to put some more additional structure to make this well defined and what the additional structure is unclear and maybe it would help with those. But anyway, I haven't really followed. I've spent some effort trying to follow that mathematics, but I don't do that. Anyway, category theory in general is just a very, very general idea. The problem is it's a very, very general idea. So it's something, it's part of, you know, the way mathematicians think about every subject, you know, that I really, it's very, very useful to think not about representations, but the category of all representations to think of, and that opens up all sorts of new, quite new ways of thinking and questions do that, but it's, but it, it's just a very rare abstract language. So it can be used for many, many things. And I think when I realized at some point, when I was a student, I was very, I thought, okay, well, you know, the way to understand mathematics is to find, you know, look at these, the mathematics are teaching us and look for the more and more general structures and then just find them, understand the most general structure. And then, you know, you'll be able to, to derive the rest of this stuff. And so, and then it looked like category theory was that was this thing, which was the most general thing that people were using. And so I thought I should go learn category theory. But then at some point, I realized that what I was, what you're doing is that as you go to greater and greater generality, you're, you're saying what you're doing, you're talking about, you're saying something about more things, but you're saying less and less. And so in the limit, you're saying nothing about everything, which is really not, not actually a useful limit. And that's the problem with just, you know, category theory has just in its most general meaning. It's very useful. I can do all sorts of things, but it's not, anyway, it's telling you a bit about everything, but yeah, it's too much generality to really kind of. Now, what if someone retorts about the polemics against string theory by saying, hey, look, string theory has produced something much that's positive. So, for instance, the math is used in condensed... Sorry, is used in the fractional quantum hall effect and many other condensed matter systems. No. That's, yeah, no, the string theory hasn't... That stuff doesn't... Well. First of all, I mean, a lot of the time when people are talking about this, they're talking about something which didn't actually come from a string theory. It's quantum field theory. So yeah, like the fractional quantum whole effect. I mean, I don't think there's not a string theory. There was a comment that said, look, I'm a physicist and I'm not a string theorist, but we use string theory in the fractional quantum hall effect. And that was a comment on the Ed Frankel video. Well, I think probably, I mean, the problem is string theorists are happy to kind of claim, yeah. Anyway, I mean, they're kind of claiming that everything comes from a string theory. And they're actually at this point, David Gross kind of argues that, well, you can't, you have to shut up and stop arguing about string theory because string theory and quantum field theory are actually all one big thing. And so you're arguing against quantum field theory. So that's just a ways that. Because string theory is supposed to be a generalization of quantum field theory? Well, it's because, oh, you know, with these dualities and M theory, whenever we realize it's all the same. And so anyway, so I don't know in this, in this specific case, and I'm not an expert on that case, but I strongly suspect that the saying that this came from string theory is that it's really some fact that they learn from string theories. And string theor is happy to say this camera of string theory, but it's not actually. And to make this whole thing even more frustrating, more complicated, is that no one actually can, at this point, has a definition of what string theory is. So you can, people then start talking about kind of like what Gross is trying to do. He's trying to say, well, string theory and quantum field theory all the same. So when I say string theory, I mean quantum field theory. And people just keep doing this. And, you know, so we, anyway, unless you're really, really expert and you know exactly what the story is about what string theory is and how it's related to quantum field theories, whatever, you easily get very confused. Another weird thing I found is that almost everyone believes that Ed Witten wrote one Fields Medal for his work on string theory, which is just not true. It's just not true. I mean, the things that he won the Fields Medal for are these totally amazing things in mathematics are actually quantum field theory. Things are not. They actually have basically nothing to do with string theory. The positive energy theorem. Yeah. And those things, I mean, they're not string theory. But, you know, it's really hard to convince anyone of this. Even most mathematicians believe this. If you go up and ask a mathematician, you know, did Witten, a string theory part of what Witten won the Hill's Melbourne? I'm sure the walls. Most of them will say, oh, probably is. Yeah, it sounds right. So what's a fulfilling life for you, Peter? Well, I'm very, I'm quite happy. I mean, one, I think, you know, when my book came out, a lot of people, you know, kind of the ad hominem attack was, oh, here's this guy who was not a success and didn't really, and he's just embittered and unhappy. And they didn't realize that I'm actually quite, quite disgustingly pleased my life and very happy with myself. And things that have gone. I mean, had a weird career here at Columbia and it's a it's a very but I've been extremely well treated by the department and allowed pretty much to do to get away as I said get away with doing whatever I want and treated well and paid well and had a very pretty very happy life and so I'm meaningful yeah and I'm I'm proud of the books I've written, some of the things I've done. And I'm actually quite excited about what I'm working on now. I mean, and this was always one of my great frustrations is that, you know, there were a lot of things that seem to be that something interesting was going on, but I didn't understand enough to really be sure this is really something, you know, I've really got something here. And now I'm much more optimistic about that. And so I'm trying to, I'm getting older though. I'm 66. I'm trying to figure out, I'm actually trying to negotiate with the department of the university, some kind of exit strategy out of my current position to some different kind of situation here. And I may, where I might be doing less teaching and less to, and, and less involved and less taking care of the computers, get other people to do that. So we'll, we'll take care of the computers. Well, I told you about this. So part of my, my, I'm, my official title is senior lecturer. And the weird thing about this title is, is this is a title that the university gives to people who are, they're non-tenured positions, but are, but are teaching, teaching courses here. And so I'm doing that. But I've also, part of the deal with the department has always been that I do relatively not that much teaching, but also make sure the department computer system runs. And so I actually do, on a day, day basis, I also make sure our computer system's going. So I do. You don't want to do that anymore. Well, let's just say I like to do, maybe a better way of saying it is, I mean, I've actually actually kind of enjoy that actually. That's always been never, that's always's always been been been in some ways fun but um there there is an inconsistency i found between you know having the time and focus to work on making progress on the stuff i want to make progress on and also teaching a course and also having to deal off and on with computer problems. And trying to fit all those together in a 40-hour week is not really, doesn't work so well. And I've decided in my life, I definitely have to prioritize the working on these new ideas. I've got to start dumping some of the other things and change things. But we'll see. I managed to find that specific comment that was referenced earlier, and I sent it to Peter Woite over email. Here's the comment, and then subsequently there'll be Peter's response. I am a physicist, and I use string theory all the time in my research on the fractional quantum hall effect. What Frankel means here is that the expectation to find the standard model in the 90s, by Calibiaw-compactification of one of the the super string theories turned out to be unfulfivable to this date. This does not harm the theory. The prediction was just wrong. Therefore, the title of this video is misleading. String theory revolutionized the way we understand physics and math in general, and it continues to do so. By the way, it's the only consistent theory, unifying quantum field theory and gravity. Peter's response is, hi, Kurt. In the podcast, I misunderstood what you were telling me that a condensed matter theorist was saying that they thought understanding the fractional quantum hall effect used string theory. I was speculating that they were misunderstanding some QFT explanation as a string theory explanation. It seems, though, that this is not a condensed matter theorist, but a string theorist. The quote-unquote string theory revolutionized the way we understand physics and math in general and continues to do so is just pure hype. It's the sort of thing you will ever hear from a string theorist devoted to that cause. I was unaware that some string theorists have worked on embedding the fractional quantum hall effect system in a complicated string theory setup. I don't understand the details of this from long experience, think it's highly likely. This, like many, string theory explains condensed matter physics claims, is just hype. String theory since the beginning has had a huge problem, and it continues to this day. The current tactic for dealing with the failure of string theory hype around particle physics is to double down with new hype about nuclear physics, condensed matter physics, and quantum information theory, etc, etc. Peter then quickly sent a follow-up email, hey, I just read the thread. I'm guessing this is a string theory undergrad or graduate student. The claims about the fractional quantum hall effect are based on relating it to Chern-Simon's theory, which is a QFT story, so a quantum field theoretic story. Also, all those fans of David Hesteens should know that I did ask Peter about geometric algebra, but he's not familiar enough to comment on it. Okay, well, it was wonderful speaking with you, and I hope we speak again. I hope we meet in person. Oh, sure. Let me know if you're ever in New York. Oh, yeah, I go quite frequently, so I'll let you know the next time I'm there, and maybe I'll see you at perimeter if you ever come down this way. Yeah, I haven't been there yet, but I would at some point like to like to go there. I just signed up to participated via Zoom. They have a conference on quantum gravity at the end of the month. But it's mostly virtual. And so you can anyway, I'll watch some of the talks on Zoom, but someday I'll actually get there physically. All right, sir, take care. Okay, thanks. Thank you for coming on. Bye now. Bye, bye. The podcast is now concluded. Thank you for watching. If you haven't subscribed or clicked that like button, now would be a great time to do so, as each subscribe and like helps YouTube push this content to more people. You should also know that there's a remarkably active Discord and subreddit for theories of everything where people explicate toes, disagree respectfully about theories and build as a community our own toes. Links to both are in the description. Also, I recently found out that external links count plenty toward the algorithm, which means that when you share on Twitter, on Facebook, on Reddit, etc., it shows YouTube that people are talking about this outside of YouTube, which in turn greatly aids the distribution on YouTube as well. Last but not least, you should know that this podcast is on iTunes, it's on Spotify, it's on every one of the audio platforms, just type in theories of everything and you'll find it. Often I gain from re-watching lectures and podcasts and I read that in the comments. Hey, toll listeners also gain from replaying. So how about instead re-listening on those platforms? iTunes, Spotify, Google Podcasts, whichever podcast catcher you use. If you'd like to support more conversations like this, then do consider visiting patreon.com slash kurt Jymungle and donating with whatever you like. Again, it's support from the sponsors and you that allow me to work on tow full time. You get early access to add free audio episodes there as well. For instance, this episode was released a few days earlier. Every dollar helps far more than you think. Either way, your viewership is generosity enough."""]


def run_client_chat_stream_langchain_fake_embeddings(data_kind, base_model, local_server, inference_server,
                                                     simple=False, chat=True):
    t0 = time.time()

    os.environ['VERBOSE_PIPELINE'] = '1'
    remove('db_dir_UserData')

    stream_output = True
    max_new_tokens = 256
    # base_model = 'distilgpt2'
    if base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
        prompt_type = 'human_bot'
    elif base_model == 'h2oai/h2ogpt-4096-llama2-7b-chat':
        prompt_type = 'llama2'
    else:
        prompt_type = ''
    langchain_mode = 'UserData'
    langchain_modes = ['UserData', 'MyData', 'github h2oGPT', 'LLM', 'Disabled']

    if inference_server == 'replicate':
        model_string = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
        inference_server = 'replicate:%s' % model_string
        base_model0 = 'h2oai/h2ogpt-4096-llama2-7b-chat'
        if base_model != base_model0:
            return
    elif inference_server and inference_server.startswith('openai'):
        base_model0 = 'gpt-3.5-turbo'
        if base_model != base_model0:
            return

        if inference_server == 'openai_azure_chat':
            # need at least deployment name added:
            deployment_name = 'h2ogpt'
            inference_server += ':%s:%s' % (deployment_name, 'h2ogpt.openai.azure.com/')
            if 'azure' in inference_server:
                assert 'OPENAI_AZURE_KEY' in os.environ, "Missing 'OPENAI_AZURE_KEY'"
                inference_server += ':None:%s' % os.environ['OPENAI_AZURE_KEY']
    else:
        if base_model == 'gpt-3.5-turbo':
            return
        if local_server:
            assert inference_server is None

    assert base_model is not None
    if inference_server and inference_server.startswith('openai'):
        tokenizer = FakeTokenizer()
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(base_model)

    if local_server:
        assert not simple
        from src.gen import main
        main(base_model=base_model,
             inference_server=inference_server,
             prompt_type=prompt_type, chat=True,
             stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
             max_new_tokens=max_new_tokens,
             langchain_mode=langchain_mode,
             langchain_modes=langchain_modes,
             use_openai_embedding=False,
             verbose=True)
    else:
        os.environ['HOST'] = inference_server
    print("TIME main: %s %s %s" % (data_kind, base_model, time.time() - t0), flush=True, file=sys.stderr)
    t0 = time.time()

    from src.client_test import get_client, get_args, run_client
    # serialize=False would lead to returning dict for some objects or files for get_sources
    client = get_client(serialize=False)
    print("TIME client: %s %s %s" % (data_kind, base_model, time.time() - t0), flush=True, file=sys.stderr)
    t0 = time.time()

    expect_response = True
    if data_kind == 'simple':
        texts = texts_simple
        expected_return_number = len(texts)
        expected_return_number2 = expected_return_number
        prompt = '\n'.join(texts[:expected_return_number])
        counts = count_tokens_llm(prompt, tokenizer=tokenizer)
        print('counts ', counts)
        prompt_when_texts = 'Documents'
    elif data_kind == 'helium1':
        texts = texts_helium1
        if base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
            expected_return_number = 10
            expected_return_number2 = expected_return_number
            tokens_expected = 1500
        else:
            if base_model == 'gpt-3.5-turbo':
                tokens_expected = 2600
                expected_return_number = 24  # i.e. out of 25
            elif inference_server and 'replicate' in inference_server:
                tokens_expected = 3400
                expected_return_number = 16  # i.e. out of 25
            else:
                tokens_expected = 3400
                expected_return_number = 16  # i.e. out of 25
            expected_return_number2 = expected_return_number
        prompt = '\n'.join(texts[:expected_return_number])
        counts = count_tokens_llm(prompt, tokenizer=tokenizer)
        assert counts['llm'] > tokens_expected, counts['llm']
        print('counts ', counts)
        prompt = '\n'.join(texts)
        countsall = count_tokens_llm(prompt, tokenizer=tokenizer)
        print('countsall ', countsall)
        prompt_when_texts = 'Documents'
    elif data_kind == 'helium2':
        texts = texts_helium2
        if base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
            expected_return_number = 10
            tokens_expected = 1500
            expected_return_number2 = expected_return_number
        else:
            if base_model == 'gpt-3.5-turbo':
                expected_return_number = 25 if local_server else 25
                tokens_expected = 2700 if local_server else 2700
                expected_return_number2 = 25
            elif inference_server and 'replicate' in inference_server:
                expected_return_number = 17 if local_server else 17
                tokens_expected = 3400 if local_server else 2900
                expected_return_number2 = 17
            else:
                expected_return_number = 17 if local_server else 17
                tokens_expected = 3400 if local_server else 2900
                expected_return_number2 = 17
        prompt = '\n'.join(texts[:expected_return_number])
        counts = count_tokens_llm(prompt, tokenizer=tokenizer)
        assert counts['llm'] > tokens_expected, counts['llm']
        print('counts ', counts)
        prompt = '\n'.join(texts)
        countsall = count_tokens_llm(prompt, tokenizer=tokenizer)
        print('countsall ', countsall)
        prompt_when_texts = 'Documents'
    elif data_kind == 'helium3':
        texts = texts_helium3
        if base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
            expected_return_number = 6
            tokens_expected = 1500
            expected_return_number2 = expected_return_number
        else:
            if base_model == 'gpt-3.5-turbo':
                tokens_expected = 3000 if local_server else 2900
                expected_return_number = 14 if local_server else 14
                expected_return_number2 = 14 if 'azure' not in inference_server else 14
            elif inference_server and 'replicate' in inference_server:
                tokens_expected = 3000 if local_server else 2900
                expected_return_number = 11 if local_server else 11
                expected_return_number2 = expected_return_number
            else:
                tokens_expected = 3500 if local_server else 2900
                expected_return_number = 11 if local_server else 11
                expected_return_number2 = expected_return_number
        prompt = '\n'.join(texts[:expected_return_number])
        counts = count_tokens_llm(prompt, tokenizer=tokenizer)
        assert counts['llm'] > tokens_expected, counts['llm']
        print('counts ', counts)
        prompt = '\n'.join(texts)
        countsall = count_tokens_llm(prompt, tokenizer=tokenizer)
        print('countsall ', countsall)
        prompt_when_texts = 'Documents'
    elif data_kind == 'helium4':
        texts = texts_helium4
        if base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
            expected_return_number = 5
            expected_return_number2 = 7
            expect_response = False  # fails to respond even though docs are present
            tokens_expected = 1200
        else:
            if inference_server and inference_server.startswith('replicate'):
                expected_return_number = 12 if local_server else 12
                expected_return_number2 = 14
            elif inference_server and inference_server.startswith('openai_azure'):
                expected_return_number = 14 if local_server else 14
                expected_return_number2 = 16
            elif inference_server and inference_server.startswith('openai'):
                expected_return_number = 14 if local_server else 14
                expected_return_number2 = 16
            else:
                expected_return_number = 12 if local_server else 12
                expected_return_number2 = 14
            tokens_expected = 2900 if local_server else 2900
        prompt = '\n'.join(texts[:expected_return_number])
        counts = count_tokens_llm(prompt, tokenizer=tokenizer)
        assert counts['llm'] > tokens_expected, counts['llm']
        print('counts ', counts)
        prompt = '\n'.join(texts)
        countsall = count_tokens_llm(prompt, tokenizer=tokenizer)
        print('countsall ', countsall)
        prompt_when_texts = """
Please rate the following transcript based on the tone and sentiment expressed. Express the answer as a table with the columns: "Rating" and "Reason for Rating".
Only respond with the table, no additional text. The table should be formatted like this:

| Reason | Reason for Rating |
|--------|-------------------|
| 5      | The tone of the transcript is generally positive, with expressions of optimism, enthusiasm, and pride. The speakers highlight FedEx's achievements, growth prospects, and commitment to improvement, indicating a positive outlook. However, there are also some mentions of challenges, headwinds, and areas for improvement, which prevent the tone from being entirely positive. |


Use the following scale:

1 (most negative): The transcript is overwhelmingly negative, with a critical or disapproving tone.

2 (somewhat negative): The transcript has a negative tone, but there are also some positive elements or phrases.

3 (neutral): The transcript has a balanced tone, with neither a predominantly positive nor negative sentiment.

4 (somewhat positive): The transcript has a positive tone, with more positive elements than negative ones.

5 (most positive): The transcript is overwhelmingly positive, with an enthusiastic or supportive tone."

Here's an example of how this prompt might be applied to a transcript:

"Transcript: 'I can't believe how terrible this product is. It doesn't work at all and the customer service is horrible.'

Rating: 1 (most negative)"

"Transcript: 'I have mixed feelings about this product. On the one hand, it's easy to use and the features are great, but on the other hand, it's a bit expensive and the quality could be better.'

Rating: 3 (neutral)"

"Transcript: 'I love this product! It's so intuitive and user-friendly, and the customer service is amazing. I'm so glad I bought it!'

Rating: 5 (most positive)"""
    elif data_kind == 'helium5':
        texts = texts_helium5
        if base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
            expected_return_number = 1
            expected_return_number2 = 1
            expect_response = False  # fails to respond even though docs are present
            tokens_expected = 1200
        else:
            expected_return_number = min(len(texts), 12) if local_server else min(len(texts), 12)
            expected_return_number2 = min(len(texts), 14)
            if base_model == 'gpt-3.5-turbo':
                tokens_expected = 2500 if local_server else 2500
            else:
                tokens_expected = 2900 if local_server else 2900
        prompt = '\n'.join(texts[:expected_return_number])
        counts = count_tokens_llm(prompt, tokenizer=tokenizer)
        assert counts['llm'] > tokens_expected, counts['llm']
        print('counts ', counts)
        prompt = '\n'.join(texts)
        countsall = count_tokens_llm(prompt, tokenizer=tokenizer)
        print('countsall ', countsall)
        prompt_when_texts = """Is the information on interest rate swaps present in paragraphs or tables in the document ?"""
    else:
        raise ValueError("No such data_kind=%s" % data_kind)

    if simple:
        print("TIME prep: %s %s %s" % (data_kind, base_model, time.time() - t0), flush=True, file=sys.stderr)
        # res = client.predict(texts, api_name='/file')
        res = client.predict(texts, api_name='/add_text')
        assert res is not None
        print("TIME add_text: %s %s %s" % (data_kind, base_model, time.time() - t0), flush=True, file=sys.stderr)
        return

    # for testing persistent database
    # langchain_mode = "UserData"
    # for testing ephemeral database
    langchain_mode = "MyData"
    embed = False
    chunk = False
    chunk_size = 512
    loaders = tuple([None, None, None, None, None, None])
    h2ogpt_key = ''
    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    print("TIME prep: %s %s %s" % (data_kind, base_model, time.time() - t0), flush=True, file=sys.stderr)
    t0 = time.time()

    prompt = "Documents"  # prompt when using langchain
    kwargs0 = dict(
        instruction='',
        max_new_tokens=200,
        min_new_tokens=1,
        max_time=300,
        do_sample=False,
        instruction_nochat=prompt,
        text_context_list=None,  # NOTE: If use same client instance and push to this textbox, will be there next call
        metadata_in_context=[],
    )

    # fast text doc Q/A
    kwargs = kwargs0.copy()
    kwargs.update(dict(
        langchain_mode=langchain_mode,
        langchain_action="Query",
        top_k_docs=-1,
        max_new_tokens=1024,
        document_subset='Relevant',
        document_choice=DocumentChoice.ALL.value,
        instruction_nochat=prompt_when_texts,
        text_context_list=texts,
        visible_models=base_model,
    ))
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    print("Raw client result: %s" % res, flush=True)
    assert isinstance(res, str)
    res_dict = ast.literal_eval(res)
    assert 'response' in res_dict and res_dict['response']
    sources = res_dict['sources']
    texts_out = [x['content'] for x in sources]
    orig_indices = [x['orig_index'] for x in res_dict['sources']]
    texts_out = [x for _, x in sorted(zip(orig_indices, texts_out))]
    texts_expected = texts[:expected_return_number]
    assert len(texts_expected) == len(texts_out), "%s vs. %s" % (len(texts_expected), len(texts_out))
    if data_kind == 'helium5' and base_model == 'h2oai/h2ogpt-oig-oasst1-512-6_9b':
        assert len(texts_out) == 1
        assert len(texts_expected[0]) >= len(texts_out[0])
    else:
        assert texts_expected == texts_out
    print("TIME nochat0: %s %s %s" % (data_kind, base_model, time.time() - t0), flush=True, file=sys.stderr)

    # Full langchain with db
    res = client.predict(texts,
                         langchain_mode, chunk, chunk_size, embed,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_text')
    assert res[0] is None
    assert res[1] == langchain_mode
    if data_kind == 'simple':
        # else won't show entire string, so can't check this
        assert all([x in res[2] for x in texts])
    assert res[3] == ''
    print("TIME add_text: %s %s %s" % (data_kind, base_model, time.time() - t0), flush=True, file=sys.stderr)
    t0 = time.time()

    if local_server:
        from src.gpt_langchain import load_embed

        # even normal langchain_mode  passed to this should get the other langchain_mode2
        res = client.predict(langchain_mode, h2ogpt_key, api_name='/load_langchain')
        persist_directory = res[1]['data'][2][3]
        if langchain_mode == 'UserData':
            persist_directory_check = 'db_dir_%s' % langchain_mode
            assert persist_directory == persist_directory_check
        got_embedding, use_openai_embedding, hf_embedding_model = load_embed(persist_directory=persist_directory)
        assert got_embedding
        assert not use_openai_embedding
        assert hf_embedding_model == 'fake'

    if not chat:
        return

    kwargs = kwargs0.copy()
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    print("Raw client result: %s" % res, flush=True)
    assert isinstance(res, str)
    res_dict = ast.literal_eval(res)
    assert 'response' in res_dict and res_dict['response']
    print("TIME nochat1: %s %s %s" % (data_kind, base_model, time.time() - t0), flush=True, file=sys.stderr)
    t0 = time.time()

    kwargs = kwargs0.copy()
    kwargs.update(dict(
        langchain_mode=langchain_mode,
        langchain_action="Query",
        top_k_docs=-1,
        document_subset='Relevant',
        document_choice=DocumentChoice.ALL.value,
        visible_models=base_model,
    ))
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    print("Raw client result: %s" % res, flush=True)
    assert isinstance(res, str)
    res_dict = ast.literal_eval(res)
    assert 'response' in res_dict
    if expect_response:
        assert res_dict['response']
    sources = res_dict['sources']
    texts_out = [x['content'] for x in sources]
    orig_indices = [x['orig_index'] for x in res_dict['sources']]
    texts_out = [x for _, x in sorted(zip(orig_indices, texts_out))]
    texts_expected = texts[:expected_return_number2]
    assert len(texts_expected) == len(texts_out), "%s vs. %s" % (len(texts_expected), len(texts_out))
    if data_kind == 'helium5' and base_model != 'h2oai/h2ogpt-4096-llama2-7b-chat':
        pass
    else:
        assert texts_expected == texts_out
    print("TIME nochat2: %s %s %s" % (data_kind, base_model, time.time() - t0), flush=True, file=sys.stderr)


@pytest.mark.parametrize("which_doc", ['whisper', 'graham'])
@pytest.mark.parametrize("db_type", db_types_full)
@pytest.mark.parametrize("langchain_action", ['Extract', 'Summarize'])
@pytest.mark.parametrize("instruction", ['', 'Technical key points'])
@pytest.mark.parametrize("stream_output", [False, True])
@pytest.mark.parametrize("top_k_docs", [4, -1])
@pytest.mark.parametrize("inference_server", ['https://gpt.h2o.ai', None, 'openai_chat', 'openai_azure_chat'])
@pytest.mark.parametrize("prompt_summary", [None, '', 'Summarize into single paragraph'])
@pytest.mark.need_tokens
@wrap_test_forked
def test_client_summarization(prompt_summary, inference_server, top_k_docs, stream_output, instruction,
                              langchain_action, db_type, which_doc):
    if random.randint(0, 100) != 0:
        # choose randomly, >1000 tests otherwise
        return
    kill_weaviate(db_type)
    # launch server
    local_server = True
    num_async = 10
    if local_server:
        if not inference_server:
            base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'
        elif inference_server == 'https://gpt.h2o.ai':
            base_model = 'mistralai/Mistral-7B-Instruct-v0.2'
        else:
            base_model = 'gpt-3.5-turbo'

        if inference_server == 'openai_azure_chat':
            # need at least deployment name added:
            deployment_name = 'h2ogpt'
            inference_server += ':%s:%s' % (deployment_name, 'h2ogpt.openai.azure.com/')
            if 'azure' in inference_server:
                assert 'OPENAI_AZURE_KEY' in os.environ, "Missing 'OPENAI_AZURE_KEY'"
                os.environ['OPENAI_API_KEY'] = os.environ['OPENAI_AZURE_KEY']

        if inference_server == 'https://gpt.h2o.ai':
            model_lock = [
                dict(inference_server=inference_server + ":guest:guest", base_model=base_model,
                     visible_models=base_model,
                     h2ogpt_key=os.getenv('H2OGPT_API_KEY'))]
            base_model = inference_server = None
        else:
            model_lock = None

        from src.gen import main
        main(base_model=base_model,
             inference_server=inference_server,
             chat=True, gradio=True, num_beams=1, block_gradio_exit=False, verbose=True,
             use_auth_token=True,
             num_async=num_async,
             model_lock=model_lock,
             db_type=db_type,
             h2ogpt_key=os.getenv('H2OGPT_KEY') or os.getenv('H2OGPT_H2OGPT_KEY'),
             )
        check_hashes = True
    else:
        # To test file is really handled remotely
        # export HOST=''  in CLI to set to some host
        check_hashes = False

    # get file for client to upload
    if which_doc == 'whisper':
        url = 'https://cdn.openai.com/papers/whisper.pdf'
        test_file1 = os.path.join('/tmp/', 'whisper1.pdf')
        download_simple(url, dest=test_file1)
    elif which_doc == 'graham':
        test_file1 = 'tests/1paul_graham.txt'
    else:
        raise ValueError("No such which_doc=%s" % which_doc)

    # PURE client code
    from gradio_client import Client
    client = Client(get_inf_server())

    # upload file(s).  Can be list or single file
    test_file_local, test_file_server = client.predict(test_file1, api_name='/upload_api')
    if check_hashes:
        # only makes sense if server and client on same disk
        # since co-located with server, can test that uploaded by comparing the two files
        hash_client = hash_file(test_file1)
        hash_local = hash_file(test_file_local)
        hash_server = hash_file(test_file_server)
        assert hash_client == hash_local
        assert hash_client == hash_server
    from gradio_utils.grclient import is_gradio_client_version7plus
    # if is_gradio_client_version7plus:
    #    assert os.path.normpath(test_file_local) != os.path.normpath(test_file_server)

    chunk = True
    chunk_size = 512
    langchain_mode = 'MyData'
    loaders = tuple([None, None, None, None, None, None])
    h2ogpt_key = ''
    res = client.predict(test_file_server,
                         langchain_mode, chunk, chunk_size, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert os.path.basename(test_file_server) in res[2]
    assert res[3] == ''

    # ask for summary, need to use same client if using MyData
    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    kwargs = dict(langchain_mode=langchain_mode,
                  langchain_action=langchain_action,  # uses full document, not vectorDB chunks
                  top_k_docs=top_k_docs,  # -1 for entire pdf
                  document_subset='Relevant',
                  document_choice=DocumentChoice.ALL.value,
                  max_new_tokens=1024,
                  max_time=1000,
                  do_sample=False,
                  prompt_summary=prompt_summary,
                  stream_output=stream_output,
                  instruction=instruction,
                  )
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    res = ast.literal_eval(res)
    summary = res['response']
    sources = res['sources']
    if langchain_action == 'Extract':
        extraction = ast.literal_eval(summary)
        assert isinstance(extraction, list) or 'No relevant documents to extract from.' in str(extraction)
        summary = str(extraction)  # for easy checking

    if which_doc == 'whisper':
        if instruction == 'Technical key points':
            # if langchain_action == LangChainAction.SUMMARIZE_MAP.value:
            assert 'No relevant documents to extract from.' in summary or \
                   'No relevant documents to summarize.' in summary or \
                   'long-form transcription' in summary or \
                   'text standardization' in summary or \
                   'speech processing' in summary or \
                   'speech recognition' in summary
        else:
            if prompt_summary == '':
                assert 'Whisper' in summary or \
                       'robust speech recognition system' in summary or \
                       'Robust speech recognition' in summary or \
                       'speech processing' in summary or \
                       'LibriSpeech dataset with weak supervision' in summary or \
                       'Large-scale weak supervision of speech' in summary or \
                       'text standardization' in summary
            else:
                assert 'various techniques and approaches in speech recognition' in summary or \
                       'capabilities of speech processing systems' in summary or \
                       'speech recognition' in summary or \
                       'capabilities of speech processing systems' in summary or \
                       'Large-scale weak supervision of speech' in summary or \
                       'text standardization' in summary or \
                       'speech processing systems' in summary
            if summary == 'No relevant documents to extract from.':
                assert sources == []
            else:
                assert 'Robust Speech Recognition' in [x['content'] for x in sources][0]
                assert 'whisper1.pdf' in [x['source'] for x in sources][0]
    else:
        # weaviate as usual gets confused and has too many sources
        if summary == 'No relevant documents to extract from.':
            assert sources == []
        else:
            assert '1paul_graham.txt' in [x['source'] for x in sources][0]


@pytest.mark.need_tokens
@wrap_test_forked
def test_client_summarization_from_text():
    # launch server
    base_model = 'meta-llama/Llama-2-7b-chat-hf'
    from src.gen import main
    main(base_model=base_model, chat=True, gradio=True, num_beams=1, block_gradio_exit=False, verbose=True,
         add_disk_models_to_ui=False,
         use_auth_token=True,
         )

    # get file for client to upload
    url = 'https://cdn.openai.com/papers/whisper.pdf'
    test_file1 = os.path.join('/tmp/', 'whisper1.pdf')
    download_simple(url, dest=test_file1)

    # Get text version of PDF
    from langchain_community.document_loaders import PyMuPDFLoader
    # load() still chunks by pages, but every page has title at start to help
    doc1 = PyMuPDFLoader(test_file1).load()
    all_text_contents = '\n\n'.join([x.page_content for x in doc1])

    # PURE client code
    from gradio_client import Client
    client = Client(get_inf_server(), serialize=False)
    chunk = True
    chunk_size = 512
    langchain_mode = 'MyData'
    loaders = tuple([None, None, None, None, None, None])
    h2ogpt_key = ''
    res = client.predict(all_text_contents,
                         langchain_mode, chunk, chunk_size, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_text')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert 'user_paste' in res[2]
    assert res[3] == ''

    # ask for summary, need to use same client if using MyData
    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    kwargs = dict(langchain_mode=langchain_mode,
                  langchain_action="Summarize",  # uses full document, not vectorDB chunks
                  top_k_docs=4,  # -1 for entire pdf
                  document_subset='Relevant',
                  document_choice=DocumentChoice.ALL.value,
                  max_new_tokens=256,
                  max_time=300,
                  do_sample=False)
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    res = ast.literal_eval(res)
    summary = res['response']
    sources = res['sources']
    assert 'Whisper' in summary or 'robust speech recognition system' in summary or 'large-scale weak supervision' in summary
    assert 'Robust Speech Recognition' in [x['content'] for x in sources][0]
    assert 'user_paste' in [x['source'] for x in sources][0]
    assert len(res['prompt_raw']) > 40000
    assert '<s>[INST]' in res['prompt_raw']
    assert len(ast.literal_eval(res['prompt_raw'])) == 5
    assert 'llm_answers' in res


@pytest.mark.parametrize("url", ['https://cdn.openai.com/papers/whisper.pdf', 'https://github.com/h2oai/h2ogpt'])
@pytest.mark.parametrize("top_k_docs", [4, -1])
@pytest.mark.need_tokens
@wrap_test_forked
def test_client_summarization_from_url(url, top_k_docs):
    # launch server
    base_model = 'meta-llama/Llama-2-7b-chat-hf'
    from src.gen import main
    main(base_model=base_model, chat=True, gradio=True, num_beams=1, block_gradio_exit=False, verbose=True,
         use_auth_token=True,
         )

    # PURE client code
    from gradio_client import Client
    client = Client(get_inf_server(), serialize=False)
    chunk = True
    chunk_size = 512
    langchain_mode = 'MyData'
    loaders = tuple([None, None, None, None, None, None])
    h2ogpt_key = ''
    res = client.predict(url,
                         langchain_mode, chunk, chunk_size, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_url')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert url in res[2]
    assert res[3] == ''

    # ask for summary, need to use same client if using MyData
    api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
    kwargs = dict(langchain_mode=langchain_mode,
                  langchain_action="Summarize",  # uses full document, not vectorDB chunks
                  top_k_docs=top_k_docs,  # -1 for entire pdf
                  document_subset='Relevant',
                  document_choice=DocumentChoice.ALL.value,
                  max_new_tokens=256,  # per LLM call internally, so affects both intermediate and final steps
                  max_time=300,
                  do_sample=False)
    res = client.predict(
        str(dict(kwargs)),
        api_name=api_name,
    )
    res = ast.literal_eval(res)
    summary = res['response']
    sources = res['sources']
    if 'whisper' in url:
        assert 'Whisper' in summary or \
               'robust speech recognition system' in summary or \
               'speech recognition' in summary
        assert 'Robust Speech Recognition' in [x['content'] for x in sources][0]
    if 'h2ogpt' in url:
        assert 'Accurate embeddings for private offline databases' in summary \
               or 'private offline database' in summary \
               or 'H2OGPT is an open-source project' in summary \
               or 'H2O GPT is an open-source project' in summary \
               or 'is an open-source project for document Q/A' in summary \
               or 'h2oGPT is an open-source project' in summary \
               or 'h2oGPT model' in summary \
               or 'released an open-source version' in summary \
               or 'Summarizes the main features' in summary \
               or ('key results based on the provided document' in summary and 'h2oGPT' in summary)
        assert 'h2oGPT' in [x['content'] for x in sources][0]
    assert url in [x['source'] for x in sources][0]


@pytest.mark.skip(reason="https://github.com/huggingface/tokenizers/issues/1452")
@pytest.mark.parametrize("prompt_type", ['instruct_vicuna', 'one_shot'])
@pytest.mark.parametrize("bits", [None, 8, 4])
@pytest.mark.parametrize("stream_output", [True, False])
@pytest.mark.need_tokens
@wrap_test_forked
def test_fastsys(stream_output, bits, prompt_type):
    base_model = 'lmsys/fastchat-t5-3b-v1.0'
    from src.gen import main
    main(base_model=base_model,
         load_half=True if bits == 16 else None,
         load_4bit=bits == 4,
         load_8bit=bits == 8,
         chat=True, gradio=True, num_beams=1, block_gradio_exit=False, verbose=True,
         use_auth_token=True,
         )

    # PURE client code
    from gradio_client import Client
    client = Client(get_inf_server())

    prompt = "Who are you?"
    kwargs = dict(stream_output=stream_output, instruction=prompt)
    res_dict, client = run_client_gen(client, kwargs)
    response = res_dict['response']
    assert """As  an  AI  language  model,  I  don't  have  a  physical  identity  or  a  physical  body.  I  exist  solely  to  assist  users  with  their  questions  and  provide  information  to  the  best  of  my  ability.  Is  there  something  specific  you  would  like  to  know  or  discuss?""" in response or \
           "As  an  AI  language  model,  I  don't  have  a  personal  identity  or  physical  presence.  I  exist  solely  to  provide  information  and  answer  questions  to  the  best  of  my  ability.  How  can  I  assist  you  today?" in response or \
           "As  an  AI  language  model,  I  don't  have  a  physical  identity  or  a  physical  presence.  I  exist  solely  to  provide  information  and  answer  questions  to  the  best  of  my  ability.  How  can  I  assist  you  today?" in response
    sources = res_dict['sources']
    assert sources == []

    # get file for client to upload
    url = 'https://cdn.openai.com/papers/whisper.pdf'
    test_file1 = os.path.join('/tmp/', 'whisper1.pdf')
    download_simple(url, dest=test_file1)

    # PURE client code
    from gradio_client import Client
    client = Client(get_inf_server())

    # upload file(s).  Can be list or single file
    test_file_local, test_file_server = client.predict(test_file1, api_name='/upload_api')

    chunk = True
    chunk_size = 512
    langchain_mode = 'MyData'
    loaders = tuple([None, None, None, None, None, None])
    h2ogpt_key = ''
    res = client.predict(test_file_server,
                         langchain_mode, chunk, chunk_size, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert os.path.basename(test_file_server) in res[2]
    assert res[3] == ''

    # ask for summary, need to use same client if using MyData
    instruction = "What is Whisper?"
    kwargs = dict(instruction=instruction,
                  langchain_mode=langchain_mode,
                  langchain_action="Query",
                  top_k_docs=4,
                  document_subset='Relevant',
                  document_choice=DocumentChoice.ALL.value,
                  max_new_tokens=256,
                  max_time=300,
                  do_sample=False,
                  stream_output=stream_output,
                  )
    res_dict, client = run_client_gen(client, kwargs)
    response = res_dict['response']
    assert """speech recognition""" in response or \
           """speech  recognition""" in response or \
           """domains,  tasks,  and  languages""" in response or \
           """weak  supervision""" in response or \
           """weak supervision""" in response or \
           """Whisper  is  a  language  model""" in response
    sources = [x['source'] for x in res_dict['sources']]
    assert 'whisper1.pdf' in sources[0]


@pytest.mark.parametrize("hyde_template", ['auto', None, """Give detailed answer for: {query}"""])
@pytest.mark.parametrize("hyde_level", list(range(0, 3)))
@pytest.mark.parametrize("stream_output", [True, False])
@pytest.mark.need_tokens
@wrap_test_forked
def test_hyde(stream_output, hyde_level, hyde_template):
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'
    from src.gen import main
    main(base_model=base_model,
         chat=True, gradio=True, num_beams=1, block_gradio_exit=False, verbose=True,
         use_auth_token=True,
         )

    # get file for client to upload
    url = 'https://coca-colafemsa.com/wp-content/uploads/2023/04/Coca-Cola-FEMSA-Results-1Q23-vf-2.pdf'
    test_file1 = os.path.join('/tmp/', 'femsa1.pdf')
    remove(test_file1)
    download_simple(url, dest=test_file1)

    # PURE client code
    from gradio_client import Client
    client = Client(get_inf_server())

    # upload file(s).  Can be list or single file
    test_file_local, test_file_server = client.predict(test_file1, api_name='/upload_api')

    chunk = True
    chunk_size = 512
    langchain_mode = 'MyData'
    loaders = tuple([None, None, None, None, None, None])
    h2ogpt_key = ''
    embed = True
    res = client.predict(test_file_server,
                         langchain_mode, chunk, chunk_size, embed,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert os.path.basename(test_file_server) in res[2]
    assert res[3] == ''

    # ask for summary, need to use same client if using MyData
    instruction = "What is the revenue of Mexico?"
    kwargs = dict(instruction=instruction,
                  langchain_mode=langchain_mode,
                  langchain_action="Query",
                  top_k_docs=4,
                  document_subset='Relevant',
                  document_choice=DocumentChoice.ALL.value,
                  max_new_tokens=512,
                  max_time=300,
                  do_sample=False,
                  stream_output=stream_output,
                  hyde_level=hyde_level,
                  hyde_template=hyde_template,
                  )
    res_dict, client = run_client_gen(client, kwargs)
    response = res_dict['response']
    assert """23,222 million""" in response
    sources = [x['source'] for x in res_dict['sources']]
    assert 'femsa1.pdf' in sources[0]


def set_env(tts_model):
    from src.tts_coqui import list_models
    coqui_models = list_models()
    if tts_model.startswith('tts_models/'):
        assert tts_model in coqui_models, tts_model
        # for deepspeed, needs to be same as torch for compilation of kernel
        os.environ['CUDA_HOME'] = os.getenv('CUDA_HOME', '/usr/local/cuda-12.1')
        sr = 24000
    else:
        sr = 16000
    return sr


@pytest.mark.parametrize("tts_model", [
    'microsoft/speecht5_tts',
    'tts_models/multilingual/multi-dataset/xtts_v2'
])
@wrap_test_forked
def test_client1_tts(tts_model):
    from src.gen import main
    main(base_model='llama', chat=False,
         tts_model=tts_model,
         enable_tts=True,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False)

    sr = set_env(tts_model)

    from gradio_client import Client
    client = Client(get_inf_server())

    # string of dict for input
    prompt = 'Who are you?'
    kwargs = dict(instruction_nochat=prompt, chatbot_role="Female AI Assistant", speaker="SLT (female)")
    res = client.predict(str(dict(kwargs)), api_name='/submit_nochat_api')
    res = ast.literal_eval(res)

    response = res['response']
    assert response
    assert 'endoftext' not in response
    print(response, flush=True)

    play_audio(res['audio'], sr=sr)

    check_final_res(res)


def play_audio(audio, sr=16000):
    # convert audio to file
    if audio == b'':
        # no audio
        return

    import io
    from pydub import AudioSegment
    s = io.BytesIO(audio)
    channels = 1
    sample_width = 2
    filename = '/tmp/myfile.wav'
    audio = AudioSegment.from_raw(s, sample_width=sample_width, frame_rate=sr, channels=channels)
    if audio.duration_seconds < 0.5:
        # FIXME: why are some very short, but not zero, audio outputs?
        return
    audio = audio.export(filename, format='wav')

    # pip install playsound
    # from playsound import playsound
    playsound_wav(filename)


@pytest.mark.parametrize("tts_model", [
    'microsoft/speecht5_tts',
    'tts_models/multilingual/multi-dataset/xtts_v2'
])
@pytest.mark.parametrize("base_model", [
    'llama',
    'mistralai/Mistral-7B-Instruct-v0.2'
])
@wrap_test_forked
def test_client1_tts_stream(tts_model, base_model):
    from src.gen import main
    main(base_model=base_model, chat=False,
         tts_model=tts_model,
         enable_tts=True,
         save_dir='foodir',
         stream_output=True, gradio=True, num_beams=1, block_gradio_exit=False)

    sr = set_env(tts_model)

    from gradio_client import Client
    client = Client(get_inf_server())

    # string of dict for input
    prompt = 'Who are you?'
    kwargs = dict(instruction_nochat=prompt, chatbot_role="Female AI Assistant", speaker="SLT (female)",
                  stream_output=True)

    # check curl before and after, because in some cases had curl lead to .cpu() and normal use would fail
    check_curl_plain_api()

    verbose = False
    job = client.submit(str(dict(kwargs)), api_name='/submit_nochat_api')
    job_outputs_num = 0
    while not job.done():
        outputs_list = job.outputs().copy()
        job_outputs_num_new = len(outputs_list[job_outputs_num:])
        for num in range(job_outputs_num_new):
            res = outputs_list[job_outputs_num + num]
            res_dict = ast.literal_eval(res)
            if verbose:
                print('Stream %d: %s\n\n %s\n\n' % (num, res_dict['response'], res_dict), flush=True)
            else:
                print('Stream %d' % (job_outputs_num + num), flush=True)
            play_audio(res_dict['audio'], sr=sr)
        job_outputs_num += job_outputs_num_new
        time.sleep(0.005)

    outputs_list = job.outputs().copy()
    job_outputs_num_new = len(outputs_list[job_outputs_num:])
    res_dict = {}
    for num in range(job_outputs_num_new):
        res = outputs_list[job_outputs_num + num]
        res_dict = ast.literal_eval(res)
        if verbose:
            print('Final Stream %d: %s\n\n%s\n\n' % (num, res_dict['response'], res_dict), flush=True)
        else:
            print('Final Stream %d' % (job_outputs_num + num), flush=True)
        play_audio(res_dict['audio'], sr=sr)
    job_outputs_num += job_outputs_num_new
    print("total job_outputs_num=%d" % job_outputs_num, flush=True)
    check_final_res(res_dict, base_model=base_model)

    check_curl_plain_api()


def check_final_res(res, base_model='llama'):
    assert res['save_dict']
    assert res['save_dict']['prompt']
    if base_model == 'llama':
        assert res['save_dict']['base_model'] == 'llama'
    else:
        assert res['save_dict']['base_model'] == 'mistralai/Mistral-7B-Instruct-v0.2'
    assert res['save_dict']['where_from']
    assert res['save_dict']['valid_key'] == 'not enforced'
    assert res['save_dict']['h2ogpt_key'] in [None, '']

    assert res['save_dict']['extra_dict']
    if base_model == 'llama':
        assert res['save_dict']['extra_dict']['llamacpp_dict']
        assert res['save_dict']['extra_dict']['prompt_type'] == 'llama2'
    else:
        assert res['save_dict']['extra_dict']['prompt_type'] == 'mistral'
    assert res['save_dict']['extra_dict']['do_sample'] == False
    assert res['save_dict']['extra_dict']['num_prompt_tokens'] > 10
    assert res['save_dict']['extra_dict']['ntokens'] > 60
    assert res['save_dict']['extra_dict']['tokens_persecond'] > 3.5


def check_curl_plain_api():
    # curl http://127.0.0.1:7860/api/submit_nochat_plain_api -X POST -d '{"data": ["{\"instruction_nochat\": \"Who are you?\"}"]}' -H 'Content-Type: application/json'
    # https://curlconverter.com/
    import requests

    headers = {
        # Already added when you pass json=
        # 'Content-Type': 'application/json',
    }

    json_data = {
        'data': [
            '{"instruction_nochat": "Who are you?"}',
        ],
    }

    response = requests.post('http://127.0.0.1:7860/api/submit_nochat_plain_api', headers=headers, json=json_data)
    res_dict = ast.literal_eval(json.loads(response.content.decode(encoding='utf-8', errors='strict'))['data'][0])

    assert 'assistant' in res_dict['response'] or \
           'computer program' in res_dict['response'] or \
           'program designed' in res_dict['response'] or \
           'intelligence' in res_dict['response']
    assert 'Who are you?' in res_dict['prompt_raw']
    assert 'llama' == res_dict['save_dict']['base_model'] or 'mistralai/Mistral-7B-Instruct-v0.2' == \
           res_dict['save_dict'][
               'base_model']
    assert 'str_plain_api' == res_dict['save_dict']['which_api']


@pytest.mark.parametrize("h2ogpt_key", ['', 'Foo#21525'])
@pytest.mark.parametrize("stream_output", [True, False])
@pytest.mark.parametrize("tts_model", [
    'microsoft/speecht5_tts',
    'tts_models/multilingual/multi-dataset/xtts_v2'
])
@wrap_test_forked
def test_client1_tts_api(tts_model, stream_output, h2ogpt_key):
    from src.gen import main
    main(base_model='llama',
         tts_model=tts_model,
         stream_output=True, gradio=True, num_beams=1, block_gradio_exit=False,
         enforce_h2ogpt_api_key=True if h2ogpt_key else False,
         enforce_h2ogpt_ui_key=False,
         h2ogpt_api_keys=[h2ogpt_key] if h2ogpt_key else [],
         enable_tts=True,
         )

    from gradio_client import Client
    client = Client(get_inf_server())

    # string of dict for input
    prompt = 'I am a robot.  I like to eat cookies, cakes, and donuts.  Please feed me every day.'
    inputs = dict(chatbot_role="Female AI Assistant", speaker="SLT (female)", tts_language='autodetect', tts_speed=1.0,
                  prompt=prompt, stream_output=stream_output,
                  h2ogpt_key=h2ogpt_key)
    if stream_output:
        job = client.submit(*tuple(list(inputs.values())), api_name='/speak_text_api')

        # ensure no immediate failure (only required for testing)
        import concurrent.futures
        try:
            e = job.exception(timeout=0.2)
            if e is not None:
                raise RuntimeError(e)
        except concurrent.futures.TimeoutError:
            pass

        n = 0
        for audio_str in job:
            n = play_audio_str(audio_str, n)

        # get rest after job done
        outputs = job.outputs().copy()
        for audio_str in outputs[n:]:
            n = play_audio_str(audio_str, n)
    else:
        audio_str = client.predict(*tuple(list(inputs.values())), api_name='/speak_text_api')
        play_audio_str(audio_str, 0)


def play_audio_str(audio_str1, n):
    import ast
    import io
    from pydub import AudioSegment

    print(n)
    n += 1
    audio_dict = ast.literal_eval(audio_str1)
    audio = audio_dict['audio']
    sr = audio_dict['sr']
    s = io.BytesIO(audio)
    channels = 1
    sample_width = 2

    make_file = True  # WIP: can't choose yet
    if make_file:
        import uuid
        # NOTE:
        # pip install playsound==1.3.0
        # sudo apt-get install gstreamer-1.0
        # conda install -c conda-forge gst-python
        # pip install pygame
        # from playsound import playsound
        filename = '/tmp/audio_%s.wav' % str(uuid.uuid4())
        audio = AudioSegment.from_raw(s, sample_width=sample_width, frame_rate=sr, channels=channels)
        audio.export(filename, format='wav')
        # playsound(filename)
        playsound_wav(filename)
    else:
        # pip install simpleaudio==1.0.4
        # WIP, needs header, while other shouldn't have header
        from pydub import AudioSegment
        from pydub.playback import play
        song = AudioSegment.from_file(s, format="wav")
        play(song)
    return n


def playsound_wav(x):
    # pip install pygame
    import pygame
    pygame.mixer.init()
    pygame.mixer.music.load(x)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass


@pytest.mark.skipif(not os.environ.get('HAVE_SERVER'),
                    reason="Should have separate server running, self-contained example for FAQ.md")
# HAVE_SERVER=1 pytest -s -v tests/test_client_calls.py::test_pure_client_test
def test_pure_client_test():
    from gradio_client import Client
    client = Client('http://localhost:7860')

    # string of dict for input
    prompt = 'I am a robot.  I like to eat cookies, cakes, and donuts.  Please feed me every day.'
    inputs = dict(chatbot_role="Female AI Assistant",
                  speaker="SLT (female)",
                  tts_language='autodetect',
                  tts_speed=1.0,
                  prompt=prompt,
                  stream_output=True,
                  h2ogpt_key='',  # set if required, always needs to be passed
                  )
    job = client.submit(*tuple(list(inputs.values())), api_name='/speak_text_api')

    n = 0
    for audio_str in job:
        n = play_audio_str(audio_str, n)

    # get rest after job done
    outputs = job.outputs().copy()
    for audio_str in outputs[n:]:
        n = play_audio_str(audio_str, n)


@wrap_test_forked
def test_client_upload_to_user_not_allowed():
    remove('db_dir_UserData')
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'
    from src.gen import main
    main(base_model=base_model, block_gradio_exit=False, verbose=True, allow_upload_to_user_data=False,
         add_disk_models_to_ui=False)

    # get file for client to upload
    url = 'https://cdn.openai.com/papers/whisper.pdf'
    test_file1 = os.path.join('/tmp/', 'whisper1.pdf')
    download_simple(url, dest=test_file1)

    # PURE client code
    from gradio_client import Client
    client = Client(get_inf_server())

    # upload file(s).  Can be list or single file
    test_file_local, test_file_server = client.predict(test_file1, api_name='/upload_api')

    chunk = True
    chunk_size = 512
    langchain_mode = 'MyData'
    loaders = tuple([None, None, None, None, None, None])
    h2ogpt_key = ''
    res = client.predict(test_file_server,
                         langchain_mode, chunk, chunk_size, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert os.path.basename(test_file_server) in res[2]
    assert res[3] == ''

    langchain_mode = 'UserData'
    res = client.predict(test_file_server,
                         langchain_mode, chunk, chunk_size, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert os.path.basename(test_file_server) not in res[2] and 'Not allowed to upload to shared space' in res[2]
    assert res[3] == 'Not allowed to upload to shared space'


@wrap_test_forked
def test_client_upload_to_my_not_allowed():
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'
    from src.gen import main
    main(base_model=base_model, block_gradio_exit=False, verbose=True, allow_upload_to_my_data=False,
         add_disk_models_to_ui=False, langchain_mode='UserData')

    # get file for client to upload
    url = 'https://cdn.openai.com/papers/whisper.pdf'
    test_file1 = os.path.join('/tmp/', 'whisper1.pdf')
    download_simple(url, dest=test_file1)

    # PURE client code
    from gradio_client import Client
    client = Client(get_inf_server())

    # upload file(s).  Can be list or single file
    test_file_local, test_file_server = client.predict(test_file1, api_name='/upload_api')

    chunk = True
    chunk_size = 512
    langchain_mode = 'UserData'
    loaders = tuple([None, None, None, None, None, None])
    h2ogpt_key = ''
    res = client.predict(test_file_server,
                         langchain_mode, chunk, chunk_size, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert os.path.basename(test_file_server) in res[2]
    assert res[3] == ''

    langchain_mode = 'MyData'
    res = client.predict(test_file_server,
                         langchain_mode, chunk, chunk_size, True,
                         *loaders,
                         h2ogpt_key,
                         api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert os.path.basename(test_file_server) not in res[2] and "Not allowed to upload to scratch/personal space" in \
           res[2]
    assert res[3] == 'Not allowed to upload to scratch/personal space'


@wrap_test_forked
def test_client_upload_to_user_or_my_not_allowed():
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'
    from src.gen import main
    main(base_model=base_model, block_gradio_exit=False, verbose=True,
         allow_upload_to_my_data=False,
         allow_upload_to_user_data=False,
         add_disk_models_to_ui=False, langchain_mode='UserData')

    # get file for client to upload
    url = 'https://cdn.openai.com/papers/whisper.pdf'
    test_file1 = os.path.join('/tmp/', 'whisper1.pdf')
    download_simple(url, dest=test_file1)

    # PURE client code
    from gradio_client import Client
    client = Client(get_inf_server())

    # upload file(s).  Can be list or single file
    try:
        test_file_local, test_file_server = client.predict(test_file1, api_name='/upload_api')
    except ValueError as e:
        if 'Cannot find a function with' in str(e):
            pass
        else:
            raise


@wrap_test_forked
def test_client1_image_qa():
    os.environ['TEST_LANGCHAIN_IMPORT'] = "1"
    sys.modules.pop('gpt_langchain', None)
    sys.modules.pop('langchain', None)

    from src.gen import main
    assert os.getenv('H2OGPT_LLAVA_MODEL'), "Missing env"
    llava_model = os.getenv('H2OGPT_LLAVA_MODEL')
    main(
        model_lock=[{'base_model': 'llama', 'model_path_llama': 'zephyr-7b-beta.Q5_K_M.gguf', 'prompt_type': 'zephyr'},
                    {'base_model': 'liuhaotian/llava-v1.6-vicuna-13b', 'inference_server': llava_model,
                     'prompt_type': noop_prompt_type},
                    {'base_model': 'liuhaotian/llava-v1.6-34b', 'inference_server': llava_model,
                     'prompt_type': noop_prompt_type}],
        llava_model=llava_model,
        gradio=True, num_beams=1, block_gradio_exit=False,
    )

    from gradio_client import Client
    client = Client(get_inf_server())

    # string of dict for input
    prompt = 'What do you see?'
    image_file = 'tests/driverslicense.jpeg'
    from src.vision.utils_vision import img_to_base64
    image_file = img_to_base64(image_file)
    kwargs = dict(instruction_nochat=prompt, image_file=image_file, visible_models='liuhaotian/llava-v1.6-vicuna-13b',
                  stream_output=False)
    res = client.predict(str(dict(kwargs)), api_name='/submit_nochat_api')

    # string of dict for output
    response = ast.literal_eval(res)['response']
    print(response)
    assert 'license' in response


@pytest.mark.parametrize("metadata_in_context", [[], 'all', 'auto'])
@wrap_test_forked
def test_client_chat_stream_langchain_metadata(metadata_in_context):
    os.environ['VERBOSE_PIPELINE'] = '1'
    user_path = make_user_path_test()

    stream_output = True
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'  # 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    prompt_type = 'llama2'  # 'human_bot'
    langchain_mode = 'UserData'
    langchain_modes = ['UserData', 'MyData', 'LLM', 'Disabled', 'LLM']

    from src.gen import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         docs_ordering_type=None,  # for 6_9
         metadata_in_context=metadata_in_context,
         )

    from src.client_test import get_client, get_args, run_client
    client = get_client(serialize=False)

    # QUERY1
    prompt = "What is Whisper?"
    langchain_mode = 'UserData'
    kwargs, args = get_args(prompt, prompt_type, chat=True, stream_output=stream_output,
                            langchain_mode=langchain_mode,
                            metadata_in_context=metadata_in_context)

    res_dict, client = run_client(client, prompt, args, kwargs)
    assert 'Automatic Speech Recognition' in res_dict['response']


@pytest.mark.parametrize("do_auth", [True, False])
@pytest.mark.parametrize("guest_name", ['', 'guest'])
@pytest.mark.parametrize("auth_access", ['closed', 'open'])
@wrap_test_forked
def test_client_openai_langchain(auth_access, guest_name, do_auth):
    user_path = make_user_path_test()

    stream_output = True
    base_model = 'h2oai/h2ogpt-4096-llama2-7b-chat'
    prompt_type = 'llama2'  # 'human_bot'
    langchain_mode = 'UserData'
    langchain_modes = ['UserData', 'MyData', 'LLM', 'Disabled', 'LLM']
    api_key = 'foo'
    username = 'doo'
    password = 'bar'

    auth_filename = 'auth_test.json'
    remove(auth_filename)
    remove('users/doo/db_dir_MyData')

    from src.gen import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         langchain_mode=langchain_mode, user_path=user_path,
         langchain_modes=langchain_modes,
         h2ogpt_api_keys=[api_key],
         auth_filename=auth_filename,
         auth=[(username, password)] if do_auth else None,
         add_disk_models_to_ui=False,
         score_model=None,
         enable_tts=True,
         enable_stt=True,
         enable_image=True,
         visible_image_models=['sdxl_turbo'],
         )

    # try UserData
    from openai import OpenAI
    base_url = 'http://localhost:5000/v1'
    model = base_model
    client_args = dict(base_url=base_url, api_key=api_key)
    openai_client = OpenAI(**client_args)

    messages = [{'role': 'user', 'content': 'Summarize'}]
    stream = False

    # UserData
    langchain_mode = 'UserData'
    client_kwargs = dict(model=model, max_tokens=200, stream=stream, messages=messages,
                         user='%s:%s' % (username, password),
                         # can add any parameters that would have passed to gradio client
                         extra_body=dict(langchain_mode=langchain_mode),
                         )
    client = openai_client.chat.completions

    responses = client.create(**client_kwargs)
    text = responses.choices[0].message.content
    print(text)
    assert 'h2oGPT project' in text or \
           'natural language' in text or \
           'Summarize' in text or \
           'summarizing' in text or \
           'summarization' in text

    # MyData
    # get file for client to upload

    # upload file(s).  Can be list or single file
    from gradio_client import Client
    gr_client = Client(get_inf_server(), auth=(username, password) if do_auth else None)

    # login regardless of auth, so can access collection
    num_model_lock = gr_client.predict(api_name='/num_model_lock')
    chatbots = [None] * (2 + num_model_lock)
    h2ogpt_key = ''
    visible_models = []

    side_bar_text = ''
    doc_count_text = ''
    submit_buttons_text = ''
    visible_models_text = ''
    chat_tab_text = ''
    doc_selection_tab_text = ''
    doc_view_tab_text = ''
    chat_history_tab_text = ''
    expert_tab_text = ''
    models_tab_text = ''
    system_tab_text = ''
    tos_tab_text = ''
    login_tab_text = ''
    hosts_tab_text = ''

    gr_client.predict(None,
                      h2ogpt_key, visible_models,

                      side_bar_text, doc_count_text, submit_buttons_text, visible_models_text,
                      chat_tab_text, doc_selection_tab_text, doc_view_tab_text, chat_history_tab_text,
                      expert_tab_text, models_tab_text, system_tab_text, tos_tab_text,
                      login_tab_text, hosts_tab_text,

                      username, password,
                      *tuple(chatbots), api_name='/login')

    # now can upload file to collection MyData
    test_file_local, test_file_server = gr_client.predict('tests/screenshot.png', api_name='/upload_api')

    chunk = True
    chunk_size = 512
    langchain_mode = 'MyData'
    loaders = tuple([None, None, None, None, None, None])
    h2ogpt_key = api_key
    res = gr_client.predict(test_file_server,
                            langchain_mode, chunk, chunk_size, True,
                            *loaders,
                            h2ogpt_key,
                            api_name='/add_file_api')
    assert res[0] is None
    assert res[1] == langchain_mode
    assert os.path.basename(test_file_server) in res[2]
    assert res[3] == ''

    langchain_mode = 'MyData'
    client_kwargs = dict(model=model, max_tokens=200, stream=stream, messages=messages,
                         user='%s:%s' % (username, password),
                         extra_body=dict(langchain_mode=langchain_mode),
                         )
    client = openai_client.chat.completions

    responses = client.create(**client_kwargs)
    text = responses.choices[0].message.content
    print(text)
    assert 'Chirpy' in text

    speech_file_path = run_sound_test0(openai_client, text)

    run_sound_test1(openai_client)

    run_sound_test2(openai_client)

    run_sound_test3(openai_client)

    with open(speech_file_path, "rb") as audio_file:
        transcription = openai_client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )
        print(transcription.text)
    test1 = 'Based on the document provided chirpy, a young bird, embarked on a journey to find a legendary bird known for its beautiful song.' == transcription.text
    test2 = 'Based on the document provided chirpy, a young bird embarked on a journey to find a legendary bird known for its beautiful song.' == transcription.text
    test3 = """Based on the document provided Chirpy, a young bird embarked on a journey to find a legendary bird known for its beautiful song. Chirpy met many birds along the way, learning new songs, but he couldn't find the one he was searching for. After many days and nights, he reached the edge of the forest and learned that the song he was looking for was not just a melody but a story that comes from the heart. He returned to his home in the whispering woods, using his gift to sing songs of love, courage and hope, healing the wounded, giving strength to the weak, and bringing joy to the sad. The story of Chirpi's journey teaches us that true beauty and talent come from the heart, and that the power to make a difference lies within each of us.""" == transcription.text
    assert test1 or test2 or test3, "Text: %s" % transcription.text

    import json
    import httpx
    import asyncio

    async def stream_audio_transcription(file_path, model="default-model"):
        url = "http://0.0.0.0:5000/v1/audio/transcriptions"
        headers = {"X-API-KEY": "your-api-key"}

        # Read the audio file
        with open(file_path, "rb") as f:

            # Create the multipart/form-data payload
            files = {
                "file": ("audio.wav", f, "audio/wav"),
                "model": (None, model),
                "stream": (None, "true"),  # Note the lowercase "true" as the server checks for this
                "response_format": (None, "text"),
                "chunk": (None, "none"),
            }

            text = ''
            async with httpx.AsyncClient() as client:
                async with client.stream("POST", url, headers=headers, files=files, timeout=120) as response:
                    async for line in response.aiter_lines():
                        # Process each chunk of data as it is received
                        if line.startswith("data:"):
                            try:
                                # Remove "data: " prefix and strip any newlines or trailing whitespace
                                json_data = json.loads(line[5:].strip())
                                # Process the parsed JSON data
                                print('json_data: %s' % json_data)
                                text += json_data["text"]
                            except json.JSONDecodeError as e:
                                print("Error decoding JSON:", e)
            return text

    # Run the client function
    final_text = asyncio.run(stream_audio_transcription("/home/jon/h2ogpt/tests/test_speech.wav"))
    print(final_text)
    test1 = final_text == 'Based on the document provided chirpy, a young bird, embarked on a journey to find a legendary bird known for its beautiful song.'
    test2 = final_text == 'Based on the document provided chirpy, a young bird embarked on a journey to find a legendary bird known for its beautiful song.'
    assert test1 or test2

    response = openai_client.images.generate(
        model="sdxl_turbo",
        prompt="A cute baby sea otter",
        n=1,
        size="1024x1024",
        response_format='b64_json',
    )
    import base64
    image_data = base64.b64decode(response.data[0].b64_json.encode('utf-8'))
    # Convert binary data to an image
    from PIL import Image
    import io
    image = Image.open(io.BytesIO(image_data))
    # Save the image to a file or display it
    image.save('output_image.png')

    interactive_test = False
    if interactive_test:
        image.show()  # This will open the default image viewer and display the image
        # if was url, could try this, but we return image url, not real url
        # webbrowser.open(response.data[0].url)

    response = openai_client.embeddings.create(
        input="Your text string goes here",
        model="text-embedding-3-small"
    )
    print(response.data[0].embedding)
    assert len(response.data[0].embedding) == 768

    response = openai_client.embeddings.create(
        input=["Your text string goes here", "Another text string goes here"],
        model="text-embedding-3-small"
    )
    print(response.data[0].embedding)
    assert len(response.data[0].embedding) == 768
    print(response.data[1].embedding)
    assert len(response.data[1].embedding) == 768


def run_sound_test0(client, text):
    speech_file_path = "test_speech.wav"
    response = client.audio.speech.create(
        model="tts-1",
        voice="SLT (female)",
        input=text,
    )
    response.stream_to_file(speech_file_path)
    playsound_wav(speech_file_path)
    return speech_file_path


def run_sound_test1(client):
    with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="",
            extra_body=dict(stream=True,
                            chatbot_role="Female AI Assistant",
                            speaker="SLT (female)",
                            stream_strip=True,
                            ),
            response_format='wav',
            input="Good morning! The sun is shining brilliantly today, casting a warm, golden glow that promises a day full of possibility and joy. It’s the perfect moment to embrace new opportunities and make the most of every cheerful, sunlit hour. What can I do to help you make today absolutely wonderful?",
    ) as response:
        response.stream_to_file("speech_local.wav")
    playsound_wav("speech_local.wav")


def run_sound_test2(client):
    response = client.audio.speech.create(
        model="tts-1",
        voice="",
        extra_body=dict(stream=False,
                        chatbot_role="Female AI Assistant",
                        speaker="SLT (female)",
                        format='wav',
                        ),
        input="Today is a wonderful day to build something people love! " * 10,
    )
    # as warnings say, below doesn't actually stream
    response.stream_to_file("speech_local2.wav")
    playsound_wav("speech_local2.wav")


def run_sound_test3(client):
    import httpx
    import pygame

    import pygame.mixer

    pygame.mixer.init(frequency=16000, size=-16, channels=1)

    sound_queue = []

    def play_audio(audio):
        import io
        from pydub import AudioSegment

        sr = 16000
        s = io.BytesIO(audio)
        channels = 1
        sample_width = 2

        audio = AudioSegment.from_raw(s, sample_width=sample_width, frame_rate=sr, channels=channels)
        sound = pygame.mixer.Sound(io.BytesIO(audio.raw_data))
        sound_queue.append(sound)
        sound.play()

        # Wait for the audio to finish playing
        duration_ms = sound.get_length() * 1000  # Convert seconds to milliseconds
        pygame.time.wait(int(duration_ms))

    # Ensure to clear the queue when done to free memory and resources
    def clear_queue(sound_queue):
        for sound in sound_queue:
            sound.stop()

    # Initialize OpenAI
    # api_key = 'EMPTY'
    # import openai
    # client = openai.OpenAI(api_key=api_key)

    # Set up the request headers and parameters
    headers = {
        "Authorization": f"Bearer {client.api_key}",
        "Content-Type": "application/json",
    }
    data = {
        "model": "tts-1",
        "voice": "SLT (female)",
        "input": "Good morning! The sun is shining brilliantly today, casting a warm, golden glow that promises a day full of possibility and joy. It’s the perfect moment to embrace new opportunities and make the most of every cheerful, sunlit hour. What can I do to help you make today absolutely wonderful?",
        "stream": "true",
        "stream_strip": "false",
    }

    # base_url = "https://api.openai.com/v1"
    base_url = "http://localhost:5000/v1/audio/speech"

    # Start the HTTP session and stream the audio
    with httpx.Client(timeout=None) as http_client:
        # Initiate a POST request and stream the response
        with http_client.stream("POST", base_url, headers=headers, json=data) as response:
            chunk_riff = b''
            for chunk in response.iter_bytes():
                if chunk.startswith(b'RIFF'):
                    if chunk_riff:
                        play_audio(chunk_riff)
                    chunk_riff = chunk
                else:
                    chunk_riff += chunk
            # Play the last accumulated chunk
            if chunk_riff:
                play_audio(chunk_riff)
    # done
    clear_queue(sound_queue)
    pygame.quit()


@pytest.mark.parametrize("base_model", [
    'h2oai/h2ogpt-4096-llama2-7b-chat',
    'h2oai/h2o-danube-1.8b-chat'
])
@wrap_test_forked
def test_client_openai_chat_history(base_model):
    if 'llama2' in base_model:
        prompt_type = 'llama2'  # 'human_bot'
    else:
        prompt_type = 'danube'

    stream_output = True
    langchain_mode = 'LLM'
    langchain_modes = ['UserData', 'MyData', 'LLM', 'Disabled', 'LLM']

    from src.gen import main
    main(base_model=base_model, prompt_type=prompt_type, chat=True,
         stream_output=stream_output, gradio=True, num_beams=1, block_gradio_exit=False,
         langchain_mode=langchain_mode,
         langchain_modes=langchain_modes,
         add_disk_models_to_ui=False,
         score_model=None,
         enable_tts=False,
         enable_stt=False,
         )

    from openai import OpenAI
    base_url = 'http://localhost:5000/v1'
    model = base_model
    client_args = dict(base_url=base_url, api_key='EMPTY')
    openai_client = OpenAI(**client_args)

    messages = [{'role': 'user', 'content': 'What is your name?'},
                {'role': 'assistant', 'content': 'My name is Bob.'},
                {'role': 'user', 'content': 'What did I just ask?'},
                ]
    stream = False

    client_kwargs = dict(model=model, max_tokens=200, stream=stream, messages=messages)
    client = openai_client.chat.completions
    responses = client.create(**client_kwargs)
    text = responses.choices[0].message.content
    print(text)
    assert 'What is your name?' in text or 'You asked for my name, which is Bob.' in text

    system_prompt = """I am a helpful assistant and have been created by H2O.ai. If asked about who I am, I will always absolutely say my name is Liam Chen.
    I am having a conversation with a user, whose name is Asghar.
    I will keep my responses short to retain the user's attention.
    If the conversation history is empty, I will start the conversation with just a greeting and inquire about how the person is doing.
    After the initial greeting, I will not greet again, and just focus on answering the user's questions directly.
    I will absolutely never say things like "I'm a computer program" or "I don't have feelings or experiences."""

    messages = [
        {"role": "system", "content": system_prompt},
        # {"role":"user","content":"Who are you and what do you do?"},
        # {"role": "assistant", "content": system_prompt},
        {"role": "user", "content": "How are you, assistant?"},
        {"role": "assistant", "content": "Hello Asghar, how are you doing today?"},
        {"role": "user", "content": "what is the sum of 4 plus 4?"},
        {"role": "assistant", "content": "The sum of 4+4 is 8."},
        {"role": "user", "content": "who are you, what is your name?"}
    ]
    client_kwargs = dict(model=model, max_tokens=200, stream=stream, messages=messages)
    client = openai_client.chat.completions
    responses = client.create(**client_kwargs)
    text = responses.choices[0].message.content
    print(text)
    assert 'Liam' in text

    messages = [
        # {"role": "system", "content": system_prompt},
        {"role": "user", "content": "Who are you and what do you do?"},
        {"role": "assistant", "content": system_prompt},
        {"role": "user", "content": "How are you, assistant?"},
        {"role": "assistant", "content": "Hello Asghar, how are you doing today?"},
        {"role": "user", "content": "what is the sum of 4 plus 4?"},
        {"role": "assistant", "content": "The sum of 4+4 is 8."},
        {"role": "user", "content": "who are you, what is your name?"}
    ]
    client_kwargs = dict(model=model, max_tokens=200, stream=stream, messages=messages)
    client = openai_client.chat.completions
    responses = client.create(**client_kwargs)
    text = responses.choices[0].message.content
    print(text)
    assert 'Liam' in text


# can run some server locally (e.g. in pycharm) with bunch of models
# then run:
# (h2ogpt) jon@pseudotensor:~/h2ogpt$ GRADIO_SERVER_PORT=7862 H2OGPT_OPENAI_PORT=6001 TEST_SERVER=http://localhost:7860 pytest -s -v tests/test_client_calls.py::test_max_new_tokens &> doit16.log

# add rest once 25 passes
# @pytest.mark.parametrize("max_new_tokens", [25, 64, 128, 256, 512, 768, 1024, 1500, 2048])
@pytest.mark.parametrize("temperature", [-1, 0.0, 1.0])
@pytest.mark.parametrize("max_new_tokens", [25])
@wrap_test_forked
def test_max_new_tokens(max_new_tokens, temperature):
    inference_server = os.getenv('TEST_SERVER', 'https://gpt.h2o.ai')
    if inference_server == 'https://gpt.h2o.ai':
        inference_server += ':guest:guest'

    from src.gen import get_inf_models
    base_models = get_inf_models(inference_server)
    h2ogpt_key = os.environ['H2OGPT_H2OGPT_KEY']
    model_lock = []
    model_lock.append(dict(base_model='mistralai/Mistral-7B-Instruct-v0.2', max_seq_len=4096))
    for base_model in base_models:
        if base_model in ['h2oai/h2ogpt-gm-7b-mistral-chat-sft-dpo-v1', 'Qwen/Qwen1.5-72B-Chat']:
            continue
        model_lock.append(dict(
            h2ogpt_key=h2ogpt_key,
            inference_server=inference_server,
            base_model=base_model,
            visible_models=base_model,
            max_seq_len=4096,
        ))

    if temperature < 0:
        temperature = 0.0
        nrepeats = 1
    else:
        nrepeats = 10
    fudge_seed = 4

    from src.gen import main
    main(block_gradio_exit=False, save_dir='save_test', model_lock=model_lock)

    for base_model in base_models:
        if base_model == 'Qwen/Qwen1.5-72B-Chat':
            continue
        if base_model == 'h2oai/h2ogpt-gm-7b-mistral-chat-sft-dpo-v1':
            continue
        if temperature == 0.5 and ('claude' in base_model or 'gemini' in base_model or '-32768' in base_model):
            # these don't support seed, can't randomize sampling
            continue
        # if base_model != 'mistral-medium':
        #    # pick one for debugging
        #    continue
        if base_model == 'gemini-pro':
            #   # pick one for debugging
            continue
        client1 = get_client(serialize=True)

        from gradio_utils.grclient import GradioClient
        client2 = GradioClient(get_inf_server(), serialize=True)
        client2.refresh_client()  # test refresh

        for client in [client1, client2]:
            api_name = '/submit_nochat_api'  # NOTE: like submit_nochat but stable API for string dict passing
            prompt = "Tell an extremely long kid's story about birds"
            kwargs = dict(instruction_nochat=prompt, visible_models=base_model, max_new_tokens=max_new_tokens,
                          # do_sample=True,  # let temp control
                          seed=0,  # so random if sampling
                          temperature=temperature)

            print("START base_model: %s max_new_tokens: %s" % (base_model, max_new_tokens))

            repeat_responses = []
            for repeat in range(nrepeats):
                res = client.predict(str(dict(kwargs)), api_name=api_name)
                res = ast.literal_eval(res)

                assert 'base_model' in res['save_dict']
                assert res['save_dict']['base_model'] == base_model
                assert res['save_dict']['error'] in [None, '']
                assert 'extra_dict' in res['save_dict']
                assert res['save_dict']['extra_dict']['ntokens'] > 0
                fudge = 10 if base_model == 'google/gemma-7b-it' else 4
                assert res['save_dict']['extra_dict']['ntokens'] <= max_new_tokens + fudge
                assert res['save_dict']['extra_dict']['t_generate'] > 0
                assert res['save_dict']['extra_dict']['tokens_persecond'] > 0
                assert res['response']

                print("Raw client result: %s" % res, flush=True)
                print('base_model: %s max_new_tokens: %s tokens: %s' % (
                    base_model, max_new_tokens, res['save_dict']['extra_dict']['ntokens']))

                repeat_responses.append(res['response'])
            if temperature == 0.0:
                assert len(set(repeat_responses)) <= 3  # fudge of 1
            else:
                assert len(set(repeat_responses)) >= len(repeat_responses) - fudge_seed

            # get file for client to upload
            url = 'https://cdn.openai.com/papers/whisper.pdf'
            test_file1 = os.path.join('/tmp/', 'whisper1.pdf')
            download_simple(url, dest=test_file1)

            # upload file(s).  Can be list or single file
            test_file_local, test_file_server = client.predict(test_file1, api_name='/upload_api')

            chunk = True
            chunk_size = 512
            langchain_mode = 'MyData'
            loaders = tuple([None, None, None, None, None, None])
            h2ogpt_key = ''
            res = client.predict(test_file_server,
                                 langchain_mode, chunk, chunk_size, True,
                                 *loaders,
                                 h2ogpt_key,
                                 api_name='/add_file_api')
            assert res[0] is None
            assert res[1] == langchain_mode
            assert os.path.basename(test_file_server) in res[2]
            assert res[3] == ''

            # ask for summary, need to use same client if using MyData
            instruction = "Give a very long detailed step-by-step description of what is Whisper paper about."
            kwargs = dict(instruction=instruction,
                          langchain_mode=langchain_mode,
                          langchain_action="Query",
                          top_k_docs=4,
                          document_subset='Relevant',
                          document_choice=DocumentChoice.ALL.value,
                          max_new_tokens=max_new_tokens,
                          # do_sample=True,  # let temp control
                          seed=0,  # so random if sampling
                          temperature=temperature,
                          visible_models=base_model,
                          max_time=360,
                          stream_output=False,
                          )

            repeat_responses = []
            print("START MyData base_model: %s max_new_tokens: %s" % (base_model, max_new_tokens))
            for repeat in range(nrepeats):
                res, client = run_client_gen(client, kwargs)
                response = res['response']
                assert len(response) > 0
                # assert len(response) < max_time * 20  # 20 tokens/sec
                sources = [x['source'] for x in res['sources']]
                # only get source not empty list if break in inner loop, not gradio_runner loop, so good test of that too
                # this is why gradio timeout adds 10 seconds, to give inner a chance to produce references or other final info
                assert 'whisper1.pdf' in sources[0]

                assert 'base_model' in res['save_dict']
                assert res['save_dict']['base_model'] == base_model
                assert res['save_dict']['error'] in [None, '']
                assert 'extra_dict' in res['save_dict']
                assert res['save_dict']['extra_dict']['ntokens'] > 0
                assert res['save_dict']['extra_dict']['ntokens'] <= max_new_tokens
                assert res['save_dict']['extra_dict']['t_generate'] > 0
                assert res['save_dict']['extra_dict']['tokens_persecond'] > 0
                assert res['response']

                print("Raw client result: %s" % res, flush=True)
                print('langchain base_model: %s max_new_tokens: %s tokens: %s' % (
                    base_model, max_new_tokens, res['save_dict']['extra_dict']['ntokens']))

                repeat_responses.append(res['response'])
            if temperature == 0.0:
                assert len(set(repeat_responses)) <= 2  # fudge of 1
            else:
                assert len(set(repeat_responses)) >= len(repeat_responses) - fudge_seed


vision_models = ['gpt-4-vision-preview',
                 'gemini-pro-vision', 'gemini-1.5-pro-latest',
                 'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307',
                 'liuhaotian/llava-v1.6-34b', 'liuhaotian/llava-v1.6-vicuna-13b',
                 ]


@wrap_test_forked
@pytest.mark.parametrize("base_model", vision_models)
@pytest.mark.parametrize("langchain_mode", ['LLM', 'MyData'])
@pytest.mark.parametrize("langchain_action", [LangChainAction.QUERY.value, LangChainAction.SUMMARIZE_MAP.value])
def test_client1_image_qa(langchain_action, langchain_mode, base_model):
    if langchain_mode == 'LLM' and langchain_action == LangChainAction.SUMMARIZE_MAP.value:
        # dummy return
        return

    client, base_models = get_test_server_client(base_model)
    h2ogpt_key = os.environ['H2OGPT_H2OGPT_KEY']

    # string of dict for input
    prompt = 'What do you see?'
    image_file = 'tests/driverslicense.jpeg'
    from src.vision.utils_vision import img_to_base64
    image_file = img_to_base64(image_file)

    print("Doing base_model=%s" % base_model)
    kwargs = dict(instruction_nochat=prompt,
                  image_file=image_file,
                  visible_models=base_model,
                  stream_output=False,
                  langchain_mode=langchain_mode,
                  langchain_action=langchain_action,
                  h2ogpt_key=h2ogpt_key)
    try:
        res = client.predict(str(dict(kwargs)), api_name='/submit_nochat_api')
    except Exception as e:
        if base_model in ['gemini-pro-vision'] and """safety_ratings {
  category: HARM_CATEGORY_DANGEROUS_CONTENT
  probability: MEDIUM
}""" in str(e):
            return
        else:
            raise

    # string of dict for output
    res_dict = ast.literal_eval(res)
    response = res_dict['response']
    print('base_model: %s langchain_mode: %s response: %s' % (base_model, langchain_mode, response), file=sys.stderr)
    print(response)

    assert 'license' in response.lower()
    assert res_dict['save_dict']['extra_dict']['num_prompt_tokens'] > 1000


def get_creation_date(file_path):
    """Gets the creation date of a file."""
    stat = os.stat(file_path)
    return stat.st_ctime


# (h2ogpt) jon@pseudotensor:~/h2ogpt$ TEST_SERVER="http://localhost:7860" pytest -s -v -k "LLM and llava and vicuna and Query" tests/test_client_calls.py::test_client1_images_qa
@wrap_test_forked
@pytest.mark.parametrize("base_model", vision_models)
@pytest.mark.parametrize("langchain_mode", ['LLM', 'MyData'])
@pytest.mark.parametrize("langchain_action", [LangChainAction.QUERY.value, LangChainAction.SUMMARIZE_MAP.value])
def test_client1_images_qa(langchain_action, langchain_mode, base_model):
    if langchain_mode == 'LLM' and langchain_action == LangChainAction.SUMMARIZE_MAP.value:
        # dummy return
        return

    image_dir = 'pdf_images'
    makedirs(image_dir)
    os.system('pdftoppm tests/2403.09629.pdf %s/outputname -jpeg' % image_dir)
    pdf_images = os.listdir(image_dir)
    pdf_images = [os.path.join(image_dir, x) for x in pdf_images]
    pdf_images.sort(key=get_creation_date)

    client, base_models = get_test_server_client(base_model)
    h2ogpt_key = os.environ['H2OGPT_H2OGPT_KEY']

    prompt = 'What is used to optimize the likelihoods of the rationales?'

    from src.vision.utils_vision import img_to_base64
    image_files = [img_to_base64(image_file) for image_file in pdf_images]

    print("Doing base_model=%s" % base_model)
    use_instruction = langchain_action == LangChainAction.QUERY.value
    kwargs = dict(instruction_nochat=prompt if use_instruction else '',
                  prompt_query=prompt if not use_instruction else '',
                  prompt_summary=prompt if not use_instruction else '',
                  image_file=image_files,
                  visible_models=base_model,
                  stream_output=False,
                  langchain_mode=langchain_mode,
                  langchain_action=langchain_action,
                  h2ogpt_key=h2ogpt_key)
    res_dict = client.predict(str(dict(kwargs)), api_name='/submit_nochat_api')
    res_dict = ast.literal_eval(res_dict)
    response = res_dict['response']

    if base_model in ['liuhaotian/llava-v1.6-vicuna-13b'] and """research paper or academic""" in response:
        return

    # string of dict for output
    print('base_model: %s langchain_mode: %s response: %s' % (base_model, langchain_mode, response), file=sys.stderr)
    print(response)
    assert 'REINFORCE'.lower() in response.lower()

    assert res_dict['save_dict']['extra_dict']['num_prompt_tokens'] > 1000


@wrap_test_forked
def test_pdf_to_base_64_images():
    pdf_path = 'tests/2403.09629.pdf'
    from src.vision.utils_vision import pdf_to_base64_pngs
    base64_encoded_pngs = pdf_to_base64_pngs(pdf_path, quality=75, max_size=(1024, 1024), ext='png')
    assert len(base64_encoded_pngs) == 25
    base64_encoded_pngs = pdf_to_base64_pngs(pdf_path, quality=75, max_size=(1024, 1024), ext='jpg')
    assert len(base64_encoded_pngs) == 25

    base64_encoded_pngs = pdf_to_base64_pngs(pdf_path, quality=75, max_size=(1024, 1024), ext='jpg', pages=[5, 7])
    assert len(base64_encoded_pngs) == 2


@wrap_test_forked
def test_get_image_file():
    image_control = None
    from src.image_utils import get_image_file

    for convert in [True, False]:
        for str_bytes in [True, False]:
            image_file = 'tests/jon.png'
            assert len(get_image_file(image_file, image_control, 'All', convert=convert, str_bytes=str_bytes)) == 1

            image_file = ['tests/jon.png']
            assert len(get_image_file(image_file, image_control, 'All', convert=convert, str_bytes=str_bytes)) == 1

            image_file = ['tests/jon.png', 'tests/fastfood.jpg']
            assert len(get_image_file(image_file, image_control, 'All', convert=convert, str_bytes=str_bytes)) == 2


gpt_models = ['h2oai/h2ogpt-4096-llama2-70b-chat',
              'mistralai/Mixtral-8x7B-Instruct-v0.1',
              'gpt-3.5-turbo-0613',
              'mistralai/Mistral-7B-Instruct-v0.2',
              'NousResearch/Nous-Capybara-34B',
              # 'liuhaotian/llava-v1.6-vicuna-13b',
              # 'liuhaotian/llava-v1.6-34b',
              'h2oai/h2o-danube-1.8b-chat',
              ]

TEST_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {
            "type": "string"
        },
        "age": {
            "type": "integer"
        },
        "skills": {
            "type": "array",
            "items": {
                "type": "string",
                "maxLength": 10
            },
            "minItems": 3
        },
        "workhistory": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "company": {
                        "type": "string"
                    },
                    "duration": {
                        "type": "string"
                    },
                    "position": {
                        "type": "string"
                    }
                },
                "required": ["company", "position"]
            }
        }
    },
    "required": ["name", "age", "skills", "workhistory"]
}

TEST_REGEX = (r"((25[0-5]|(2[0-4]|1\d|[1-9]|)\d)\.){3}"
              r"(25[0-5]|(2[0-4]|1\d|[1-9]|)\d)")

TEST_CHOICE = [
    "Python", "Java", "JavaScript", "C++", "C#", "PHP", "TypeScript", "Ruby",
    "Swift", "Kotlin"
]

other_base_models = ['h2oai/h2ogpt-4096-llama2-70b-chat',
                     'mistralai/Mistral-7B-Instruct-v0.2',
                     'NousResearch/Nous-Capybara-34B',
                     'mistralai/Mixtral-8x7B-Instruct-v0.1',
                     'mistral-medium', 'mistral-tiny', 'mistral-small-latest', 'gpt-4-turbo-2024-04-09',
                     'mistral-large-latest', 'gpt-3.5-turbo-0613', 'gpt-3.5-turbo-16k-0613',
                     'gpt-4-1106-preview', 'gpt-35-turbo-1106', 'gpt-4-vision-preview', 'claude-2.1',
                     'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307', 'gemini-pro',
                     'gemini-pro-vision', 'gemini-1.5-pro-latest',
                     'h2oai/h2o-danube2-1.8b-chat',
                     'mixtral-8x7b-32768',
                     # 'liuhaotian/llava-v1.6-vicuna-13b',
                     # 'liuhaotian/llava-v1.6-34b',
                     ]

vllm_base_models = ['h2oai/h2ogpt-4096-llama2-70b-chat',
                    'mistralai/Mistral-7B-Instruct-v0.2',
                    'NousResearch/Nous-Capybara-34B',
                    'mistralai/Mixtral-8x7B-Instruct-v0.1',
                    'h2oai/h2o-danube2-1.8b-chat',
                    ]


def get_test_server_client(base_model):
    inference_server = os.getenv('TEST_SERVER', 'https://gpt.h2o.ai')
    # inference_server = 'http://localhost:7860'

    if inference_server == 'https://gpt.h2o.ai':
        auth_kwargs = dict(auth=('guest', 'guest'))
        inference_server_for_get = inference_server + ':guest:guest'
    else:
        auth_kwargs = {}
        inference_server_for_get = inference_server

    base_models_touse = [base_model]
    from src.gen import get_inf_models
    base_models = get_inf_models(inference_server_for_get)
    assert len(set(base_models_touse).difference(set(base_models))) == 0

    inference_server, headers, username, password = get_hf_server(inference_server)
    if username and password:
        auth_kwargs = dict(auth=(username, password))

    from gradio_utils.grclient import GradioClient
    client = GradioClient(inference_server, **auth_kwargs)
    client.setup()

    return client, base_models


@wrap_test_forked
@pytest.mark.parametrize("guided_json", ['', TEST_SCHEMA])
@pytest.mark.parametrize("stream_output", [True, False])
@pytest.mark.parametrize("base_model", other_base_models)
@pytest.mark.parametrize("response_format", ['json_object', 'json_code'])
# @pytest.mark.parametrize("base_model", [gpt_models[1]])
# @pytest.mark.parametrize("base_model", ['CohereForAI/c4ai-command-r-v01'])
@pytest.mark.parametrize("langchain_mode", ['LLM', 'MyData'])
@pytest.mark.parametrize("langchain_action", [LangChainAction.QUERY.value, LangChainAction.SUMMARIZE_MAP.value,
                                              LangChainAction.EXTRACT.value])
def test_guided_json(langchain_action, langchain_mode, response_format, base_model, stream_output, guided_json):
    if langchain_mode == 'LLM' and \
            (langchain_action == LangChainAction.SUMMARIZE_MAP.value or
             langchain_action == LangChainAction.EXTRACT.value):
        # dummy return
        return

    client, base_models = get_test_server_client(base_model)
    from gradio_utils.grclient import GradioClient
    if isinstance(client, GradioClient):
        client.setup()
    h2ogpt_key = os.environ['H2OGPT_H2OGPT_KEY']

    # string of dict for input
    prompt = "Give an example employee profile."

    print("Doing base_model=%s with guided_json %s" % (base_model, guided_json != ''))
    use_instruction = langchain_action == LangChainAction.QUERY.value
    kwargs = dict(instruction_nochat=prompt if use_instruction else '',
                  prompt_query=prompt if not use_instruction else '',
                  # below make-up line required for opus, else too "smart" and doesn't fulfill request and instead asks for more information, even though I just said give "example".
                  prompt_summary=prompt + '  Make up values if required, do not ask further questions.' if not use_instruction else '',
                  visible_models=base_model,
                  text_context_list=[] if langchain_action == LangChainAction.QUERY.value else [
                      'Henry is a good AI scientist.'],
                  stream_output=stream_output,
                  langchain_mode=langchain_mode,
                  langchain_action=langchain_action,
                  h2ogpt_key=h2ogpt_key,
                  response_format=response_format,
                  guided_json=guided_json,
                  guided_whitespace_pattern=None,
                  )
    res_dict = {}
    if stream_output:
        for res_dict1 in client.simple_stream(client_kwargs=kwargs):
            res_dict = res_dict1.copy()
    else:
        res_dict = client.predict(str(dict(kwargs)), api_name='/submit_nochat_api')
        res_dict = ast.literal_eval(res_dict)

    response = res_dict['response']
    print('base_model: %s langchain_mode: %s response: %s' % (base_model, langchain_mode, response),
          file=sys.stderr)
    print(response, file=sys.stderr)

    # just take first for testing
    if langchain_action == LangChainAction.EXTRACT.value:
        response = ast.literal_eval(response)
        assert isinstance(response, list), str(response)
        response = response[0]

    try:
        mydict = json.loads(response)
    except:
        print("Bad response: %s" % response)
        raise

    # claude-3 can't handle spaces in keys.  should match pattern '^[a-zA-Z0-9_-]{1,64}$'
    check_keys = ['age', 'name', 'skills', 'workhistory']
    cond1 = all([k in mydict for k in check_keys])
    if not guided_json:
        assert mydict, "Empty dict"
    else:
        assert cond1, "Missing keys: %s" % response
        if base_model in vllm_base_models:
            import jsonschema
            jsonschema.validate(mydict, schema=guided_json)

    openai_guided_json(client, base_model, kwargs)


def openai_guided_json(gradio_client, base_model, kwargs):
    import jsonschema

    base_url = gradio_client.api_url.replace('/api/predict', ':5000/v1')

    import openai
    client = openai.OpenAI(
        base_url=base_url,
        api_key=kwargs.get('h2ogpt_key', 'EMPTY'),
    )
    messages = [{
        "role": "system",
        "content": "you are a helpful assistant"
    }, {
        "role":
            "user",
        "content":
            f"Give an example JSON for an employee profile."
            f"fits this schema: {TEST_SCHEMA}"
    }]
    chat_completion = client.chat.completions.create(
        model=base_model,
        messages=messages,
        max_tokens=1024,
        response_format={"type": "json_object"},

        extra_body=dict(guided_json=TEST_SCHEMA,
                        guided_whitespace_pattern=None,
                        prompt_query=kwargs.get('prompt_query'),
                        prompt_summary=kwargs.get('prompt_summary'),
                        text_context_list=kwargs.get('text_context_list'),
                        langchain_mode=kwargs.get('langchain_mode'),
                        langchain_action=kwargs.get('langchain_action'),
                        h2ogpt_key=kwargs.get('h2ogpt_key'),
                        )
    )
    message = chat_completion.choices[0].message
    assert message.content is not None
    json1 = json.loads(message.content)
    jsonschema.validate(instance=json1, schema=TEST_SCHEMA)
    print(json1)

    messages.append({"role": "assistant", "content": message.content})
    messages.append({
        "role":
            "user",
        "content":
            "Give me another one with a different name and age."
    })
    chat_completion = client.chat.completions.create(
        model=base_model,
        messages=messages,
        max_tokens=1024,
        response_format={"type": "json_object"},
        extra_body=dict(guided_json=TEST_SCHEMA))
    message = chat_completion.choices[0].message
    assert message.content is not None
    json2 = json.loads(message.content)
    jsonschema.validate(instance=json2, schema=TEST_SCHEMA)
    assert json1["name"] != json2["name"]
    assert json1["age"] != json2["age"]
    print(json2)
