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
    db_types_full
from src.utils import get_githash, remove, download_simple, hash_file, makedirs, lg_to_gr, FakeTokenizer, \
    is_gradio_version4
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
    base2 = 'distilgpt2'
    model_lock = [dict(base_model=base1, prompt_type='human_bot'),
                  dict(base_model=base2, prompt_type='plain')]
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

    for prompt_type in ['plain', None, '']:
        for visible_models in [1, base2]:
            prompt = 'The sky is'
            res_dict, _ = test_client_basic(visible_models=visible_models, prompt=prompt,
                                            prompt_type=prompt_type)
            assert res_dict['prompt'] == prompt
            assert res_dict['iinput'] == ''
            assert 'the limit of time' in res_dict['response']


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
    main(base_model=base_model, prompt_type='prompt_answer', chat=False,
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
    assert """I am a mischievous pixie, always up to no good! *wink* But don't worry, I won't play any tricks on you... unless you want me to, that is. *giggles*
As for my fairy dust, it's a special blend of sparkly, shimmering magic that can grant wishes and make dreams come true. *twinkle eyes* Would you like some? *offers a tiny vial of sparkles*""" in response or \
           """I am a mischievous pixie, always up to no good! *winks* But don't worry, I won't play any tricks on you... unless you want me to, that is. *giggles*
   As for my fairy dust, it's a special blend of sparkly, shimmering magic that can grant wishes and make dreams come true. *twinkle* Would you like some? *offers a tiny vial of sparkles*""" in response or \
           """I am a mischievous pixie""" in response


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
    os.environ['GET_GITHASH'] = '1'
    main(base_model=base_model, prompt_type='human_bot', chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False,
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
    assert res in [get_githash(), 'GET_GITHASH']

    res = client2.get_server_hash()
    assert res in [get_githash(), 'GET_GITHASH']


@wrap_test_forked
def test_client1api_lean_lock_choose_model():
    from src.gen import main
    base1 = 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    base2 = 'distilgpt2'
    model_lock = [dict(base_model=base1, prompt_type='human_bot'),
                  dict(base_model=base2, prompt_type='plain')]
    save_dir = 'save_test'
    main(model_lock=model_lock, chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False,
         save_dir=save_dir)

    client = get_client(serialize=not is_gradio_version4)
    for prompt_type in ['human_bot', None, '', 'plain']:
        for visible_models in [None, 0, base1, 1, base2]:
            base_model = base1 if visible_models in [None, 0, base1] else base2
            if base_model == base1 and prompt_type == 'plain':
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
                   {'base_model': 'distilgpt2', 'prompt_type': 'plain', 'prompt_dict': None, 'load_8bit': False,
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
           'As an AI assistant'  in res_dict['response']


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
    assert 'Birds' in res_dict['response'] or \
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
    assert 'h2oGPT is a variant of the popular GPT' in res_dict['response'] and '.md' not in res_dict['response']

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
                assert 'ROAR!' in res_dict['response'] and 'respectful' not in res_dict[
                    'response'] and 'developed by Meta' not in res_dict['response']
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
                       "awesome lion"  in res_dict['response']
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
@pytest.mark.parametrize("auto_migrate_db", [False, True])
@wrap_test_forked
def test_client_chat_stream_langchain_steps2(max_new_tokens, top_k_docs, auto_migrate_db):
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
         verbose=True,
         auto_migrate_db=auto_migrate_db)

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
                'response']
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
           'The story begins with'  in res_dict['response']


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
    prompt = 'Who are you?'
    stream_output = False
    max_new_tokens = 256
    base_model = 'TheBloke/Llama-2-13B-chat-AWQ'
    load_awq = 'model'
    use_safetensors = True
    prompt_type = 'llama2'
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
         docs_ordering_type=docs_ordering_type)

    from src.client_test import run_client_chat
    res_dict, client = run_client_chat(prompt=prompt, prompt_type=prompt_type, stream_output=stream_output,
                                       max_new_tokens=max_new_tokens, langchain_mode=langchain_mode,
                                       langchain_action=langchain_action, langchain_agents=langchain_agents)
    assert res_dict['prompt'] == prompt
    assert res_dict['iinput'] == ''
    assert "am a virtual assistant" in res_dict['response'] or \
           "Hello! My name is LLaMA, I'm a large language model trained by a team" in res_dict['response']

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
    main(base_model='mosaicml/mpt-7b-storywriter', prompt_type='plain', chat=False,
         stream_output=False, gradio=True, num_beams=1, block_gradio_exit=False)

    with open("/home/jon/Downloads/Gatsby_PDF_FullText.txt") as f:
        prompt = f.readlines()

    from src.client_test import run_client_nochat
    res_dict, _ = run_client_nochat(prompt=prompt, prompt_type='plain', max_new_tokens=86000)
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
         append_sources_to_answer=True,
         append_sources_to_chat=False,
         **main_kwargs,
         verbose=True)

    from src.client_test import get_client, get_args, run_client
    # serialize=False would lead to returning dict for some objects or files for get_sources
    client = get_client(serialize=False)

    url = 'https://www.africau.edu/images/default/sample.pdf'
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
            "more text is boring" in res_dict['response']) \
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
    sources_expected = f'{user_path}/FAQ.md\n{user_path}/README.md\n{user_path}/next.txt\n{user_path}/pexels-evg-kowalievska-1170986_small.jpg\n{user_path}/sample1.pdf'
    assert sources == sources_expected or sources.replace('\\', '/').replace('\r', '') == sources_expected.replace(
        '\\', '/').replace('\r', '')

    # check sources, and do after so would detect leakage
    sources = ast.literal_eval(client.predict(langchain_mode, h2ogpt_key, api_name='/get_sources_api'))
    assert isinstance(sources, list)
    sources_expected = ['user_path_test/FAQ.md', 'user_path_test/README.md', 'user_path_test/next.txt',
                        'user_path_test/pexels-evg-kowalievska-1170986_small.jpg', 'user_path_test/sample1.pdf']
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
    url = 'https://www.africau.edu/images/default/sample.pdf'
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

    sources_state_after_delete = ast.literal_eval(client.predict(langchain_mode3, h2ogpt_key, api_name='/get_sources_api'))
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
        max_seq_len_ex = 32768.0
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
    main(base_model=base_model,
         inference_server='' if base_model not in openai_gpts else 'openai_chat',
         chat=True,
         stream_output=stream_output,
         gradio=True, num_beams=1, block_gradio_exit=False,
         score_model='',
         verbose=True)

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

    url = 'https://www.africau.edu/images/default/sample.pdf'
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
    assert hf_embedding_model == 'hkunlp/instructor-large'  # but not used
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
@pytest.mark.parametrize("inference_server", [None, 'openai', 'openai_chat', 'openai_azure_chat', 'replicate'])
# local_server=True
@pytest.mark.parametrize("base_model",
                         ['h2oai/h2ogpt-oig-oasst1-512-6_9b', 'h2oai/h2ogpt-4096-llama2-7b-chat', 'gpt-3.5-turbo'])
# local_server=False or True if inference_server used
# @pytest.mark.parametrize("base_model", ['h2oai/h2ogpt-4096-llama2-70b-chat'])
@pytest.mark.parametrize("data_kind", [
    'simple',
    'helium1',
    'helium2',
    'helium3',
    'helium4',
    'helium5',
])
@wrap_test_forked
def test_client_chat_stream_langchain_fake_embeddings(data_kind, base_model, inference_server):
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
                inference_server += ':None:%s' %  os.environ['OPENAI_AZURE_KEY']
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
            base_model = 'HuggingFaceH4/zephyr-7b-beta'
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
                dict(inference_server=inference_server, base_model=base_model, visible_models=base_model,
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
    if is_gradio_client_version7plus:
        assert os.path.normpath(test_file_local) != os.path.normpath(test_file_server)

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
         use_auth_token=True,
         )

    # get file for client to upload
    url = 'https://cdn.openai.com/papers/whisper.pdf'
    test_file1 = os.path.join('/tmp/', 'whisper1.pdf')
    download_simple(url, dest=test_file1)

    # Get text version of PDF
    from langchain.document_loaders import PyMuPDFLoader
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
               or ('key results based on the provided document' in summary and 'h2oGPT' in summary)
        assert 'h2oGPT' in [x['content'] for x in sources][0]
    assert url in [x['source'] for x in sources][0]


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
        os.environ['CUDA_HOME'] = '/usr/local/cuda-11.7'
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
    from playsound import playsound
    playsound(filename)


@pytest.mark.parametrize("tts_model", [
    'microsoft/speecht5_tts',
    'tts_models/multilingual/multi-dataset/xtts_v2'
])
@pytest.mark.parametrize("base_model", [
    'llama',
    'HuggingFaceH4/zephyr-7b-beta'
])
@wrap_test_forked
def test_client1_tts_stream(tts_model, base_model):
    from src.gen import main
    main(base_model=base_model, chat=False,
         tts_model=tts_model,
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
        time.sleep(0.01)

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
        assert res['save_dict']['base_model'] == 'HuggingFaceH4/zephyr-7b-beta'
    assert res['save_dict']['where_from']
    assert res['save_dict']['valid_key'] == 'not enforced'
    assert res['save_dict']['h2ogpt_key'] in [None, '']

    assert res['save_dict']['extra_dict']
    if base_model == 'llama':
        assert res['save_dict']['extra_dict']['llamacpp_dict']
        assert res['save_dict']['extra_dict']['prompt_type'] == 'llama2'
    else:
        assert res['save_dict']['extra_dict']['prompt_type'] == 'zephyr'
    assert res['save_dict']['extra_dict']['do_sample'] == False
    assert res['save_dict']['extra_dict']['num_prompt_tokens'] > 10
    assert res['save_dict']['extra_dict']['ntokens'] > 60
    assert res['save_dict']['extra_dict']['tokens_persecond'] > 5


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

    assert 'assistant' in res_dict['response'] or 'computer program' in res_dict['response'] or 'program designed' in \
           res_dict['response']
    assert 'Who are you?' in res_dict['prompt_raw']
    assert 'llama' == res_dict['save_dict']['base_model'] or 'HuggingFaceH4/zephyr-7b-beta' == res_dict['save_dict'][
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
         h2ogpt_api_keys=[h2ogpt_key] if h2ogpt_key else [])

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
        # NOTE: pip install playsound
        from playsound import playsound
        filename = '/tmp/audio_%s.wav' % str(uuid.uuid4())
        audio = AudioSegment.from_raw(s, sample_width=sample_width, frame_rate=sr, channels=channels)
        audio.export(filename, format='wav')
        playsound(filename)
    else:
        # pip install simpleaudio==1.0.4
        # WIP, needs header, while other shouldn't have header
        from pydub import AudioSegment
        from pydub.playback import play
        song = AudioSegment.from_file(s, format="wav")
        play(song)
    return n


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
