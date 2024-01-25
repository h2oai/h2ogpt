import pandas as pd
import pytest

from tests.utils import wrap_test_forked, make_user_path_test
from src.enums import DocumentSubset, LangChainAction, docs_joiner_default
from src.utils import remove


@pytest.mark.parametrize("base_model", ['h2oai/h2ogpt-oig-oasst1-512-6_9b', 'junelee/wizard-vicuna-13b'])
@pytest.mark.parametrize("bits", [4, 8, 16, 32])
@pytest.mark.parametrize("cpu", [False, True])
@wrap_test_forked
def test_eval1(cpu, bits, base_model):
    if cpu and bits != 32:
        return
    run_eval1(cpu=cpu, bits=bits, base_model=base_model)


@wrap_test_forked
def test_eval_json():
    base_model = 'h2oai/h2ogpt-oig-oasst1-512-6_9b'
    cpu = False
    bits = 8

    # make 2 rows of json
    prompts = [dict(instruction="Who are you?", output="I'm h2oGPT"),
               dict(instruction="What is 2+2?", output="4"),
               ]
    eval_filename = 'test_prompts.json'
    remove(eval_filename)
    import json
    with open(eval_filename, "wt") as f:
        f.write(json.dumps(prompts, indent=2))

    eval_out_filename = run_eval1(cpu=cpu, bits=bits, base_model=base_model, eval_filename=eval_filename,
                                  eval_prompts_only_num=len(prompts))
    df = pd.read_parquet(eval_out_filename)
    val0 = "My name is h2oGPT. I'm a large language model trained by H2O.ai. How may I assist you?"
    val1 = """Hi! I'm h2oGPT, a large language model by H2O.ai, the visionary leader in democratizing AI. How may I assist you?"""
    val2 = """Hi! I'm h2oGPT, a large language model by H2O.ai"""
    val3 = """My name is h2oGPT. I'm a large language model trained by H2O.ai. How may I assist you?"""
    val4 = """ I'm h2oGPT, a large language model by H2O.ai. How may I assist you?"""
    assert df['response'].values[0] == val0 or \
           df['response'].values[0] == ' ' + val0 or \
           df['response'].values[0] == val1 or \
           df['response'].values[0] == val3 or \
           df['response'].values[0] == val4 or \
           val2 in df['response'].values[0]
    assert df['score'].values[0] > 0.03  # odd score IMO
    assert df['response'].values[1] in ["2 + 2 = 4\n", "2+2 = 4\n", " 2 + 2 = 4\n", ' 4\n']
    assert df['score'].values[1] > 0.5


def run_eval1(cpu=False, bits=None, base_model='h2oai/h2ogpt-oig-oasst1-512-6_9b', eval_filename=None,
              eval_prompts_only_num=1,
              langchain_mode='Disabled'):
    if base_model == 'junelee/wizard-vicuna-13b' and (bits != 8 or cpu):
        # Too much CPU memory or GPU memory
        return

    import os, sys
    os.environ['TEST_LANGCHAIN_IMPORT'] = "1"
    sys.modules.pop('gpt_langchain', None)
    sys.modules.pop('langchain', None)

    prompt_type = None
    if 'h2oai/h2ogpt-' in base_model:
        prompt_type = 'human_bot'
    if 'junelee/wizard-vicuna-13b' == base_model:
        prompt_type = 'instruct_vicuna'
    assert prompt_type is not None

    if cpu:
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import pandas as pd
    from src.evaluate_params import eval_func_param_names, eval_extra_columns
    from src.gen import main
    kwargs = dict(
        stream_output=False, prompt_type=prompt_type, prompt_dict='',
        temperature=0.4, top_p=0.85, top_k=70, penalty_alpha=0.0, num_beams=1, max_new_tokens=256,
        min_new_tokens=0, early_stopping=False, max_time=180, repetition_penalty=1.07,
        num_return_sequences=1, do_sample=True, chat=False,
        langchain_mode=langchain_mode, add_chat_history_to_context=True,
        add_search_to_context=False,
        langchain_action=LangChainAction.QUERY.value, langchain_agents=[],
        chunk=True, chunk_size=512,
        load_half=False, load_4bit=False, load_8bit=False,
        load_gptq='', load_awq='', load_exllama=False, use_safetensors=False,
    )
    if bits == 4:
        kwargs['load_4bit'] = True
    elif bits == 8:
        kwargs['load_8bit'] = True
    elif bits == 16:
        kwargs['load_half'] = True
    elif bits == 32:
        pass
    kwargs['load_gptq'] = ''
    kwargs['load_awq'] = ''
    kwargs['load_exllama'] = False
    kwargs['use_safetensors'] = False
    eval_out_filename = main(base_model=base_model,
                             gradio=False,
                             eval_filename=eval_filename,
                             eval_prompts_only_num=eval_prompts_only_num,
                             eval_as_output=False,
                             eval_prompts_only_seed=1235,
                             score_model='OpenAssistant/reward-model-deberta-v3-large-v2',
                             **kwargs)
    if eval_filename is not None:
        # then not sharegpt
        return eval_out_filename
    import numpy as np

    df = pd.read_parquet(eval_out_filename)
    assert df.shape[0] == 1
    columns = eval_func_param_names + eval_extra_columns
    assert df.shape[1] == len(columns)
    # assumes SEED = 1236 in generate.py
    result_list = list(df.values[0])
    key_separate = ['response', 'score']
    actual1 = {k: v for k, v in zip(columns, result_list) if k not in key_separate}
    expected1 = {'instruction': '', 'iinput': '', 'context': '',
                 'instruction_nochat': 'I confess, with only a touch of embarrassment, that I had no idea until we started working on this book that each vertebra was really a two-part bone. There is the more or less solid and weight-bearing portion toward your front, called “the vertebral body” (with the discs in between). And then there’s this flying buttress–looking piece that sticks off the back (the “vertebral arch”). And there is a very important space between the two sections. The “hollow” down the middle of your spine is really a space between these two segments of the vertebra. The vertebra is one solid piece of bone but with two very distinct segments and a hole down the middle where the spinal cord goes. \nThe Spinal Column\n\nDo you see the spiny-looking pieces in the picture, above, sticking off the vertebrae? Those are pieces of the vertebral arch. They are called “the spinous processes” (no one cares), and they are basically anchor points. That’s where ligaments can attach muscles to the vertebrae. If you’re a sailor, think of cleats on the deck, for ropes. When you reach back and feel the spiny part of your back, you’re feeling the “spinous processes” or the cleats. By the way, the ligaments or sinews are the lines (think “stays” on a sailboat) that hold your spine erect. Without stays, the mast on a sailboat would flop around and break in no time; with stays, the mast on a well-designed sailboat is remarkably stable. Flexible, like your spine, but stable and strong, too. \nOkeydoke, on to the discs. This is familiar territory to most of us. You hear about discs all the time. “Bulging discs,” “pinched discs,” “slipped discs,” “ruptured discs” and so on. They are basically washers to keep the weight-bearing parts of the vertebrae from rubbing on one another and to put some “give” into your back. You cannot have an articulated stack of bones without a wonderfully effective stack of washers to keep ’em apart, and you do. Think of them as very tough jelly doughnuts, as I mentioned before. There is a tough, fibrous layer on the outside and a gooey or liquid core on the inside. They act as shock absorbers and have a lot to do with letting you bend. Dysfunctional discs can be a major source of problems and pain. \nA YOUNG PERSON’S PROBLEMS\nThis is interesting. Bulging and actually ruptured discs are mostly a young person’s problem, people in their thirties (and you kids are welcome to them; they really hurt). Older people have horrendous problems, too; after all, some 35 percent of people from ages forty-five to sixty-five have serious back pain. But usually not this particular horror. Which also means that more younger people are going to get bundled off to the surgeon, if the problem is grim enough. Older people have disc problems, too, but nowhere near as often. \nTake a long look at the pictures on the next pages. They show you how the spinal cord, spine, and discs work together. First is a side view depicting how the brain, spinal cord, and spine are positioned in the body. Second is a close-up of a segment made up of two vertebrae with their disc (in gray) in between and the spinal cord and nerve roots visible. Notice how the rear parts of the adjoining vertebrae form a canal through which the spinal cord runs from top to bottom. Also notice how the two adjoining vertebrae form holes, or “foramina,” on the sides where the nerve roots come out of the spine. Those holes are super-important: The holes can become smaller from disc degeneration or movement of the vertebrae on top of each other. And the nerve that comes out of the hole is pressured, and it hurts like blazes. Not to get too scary, but when things really go to hell and you actually rupture or split the disc open with your ridiculous posture or whatnot, the pain really goes over the top. (Good news: You hear about ruptured discs all the time, but they are comparatively rare.) Bones wear on bones, discs hurt like crazy, and the stuff in the middle squirts all over the place. Which is bad because it causes severe chemical pain in the nerves. Not so good. When we say that there are times when traditional medicine (surgery) has a critical role, this is one of them. \nNote the bits of bone to the left in the close-up side view vertebral segment. These are “the facet joints.” The point of this picture is to show how they are right next to the spinal cord and near one of the nerve exit spots. They are well placed, in other words, to raise hell if things go wrong with them. I forgot to mention this: The surfaces of the facet joints are covered in cartilage, which allows smooth movement in a healthy spine. So what? The point is that this cartilage can be abraded or torn by dumb moves, too, and that hurts as well. \nHere are two more views, below. Note the sort of circular thing with the lighter insides. That’s a cross section of a disc, seen from the top. \n\nLigaments and Tendons\nAll right, that‘s the spinal cord and the spinal column. But they would not stand alone without a ton of support. Think of the spinal column as a slender reed. If you press down on it at all from above (or the sides), it will bend crazily. Indeed, it cannot sustain any weight at all to speak of. But now, add a bunch of support lines from the pole to a solid support, and it’s a different story. Our backbone has a lot of very sturdy support lines called ligaments and tendons (ligaments connect bone to bone; tendons connect bone to muscle.) There are an awful lot of ligaments connected to the spine. The following picture gives you the idea. \nHere’s another thing you need to know: Ligaments can become deformed or sprained because of bad posture, a persistent pattern of bad movements, or an injury. When that happens to a ligament, the joints those ligaments were supporting “get loose” and can slip around. That is really bad. Here is a language alert: A “sprain” is an unhealthy stretch or tear of a ligament, and a “strain” is an unhealthy stretch or tear in a tendon or muscle. Look at the picture on the opposite page: there are a ton of ligaments here, all waiting to go haywire if you are foolish or unlucky. \nSpinal Ligaments',
                 'iinput_nochat': '',
                 'prompt': 'I confess, with only a touch of embarrassment, that I had no idea until we started working on this book that each vertebra was really a two-part bone. There is the more or less solid and weight-bearing portion toward your front, called “the vertebral body” (with the discs in between). And then there’s this flying buttress–looking piece that sticks off the back (the “vertebral arch”). And there is a very important space between the two sections. The “hollow” down the middle of your spine is really a space between these two segments of the vertebra. The vertebra is one solid piece of bone but with two very distinct segments and a hole down the middle where the spinal cord goes. \nThe Spinal Column\n\nDo you see the spiny-looking pieces in the picture, above, sticking off the vertebrae? Those are pieces of the vertebral arch. They are called “the spinous processes” (no one cares), and they are basically anchor points. That’s where ligaments can attach muscles to the vertebrae. If you’re a sailor, think of cleats on the deck, for ropes. When you reach back and feel the spiny part of your back, you’re feeling the “spinous processes” or the cleats. By the way, the ligaments or sinews are the lines (think “stays” on a sailboat) that hold your spine erect. Without stays, the mast on a sailboat would flop around and break in no time; with stays, the mast on a well-designed sailboat is remarkably stable. Flexible, like your spine, but stable and strong, too. \nOkeydoke, on to the discs. This is familiar territory to most of us. You hear about discs all the time. “Bulging discs,” “pinched discs,” “slipped discs,” “ruptured discs” and so on. They are basically washers to keep the weight-bearing parts of the vertebrae from rubbing on one another and to put some “give” into your back. You cannot have an articulated stack of bones without a wonderfully effective stack of washers to keep ’em apart, and you do. Think of them as very tough jelly doughnuts, as I mentioned before. There is a tough, fibrous layer on the outside and a gooey or liquid core on the inside. They act as shock absorbers and have a lot to do with letting you bend. Dysfunctional discs can be a major source of problems and pain. \nA YOUNG PERSON’S PROBLEMS\nThis is interesting. Bulging and actually ruptured discs are mostly a young person’s problem, people in their thirties (and you kids are welcome to them; they really hurt). Older people have horrendous problems, too; after all, some 35 percent of people from ages forty-five to sixty-five have serious back pain. But usually not this particular horror. Which also means that more younger people are going to get bundled off to the surgeon, if the problem is grim enough. Older people have disc problems, too, but nowhere near as often. \nTake a long look at the pictures on the next pages. They show you how the spinal cord, spine, and discs work together. First is a side view depicting how the brain, spinal cord, and spine are positioned in the body. Second is a close-up of a segment made up of two vertebrae with their disc (in gray) in between and the spinal cord and nerve roots visible. Notice how the rear parts of the adjoining vertebrae form a canal through which the spinal cord runs from top to bottom. Also notice how the two adjoining vertebrae form holes, or “foramina,” on the sides where the nerve roots come out of the spine. Those holes are super-important: The holes can become smaller from disc degeneration or movement of the vertebrae on top of each other. And the nerve that comes out of the hole is pressured, and it hurts like blazes. Not to get too scary, but when things really go to hell and you actually rupture or split the disc open with your ridiculous posture or whatnot, the pain really goes over the top. (Good news: You hear about ruptured discs all the time, but they are comparatively rare.) Bones wear on bones, discs hurt like crazy, and the stuff in the middle squirts all over the place. Which is bad because it causes severe chemical pain in the nerves. Not so good. When we say that there are times when traditional medicine (surgery) has a critical role, this is one of them. \nNote the bits of bone to the left in the close-up side view vertebral segment. These are “the facet joints.” The point of this picture is to show how they are right next to the spinal cord and near one of the nerve exit spots. They are well placed, in other words, to raise hell if things go wrong with them. I forgot to mention this: The surfaces of the facet joints are covered in cartilage, which allows smooth movement in a healthy spine. So what? The point is that this cartilage can be abraded or torn by dumb moves, too, and that hurts as well. \nHere are two more views, below. Note the sort of circular thing with the lighter insides. That’s a cross section of a disc, seen from the top. \n\nLigaments and Tendons\nAll right, that‘s the spinal cord and the spinal column. But they would not stand alone without a ton of support. Think of the spinal column as a slender reed. If you press down on it at all from above (or the sides), it will bend crazily. Indeed, it cannot sustain any weight at all to speak of. But now, add a bunch of support lines from the pole to a solid support, and it’s a different story. Our backbone has a lot of very sturdy support lines called ligaments and tendons (ligaments connect bone to bone; tendons connect bone to muscle.) There are an awful lot of ligaments connected to the spine. The following picture gives you the idea. \nHere’s another thing you need to know: Ligaments can become deformed or sprained because of bad posture, a persistent pattern of bad movements, or an injury. When that happens to a ligament, the joints those ligaments were supporting “get loose” and can slip around. That is really bad. Here is a language alert: A “sprain” is an unhealthy stretch or tear of a ligament, and a “strain” is an unhealthy stretch or tear in a tendon or muscle. Look at the picture on the opposite page: there are a ton of ligaments here, all waiting to go haywire if you are foolish or unlucky. \nSpinal Ligaments',
                 'top_k_docs': 10,
                 'document_subset': DocumentSubset.Relevant.name,  # matches return
                 'document_choice': np.array([]),  # matches return
                 'document_content_substrings': np.array([]),  # matches return
                 'document_source_substrings_op': 'and',
                 'document_source_substrings': np.array([]),  # matches return
                 'document_content_substrings_op': 'and',
                 'langchain_agents': np.array([]),  # matches return
                 'pre_prompt_query': None,
                 'prompt_query': None,
                 'pre_prompt_summary': None,
                 'prompt_summary': None,
                 'hyde_llm_prompt': None,
                 'system_prompt': 'auto',
                 'pdf_loaders': np.array(['PyMuPDF'], dtype=object),
                 'url_loaders': np.array(['Unstructured'], dtype=object),
                 'jq_schema': '.[]',
                 'extract_frames': 10,
                 'visible_models': None,
                 'h2ogpt_key': None,
                 'chat_conversation': None,
                 'text_context_list': None,
                 'docs_ordering_type': 'reverse_ucurve_sort',
                 'min_max_new_tokens': 512,
                 'max_input_tokens': -1,
                 'llava_prompt': 'auto',
                 'max_total_input_tokens': -1,
                 'docs_token_handling': 'split_or_merge',
                 'docs_joiner': docs_joiner_default,
                 'hyde_level': 0,
                 'hyde_template': None,
                 'hyde_show_only_final': False,
                 'doc_json_mode': False,
                 'chatbot_role': 'None',
                 'speaker': 'None',
                 'tts_language': 'autodetect',
                 'tts_speed': 1.0,
                 }
    if cpu and bits == 32:
        expected1.update({'image_audio_loaders': np.array([], dtype=object)})
    else:
        expected1.update({'image_audio_loaders': np.array(['Caption'], dtype=object)})

    expected1.update({k: v for k, v in kwargs.items() if
                      k not in ['load_half', 'load_4bit', 'load_8bit', 'load_gptq', 'load_awq', 'load_exllama', 'use_safetensors']})
    drop_keys = ['document_choice',
                 'document_source_substrings', 'document_source_substrings_op', 'document_content_substrings', 'document_content_substrings_op',
                 'langchain_agents', 'image_audio_loaders']  # some numpy things annoying to match
    expected1 = {k: v for k, v in expected1.items() if k not in drop_keys}
    actual1 = {k: v for k, v in actual1.items() if k not in drop_keys}
    assert sorted(actual1.items()) == sorted(expected1.items())
    actual2 = {k: v for k, v in zip(columns, result_list) if k in key_separate}

    import torch
    if torch.cuda.is_available():
        if bits == 4:
            expected2 = {
                'response': """The spinal ligaments are the thick bands of tissue that connect the vertebrae of the spine. They are there to keep the vertebrae in place and to protect the spinal cord.""",
                'score': 0.7533428072929382}
        elif bits == 8:
            if base_model == 'junelee/wizard-vicuna-13b':
                expected2 = {
                    'response': """The human spine is made up of individual vertebrae, each consisting of two distinct segments - the vertebral body and the vertebral arch. The vertebral body is a weight-bearing segment while the vertebral arch contains the spinous processes, which serve as anchor points for muscles and ligaments. The discs between the vertebrae act as shock absorbers and help with flexibility. However, dysfunctional discs can cause problems and pain. Bulging and ruptured discs are mostly a young person's issue, while older people are more likely to have serious back pain due to other factors. The ligaments and tendons provide support to the spine and prevent it from bending too much. Bad posture, injuries, and persistent poor movements can cause ligament sprains and tendon strains, leading to joint instability.""",
                    'score': 0.7533428072929382}
            else:
                expected2 = {
                    'response': """The ligaments are the bands of tissue that connect the vertebrae together. The ligaments help to stabilize the spine and protect the spinal cord.""",
                    'score': 0.7533428072929382}

        elif bits == 16:
            expected2 = {
                'response': """The spinal ligaments are like the supports on a bridge. They hold the spinal column in place, and they are very important. If you pull on the spinal column, the ligaments will try to keep the column straight. If you push on the spinal column, the ligaments will try to keep the column straight. If you twist the spinal column, the ligaments will try to keep the column straight. If you pull on the ligaments themselves, they will try to keep the column straight. If you twist the ligaments, they will try to keep the column straight. If you twist the spinal column, the ligaments will try to keep the column straight. If you twist the spinal column, the ligaments will try to keep the column straight. If you twist the spinal column, the ligaments will try to keep the column straight. If you twist the spinal column, the ligaments will try to keep the column straight. If you twist the spinal column, the ligaments will try to keep the column straight. If you twist the spinal column, the ligaments will try to keep the column straight. If you twist the spinal column, the ligaments will try to keep the column straight. If you twist the spinal column, the ligaments will try to keep""",
                'score': 0.65}
        else:
            expected2 = {
                'response': """The spinal ligaments are like the webbing on a tree branch. They are there to help the spinal cord stay upright and prevent it from flopping around. If the spinal cord gets twisted or bent, the ligaments can get stretched or torn. That can cause pain and sometimes paralysis. \nTendons""",
                'score': 0.65}
    else:
        expected2 = {
            'response': 'The ligaments that support the spine are called the “spinal ligaments.” They are there to help keep the spine straight and upright. They are made up of tough fibers that run from the pelvis to the skull. They are like the stays on a sailboat, except that they are much thicker and stronger. \nThe spinal ligaments are divided into two groups: anterior and posterior. The anterior ligaments are attached to the front of the vertebrae, while the posterior ligaments are attached to the back. The anterior ligaments are called the “anterior longitudinal ligaments”',
            'score': 0.77}
    if bits == 32 and cpu:
        expected2 = {
            'response': """The ligaments that support the spine are called the ?sp
inal ligaments.? They are there to help keep the spine straight and upright. They are made up of tough fibers that run from the pelvis to the skull. They are like the stays on a sailboat, except that they are much thicker and stronger. \nThe spin
al ligaments are divided into two groups: anterior and posterior. The anterior ligaments are attached to the front of the vertebrae, while the posterior ligaments are attached to the back. The anterior ligaments are called the ?anterior longitudi
nal ligaments? because they run along the length of the spine. The posterior ligaments are called the ?transverse ligaments? because they run across the width of the spine. \nThe anterior ligaments are attached to the front of the vertebrae, whil
e the posterior ligaments are attached to the back. The anterior ligaments are called the ?anterior longitudinal ligaments? because they run along the length of the spine. The posterior ligaments are called the ?transverse ligaments? because they
 run across the width of the spine. \nThe anterior ligaments are attached to the front of the vertebrae, while the posterior ligaments are attached to the back. The anterior ligaments are""",
            'score': 0.77}

    assert np.isclose(actual2['score'], expected2['score'], rtol=0.3), "Score is not as expected: %s %s" % (
        actual2['score'], expected2['score'])

    from sacrebleu.metrics import BLEU
    bleu = BLEU()
    assert bleu.sentence_score(actual2['response'], [expected2['response']]).score > 10
    return eval_out_filename


@wrap_test_forked
def test_eval_json_langchain():
    base_model = 'llama'
    user_path = make_user_path_test()

    # make 2 rows of json
    prompts = [dict(instruction="What is Whisper?", response="""According to the document sources provided in the context, Whisper is a large language model (LLM) that can be used for various tasks such as text-to-speech (TTS), voice cloning, and speech recognition (ASR). It is a powerful tool for generating human-like speech and can be trained on a wide range of data sources."""),
               dict(instruction="Who made Whisper?", response="""According to the document sources provided within the context, Whisper was made by OpenAI."""),
               ]
    eval_prompts_only_num = len(prompts)
    eval_filename = 'test_prompts.json'
    remove(eval_filename)
    import json
    with open(eval_filename, "wt") as f:
        f.write(json.dumps(prompts, indent=2))

    import pandas as pd
    from src.evaluate_params import eval_func_param_names, eval_extra_columns
    from src.gen import main
    kwargs = dict(
        stream_output=False,
        langchain_mode='UserData',
        user_path=user_path,
    )
    eval_out_filename = main(base_model=base_model,
                             gradio=False,
                             eval_filename=eval_filename,
                             eval_prompts_only_num=eval_prompts_only_num,
                             eval_as_output=False,
                             asr_model='',
                             answer_with_sources=False,
                             show_link_in_sources=False,
                             append_sources_to_answer=False,
                             append_sources_to_chat=False,
                             eval_prompts_only_seed=1235,
                             score_model='OpenAssistant/reward-model-deberta-v3-large-v2',
                             **kwargs)
    df = pd.read_parquet(eval_out_filename)
    assert df.shape[0] == 2
    columns = eval_func_param_names + eval_extra_columns
    assert df.shape[1] == len(columns)
    print(df.values)
    actuals = [dict(score=df['score'].values[ii], response=df['response'].values[ii]) for ii in range(df.shape[0])]
    expecteds = [0.05, 0.01]

    for prompt, expected, actual in zip(prompts, expecteds, actuals):
        import numpy as np
        print("actual: %s" % actual)
        print("expected: %s" % expected)
        assert actual['score'] > expected, "Assert: %s %s" % (actual, expected)

        from sacrebleu.metrics import BLEU
        bleu = BLEU()
        assert bleu.sentence_score(actual['response'], [prompt['response']]).score > 25
