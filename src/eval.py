import inspect
import os
import traceback
import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt

from evaluate_params import eval_func_param_names, eval_extra_columns
from gen import get_context, get_score_model, get_model, evaluate, check_locals
from prompter import Prompter
from utils import clear_torch_cache, NullContext, get_kwargs


def run_eval(  # for local function:
        base_model=None, lora_weights=None, inference_server=None,
        prompt_type=None, prompt_dict=None,
        debug=None, chat=False, chat_context=None, stream_output=None,
        eval_filename=None, eval_prompts_only_num=None, eval_prompts_only_seed=None, eval_as_output=None,
        examples=None, memory_restriction_level=None,
        # for get_model:
        score_model=None, load_8bit=None, load_4bit=None, load_half=None, load_gptq=None, use_safetensors=None,
        use_gpu_id=None, tokenizer_base_model=None,
        gpu_id=None, local_files_only=None, resume_download=None, use_auth_token=None,
        trust_remote_code=None, offload_folder=None, compile_model=None,
        # for evaluate args beyond what's already above, or things that are always dynamic and locally created
        temperature=None,
        top_p=None,
        top_k=None,
        num_beams=None,
        max_new_tokens=None,
        min_new_tokens=None,
        early_stopping=None,
        max_time=None,
        repetition_penalty=None,
        num_return_sequences=None,
        do_sample=None,
        langchain_mode=None,
        langchain_action=None,
        langchain_agents=[],
        top_k_docs=None,
        chunk=None,
        chunk_size=None,
        document_subset=None,
        document_choice=None,
        # for evaluate kwargs:
        src_lang=None, tgt_lang=None, concurrency_count=None, save_dir=None, sanitize_bot_response=None,
        model_state0=None,
        max_max_new_tokens=None,
        is_public=None,
        max_max_time=None,
        raise_generate_gpu_exceptions=None, load_db_if_exists=None, use_llm_if_no_docs=None,
        dbs=None, user_path=None,
        detect_user_path_changes_every_query=None,
        use_openai_embedding=None, use_openai_model=None, hf_embedding_model=None,
        db_type=None, n_jobs=None, first_para=None, text_limit=None, verbose=None, cli=None, reverse_docs=None,
        use_cache=None,
        auto_reduce_chunks=None, max_chunks=None,
        model_lock=None, force_langchain_evaluate=None,
        model_state_none=None,
):
    check_locals(**locals())

    if eval_prompts_only_num > 0:
        np.random.seed(eval_prompts_only_seed)
        example1 = examples[-1]  # pick reference example
        examples = []
        responses = []
        if eval_filename is None:
            # override default examples with shareGPT ones for human-level eval purposes only
            eval_filename = 'ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json'
            if not os.path.isfile(eval_filename):
                os.system(
                    'wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/%s' % eval_filename)
            import json
            data = json.load(open(eval_filename, 'rt'))
            # focus on data that starts with human, else likely chopped from other data
            turn_start = 0  # odd in general
            data = [x for x in data if len(x['conversations']) > turn_start + 1 and
                    x['conversations'][turn_start]['from'] == 'human' and
                    x['conversations'][turn_start + 1]['from'] == 'gpt']
            for i in sorted(np.random.randint(0, len(data), size=eval_prompts_only_num)):
                assert data[i]['conversations'][turn_start]['from'] == 'human'
                instruction = data[i]['conversations'][turn_start]['value']
                assert data[i]['conversations'][turn_start + 1]['from'] == 'gpt'
                output = data[i]['conversations'][turn_start + 1]['value']
                examplenew = example1.copy()
                assert not chat, "No gradio must use chat=False, uses nochat instruct"
                examplenew[eval_func_param_names.index('instruction_nochat')] = instruction
                examplenew[eval_func_param_names.index('iinput_nochat')] = ''  # no input
                examplenew[eval_func_param_names.index('context')] = get_context(chat_context, prompt_type)
                examples.append(examplenew)
                responses.append(output)
        else:
            # get data, assume in correct format: json of rows of dict of instruction and output
            # only instruction is required
            import json
            data = json.load(open(eval_filename, 'rt'))
            for i in sorted(np.random.randint(0, len(data), size=eval_prompts_only_num)):
                examplenew = example1.copy()
                instruction = data[i]['instruction']
                output = data[i].get('output', '')  # not required
                assert not chat, "No gradio must use chat=False, uses nochat instruct"
                examplenew[eval_func_param_names.index('instruction_nochat')] = instruction
                examplenew[eval_func_param_names.index('iinput_nochat')] = ''  # no input
                examplenew[eval_func_param_names.index('context')] = get_context(chat_context, prompt_type)
                examples.append(examplenew)
                responses.append(output)

    num_examples = len(examples)
    scoring_path = 'scoring'
    os.makedirs(scoring_path, exist_ok=True)
    if eval_as_output:
        used_base_model = 'gpt35'
        used_lora_weights = ''
        used_inference_server = ''
    else:
        used_base_model = str(base_model.split('/')[-1])
        used_lora_weights = str(lora_weights.split('/')[-1])
        used_inference_server = str(inference_server.split('/')[-1])
    eval_out_filename = "df_scores_%s_%s_%s_%s_%s_%s_%s.parquet" % (num_examples, eval_prompts_only_num,
                                                                    eval_prompts_only_seed,
                                                                    eval_as_output,
                                                                    used_base_model,
                                                                    used_lora_weights,
                                                                    used_inference_server,
                                                                    )
    eval_out_filename = os.path.join(scoring_path, eval_out_filename)

    # torch.device("cuda") leads to cuda:x cuda:y mismatches for multi-GPU consistently
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available else 0
    device = 'cpu' if n_gpus == 0 else 'cuda'
    context_class = NullContext if n_gpus > 1 or n_gpus == 0 else torch.device

    with context_class(device):
        # ensure was set right above before examples generated
        assert not stream_output, "stream_output=True does not make sense with example loop"
        import time
        from functools import partial

        # get score model
        smodel, stokenizer, sdevice = get_score_model(reward_type=True,
                                                      **get_kwargs(get_score_model, exclude_names=['reward_type'],
                                                                   **locals()))

        if not eval_as_output:
            model, tokenizer, device = get_model(reward_type=False,
                                                 **get_kwargs(get_model, exclude_names=['reward_type'], **locals()))
            model_dict = dict(base_model=base_model, tokenizer_base_model=tokenizer_base_model,
                              lora_weights=lora_weights,
                              inference_server=inference_server, prompt_type=prompt_type, prompt_dict=prompt_dict)
            model_state = dict(model=model, tokenizer=tokenizer, device=device)
            model_state.update(model_dict)
            my_db_state = [None]
            fun = partial(evaluate, model_state, my_db_state,
                          **get_kwargs(evaluate, exclude_names=['model_state', 'my_db_state'] + eval_func_param_names,
                                       **locals()))
        else:
            assert eval_prompts_only_num > 0

            def get_response(*args, exi=0):
                # assumes same ordering of examples and responses
                yield responses[exi]

            fun = get_response
        t0 = time.time()
        score_dump = []
        score_avg = 0
        score_median = 0

        for exi, ex in enumerate(examples):
            clear_torch_cache()

            instruction = ex[eval_func_param_names.index('instruction_nochat')]
            iinput = ex[eval_func_param_names.index('iinput_nochat')]
            context = ex[eval_func_param_names.index('context')]
            clear_torch_cache()
            print("")
            print("START" + "=" * 100)
            print("Question: %s %s" % (instruction, ('input=%s' % iinput if iinput else '')))
            print("-" * 105)
            # fun yields as generator, so have to iterate over it
            # Also means likely do NOT want --stream_output=True, else would show all generations
            t1 = time.time()
            gener = fun(*tuple(ex), exi=exi) if eval_as_output else fun(*tuple(ex))
            for res_fun in gener:
                res = res_fun['response']
                extra = res_fun['sources']
                print(res)
                if smodel:
                    score_with_prompt = False
                    if score_with_prompt:
                        data_point = dict(instruction=instruction, input=iinput, context=context)
                        prompter = Prompter(prompt_type, prompt_dict,
                                            debug=debug, chat=chat, stream_output=stream_output)
                        prompt = prompter.generate_prompt(data_point)
                    else:
                        # just raw input and output
                        if eval_prompts_only_num > 0:
                            # only our own examples have this filled at moment
                            assert iinput in [None, ''], iinput  # should be no iinput
                        if not (chat_context and prompt_type == 'human_bot'):
                            assert context in [None, ''], context  # should be no context
                        prompt = instruction
                    if memory_restriction_level > 0:
                        cutoff_len = 768 if memory_restriction_level <= 2 else 512
                    else:
                        cutoff_len = tokenizer.model_max_length
                    inputs = stokenizer(prompt, res,
                                        return_tensors="pt",
                                        truncation=True,
                                        max_length=cutoff_len)
                    try:
                        score = torch.sigmoid(smodel(**inputs).logits[0].float()).cpu().detach().numpy()[0]
                    except torch.cuda.OutOfMemoryError as e:
                        print("GPU OOM 1: question: %s answer: %s exception: %s" % (prompt, res, str(e)),
                              flush=True)
                        traceback.print_exc()
                        score = 0.0
                        clear_torch_cache()
                    except (Exception, RuntimeError) as e:
                        if 'Expected all tensors to be on the same device' in str(e) or \
                                'expected scalar type Half but found Float' in str(e) or \
                                'probability tensor contains either' in str(e) or \
                                'cublasLt ran into an error!' in str(e):
                            print("GPU error: question: %s answer: %s exception: %s" % (prompt, res, str(e)),
                                  flush=True)
                            traceback.print_exc()
                            score = 0.0
                            clear_torch_cache()
                        else:
                            raise
                    score_dump.append(ex + [prompt, res, score])
                    # dump every score in case abort
                    df_scores = pd.DataFrame(score_dump,
                                             columns=eval_func_param_names + eval_extra_columns)
                    df_scores.to_parquet(eval_out_filename, index=False)
                    # plot histogram so far
                    plt.figure(figsize=(10, 10))
                    plt.hist(df_scores['score'], bins=20)
                    score_avg = np.mean(df_scores['score'])
                    score_median = np.median(df_scores['score'])
                    print("SCORE %s: %s  So far: AVG: %s MEDIAN: %s" % (exi, score, score_avg, score_median),
                          flush=True)
                    plt.title("Score avg: %s median: %s" % (score_avg, score_median))
                    plt.savefig(eval_out_filename.replace('.parquet', '.png'))
                    plt.close()

            print("END" + "=" * 102)
            print("")
            t2 = time.time()
            print("Time taken for example: %s Time taken so far: %.4f about %.4g per example" % (
                t2 - t1, t2 - t0, (t2 - t0) / (1 + exi)))
        t1 = time.time()
        print("Total time taken: %.4f about %.4g per example" % (t1 - t0, (t1 - t0) / num_examples))
        print("Score avg: %s median: %s" % (score_avg, score_median), flush=True)
    return eval_out_filename
