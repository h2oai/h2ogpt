import pytest
from tests.utils import wrap_test_forked


@pytest.mark.need_tokens
@wrap_test_forked
def test_langchain_simple_h2ogpt():
    run_langchain_simple(base_model='h2oai/h2ogpt-oasst1-512-12b', prompt_type='human_bot')


@pytest.mark.need_tokens
@wrap_test_forked
def test_langchain_simple_vicuna():
    run_langchain_simple(base_model='junelee/wizard-vicuna-13b', prompt_type='instruct_vicuna')


def run_langchain_simple(base_model='h2oai/h2ogpt-oasst1-512-12b', prompt_type='human_bot'):
    """
    :param base_model:
    :param prompt_type: prompt_type required for stopping support and correct handling of instruction prompting
    :return:
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from src.h2oai_pipeline import H2OTextGenerationPipeline

    model_name = base_model

    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(base_model, use_auth_token=True,
                                        trust_remote_code=True,
                                        offload_folder="./")

    llama_type_from_config = 'llama' in str(config).lower()
    llama_type_from_name = "llama" in base_model.lower()
    llama_type = llama_type_from_config or llama_type_from_name

    if llama_type:
        from transformers import LlamaForCausalLM, LlamaTokenizer
        model_loader = LlamaForCausalLM
        tokenizer_loader = LlamaTokenizer
    else:
        model_loader = AutoModelForCausalLM
        tokenizer_loader = AutoTokenizer

    load_in_8bit = True
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available else 0
    device = 'cpu' if n_gpus == 0 else 'cuda'
    device_map = {"": 0} if device == 'cuda' else "auto"
    tokenizer = tokenizer_loader.from_pretrained(model_name, padding_side="left")

    model = model_loader.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device_map,
                                         load_in_8bit=load_in_8bit)

    gen_kwargs = dict(max_new_tokens=512, return_full_text=True, early_stopping=False)
    pipe = H2OTextGenerationPipeline(model=model, tokenizer=tokenizer, prompt_type=prompt_type, **gen_kwargs)
    # below makes it listen only to our prompt removal,
    # not built in prompt removal that is less general and not specific for our model
    pipe.task = "text2text-generation"

    # create llm for LangChain
    from langchain.llms import HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=pipe)

    # Setup QA
    from langchain import PromptTemplate
    from langchain.chains.question_answering import load_qa_chain
    # NOTE: Instruct-tuned models don't need excessive many-shot examples that waste context space
    template = """
    ==
    {context}
    ==
    {question}"""
    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=template,
    )
    chain = load_qa_chain(llm, prompt=prompt)
    docs = []  # could have been some Documents from LangChain inputted from some sources
    query = "Give detailed list of reasons for who is smarter, Einstein or Newton."
    chain_kwargs = dict(input_documents=docs, question=query)
    answer = chain(chain_kwargs)
    print(answer)

    if 'vicuna' in base_model:
        res1 = 'Both Albert Einstein and Sir Isaac Newton were brilliant scientists' in answer[
            'output_text'] and "Newton" in answer['output_text']
        res2 = 'Both Albert Einstein and Sir Isaac Newton are considered two' in answer[
            'output_text'] and "Newton" in answer['output_text']
    else:
        res1 = 'Einstein was a genius who revolutionized physics' in answer['output_text'] and "Newton" in answer[
            'output_text']
        res2 = 'Einstein and Newton are two of the most famous scientists in history' in answer[
            'output_text'] and "Newton" in answer['output_text']
    assert res1 or res2
