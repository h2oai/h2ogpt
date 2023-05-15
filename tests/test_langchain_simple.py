def test_langchain_simple():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from h2oai_pipeline import H2OTextGenerationPipeline

    model_name = 'h2oai/h2ogpt-oasst1-512-12b'
    load_in_8bit = True
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available else 0
    device = 'cpu' if n_gpus == 0 else 'cuda'
    device_map = {"": 0} if device == 'cuda' else "auto"
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map=device_map, load_in_8bit=load_in_8bit)

    gen_kwargs = dict(max_new_tokens=512, return_full_text=True, early_stopping=False)
    pipe = H2OTextGenerationPipeline(model=model, tokenizer=tokenizer, **gen_kwargs)
    # below makes it listen only to our prompt removal, not built in prompt removal that is less general and not specific for our model
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
