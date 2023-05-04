import os
import pandas as pd
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import ConversationalRetrievalChain


def get_db(pdf_filename, split_method='chunk', use_openai=False):
    if split_method == 'page':
        # Simple method - Split by pages
        loader = PyPDFLoader(pdf_filename)
        pages = loader.load_and_split()
        print(pages[0])

        # SKIP TO STEP 2 IF YOU'RE USING THIS METHOD
        chunks = pages
    elif split_method == 'chunk':
        # Advanced method - Split by chunk

        # Step 1: Convert PDF to text
        import textract
        doc = textract.process(pdf_filename)

        # Step 2: Save to .txt and reopen (helps prevent issues)
        txt_filename = pdf_filename.replace('.pdf', '.txt')
        with open(txt_filename, 'w') as f:
            f.write(doc.decode('utf-8'))

        with open(txt_filename, 'r') as f:
            text = f.read()

        # Step 3: Create function to count tokens
        from transformers import GPT2TokenizerFast
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        def count_tokens(textin: str) -> int:
            return len(tokenizer.encode(textin))

        # Step 4: Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            # Set a really small chunk size, just to show.
            chunk_size=512,
            chunk_overlap=24,
            length_function=count_tokens,
        )

        chunks = text_splitter.create_documents([text])

        # show_counts(chunks, count_tokens)
    else:
        raise RuntimeError("No such split_method=%s" % split_method)

    # Result is many LangChain 'Documents' around 500 tokens or less (Recursive splitter sometimes allows more tokens to retain context)
    type(chunks[0])

    # Get embedding model
    if use_openai:
        assert os.getenv("OPENAI_API_KEY") is not None, "Set ENV OPENAI_API_KEY"
        from langchain.embeddings import OpenAIEmbeddings
        embedding = OpenAIEmbeddings()
    else:
        from langchain.embeddings import HuggingFaceEmbeddings

        model_name = "sentence-transformers/all-mpnet-base-v2"
        device, torch_dtype, context_class = get_device_dtype()
        model_kwargs = {'device': device}
        embedding = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    # Create vector database
    db = FAISS.from_documents(chunks, embedding)

    return db


def get_context(db, query="Who created transformers?", chat_history='', use_openai=False):
    # Check similarity search is working
    docs = db.similarity_search(query)
    print(docs[0])

    # Create QA chain to integrate similarity search with user queries (answer query from knowledge base)

    if use_openai:
        from langchain.llms import OpenAI  # FIXME remove and use our model
        llm = OpenAI(temperature=0)
        chain = load_qa_chain(llm, chain_type="stuff")
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM

        #model_name = "cerebras/Cerebras-GPT-2.7B"
        model_name = "cerebras/Cerebras-GPT-13B"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        device, torch_dtype, context_class = get_device_dtype()

        with context_class(device):
            load_8bit = False
            # FIXME: for now not to spread across hetero GPUs
            # device_map={"": 0} if load_8bit and device == 'cuda' else "auto"
            device_map = {"": 0} if device == 'cuda' else "auto"
            model = AutoModelForCausalLM.from_pretrained(model_name,
                                                         device_map=device_map,
                                                         torch_dtype=torch_dtype,
                                                         load_in_8bit=load_8bit)
            from transformers import pipeline
            from langchain.llms import HuggingFacePipeline
            pipe = pipeline(
                "text-generation", model=model, tokenizer=tokenizer,
                max_new_tokens=100, early_stopping=True, no_repeat_ngram_size=2
            )
            llm = HuggingFacePipeline(pipeline=pipe)
        from langchain import PromptTemplate
        from langchain import LLMChain

        template = """
        {question}
        """

        prompt = PromptTemplate(
            input_variables=["question"],
            template=template,
        )

        chain = LLMChain(
            llm=llm,
            verbose=True,
            prompt=prompt
        )
        # WIP OPTIONAL
        if False:
            from langchain import SerpAPIWrapper
            from langchain.agents import Tool
            from langchain.agents import initialize_agent

            serpapi = SerpAPIWrapper(serpapi_api_key='...')
            tools = [
                Tool(
                    name="Search",
                    func=serpapi.run,
                    description="useful for when you need to get a weather forecast"
                )
            ]

            agent = initialize_agent(tools, llm, agent="zero-shot-react-description", verbose=True)
            res = agent.run(input="What is the weather forecast for Poznan, Poland")
            print(res)

    query = "Who created transformers?"
    docs = db.similarity_search(query)

    chain.run(input_documents=docs, question=query)

    # Create conversation chain that uses our vectordb as retriever, this also allows for chat history management
    qa = ConversationalRetrievalChain.from_llm(llm, db.as_retriever())

    result = qa({"question": query, "chat_history": chat_history})

    return result['answer']


def get_device_dtype():
    # torch.device("cuda") leads to cuda:x cuda:y mismatches for multi-GPU consistently
    import torch
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available else 0
    device = 'cpu' if n_gpus == 0 else 'cuda'
    # from utils import NullContext
    # context_class = NullContext if n_gpus > 1 or n_gpus == 0 else context_class
    context_class = torch.device
    torch_dtype = torch.float16 if device == 'cuda' else torch.float32
    return device, torch_dtype, context_class


def show_counts(chunks, count_tokens):
    # Quick data visualization to ensure chunking was successful

    # Create a list of token counts
    token_counts = [count_tokens(chunk.page_content) for chunk in chunks]

    # Create a DataFrame from the token counts
    df = pd.DataFrame({'Token Count': token_counts})

    # Create a histogram of the token count distribution
    df.hist(bins=40, )

    # Show the plot
    import matplotlib.pyplot as plt
    plt.show()


def test_demo_openai():
    return run_demo(use_openai=True)


def test_demo_hf():
    return run_demo(use_openai=False)


def run_demo(use_openai=False):
    # quick test
    pdf_filename = '1706.03762.pdf'
    if not os.path.isfile(pdf_filename):
        if os.path.isfile('1706.03762'):
            os.remove('1706.03762')
        os.system("wget --user-agent TryToStopMeFromUsingWgetNow https://arxiv.org/pdf/1706.03762")
        os.rename('1706.03762', '1706.03762.pdf')
    db = get_db(pdf_filename, split_method='chunk', use_openai=use_openai)
    answer = get_context(db, query="Who created transformers?", chat_history='', use_openai=use_openai)
    print(answer)


if __name__ == '__main__':
    test_demo_hf()
