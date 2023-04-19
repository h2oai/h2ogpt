# Building the World's Best Open-Source Large Language Model: H2O.ai's Journey

Author: Arno Candel, PhD CTO H2O.ai

At H2O.ai, we pride ourselves on developing world-class Machine Learning, Deep Learning, and AI platforms. Our commitment to open-source technology has led us to create yet another groundbreaking innovation: an open-source, commercially usable large language model (LLM) that can be easily integrated with downstream applications and low/no-code platforms.

In this blog, we'll explore our journey in building this state-of-the-art LLM and share the exciting features it offers.

## Why Build an Open-Source LLM?

In the rapidly growing field of AI, large language models have proven their immense potential in various applications such as natural language processing, code completion, reasoning, mathematics, and more. At H2O.ai, we wanted to build an LLM that not only excelled in performance but was also open-source and commercially usable, providing a valuable resource for developers, researchers, and organizations worldwide.

## The H2O.ai LLM Ecosystem

Our open-source LLM ecosystem includes the following components:

1. **Code, data, and models**: We provide a fully permissive, commercially usable open-source repository that includes code, data, and models.
2. **Fine-tuning**: Our ecosystem features code for preparing large open-source datasets for fine-tuning, including prompt engineering, making it easy to fine-tune LLMs up to 20 billion parameters on commodity hardware and enterprise GPU servers.
3. **Chatbot**: We offer code to run a chatbot on GPU servers, with a shareable end-point and Python client API, allowing you to evaluate and compare the performance of fine-tuned LLMs.
4. **H2O LLM Studio**: Our no-code LLM fine-tuning framework makes it even easier to work with our models.

## Roadmap and Future Plans

We have an ambitious roadmap for our LLM ecosystem, including:

- Integration with downstream applications and low/no-code platforms
- Complementing our chatbot with search and other APIs
- High-performance distributed training of larger models on trillion tokens
- Improvements in code completion, reasoning, mathematics, factual correctness, hallucinations, and reducing repetitions

## Getting Started with H2O.ai's LLM

To start using our LLM, follow the steps below:

1. Clone the repository: `git clone https://github.com/h2oai/h2ogpt.git`
2. Change to the repository directory: `cd h2ogpt`
3. Install the requirements: `pip install -r requirements.txt`
4. Run the chatbot: `python generate.py --base_model=h2oai/h2ogpt-oig-oasst1-256-6.9b`
5. Open your browser at `http://0.0.0.0:7860` or the public live URL printed by the server.

For more information, visit [H2O.ai's Hugging Face page](https://huggingface.co/h2oai).

## Why Choose H2O.ai?

Our team at H2O.ai has a long history of developing groundbreaking AI platforms, including H2O-3, H2O Driverless AI, H2O Hydrogen Torch, and Document AI. Our platforms for deployment, monitoring, data wrangling, and governance have garnered immense trust from our customers.

With H2O.ai's LLM, you're not only getting a powerful, open-source large language model, but also the backing of a company that's passionate about AI innovation and committed to helping you succeed.

Join us on this exciting journey as we continue to improve and expand the capabilities of our open-source LLM ecosystem!

