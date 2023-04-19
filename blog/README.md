# Building the World's Best Open-Source Large Language Model: H2O.ai's Journey

by Arno Candel, PhD, CTO H2O.ai, April 19 2023

At H2O.ai, we pride ourselves on developing world-class Machine Learning, Deep Learning, and AI platforms.
In the rapidly growing field of AI, large language models have proven their immense potential in various applications such as natural language processing, code completion, reasoning, mathematics, and more.

We are proud to announce that we are building h2oGPT, an LLM that not only excels in performance but is also fully open-source and commercially usable, providing a valuable resource for developers, researchers, and organizations worldwide.

In this blog, we'll explore our journey in building this state-of-the-art LLM.

## Why Build an Open-Source LLM?

Open-source Large Language Models (LLMs) are needed today for several reasons:
1. **Innovation and Customization**: Open-source LLMs foster collaboration and innovation, enabling users to adapt models to their unique needs and drive advancements in AI and natural language processing.
2. **Transparency and Skill Development**: Open-source LLMs provide transparency, helping users identify biases or limitations while serving as educational resources to develop skills and grow the AI community.
3. **Competitive Landscape and Data Security**: Open-source LLMs promote competition in the AI industry and allow on-premise deployment, ensuring data privacy and security while leveraging the power of large language models.

The main goal of building h2oGPT was to overcome limitations in existing models with non-permissive licenses or data restrictions for commercial use. We aimed to create a powerful LLM using fully permissive data and models, enabling broader access for businesses and commercial products without legal concerns, thus expanding access to cutting-edge AI while adhering to licensing requirements.

## The H2O.ai LLM Ecosystem

Our open-source LLM ecosystem currently includes the following components:

1. **Code, data, and models**: We provide a fully permissive, commercially usable open-source repository that includes code, data, and models.
2. **Fine-tuning**: Our ecosystem features code for preparing large open-source datasets for fine-tuning, including prompt engineering, making it easy to fine-tune LLMs up to 20 billion parameters (hopefully more soon) on commodity hardware and enterprise GPU servers.
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

For more information, visit [H2O.ai's Hugging Face page](https://huggingface.co/h2oai) and [H2O LLM Studio](https://github.com/h2oai/h2o-llmstudio) GitHub repository.

## Why Choose H2O.ai?

Join us on this exciting journey as we continue to improve and expand the capabilities of our open-source LLM ecosystem!

