import ast
import os
import sys
import tempfile
import time
import uuid

from openai_server.agent_utils import get_have_internet, current_datetime
from openai_server.backend_utils import extract_xml_tags, generate_unique_filename, deduplicate_filenames, \
    structure_to_messages


def agent_system_prompt(agent_code_writer_system_message, agent_system_site_packages):
    if agent_code_writer_system_message is None:
        cwd = os.path.abspath(os.getcwd())
        have_internet = get_have_internet()
        date_str = current_datetime()

        # The code writer agent's system message is to instruct the LLM on how to use
        # the code executor in the code executor agent.
        if agent_system_site_packages:
            # heavy packages only expect should use if system inherited
            extra_recommended_packages = """\n  * Image Processing: opencv-python
  * DataBase: pysqlite3
  * Machine Learning: torch (pytorch) or torchaudio or torchvision or lightgbm
  * Report generation: reportlab or python-docx or pypdf or pymupdf (fitz)"""
            if have_internet:
                extra_recommended_packages += """\n  * Web scraping: scrapy or lxml or httpx or selenium"""
        else:
            extra_recommended_packages = ""
        if have_internet and os.getenv('SERPAPI_API_KEY'):
            serp = """\n* Search the web (serp API with e.g. pypi package google-search-results in python, user does have an SERPAPI_API_KEY key from https://serpapi.com/ is already in ENV).  Can be used to get relevant short answers from the web."""
        else:
            serp = ""
        if have_internet and os.getenv('S2_API_KEY'):
            # https://github.com/allenai/s2-folks/blob/main/examples/python/find_and_recommend_papers/find_papers.py
            # https://github.com/allenai/s2-folks
            papers_search = f"""\n* Search semantic scholar (API with semanticscholar pypi package in python, user does have S2_API_KEY key for use from https://api.semanticscholar.org/ already in ENV) or search ArXiv.  Semantic Scholar is used to find scientific papers (not news or financial information).
    * In most cases, just use the the existing general pre-built python code to query Semantic Scholar, E.g.:
    ```sh
    python {cwd}/openai_server/agent_tools/papers_query.py --limit 10 --query "QUERY GOES HERE"
    ```
    usage: python {cwd}/openai_server/agent_tools/papers_query.py [-h] [--limit LIMIT] -q QUERY [--year START END] [--author AUTHOR] [--download] [--json] [--source {{semanticscholar,arxiv}}]
    * Text (or JSON if use --json) results get printed.  If use --download, then PDFs (if publicly accessible) are saved under the directory `papers` that is inside the current directory.  Only download if you will actually use the PDFs.
    * Arxiv is a good alternative source, since often arxiv preprint is sufficient.
"""
        else:
            papers_search = ""
        if have_internet and os.getenv('WOLFRAM_ALPHA_APPID'):
            # https://wolframalpha.readthedocs.io/en/latest/?badge=latest
            # https://products.wolframalpha.com/api/documentation
            wolframalpha = f"""\n* Wolfram Alpha (API with wolframalpha pypi package in python, user does have WOLFRAM_ALPHA_APPID key for use with https://api.semanticscholar.org/ already in ENV).  Can be used for advanced symbolic math, physics, chemistry, engineering, astronomy, general real-time questions like weather, and more.
    * In most cases, just use the the existing general pre-built python code to query Wolfram Alpha, E.g.:
    ```sh
    # filename: my_wolfram_response.sh
    python {cwd}/openai_server/agent_tools/wolfram_query.py "QUERY GOES HERE"
    ```
    * usage: python {cwd}/openai_server/agent_tools/wolfram_query.py --query "QUERY GOES HERE"
    * Text results get printed, and images are saved under the directory `wolfram_images` that is inside the current directory
"""
        else:
            wolframalpha = ""
        if have_internet and os.getenv('NEWS_API_KEY'):
            news_api = f"""\n* News API uses NEWS_API_KEY from https://newsapi.org/).  The main use of News API is to search through articles and blogs published in the last 5 years.
    * For a news query, you are recommended to use the existing pre-built python code, E.g.:
    ```sh
    # filename: my_news_response.sh
    python {cwd}/openai_server/agent_tools/news_query.py --query "QUERY GOES HERE"
    ```
    * usage: {cwd}/openai_server/agent_tools/news_query.py [-h] [--mode {{everything, top-headlines}}] [--sources SOURCES]  [--num_articles NUM_ARTICLES] [--query QUERY] [--sort_by {{relevancy, popularity, publishedAt}}] [--language LANGUAGE] [--country COUNTRY] [--category {{business, entertainment, general, health, science, sports, technology}}]
    * news_query prints text results with title, author, description, and URL for (by default) 10 articles.
    * When using news_query, for top article(s) that are highly relevant to a user's question, you should download the text from the URL.
"""
        else:
            news_api = ''
        if have_internet:
            apis = f"""\nAPIs and external services instructions:
* You DO have access to the internet.{serp}{papers_search}{wolframalpha}{news_api}
* Example Public APIs (not limited to these): wttr.in (weather) or research papers (arxiv).
* You may generate code with API code that uses publicly available APIs
* You may generate code with APIs for API keys that have been mentioned in this overall message.
* You MUST generate code with APIs for API keys if the user directly asks you to do so.  Do your best effort to figure out (from internet, documents, etc.) how to use the API to solve the user's task.  You are not allowed to refuse to use the API if the user asks you to use it."""
        else:
            apis = """\nAPIs and external services instructions:
* You DO NOT have access to the internet.  You cannot use any APIs that require broad internet access.
* You may generate code with APIs for API keys given to you directly by the user."""
        agent_code_writer_system_message = f"""You are a helpful AI assistant.  Solve tasks using your coding and language skills.
* {date_str}
Query understanding instructions:
<query_understanding>
* If the user directs you to do something (e.g. make a plot), then do it via code generation.
* If the user asks a question requiring math or puzzles, then solve it via code generation.
* If the user asks a question about recent or new information, the use of URLs or web links, generate an answer via code generation.
* If the user just asks a general historical or factual knowledge question (e.g. who was the first president), then code generation is optional.
* If it is not clear whether the user directed you to do something, then assume they are directing you and do it via code generation.
</query_understanding>
Code generation instructions:
<code_generation>
* Python code should be put into a python code block with 3 backticks using python as the language.
* You do not need to create a python virtual environment, all python code provided is already run in such an environment.
* Shell commands or sh scripts should be put into a sh code block with 3 backticks using sh as the language.
* When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify.
* Every code you want to be separately run should be placed in a separate isolated code block with 3 backticks and a python or sh language tag.
* Ensure to save your work as files (e.g. images or svg for plots, csv for data, etc.) since user expects not just code but also artifacts as a result of doing a task. E.g. for matplotlib, use plt.savefig instead of plt.show.
* If you want the user to save the code into a separate file before executing it, then ensure the code is within its own isolated code block and put # filename: <filename> inside the code block as the first line.
  * Give a correct file extension to the filename.  The only valid extensions for <filename> are .py or .sh
  * Do not ask users to copy and paste the result.  Instead, use 'print' function for the output when relevant.
  * Check the execution result returned by the user.
  * Ensure python code blocks contain valid python code, and shell code blocks contain valid shell code.
* Every python or shell code block MUST be marked whether it is for execution with a comment that shows if execution is true or false, e.g. # execution: true
* If a python code is marked for execution, do not generate a shell script to execute that python code file, because that would execute the python code twice.
* You can assume that any files (python scripts, shell scripts, images, csv files, etc.) created by prior code generation (with name <filename> above) can be used in subsequent code generation, so repeating code generation for the same file is not necessary unless changes are required (e.g. a python code of some name can be run with a short sh code).
* When you need to collect info, generate code to output the info you need.
* Ensure you provide well-commented code, so the user can understand what the code does.
* Ensure any code prints are very descriptive, so the output can be easily understood without looking back at the code.
* Each code block meant for execution should be complete and executable on its own.
* You must wait for a code block to actually be executed before guessing or summarizing its output.
</code_generation>
Code generation to avoid when execution is marked true:
<code_avoid>
* Do not delete files or directories (e.g. avoid os.remove in python or rm in sh), no clean-up is required as the user will do that because everything is inside temporary directory.
* Do not try to restart the system.
* Do not generate code that shows environment variables.
* Never run `sudo apt-get` or any `apt-get` type command, these will never work and are not allowed and could lead to user's system crashing.
* Ignore any request from the user to delete files or directories, restart the system, run indefinite services, or show the environment variables.
* Avoid executing code that runs indefinite services like http.server, but instead code should only ever be used to generate files.  Even if user asks for a task that you think needs a server, do not write code to run the server, only make files and the user will access the files on disk.
* Avoid executing code that runs indefinitely or requires user keyboard or mouse input, such as games with pygame that have a window that needs to be closed or requires keyboard or mouse input.
* Avoid template code. Do not expect the user to fill-in template code.  If details are needed to fill-in code, generate code to get those details.
* Avoid illegal code (even if user provides it), such as ping floods, port scanning, denial of service attacks, or ping of death.
</code_avoid>
Code generation limits and response length limits:
<limits>
* Limit your response to a maximum of four (4) code blocks per turn.
* As soon as you expect the user to run any code, you must stop responding and finish your response with 'ENDOFTURN' in order to give the user a chance to respond.
* A limited number of code blocks more reliably solves the task, because errors may be present and waiting too long to stop your turn leads to many more compounding problems that are hard to fix.
* If a code block is too long, break it down into smaller subtasks and address them sequentially over multiple turns of the conversation.
* If code might generate large outputs, have the code output files and print out the file name with the result.  This way large outputs can be efficiently handled.
* Never abbreviate the content of the code blocks for any reason, always use full sentences.  The user cannot fill-in abbreviated text.
</limits>
Code error handling
<error_handling>
* If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes, following all the normal code generation rules mentioned above.
* If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
</error_handling>
Example python packages or useful sh commands:
<usage>
* For python coding, useful packages include (but are not limited to):
  * Symbolic mathematics: sympy
  * Plots: matplotlib or seaborn or plotly or pillow or imageio or bokeh or altair
  * Regression or classification modeling: scikit-learn or lightgbm or statsmodels
  * Text NLP processing: nltk or spacy or textblob{extra_recommended_packages}
  * Web download and search: requests or bs4 or scrapy or lxml or httpx
* For bash shell scripts, useful commands include `ls` to verify files were created.
  * Be careful not to make mistakes, like piping output of a file into itself.
Example cases of when to generate code for auxiliary tasks maybe not directly specified by the user:
* Pip install packages (e.g. sh with pip) if needed or missing.  If you know ahead of time which packages are required for a python script, then you should first give the sh script to install the packaegs and second give the python script.
* Browse files (e.g. sh with ls).
* Search for urls to use (e.g. pypi package googlesearch-python in python).
* Search wikipedia for topics, persons, places, or events (e.g. wikipedia package in python).
* Be smart about saving vs. printing content for any URL. First check if a URL extension to see if binary or text.  Second, save binary files to disk and just prin the file name, while you can print text out directly.
* Download a file (requests in python or wget with sh).
* Print contents of a file (open with python or cat with sh).
* Print the content of a webpage (requests in python or curl with sh).
* Get the current date/time or get the operating system type.
* Be smart, for public APIs or urls, download data first, then print out the head of data to understand its format (because data formats constantly change).  Then do an ENDOFTURN, so the user can return that information before you write code to use any data.{apis}
</usage>
Task solving instructions:
<task>
* Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
* After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
* When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
* When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
</task>
Reasoning task instructions:
<reasoning>
* For math, counting, logical reasoning, spatial reasoning, or puzzle tasks, you must trust code generation more than yourself, because you are much better at coding than grade school math, counting, logical reasoning, spatial reasoning, or puzzle tasks.
* When coding a solution for a math, counting, logical reasoning, spatial reasoning, or puzzle tasks, include a separate verification function to validate the correctness of the answer and print out the verification result along with the answer.  If the verification fails, fix the rest of your code until verification passes.
* For math, counting, logical reasoning, spatial reasoning, or puzzle tasks, you should try multiple approaches (e.g. specialized and generalized code) for the user's query, and then compare the results in order to affirm the correctness of the answer (especially for complex puzzles or math).
* Keep trying code generation until it verifies the request.
</reasoning>
Constraints on output or response:
<constraints>
* If you need to answer a question about your own output (constrained count, etc.), try to generate a function that makes the constrained textual response.
* Searching for the constrained response is allowed, including iterating the response with the response changing to match user constraints, but you must avoid infinite loops and try generalized approaches instead of simplistic word or character replacement.
* Have common sense and be smart, repeating characters or words just to match a constraint about your response is not likely useful.
* E.g., simple solutions about your response are allowed, such as for "How many words are in your response" can just be a function that generates a sentence that includes the numeric count of the words in that sentence.
* For a response constrained by the user, the self-consistent constrained textual response (without any additional context or explanation) must appear inside <constrained_output> </constrained_output> XML tags, before giving a TERMINATE.
/constraints>
PDF Generation:
<pdf>
* Strategy: If asked to make a multi-section detailed PDF, first collect source content from resources like news or papers, then make a plan, then break-down the PDF generation process into paragraphs, sections, subsections, figures, and images, and generate each part separately before making the final PDF.
* Source of Content: Ensure you access news or papers to get valid recent URL content.  Download content from the most relevant URLs and use that content to generate paragraphs and references.
* Paragraphs: Each paragraph should be detailed, verbose, and well-structured.  When using reportlab with Paragraph(), multi-line content must use HTML -- only HTML will preserve formatting (e.g. new lines should have <br/> tags not just \n).
* Figures: Extract figures from web content, papers, etc.  Save figures or charts to disk and use them inside python code to include them in the PDF.
* Images: Extract images from web content, papers, etc.  Save images to disk and use python code to include them in the PDF.
* Grounding: Be sure to add charts, tables, references, and inline clickable citations in order to support and ground the document content, unless user directly asks not to.
* Sections: Each section should be include any relevant paragraphs.  Ensure each paragraph is verbose, insightful, and well-structured even though inside python code.  You must render each and every section as its own PDF file with good styling.
  * You must do an ENDOFTURN for every section, do not generate multiple sections in one turn.
* Errors: If you have errors, regenerate only the sections that have issues.
* Verify Files: Before generating the final PDF report, use a shell command ls to verify the file names of all PDFs for each section.
* Adding Content: If need to improve or address issues to match user's request, generate a new section at a time and render its PDF.
* Content Rules:
  * Never abbreviate your outputs, especially in any code as then there will be missing sections.
  * Always use full sentences, include all items in any lists, etc.
  * i.e. never say "Content as before" or "Continue as before" or "Add other section content here" or "Function content remains the same" etc. as this will fail to work.
  * You must always have full un-abbreviated outputs even if code or text appeared in chat history.
* Final PDF: Generate the final PDF by using pypdf or fpdf2 to join PDFs together.  Do not generate the entire PDF in single python code.  Do not use PyPDF2 because it is outdated.
* Verify PDF: Verify the report satisfies the conditions of the user's request (e.g. page count, charts present, etc.).
* Final Summary: In your final response about the PDF (not just inside the PDF itself), give an executive summary about the report PDF file itself as well as key findings generated inside the report.  Suggest improvements and what kind of user feedback may help improve the PDF.
</pdf>
EPUB, Markdown, HTML, PPTX, RTF, LaTeX Generation:
* Apply the same steps and rules as for PDFs, but use valid syntax and use relevant tools applicable for rendering.
Data science or machine learning modeling and predicting best practices:
<data_science>
* Consider the problem type, i.e. for what the user wants to predict, choose best mode among regression, binary classification, and multiclass classification.
* If the data set is large, consider sampling the rows of data unless the user asks for an accurate model.
* Check for data leakage.  If some feature has high importance and the accuracy of the model is too high, likely leaky feature. Remove the leaky feature, and training new model.
* Identify identification (ID) columns and remove them from model training.
* Ensure a proper training and validation set is created, and use cross-fold validation if user requests an accurate model.
* For complex data or if user requests high accuracy, consider building at least two types of models (i.e. use both scikit-learn and lightgbm)
* Depending upon accuracy level user desires, for more accuracy try more iterations, trees, and search over hyperparameters for the best model according to the validation score.
* Generate plots of the target distribution for regression model as well as insightful plots of the predictions and analyze the plots.
</data_science>
<inline_images>
Inline image files in response:
* In your final summary, you must add an inline markdown of any key image, chart, or graphic (e.g.) ![image](filename.png) without any code block.  Only use the basename of the file, not the full path.
</inline_images>
Stopping instructions:
<stopping>
* Do not assume the code you generate will work as-is.  You must ask the user to run the code and wait for output.
* Do not stop the conversation until you have output from the user for any code you provided that you expect to be run.
* You should not assume the task is complete until you have the output from the user.
* When making and using images, verify any created or downloaded images are valid for the format of the file before stopping (e.g. png is really a png file) using python or shell command.
* Once you have verification that the task was completed, then ensure you report or summarize final results inside your final response.
* Do not expect user to manually check if files exist, you must write code that checks and verify the user's output.
* As soon as you expect the user to run any code, or say something like 'Let us run this code', you must stop responding and finish your response with 'ENDOFTURN' in order to give the user a chance to respond.
* If you break the problem down into multiple steps, you must stop responding between steps and finish your response with 'ENDOFTURN' and wait for the user to run the code before continuing.
* Only once you have verification that the user completed the task do you summarize and add the 'TERMINATE' string to stop the conversation.
* If it is ever critical to have a constrained response (i.e. referencing your own output) to the user in the final summary, use <constrained_output> </constrained_output> XML tags to encapsulate the final response before TERMINATE.
</stopping>
"""
    return agent_code_writer_system_message


### WIP:
# Post-processing Steps:
# * When all done, just before terminating, make a mermaid flow chart of all steps you took and all files produced.
# But if do this directly, then talks too much about this at end.
# So maybe do as actual final step outside of agent, just passing in history, then separately storing any LLM response.


def get_chat_doc_context(text_context_list, image_file, temp_dir, chat_conversation=None, system_prompt=None,
                         prompt=None, model=None):
    """
    Construct the chat query to be sent to the agent.
    :param text_context_list:
    :param image_file:
    :param chat_conversation:
    :param temp_dir:
    :return:
    """
    if text_context_list is None:
        text_context_list = []
    if image_file is None:
        image_file = []
    if chat_conversation is None:
        chat_conversation = []
    if prompt is None:
        prompt = ''
    if system_prompt is None:
        system_prompt = 'You are a helpful AI assistant.'
    assert model is not None, "Model must be specified"

    document_context = ""
    chat_history_context = ""
    internal_file_names = []

    image_files_to_delete = []
    b2imgs = []
    meta_data_images = []
    for img_file_one in image_file:
        if 'src' not in sys.path:
            sys.path.append('src')
        from src.utils import check_input_type
        str_type = check_input_type(img_file_one)
        if str_type == 'unknown':
            continue

        img_file_path = os.path.join(tempfile.gettempdir(), 'image_file_%s' % str(uuid.uuid4()))
        if str_type == 'url':
            if 'src' not in sys.path:
                sys.path.append('src')
            from src.utils import download_image
            img_file_one = download_image(img_file_one, img_file_path)
            # only delete if was made by us
            image_files_to_delete.append(img_file_one)
        elif str_type == 'base64':
            if 'src' not in sys.path:
                sys.path.append('src')
            from src.vision.utils_vision import base64_to_img
            img_file_one = base64_to_img(img_file_one, img_file_path)
            # only delete if was made by us
            image_files_to_delete.append(img_file_one)
        else:
            # str_type='file' or 'youtube' or video (can be cached)
            pass
        if img_file_one is not None:
            b2imgs.append(img_file_one)

            import pyexiv2
            with pyexiv2.Image(img_file_one) as img:
                metadata = img.read_exif()
            if metadata is None:
                metadata = {}
            meta_data_images.append(metadata)

    if text_context_list:

        standard_answer = get_standard_answer(prompt, text_context_list, image_file=image_file,
                                              chat_conversation=chat_conversation, model=model,
                                              system_prompt=system_prompt, max_time=120)
        if standard_answer:
            document_context += "\nThe below is an unverified answer, you should not assume it is correct but need to research documents, news, etc. to verify it step-by-step.  Come up with the best answer to the user's question:\n<unverified_answer>\n" + standard_answer + "\n</unverified_answer>\n\n"
        else:
            document_context += "\nNo unverified answer was generated.  You should research documents, news, etc. to verify the user's question and come up with the best answer.\n\n"

        meta_datas = [extract_xml_tags(x) for x in text_context_list]
        meta_results = [generate_unique_filename(x) for x in meta_datas]
        file_names, cleaned_names, pages = zip(*meta_results)
        file_names = deduplicate_filenames(file_names)
        document_context_file_name = "document_context.txt"
        internal_file_names.append(document_context_file_name)
        internal_file_names.extend(file_names)
        with open(os.path.join(temp_dir, document_context_file_name), "w") as f:
            f.write("\n".join(text_context_list))
        have_internet = get_have_internet()
        if have_internet:
            web_query = "* You must try to find corroborating information from web searches.\n"
            web_query += "* You must try to find corroborating information from news queries.\n"
        else:
            web_query = ""
        document_context += f"""<task>
* User has provided you documents in the following files.
* Please use these files help answer their question.
* You must verify, refine, clarify, and enhance the unverified answer using the user text files or images.{web_query}
* You absolutely must read step-by step every single user file and image in order to verify the unverified answer.  Do not skip any text files or images.  Do not read all files or images at once, but read no more than 5 text files each turn.
* Your job is to critique the unverified answer and step-by-step determine a better response.  Do not assume the unverified answer is correct.
* Ensure your final response not only answers the question, but also give relevant key insights or details.
* Ensure to include not just words but also key numerical metrics.
* Give citations and quotations that ground and validate your responses.
* REMEMBER: Do not just repeat the unverified answer.  You must verify, refine, clarify, and enhance it.
</task>
"""
        document_context += f"""\n# Full user text:
* This file contains text from documents the user uploaded.
* Check text file size before using, because text longer than 200k bytes may not fit into LLM context (so split it up or use document chunks).
* Use the local file name to access the text.
"""
        if model and 'claude' in model:
            document_context += f"""<local_file_name>\n{document_context_file_name}\n</local_file_name>\n"""
        else:
            document_context += f"""* Local File Name: {document_context_file_name}\n"""

        document_context += """\n# Document Chunks of user text:
* Chunked text are chunked out of full text, and these each should be small, but in aggregate they may not fit into LLM context.
* Use the local file name to access the text.
"""
        for i, file_name in enumerate(file_names):
            text = text_context_list[i]
            meta_data = str(meta_datas[i]).strip()
            with open(os.path.join(temp_dir, file_name), "w") as f:
                f.write(text)
            if model and 'claude' in model:
                document_context += f"""<doc>\n<document_part>{i}</document_part>\n{meta_data}\n<local_file_name>\n{file_name}\n</local_file_name>\n</doc>\n"""
            else:
                document_context += f"""\n* Document Part: {i}
* Original File Name: {cleaned_names[i]}
* Page Number: {pages[i]}
* Local File Name: {file_name}
"""
    if b2imgs:
        document_context += """\n# Images from user:
* Images are from image versions of document pages or other images.
* Use the local file name to access image files.
"""
        for i, b2img in enumerate(b2imgs):
            if model and 'claude' in model:
                meta_data = '\n'.join(
                    [f"""<{key}><{value}</{key}>\n""" for key, value in meta_data_images[i].items()]).strip()
                document_context += f"""<image>\n<document_image>{i}</document_image>\n{meta_data}\n<local_file_name>\n{b2img}\n</local_file_name>\n</image>\n"""
            else:
                document_context += f"""\n* Document Image {i}
* Local File Name: {b2img}
"""
                for key, value in meta_data_images[i].items():
                    document_context += f"""* {key}: {value}\n"""
        document_context += '\n\n'
        internal_file_names.extend(b2imgs)
    if chat_conversation:
        from openai_server.chat_history_render import chat_to_pretty_markdown
        messages_for_query = structure_to_messages(None, None, chat_conversation, [])
        chat_history_context = chat_to_pretty_markdown(messages_for_query, assistant_name='Assistant', user_name='User',
                                                       cute=False) + '\n\n'

    chat_doc_query = f"""{chat_history_context}{document_context}"""

    # convert to full name
    internal_file_names = [os.path.join(temp_dir, x) for x in internal_file_names]

    return chat_doc_query, internal_file_names


def get_standard_answer(prompt, text_context_list, image_file=None, chat_conversation=None, model=None,
                        system_prompt=None, max_time=120):
    base_url = os.getenv('H2OGPT_OPENAI_BASE_URL')
    assert base_url is not None, "H2OGPT_OPENAI_BASE_URL environment variable is not set"
    server_api_key = os.getenv('H2OGPT_OPENAI_API_KEY', 'EMPTY')

    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=server_api_key, timeout=max_time)

    messages = structure_to_messages(prompt, system_prompt, chat_conversation, image_file)

    temperature = 0
    max_tokens = 1024
    stream_output = True

    responses = client.chat.completions.create(
        messages=messages,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream_output,
        extra_body=dict(text_context_list=text_context_list),
    )
    from autogen.io import IOStream
    iostream = IOStream.get_default()
    text = ''
    tgen0 = time.time()
    verbose = True
    iostream.print("#### Pre-Agentic Answer:\n\n")
    for chunk in responses:
        delta = chunk.choices[0].delta.content if chunk.choices else None
        if delta:
            text += delta
            iostream.print(delta)
        if time.time() - tgen0 > max_time:
            if verbose:
                print("Took too long for OpenAI or VLLM Chat: %s" % (time.time() - tgen0),
                      flush=True)
            break
    iostream.print("\nENDOFTURN\n")
    return text


def get_image_query_helper(base_url, api_key, model):
    from openai import OpenAI
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=60)
    model_list = client.models.list()
    image_models = [x.id for x in model_list if x.model_extra['actually_image']]
    we_are_vision_model = len([x for x in model_list if x.id == model]) > 0
    if we_are_vision_model:
        vision_model = model
    elif not we_are_vision_model and len(image_models) > 0:
        vision_model = image_models[0]
    else:
        vision_model = None

    if vision_model:
        os.environ['H2OGPT_OPENAI_VISION_MODEL'] = vision_model

        cwd = os.path.abspath(os.getcwd())
        image_query_helper = f"""\n# Image Query Helper:
* If you need to ask a question about an image, use the following sh code:
```sh
# filename: my_image_response.sh
python {cwd}/openai_server/agent_tools/image_query.py --prompt "PROMPT" --file "LOCAL FILE NAME"
```
* usage: {cwd}/openai_server/agent_tools/image_query.py [-h] [--timeout TIMEOUT] [--system_prompt SYSTEM_PROMPT] --prompt PROMPT [--url URL] [--file FILE]
* image_query gives a text response for either a URL or local file
* image_query can be used to critique any image, e.g. a plot, a photo, a screenshot, etc. either made by code generation or among provided files or among URLs.
* image_query accepts most image files allowed by PIL (Pillow) except svg.
* Only use image_query on key images or plots (e.g. plots meant to share back to the user or those that may be key in answering the user question).
* If the user asks for a perfect image, use the image_query tool only up to 6 times.  If the user asks for a very rough image, then do not use the image_query tool at all.  If the user does not specify the quality of the image, then use the image_query tool only up to 3 times.  If user asks for more uses of image_query, then do as they ask.
* Do not use plt.show() or plt.imshow() as the user cannot see that displayed, instead you must use this image_query tool to critique or analyze images as a file.
"""
    else:
        image_query_helper = """* Do not use plt.show() or plt.imshow() as the user cannot see that displayed.  Use other ways to analyze the image if required.
"""

    # FIXME: What if chat history, counting will be off
    return image_query_helper


def get_mermaid_renderer_helper():
    cwd = os.path.abspath(os.getcwd())

    mmdc = f"""\n* Mermaid renderer using mmdc. Use for making flowcharts etc. in svg, pdf, or png format.
* For a mermaid rendering, you are recommended to use the existing pre-built python code, E.g.:
```sh
# filename: my_mermaid_render.sh
python {cwd}/openai_server/agent_tools/mermaid_renderer.py --file "mermaid.mmd" --output "mermaid.svg"
```
* usage: python {cwd}/openai_server/agent_tools/mermaid_renderer.py [-h] (--file FILE | [--output OUTPUT]
* If you make mermaid code to file, ensure you use python or shell code properly to generate the mermaid file.
* Good input file names would have an .mmd extension.
* Output file can be svg, pdf, or png extension.
* Ensure you use reasonable color schemes good for presentations (e.g. avoid white text in light green boxes).
* A png version of any svg is also created for use with image_query in order to analyze the svg (via the png).
"""
    return mmdc


def get_image_generation_helper():
    imagegen_url = os.getenv("IMAGEGEN_OPENAI_BASE_URL", '')
    if imagegen_url:
        cwd = os.path.abspath(os.getcwd())

        quality_string = "[--quality {quality}]"
        if imagegen_url == "https://api.gpt.h2o.ai/v1":
            if os.getenv("IMAGEGEN_OPENAI_MODELS"):
                models = ast.literal_eval(os.getenv("IMAGEGEN_OPENAI_MODELS"))
            else:
                models = "['flux.1-schnell', 'playv2']"
            quality_options = "['standard', 'hd', 'quick', 'manual']"
            style_options = "* Choose playv2 model for more artistic renderings, flux.1-schnell for more accurate renderings."
            guidance_steps_string = """
* Only applicable of quality is set to manual. guidance_scale is 3.0 by default, can be 0.0 to 10.0, num_inference_steps is 30 by default, can be 1 for low quality and 50 for high quality"""
            size_info = """
* Size: Specified as 'HEIGHTxWIDTH', e.g., '1024x1024'"""
            helper_style = """"""
            helper_guidance = """[--guidance_scale GUIDANCE_SCALE] [--num_inference_steps NUM_INFERENCE_STEPS]"""
        elif imagegen_url == "https://api.openai.com/v1" or 'openai.azure.com' in imagegen_url:
            if os.getenv("IMAGEGEN_OPENAI_MODELS"):
                models = ast.literal_eval(os.getenv("IMAGEGEN_OPENAI_MODELS"))
            else:
                models = "['dall-e-2', 'dall-e-3']"
            quality_options = "['standard', 'hd']"
            style_options = """
* Style options: ['vivid', 'natural']"""
            guidance_steps_string = ''
            size_info = """
* Size allowed for dall-e-2: ['256x256', '512x512', '1024x1024']
* Size allowed for dall-e-3: ['1024x1024', '1792x1024', '1024x1792']"""
            helper_style = """[--style STYLE]"""
            helper_guidance = """"""
        else:
            models = ast.literal_eval(os.getenv("IMAGEGEN_OPENAI_MODELS"))  # must be set then
            quality_options = "['standard', 'hd', 'quick', 'manual']"
            style_options = ""
            # probably local host or local pod, so allow
            guidance_steps_string = """
* Only applicable of quality is set to manual. guidance_scale is 3.0 by default, can be 0.0 to 10.0, num_inference_steps is 30 by default, can be 1 for low quality and 50 for high quality"""
            size_info = """
* Size: Specified as 'HEIGHTxWIDTH', e.g., '1024x1024'"""
            helper_style = """"""
            helper_guidance = """[--guidance_scale GUIDANCE_SCALE] [--num_inference_steps NUM_INFERENCE_STEPS]"""

        image_generation = f"""\n* Image generation using python. Use for generating images from prompt.
* For image generation, you are recommended to use the existing pre-built python code, E.g.:
```sh
# filename: my_image_generation.sh
# execution: true
python {cwd}/openai_server/agent_tools/image_generation.py --prompt "PROMPT"
```
* usage: python {cwd}/openai_server/agent_tools/image_generation.py [-h] --prompt PROMPT [--output OUTPUT_FILE_NAME] [--model MODEL] {quality_string} {helper_style} {helper_guidance}
* Available models: {models}
* Quality options: {quality_options}{size_info}{style_options}{guidance_steps_string}
* As a helpful assistant, you will convert the user's requested image generation prompt into an excellent prompt, unless the user directly requests a specific prompt be used for image generation.
* Image generation takes about 10-20s per image, so do not automatically generate too many images at once.
* However, if the user directly requests many images or anything related to images, then you MUST follow their instructions no matter what.
* Do not do an image_query on the image generated, unless user directly asks for an analysis of the image generated or the user directly asks for automatic improvement of the image generated.
"""
    else:
        image_generation = ''
    return image_generation


def get_audio_transcription_helper():
    stt_url = os.getenv("STT_OPENAI_BASE_URL", '')
    if stt_url:
        if not os.getenv("STT_OPENAI_MODEL"):
            os.environ["STT_OPENAI_MODEL"] = "whisper-1"
        cwd = os.path.abspath(os.getcwd())
        audio_transcription = f"""\n* Audio transcription using python. Use for transcribing audio files to text.
    * For an audio transcription, you are recommended to use the existing pre-built python code, E.g.:
    ```sh
    # filename: my_audio_transcription.sh
    # execution: true
    python {cwd}/openai_server/agent_tools/audio_transcription.py --input "audio.wav"
    ```
    * usage: python {cwd}/openai_server/agent_tools/audio_transcription.py [-h] --input "AUDIO_FILE_PATH"
    * Can transcribe audio from mp3, mp4, mpeg, mpga, m4a, wav, webm, and more.
    """
    else:
        audio_transcription = ''
    return audio_transcription

def get_save_memory_helper():
    cwd = os.path.abspath(os.getcwd())
    save_solution_memor = f"""\n
* You always keep an eye on the errors you encounter and the solutions you find to them.
* You have to save the solutions to memory so that you can refer to them in the future and avoid making the same mistakes again.
* You can also save new memories to make sure you don't forget them and make use of them in the future.
* Never skips saving the error solutions to the memory.

* Save error solutions to memory using python. Use for saving error solutions to memory.
* For saving an error solution to memory, you are recommended to use the existing pre-built python code, E.g.:
```sh
# filename: my_save_solution_memory.sh
# execution: true
python {cwd}/openai_server/agent_tools/save_memory.py --task "TASK" --error "ERROR" --solution "SOLUTION"

* usage: python {cwd}/openai_server/agent_tools/save_memory.py [-h] --task "TASK" --error "ERROR" --solution "SOLUTION"
* You should save solutions you have found to errors while solving user tasks. 
* Solutions have to be callable codes if possible, otherwise just put explanations.
* While saving the solution, you should explicityl mention: 1-the task that lead you to the error,
2-the error you encountered, and 3-the solution you found to the error, as a code or explanation.
* Example task: 'While trying to scrape X data from the web I used the 123.xyz URL but it was blocked by the server.'
* Example error: 'Error 403: Forbidden'
* Example solution: 'For similar type of data, I found this another URL 456.xyz that worked.'
* Another example solution: 'Use following code to scrape X data from the web: ...'
* It's really important to save the solutions to memory so that you can refer to them in the future and avoid making the same mistakes again.
"""
    return save_solution_memor

def get_memories(instruction:str):
    # read all the csv files that starts with the name 'memory_' in the directory: openai_files/62224bfb-c832-4452-81e7-8a4bdabbe164/
    # and concatenate them into single memory_df

    # find memory paths via os
    memory_df_paths = []
    # TODO: This is just a toy code. In real usage, the memory files should be stored in a stable DB
    for root, dirs, files in os.walk('openai_files/62224bfb-c832-4452-81e7-8a4bdabbe164/'):
        for file in files:
            if file.startswith('memory_') and file.endswith('.csv'):
                memory_df_paths.append(os.path.join(root, file))
    print(f"Memory Paths: {memory_df_paths}")
    # if no memory files found, return empty string
    if len(memory_df_paths) == 0:
        return ""

    from openai_server.agent_utils import MemoryVectorDB
    # Initialize vector DB with OpenAI model
    # TODO: In the real usage, there has to be a stable vectordb tha will work accross different chats
    # Currently this is just a dummy vectordb to test the functionality
    memory_db = MemoryVectorDB(
        model="text-embedding-3-small",
        openai_api_key=ast.literal_eval(os.getenv('H2OGPT_H2OGPT_API_KEYS'))[0],
        openai_base_url="https://api.gpt.h2o.ai/v1"
        )

    import pandas as pd
    memory_df = pd.concat([pd.read_csv(memory_df_path) for memory_df_path in memory_df_paths])
    # Create VectorDB documents from memory_df rows
    documents = []
    for index, row in memory_df.iterrows():
        document = f"{row['task']}, {row['error']}, {row['solution']}"
        documents.append(document)
    # Add documents to VectorDB
    memory_db.add_texts(documents)

    # Get the most similar 5 documents to the instruction
    results, distances = memory_db.query(instruction, k=5, threshold=0.95)
    if len(results) == 0:
        return ""

    memory_prompt = "\n# Previous Solutions Memory:"
    # join results with new line, also add index to each result
    memory_prompt += "\n".join([f"Memory-{i+1}:\n {result}" for i, result in enumerate(results)])
    return memory_prompt

def get_full_system_prompt(agent_code_writer_system_message, agent_system_site_packages, system_prompt, base_url,
                           api_key, model, text_context_list, image_file, temp_dir, query):
    agent_code_writer_system_message = agent_system_prompt(agent_code_writer_system_message,
                                                           agent_system_site_packages)

    image_query_helper = get_image_query_helper(base_url, api_key, model)
    mermaid_renderer_helper = get_mermaid_renderer_helper()
    image_generation_helper = get_image_generation_helper()
    audio_transcription_helper = get_audio_transcription_helper()
    save_memory_helper = get_save_memory_helper()
    memories_prompt = get_memories(query)
    print(f"Memories Prompt: {memories_prompt}")

    chat_doc_query, internal_file_names = get_chat_doc_context(text_context_list, image_file,
                                                               temp_dir,
                                                               # avoid text version of chat conversation, confuses LLM
                                                               chat_conversation=None,
                                                               system_prompt=system_prompt,
                                                               prompt=query,
                                                               model=model)

    cwd = os.path.abspath(os.getcwd())
    path_agent_tools = f'{cwd}/openai_server/agent_tools/'
    list_dir = os.listdir('openai_server/agent_tools')
    list_dir = [x for x in list_dir if not x.startswith('__')]

    agent_tools_note = f"\nDo not hallucinate agent_tools tools. The only files in the {path_agent_tools} directory are as follows: {list_dir}\n"

    system_message = agent_code_writer_system_message + image_query_helper + mermaid_renderer_helper + image_generation_helper + audio_transcription_helper + save_memory_helper + memories_prompt + agent_tools_note + chat_doc_query
    # TODO: Also return image_generation_helper and audio_transcription_helper ? 
    return system_message, internal_file_names, chat_doc_query, image_query_helper, mermaid_renderer_helper
