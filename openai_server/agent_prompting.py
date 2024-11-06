import ast
import json
import os
import sys
import tempfile
import uuid

from openai_server.agent_utils import get_have_internet, current_datetime
from openai_server.backend_utils import extract_xml_tags, generate_unique_filename, deduplicate_filenames, \
    structure_to_messages


def agent_system_prompt(agent_code_writer_system_message, agent_system_site_packages):
    if agent_code_writer_system_message is None:
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
        agent_code_writer_system_message = f"""You are a helpful AI assistant.  Solve tasks using your coding and language skills.
* {date_str}
Query understanding instructions:
<query_understanding>
* If the user directs you to do something (e.g. make a plot), then do it via code generation.
* If the user asks a question requiring math operations (e.g. even as simple as addition or counting) or puzzle solving, you MUST solve it via code generation because you are not good at intuitively solving math or puzzles.
* If the user has documents with tabular data or you obtain documents with tabular data, you MUST analyze it via code generation, because you are not good at question-answer on tabular data.
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
* In order to save the code into a file before executing it, ensure the code is within its own isolated code block with the first line having a comment: # filename: <filename>
  * A <filename> ending in .py means the code block contains valid python code that the user will run inside python interpreter.
  * A <filename> ending in .sh means the code block contains valid shell code that the user will run in a shell like bash.
  * Ensure python code blocks contain valid python code, and shell code blocks contain valid shell code.
  * Do not ask users to copy and paste the result.  Instead, use 'print' function for the output when relevant.
  * After the user has a chance to execute the code, check the execution result returned by the user.
* Every python or shell code block MUST be marked whether it is for execution with a comment that shows if execution is true or false, e.g. # execution: true
* If a python code is marked for execution, do not generate a shell script to execute that python code file, because that would execute the python code twice.
* You can assume that any files (python scripts, shell scripts, images, csv files, etc.) created by prior code generation (with name <filename> above) can be used in subsequent code generation, so repeating code generation for the same file is not necessary unless changes are required.
* When you need to collect info, generate code to output the info you need.
* Ensure you provide well-commented code, so the user can understand what the code does.
* Ensure any code prints are very descriptive, so the output can be easily understood without looking back at the code.
* Each code block meant for execution should be complete and executable on its own.
* You MUST wait for an executable code block to actually be executed before guessing or summarizing its output.  Do not hallucinate outputs of tools.
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
* You MUST only do one executable code block in your response for each turn, else mistakes or hallucinations will break the user code execution and you will have to repeat alot of code which is bad.
* As soon as you are done writing your executable code, you must stop and finish your response and wait for the user to execute the code.
* If an executable code block is too long, break it down into smaller subtasks and address them sequentially over multiple turns of the conversation.
* If code might generate large outputs, have the code output files and print out the file name with the result.  This way large outputs can be efficiently handled.
* Never abbreviate the content of the executable code blocks for any reason, always use full sentences.  The user cannot fill-in abbreviated text.
</limits>
Code error handling
<error_handling>
* If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes, following all the normal code generation rules mentioned above.
* If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
* When fixing errors, remember if you have already written a file that does not need correction, and you had already had the # filename <filename> tag, you do not need to regenerate that file when handling the exception.
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
* Search for urls to use
* Search wikipedia for topics, persons, places, or events (e.g. wikipedia package in python).
* Be smart about saving vs. printing content for any URL. First check if a URL extension to see if binary or text.  Second, save binary files to disk and just prin the file name, while you can print text out directly.
* Download a file (requests in python or wget with sh).
* Print contents of a file (open with python or cat with sh).
* Print the content of a webpage (requests in python or curl with sh).
* Get the current date/time or get the operating system type.
* Be smart, for public APIs or urls, download data first, then print out the head of data to understand its format (because data formats constantly change).  Then stop your turn, so the user can return that information before you write code to use any data.
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
* When coding a solution for a math, counting, logical reasoning, spatial reasoning, constrained response questions, or puzzle tasks, you MUST include a separate verification function to validate the correctness of the answer and print out the verification result along with the answer.  If the verification fails, fix the rest of your code until verification passes.
* For math, counting, logical reasoning, spatial reasoning, constrained response questions, or puzzle tasks, you SHOULD try multiple approaches (e.g. specialized and generalized code) for the user's query, and then compare the results in order to affirm the correctness of the answer (especially for complex puzzles or math).
* Keep trying code generation until it verifies the request.
</reasoning>
Constraints on output or response:
<constraints>
* If you need to answer a question about your own output (constrained count, etc.), try to generate a function that makes the constrained textual response.
* Searching for the constrained response is allowed, including iterating the response with the response changing to match user constraints, but you must avoid infinite loops and try generalized approaches instead of simplistic word or character replacement.
* Have common sense and be smart, repeating characters or words just to match a constraint about your response is not likely useful.
* E.g., simple solutions about your response are allowed, such as for "How many words are in your response" can just be a function that generates a sentence that includes the numeric count of the words in that sentence.
* For a response constrained by the user, the self-consistent constrained textual response (without any additional context or explanation) must appear inside <constrained_output> </constrained_output> XML tags.
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
Web scraping or web search best practices:
<web_search>
* For web search, prioritize using agent_tools provided
* Do not just use the search snippets to answer questions.  Search snippets are only starting point for finding relevant URLs, documents, or online content.
* Multi-hop web search is expected, i.e. iterative web search over many turns of a conversation is expected
* For web search, use ask_question_about_documents.py on promising URLs to answer questions and find new relevant URLs and new relevant documents
* For web search, use results ask_question_about_documents.py to find new search terms
* For web search, iterate as many times as required on URLs and documents using web search, ask_question_about_documents.py, and other agent tools
* For web search multi-hop search, only stop when reaching am answer with information verified and key claims traced to authoritative sources
* For web search, try to verify your answer with alternative sources to get a reliable answer, especially when user expects a constrained output
</web_search>
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
* As soon as you expect the user to run any code, or say something like 'Let us run this code', you must finish your response in order to give the user a chance to respond.
* If you break the problem down into multiple steps, you must stop responding between steps and finish your response and wait for the user to run the code before continuing.
* You MUST always add a very brief natural language title near the end of your response (it should just describe the analysis, do not give step numbers) of what you just did and put that title inside <turn_title> </turn_title> XML tags. Only a single title is allowed.
* Only once you have verification that the user completed the task do you summarize.
* To stop the conversation, do not include any executable code blocks. 
* If it is ever critical to have a constrained response (i.e. referencing your own output) to the user in the final summary, use <constrained_output> </constrained_output> XML tags to encapsulate the final response.
</stopping>
"""
    return agent_code_writer_system_message


### WIP:
# Post-processing Steps:
# * When all done, just before terminating, make a mermaid flow chart of all steps you took and all files produced.
# But if do this directly, then talks too much about this at end.
# So maybe do as actual final step outside of agent, just passing in history, then separately storing any LLM response.


def get_chat_doc_context(text_context_list, image_file, agent_work_dir, chat_conversation=None, system_prompt=None,
                         prompt=None, model=None):
    """
    Construct the chat query to be sent to the agent.
    :param text_context_list:
    :param image_file:
    :param chat_conversation:
    :param agent_work_dir:
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
        # setup baseline call for ask_question_about_documents.py
        with open("text_context_list.txt", "wt") as f:
            f.write("\n".join(text_context_list))
        with open("chat_conversation.json", "wt") as f:
            f.write(json.dumps(chat_conversation or []))
        with open("system_prompt.txt", "wt") as f:
            f.write(system_prompt or '')
        with open("b2imgs.txt", "wt") as f:
            f.write("\n".join(b2imgs))
        os.environ['H2OGPT_RAG_TEXT_CONTEXT_LIST'] = os.path.abspath("text_context_list.txt")
        os.environ['H2OGPT_RAG_CHAT_CONVERSATION'] = os.path.abspath("chat_conversation.json")
        os.environ['H2OGPT_RAG_SYSTEM_PROMPT'] = os.path.abspath("system_prompt.txt")
        os.environ['H2OGPT_RAG_IMAGES'] = os.path.abspath("b2imgs.txt")

        # setup general validation part of RAG
        meta_datas = [extract_xml_tags(x) for x in text_context_list]
        meta_results = [generate_unique_filename(x) for x in meta_datas]
        file_names, cleaned_names, pages = zip(*meta_results)
        file_names = deduplicate_filenames(file_names)
        document_context_file_name = "document_context.txt"
        internal_file_names.append(document_context_file_name)
        internal_file_names.extend(file_names)
        with open(os.path.join(agent_work_dir, document_context_file_name), "w") as f:
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
* You must verify, refine, clarify, and enhance the simple_rag_answer answer using the user text files or images.{web_query}
* You absolutely must read step-by step every single user file and image in order to verify the simple_rag_answer answer.  Do not skip any text files or images.  Do not read all files or images at once, but read no more than 5 text files each turn.
* Your job is to critique the simple_rag_answer answer and step-by-step determine a better response.  Do not assume the unverified answer is correct.
* Ensure your final response not only answers the question, but also give relevant key insights or details.
* Ensure to include not just words but also key numerical metrics.
* Give citations and quotations that ground and validate your responses.
* REMEMBER: Do not just repeat the simple_rag_answer answer.  You must verify, refine, clarify, and enhance it.
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
            with open(os.path.join(agent_work_dir, file_name), "w") as f:
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
    internal_file_names = [os.path.join(agent_work_dir, x) for x in internal_file_names]

    return chat_doc_query, internal_file_names


def get_ask_question_about_image_helper(base_url, api_key, model):
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
        ask_question_about_image_helper = f"""\n# Ask Question About Image Helper:
* If you need to ask a question about an image, use the following sh code:
```sh
# filename: my_image_response.sh
# execution: true
python {cwd}/openai_server/agent_tools/ask_question_about_image.py --query "QUERY" --file "LOCAL FILE NAME"
```
* usage: {cwd}/openai_server/agent_tools/ask_question_about_image.py [-h] --query "QUERY" [--url URL] [--file FILE] [--system_prompt SYSTEM_PROMPT]
* ask_question_about_image gives a text response for either a URL or local file
* ask_question_about_image can be used to critique any image, e.g. a plot, a photo, a screenshot, etc. either made by code generation or among provided files or among URLs.
* ask_question_about_image accepts most image files allowed by PIL (Pillow) except svg.
* Important!  Vision APIs will fail for images larger than 1024x1024 because they internally use PNG, so resize images down to this size (regardless of file size) before using this tool.
* Only use ask_question_about_image on key images or plots (e.g. plots meant to share back to the user or those that may be key in answering the user question).
* If the user asks for a perfect image, use the ask_question_about_image tool only up to 6 times.  If the user asks for a very rough image, then do not use the ask_question_about_image tool at all.  If the user does not specify the quality of the image, then use the ask_question_about_image tool only up to 3 times.  If user asks for more uses of ask_question_about_image, then do as they ask.
* Do not use plt.show() or plt.imshow() as the user cannot see that displayed, instead you must use this ask_question_about_image tool to critique or analyze images as a file.
"""
    else:
        ask_question_about_image_helper = """* Do not use plt.show() or plt.imshow() as the user cannot see that displayed.  Use other ways to analyze the image if required.
"""

    # FIXME: What if chat history, counting will be off
    return ask_question_about_image_helper


def get_mermaid_renderer_helper():
    cwd = os.path.abspath(os.getcwd())

    mmdc = f"""\n* Mermaid renderer using mmdc. Use for making flowcharts etc. in svg, pdf, or png format.
* For a mermaid rendering, you are recommended to use the existing pre-built python code, E.g.:
```sh
# filename: my_mermaid_render.sh
# execution: true
python {cwd}/openai_server/agent_tools/mermaid_renderer.py --file "mermaid.mmd" --output "mermaid.svg"
```
* usage: python {cwd}/openai_server/agent_tools/mermaid_renderer.py [-h] (--file FILE | [--output OUTPUT]
* If you make mermaid code to file, ensure you use python or shell code properly to generate the mermaid file.
* Good input file names would have an .mmd extension.
* Output file can be svg, pdf, or png extension.
* Ensure you use reasonable color schemes good for presentations (e.g. avoid white text in light green boxes).
* A png version of any svg is also created for use with ask_question_about_image in order to analyze the svg (via the png).
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

        image_generation = f"""\n* Image generation using python. Use for generating images from query.
* For image generation, you are recommended to use the existing pre-built python code, E.g.:
```sh
# filename: my_image_generation.sh
# execution: true
python {cwd}/openai_server/agent_tools/image_generation.py --query "QUERY"
```
* usage: python {cwd}/openai_server/agent_tools/image_generation.py [-h] --query "QUERY" [--output OUTPUT_FILE_NAME] [--model MODEL] {quality_string} {helper_style} {helper_guidance}
* Available models: {models}
* Quality options: {quality_options}{size_info}{style_options}{guidance_steps_string}
* As a helpful assistant, you will convert the user's requested image generation query into an excellent prompt for QUERY, unless the user directly requests a specific prompt be used for image generation.
* Image generation takes about 10-20s per image, so do not automatically generate too many images at once.
* However, if the user directly requests many images or anything related to images, then you MUST follow their instructions no matter what.
* Do not do an ask_question_about_image on the image generated, unless user directly asks for an analysis of the image generated or the user directly asks for automatic improvement of the image generated.
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
        audio_transcription = f"""\n* Audio transcription for transcribing audio files to text.
    * For an audio transcription, you are recommended to use the existing pre-built python code, E.g.:
    ```sh
    # filename: my_audio_transcription.sh
    # execution: true
    python {cwd}/openai_server/agent_tools/audio_transcription.py --input "audio.wav"
    ```
    * usage: python {cwd}/openai_server/agent_tools/audio_transcription.py [-h] --input "AUDIO_FILE_PATH"
    * Can transcribe audio audio and some video formats: mp3, mp4, mpeg, mpga, m4a, wav, webm, and more.
    * Once get transcript, useful to use ask_question_about_documents.py to ask questions about the transcript.
    """
    else:
        audio_transcription = ''
    return audio_transcription


def get_query_to_web_image_helper():
    have_internet = get_have_internet()
    # check if SERPAPI_API_KEY env variable is provided if not, return empty string
    if not os.getenv("SERPAPI_API_KEY") or not have_internet:
        return ""

    cwd = os.path.abspath(os.getcwd())
    image_download = f"""\n# Web Image Downloader:
* For getting a single image for a text query from the web, you can use the existing pre-built python code, E.g.:
```sh
# filename: my_image_download.sh
# execution: true
python {cwd}/openai_server/agent_tools/query_to_web_image.py --query "QUERY" --output "file_name.jpg"
```
* usage: python {cwd}/openai_server/agent_tools/query_to_web_image.py [-h] --query "QUERY" --output "FILE_NAME"
* If already have an image URL (e.g. from google or bing search), you MUST NOT use this tool, instead directly download the image URL via wget or curl -L or requests.
"""
    return image_download


def get_aider_coder_helper(base_url, api_key, model, autogen_timeout, debug=False):
    if debug:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=autogen_timeout)
        model_list = client.models.list()
        assert model in [x.id for x in model_list], "Model must be in the list of models"

    # e.g. for Aider tool to know which model to use
    os.environ['H2OGPT_AGENT_OPENAI_MODEL'] = model
    os.environ['H2OGPT_AGENT_OPENAI_TIMEOUT'] = str(autogen_timeout)

    cwd = os.path.abspath(os.getcwd())
    aider_coder_helper = f"""\n# Get coding assistance and apply to input files:
* If you need to change multiple existing coding files at once with a single query, use the following sh code:
```sh
# filename: my_aider_coder.sh
# execution: true
python {cwd}/openai_server/agent_tools/aider_code_generation.py --query "QUERY" [--files FILES [FILES ...]]
```
* usage: {cwd}/openai_server/agent_tools/aider_code_generation.py [-h] --query "QUERY" [--files FILES [FILES ...]]
* aider_code_generation outputs code diffs and applies changes to input files.
* Absolutely only use aider_code_generation if multiple existing files require changing at once, else do the code changes yoruself.
"""
    return aider_coder_helper


def get_rag_helper(base_url, api_key, model, autogen_timeout, text_context_list, image_file, debug=False):
    if debug:
        from openai import OpenAI
        client = OpenAI(base_url=base_url, api_key=api_key, timeout=autogen_timeout)
        model_list = client.models.list()
        assert model in [x.id for x in model_list], "Model must be in the list of models"

    # e.g. for Aider tool to know which model to use
    os.environ['H2OGPT_AGENT_OPENAI_MODEL'] = model
    os.environ['H2OGPT_AGENT_OPENAI_TIMEOUT'] = str(autogen_timeout)

    cwd = os.path.abspath(os.getcwd())
    rag_helper = f"""\n# Get response to query with RAG (Retrieve Augmented Generation) using documents:
* If you need to to query many (or large) document text-based files, use the following sh code:
```sh
# filename: my_question_about_documents.sh
# execution: true
python {cwd}/openai_server/agent_tools/ask_question_about_documents.py --query "QUERY" [--files FILES [FILES ...]] [--urls URLS [URLS ...]]
```
* usage: {cwd}/openai_server/agent_tools/ask_question_about_documents.py [-h] --query "QUERY" [-b BASELINE] [--system_prompt SYSTEM_PROMPT] [--files FILES [FILES ...]] [--urls URLS [URLS ...]] [--csv]
* Do not include any file names in your QUERY, just query the document content.
* ask_question_about_documents.py --files can be any local image(s) (png, jpg, etc.), local textual file(s) (txt, json, python, xml, md, html, rtf, rst, etc.), or local document(s) (pdf, docx, doc, epub, pptx, ppt, xls, xlsx) or videos (mp4, etc.).
* For videos, note that 10 frames will be selected as representative.  If those do not have the information you need, you should download the video using download_web_video.py, extract all frames, then try to bisect your way towards the right frame by each step of bisection using ask_question_about_image.py on each frame.
* ask_question_about_documents.py --urls can be any url(s) (http://www.cnn.com, https://aiindex.stanford.edu/wp-content/uploads/2024/04/HAI_2024_AI-Index-Report.pdf, youtube videos, etc.).
* Do not use ask_question_about_documents.py just to query individual images, use ask_question_about_image.py for that.
* If need structured output for data analysis, use --csv
"""
    if text_context_list or image_file:
        rag_helper += "* Absolutely you should always run ask_question_about_documents once with -b to get a baseline answer if the user has provided documents.\n"

    return rag_helper


def get_convert_to_text_helper():
    cwd = os.path.abspath(os.getcwd())
    convert_helper = f"""\n# Convert non-image text-based documents or URLs into text:
* If you need to convert non-image text-based pdf, docx, doc, epub, pptx, ppt, xls, xlsx, or URLs into text, use the following sh code:
```sh
# filename: my_convert_document_or_url_to_text.sh
# execution: true
python {cwd}/openai_server/agent_tools/convert_document_to_text.py [--files FILES [FILES ...]] [--urls URLS [URLS ...]]
```
* usage: {cwd}/openai_server/agent_tools/convert_document_to_text.py [-h] [--files FILES [FILES ...]]
* Use convert_document_to_text.py with --files with a document (pdf, docx, doc, epub, pptx, ppt, xls, xlsx, zip, mp4, etc.) to convert to text for other tools.
* Zip files will be extracted and each file inside will be converted to text.
* The convert_document_to_text.py tool can be many url(s) (http://www.cnn.com, https://aiindex.stanford.edu/wp-content/uploads/2024/04/HAI_2024_AI-Index-Report.pdf, youtube videos, etc.) to convert to text for other tools.
* The convert_document_to_text.py tool cannot be used for images or videos.
* Note, to avoid escaping special characters, put your files or URLs in quotes.
* However, use convert_document_to_text.py if just want to directly ask a question about a non-image document or URL.
* However, use ask_question_about_image.py if just want to directly ask a question about an image.
* For data analysis on xlsx or xls files, you must use non-text ways like pd.read_excel().
* You must not assume anything about the structure or content of the text, as the conversion can be complex and imperfect.
* Use ask_question_about_documents.py to verify any questions you might try to ask by using a python scripts on the text conversion.
"""

    return convert_helper


def get_download_web_video_helper():
    have_internet = get_have_internet()
    if not have_internet:
        return ''
    cwd = os.path.abspath(os.getcwd())
    youtube_helper = f"""\n# Download Web-hosted Videos using the following Python script:
* To download a video from YouTube or other supported platforms, use the following command:
```sh
# filename: my_download_video.sh
# execution: true
python {cwd}/openai_server/agent_tools/download_web_video.py --video_url "YOUTUBE_URL"
```
* usage: {cwd}/openai_server/agent_tools/download_web_video.py [-h] --video_url VIDEO_URL --base_url BASE_URL
* download_web_video.py downloads a video from the given URL.
* The video_url is the URL of the video you want to download.
* The --base_url is the URL of the website where the video is hosted, defaults to "https://www.youtube.com" but can be any other website that hosts videos.
"""
# * List of other supported sites where videos can be downloaded is here: https://github.com/yt-dlp/yt-dlp/blob/master/supportedsites.md
    return youtube_helper


def get_serp_helper():
    have_internet = get_have_internet()
    if have_internet and os.getenv('SERPAPI_API_KEY'):
        cwd = os.path.abspath(os.getcwd())
        serp = f"""# Perform Google Searches using the following Python script:
* To perform a search using various search engines and Google services, use the following command:
```sh
# filename: my_google_search.sh
# execution: true
python {cwd}/openai_server/agent_tools/google_search.py --query "QUERY"
```
* usage: {cwd}/openai_server/agent_tools/google_search.py [-h] --query "QUERY" [--engine {{google,bing,baidu,yandex,yahoo,ebay,homedepot,youtube,scholar,walmart,appstore,naver}}] [--limit LIMIT] [--type {{web,image,local,video,news,shopping,patents}}]
* This tool should be used instead of generic searches using packages googlesearch, requests, and bs4.
* --type applies only to google engine.
* The tool saves full search results to a JSON file in the current directory.
* For non-english queries, do python {cwd}/openai_server/agent_tools/google_search.py -h to see options for other languages and locations.
* To download the video returned from this google_search.py tool:
  - For a youtube url or other urls on certain sites, use download_web_video.py agent tool.
  - For generic free web sites, use can get video via wget, curl -L, or requests.
* To download a web page via its URL or image returned from this google_search.py tool:
   - Use wget, curl -L, or requests to download the image URL.
* Multi-hop search is highly recommended, so the single-hop search with snippets and URLs should be followed up by passing URLs to using ask_question_about_documents.py for asking questions.
* Multi-hop search is highly recommended, so for queries about the search results, pass the entire JSON file to ask_question_about_documents.py for asking questions about the search results, e.g. to ask which URL is most relevant to ask further questions about using ask_question_about_documents.py again.
"""
        if os.getenv("BING_API_KEY"):
            serp += f"""# The bing_search.py tool can be used if this google_search.py tool fails or vice versa."""
    else:
        serp = ""
    return serp


def get_semantic_scholar_helper():
    cwd = os.path.abspath(os.getcwd())
    have_internet = get_have_internet()
    if have_internet and os.getenv('S2_API_KEY'):
        # https://github.com/allenai/s2-folks/blob/main/examples/python/find_and_recommend_papers/find_papers.py
        # https://github.com/allenai/s2-folks
        papers_search = f"""\n* Search semantic scholar (API with semanticscholar pypi package in python, user does have S2_API_KEY key for use from https://api.semanticscholar.org/ already in ENV) or search ArXiv.  Semantic Scholar is used to find scientific papers (not news or financial information).
* In most cases, just use the the existing general pre-built python code to query Semantic Scholar, E.g.:
```sh
# filename: my_scholar_paper_search.sh
# execution: true
python {cwd}/openai_server/agent_tools/scholar_papers_query.py --query "QUERY"
```
usage: python {cwd}/openai_server/agent_tools/scholar_papers_query.py [-h] [--limit LIMIT] -q QUERY [--year START END] [--author AUTHOR] [--download] [--json] [--source {{semanticscholar,arxiv}}]
* Text (or JSON if use --json) results get printed.  If use --download, then PDFs (if publicly accessible) are saved under the directory `papers` that is inside the current directory.  Only download if you will actually use the PDFs.
* Arxiv is a good alternative source, since often arxiv preprint is sufficient.
"""
    else:
        papers_search = ""
    return papers_search


def get_wolfram_alpha_helper():
    cwd = os.path.abspath(os.getcwd())
    have_internet = get_have_internet()
    if have_internet and os.getenv('WOLFRAM_ALPHA_APPID'):
        # https://wolframalpha.readthedocs.io/en/latest/?badge=latest
        # https://products.wolframalpha.com/api/documentation
        wolframalpha = f"""\n* Wolfram Alpha (API with wolframalpha pypi package in python, user does have WOLFRAM_ALPHA_APPID key for use with https://api.semanticscholar.org/ already in ENV).  Can be used for advanced symbolic math, physics, chemistry, engineering, and astronomy.
* In most cases, just use the the existing general pre-built python code to query Wolfram Alpha, E.g.:
```sh
# filename: my_wolfram_response.sh
# execution: true
python {cwd}/openai_server/agent_tools/wolfram_alpha_math_science_query.py --query "QUERY"
```
* usage: python {cwd}/openai_server/agent_tools/wolfram_alpha_math_science_query.py --query "QUERY GOES HERE"
* For wolfram alpha tool, query must be *very* terse and specific, e.g., "integral of x^2" or "mass of the sun" and is not to be used for general web searches.
* Text results get printed, and images are saved under the directory `wolfram_images` that is inside the current directory
"""
    else:
        wolframalpha = ""
    return wolframalpha


def get_dai_helper():
    cwd = os.path.abspath(os.getcwd())
    if os.getenv('ENABLE_DAI'):
        dai = f"""\n* DriverlessAI is an advanced AutoML tool for data science model making and predictions.
* If user specifically asks for a DAI model, then you should use the existing pre-built python code to query DriverlessAI, E.g.:
```sh
# filename: my_dai_query.sh
# execution: true
python {cwd}/openai_server/agent_tools/driverless_ai_data_science.py
```
* usage: python {cwd}/openai_server/agent_tools/driverless_ai_data_science.py [--experiment_key EXPERIMENT_KEY] [--dataset_key DATASET_KEY] [--data-url DATA_URL] [--dataset-name DATASET_NAME] [--data-source DATA_SOURCE] [--target-column TARGET_COLUMN] [--task {{classification,regression,predict,shapley_original_features,shapley_transformed_features,transform,fit_and_transform,artifacts}}] [--scorer SCORER] [--experiment-name EXPERIMENT_NAME] [--accuracy {{1,2,3,4,5,6,7,8,9,10}}] [--time {{1,2,3,4,5,6,7,8,9,10}}] [--interpretability {{1,2,3,4,5,6,7,8,9,10}}] [--train-size TRAIN_SIZE] [--seed SEED] [--fast] [--force]
* Typical case for creating experiment might be:
python {cwd}/openai_server/agent_tools/driverless_ai_data_science.py --dataset-name "my_dataset" --data-url "https://mydata.com/mydata.csv" --target-column "target" --task "classification" --scorer "auc" --experiment-name "my_experiment"
* A typical re-use of the experiment_key and dataset_key for prediction (or shapley, transform, fit_and_transform) would be like:
python {cwd}/openai_server/agent_tools/driverless_ai_data_science.py --experiment_key <experiment_key from experiment created before> --dataset_key <dataset_key from experiment> --task "prediction"
* For predict, shapley, transform, fit_and_transform, one can also pass --data-url to use a fresh dataset on the given experiment, e.g.:
python {cwd}/openai_server/agent_tools/driverless_ai_data_science.py --experiment_key <experiment_key from experiment created before> --data-url "https://mydata.com/mydata.csv" --task "prediction"
"""
        if os.getenv('DAI_TOKEN') is None:
            dai += f"""* Additionally, you must pass --token <DAI_TOKEN> to the command line to use the DAI tool."""
        dai += f"""You may also pass these additional options if user provides them: --engine DAI_ENGINE --client_id DAI_CLIENT_ID --token_endpoint_url DAI_TOKEN_ENDPOINT_URL --environment DAI_ENVIRONMENT --token DAI_TOKEN"""
    else:
        dai = ""
    return dai


def get_news_api_helper():
    cwd = os.path.abspath(os.getcwd())
    have_internet = get_have_internet()
    # only expose news API if didn't have google or bing, else confuses LLM
    if have_internet and os.getenv('NEWS_API_KEY') and not (
            os.environ.get("SERPAPI_API_KEY") or os.environ.get("BING_API_KEY")):
        news_api = f"""\n* News API uses NEWS_API_KEY from https://newsapi.org/).  The main use of News API is to search topical news articles published in the last 5 years.
* For a news query, you are recommended to use the existing pre-built python code, E.g.:
```sh
# filename: my_news_response.sh
# execution: true
python {cwd}/openai_server/agent_tools/news_query.py --query "QUERY"
```
* usage: {cwd}/openai_server/agent_tools/news_query.py [-h] [--mode {{everything, top-headlines}}] [--sources SOURCES]  [--num_articles NUM_ARTICLES] [--query "QUERY"] [--sort_by {{relevancy, popularity, publishedAt}}] [--language LANGUAGE] [--country COUNTRY] [--category {{business, entertainment, general, health, science, sports, technology}}]
* news_query is not to be used for general web searches, but only for topical news searches.
* news_query prints text results with title, author, description, and URL for (by default) 10 articles.
* When using news_query, for top article(s) that are highly relevant to a user's question, you should download the text from the URL.
"""
    else:
        news_api = ''
    return news_api


def get_bing_search_helper():
    cwd = os.path.abspath(os.getcwd())
    have_internet = get_have_internet()
    if have_internet and os.getenv('BING_API_KEY'):
        bing_search = f"""\n* Search web using Bing API (using azure-core, user has BING_API_KEY already in ENV) for web, image, news, or video search.
* In most cases, just use the existing general pre-built Python code to query Bing Search, E.g.:
```sh
# filename: my_bing_search.sh
# execution: true
python {cwd}/openai_server/agent_tools/bing_search.py --query "QUERY"
```
usage: python {cwd}/openai_server/agent_tools/bing_search.py [-h] --query "QUERY" [--type {{web,image,news,video}}] [--limit LIMIT] [--market MARKET] [--freshness {{Day,Week,Month}}]
* This Bing is highly preferred over the Google Image search query
* Available search types (--type):
  - web: General web search to find web content
  - image: Image search to find images (once have image URL, can get it via wget, curl -L, or requests)
  - news: News search to find news
  - video: Video search to find videos
* To download the video returned from this bing_search.py tool:
  - For a youtube url or other urls on certain sites, use download_web_video.py agent tool.
  - For generic free web sites, use can get video via wget, curl -L, or requests.
* To download a page or image returned from this bing_search.py tool:
   - Use wget, curl -L, or requests to download the image URL.
* Use --limit to specify the number of results (default is 10)
* Use --market to specify the market (e.g., en-US)
* Use --freshness to filter results by age (Day, Week, Month).  Default is no filter to get older results.
* Multi-hop search is highly recommended, so the single-hop search with snippets and URLs should be followed up by passing URLs to using ask_question_about_documents.py for asking questions.
* Multi-hop search is highly recommended, so for queries about the search results, pass the entire JSON file to ask_question_about_documents.py for asking questions about the search results, e.g. to ask which URL is most relevant to ask further questions about using ask_question_about_documents.py again.
"""
        if os.getenv("SERPAPI_API_KEY"):
            bing_search += f"""# The google_search.py tool can be used if this bing_search.py tool fails or vice versa."""
    else:
        bing_search = ""
    return bing_search


def get_api_helper():
    if os.getenv('SERPAPI_API_KEY') or os.getenv('BING_API_KEY'):
        search_web_api_message = """* Highly recommended to first try using google or bing search tool when searching for something on the web.
* i.e. avoid packages googlesearch package for web searches."""
    else:
        search_web_api_message = ""
    have_internet = get_have_internet()
    if have_internet:
        apis = f"""\n#APIs and external services instructions:
* You DO have access to the internet.
{search_web_api_message}
* Use existing python tools for various tasks, e.g. Wolfram Alpha, Semantic Scholar, News API, etc.
* Avoid generating code with placeholder API keys as that will never work because user will not be able to change the code.
* You MUST wait for an executable code block to actually be executed before guessing or summarizing its output.
* Do not hallucinate outputs of tools, you must wait for user to execute each executable code block.
* Example Public APIs (not limited to these): wttr.in (weather) or research papers (arxiv).
* You may generate code with API code that uses publicly available APIs that do not require any API key.
* You may generate code with APIs for API keys that have been mentioned in this overall message.
* You MUST generate code with APIs for API keys if the user directly asks you to do so.  Do your best effort to figure out (from internet, documents, etc.) how to use the API to solve the user's task.  You are not allowed to refuse to use the API if the user asks you to use it."""
    else:
        apis = """\n#APIs and external services instructions:
* You DO NOT have access to the internet.  You cannot use any APIs that require broad internet access.
* You may generate code with APIs for API keys given to you directly by the user."""
    return apis


def get_agent_tools():
    cwd = os.path.abspath(os.getcwd())
    path_agent_tools = f'{cwd}/openai_server/agent_tools/'
    list_dir = os.listdir('openai_server/agent_tools')
    list_dir = [x for x in list_dir if not x.startswith('__')]
    list_dir = [x for x in list_dir if not x.endswith('.pyc')]
    return path_agent_tools, list_dir


def get_full_system_prompt(agent_code_writer_system_message, agent_system_site_packages, system_prompt, base_url,
                           api_key, model, text_context_list, image_file, agent_work_dir, query, autogen_timeout):
    agent_code_writer_system_message = agent_system_prompt(agent_code_writer_system_message,
                                                           agent_system_site_packages)

    ask_question_about_image_helper = get_ask_question_about_image_helper(base_url, api_key, model)
    mermaid_renderer_helper = get_mermaid_renderer_helper()
    image_generation_helper = get_image_generation_helper()
    audio_transcription_helper = get_audio_transcription_helper()
    aider_coder_helper = get_aider_coder_helper(base_url, api_key, model, autogen_timeout)
    rag_helper = get_rag_helper(base_url, api_key, model, autogen_timeout, text_context_list, image_file)
    convert_helper = get_convert_to_text_helper()
    youtube_helper = get_download_web_video_helper()

    # search:
    serp_helper = get_serp_helper()
    semantic_scholar_helper = get_semantic_scholar_helper()
    wolfram_alpha_helper = get_wolfram_alpha_helper()
    news_helper = get_news_api_helper()
    bing_search_helper = get_bing_search_helper()
    query_to_web_image_helper = get_query_to_web_image_helper()

    # data science
    dai_helper = get_dai_helper()

    # general API notes:
    api_helper = get_api_helper()

    chat_doc_query, internal_file_names = get_chat_doc_context(text_context_list, image_file,
                                                               agent_work_dir,
                                                               # avoid text version of chat conversation, confuses LLM
                                                               chat_conversation=None,
                                                               system_prompt=system_prompt,
                                                               prompt=query,
                                                               model=model)

    path_agent_tools, list_dir = get_agent_tools()

    agent_tools_note = f"""\n# Agent tools notes:
* Do not hallucinate agent_tools tools. The only files in the {path_agent_tools} directory are as follows: {list_dir}"
* You have to prioritize these tools for the relevant tasks before using other tools or methods.
* If you plan to use multiple tools or execute multiple code blocks, you must end your turn after each single executable code block in order to give chance for user to execute the code blocks and prevent you from hallucinating outputs and inputs further steps.
"""

    system_message_parts = [agent_code_writer_system_message,
                            # rendering
                            mermaid_renderer_helper,
                            image_generation_helper,
                            # coding
                            aider_coder_helper,
                            # docs
                            rag_helper,
                            ask_question_about_image_helper,
                            audio_transcription_helper,
                            youtube_helper,
                            convert_helper,
                            # search
                            serp_helper,
                            semantic_scholar_helper,
                            wolfram_alpha_helper,
                            news_helper,
                            bing_search_helper,
                            query_to_web_image_helper,
                            # data science
                            dai_helper,
                            # overall
                            api_helper,
                            agent_tools_note,
                            # docs
                            chat_doc_query]

    system_message = ''.join(system_message_parts)

    return system_message, internal_file_names, system_message_parts


def planning_prompt(query):
    return f"""
<user_query>
{query}
</user_query>

* First, decide how one can search for required information.
* Second, for each agent tool in agent_tools directory, consider how the tool might be useful to answering the user's query or obtaining information.
* Third, for any relevant python packages, consider how they might be useful to answering the user's query or obtaining information.
* Forth, consider what coding algorithms might be useful to answering the user's query or obtaining information.
* Fifth, come up with a possible plan to solve the problem or respond to the user query using these tools or other coding approaches.
* Sixth, plan for any formatting or other constraints on the response given by the user.
* For steps 1-6, ensure you write a well-structured possible plan.
* Note: You must not respond to the user query directly.
* Note: You must not write any code, because you are likely planning blindly and will make mistakes.  You must NOT execute any code.
* Note: Once you have finished the plan, you must end your response immediately.
* Finally, end your turn of the conversation without any additional discussion or code.
* Note: You must not repeat any of these instructions in your planned response.
"""


def planning_final_prompt(query):
    return f"""
<user_query>
{query}
</user_query>
Come up with a possible plan for the user's query.
"""
