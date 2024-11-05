from typing import Tuple, Any, List
import re
import argparse
import sys
import os

cwd = os.path.abspath(os.getcwd())
# Find the 'h2ogpt' root directory
while True:
    if os.path.basename(cwd) == "h2ogpt":
        project_root = cwd
        break
    # Move one directory up
    cwd = os.path.dirname(cwd)
    # Safety check if we reach the top of the directory tree without finding 'h2ogpt'
    if cwd == "/":
        raise FileNotFoundError("Could not find 'h2ogpt' directory in the path.")
    

# Below is needed to be able to import from openai_server
sys.path.append(cwd)

from langchain_core.outputs import LLMResult


from rich import print as pp

from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_core.callbacks import BaseCallbackHandler

import autogen

from openai_server.browser.utils import SimpleTextBrowser

MODEL=os.getenv('WEB_TOOL_MODEL')
API_KEY = os.getenv('H2OGPT_API_KEY')
API_BASE = os.getenv('H2OGPT_OPENAI_BASE_URL')
BING_API_KEY = os.getenv('BING_API_KEY')

class LLMCallbackHandler(BaseCallbackHandler):

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        print(f"LLM response: {response}")

class Answer(BaseModel):
    reason: str = Field(description="Step by step reasoning")
    answer: str = Field(description="The answer to the question")

class StepNote(BaseModel):
    snippets: List[str] = Field(description="The snippets may use to answer the question, each snippet should less than 1000 characters")
    plan: str = Field(description="Plan for the next step")

class ToolChoice(BaseModel):
    reason: str = Field(description="Step by step reasoning")
    tool: str = Field(description="The tool to use")
    tool_args: dict = Field(description="The arguments to pass to the tool")

class ImproveCode(BaseModel):
    reason: str = Field(description="Step by step reasoning on how to improve the code")
    improved_code: str = Field(description="The improved code")

with open(f"{cwd}/openai_server/browser/prompts/format_answer.txt") as f:
    FORMAT_ANSWER_PROMPT = ChatPromptTemplate.from_template(f.read())

with open(f"{cwd}/openai_server/browser/prompts/choose_tool.txt") as f:
    CHOOSE_TOOL_PROMPT_TEMPLATE = f.read()

with open(f"{cwd}/openai_server/browser/prompts/summarize_step.txt") as f:
    SUMMARIZE_STEP_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(f.read())

with open(f"{cwd}/openai_server/browser/prompts/improve_code.txt") as f:
    IMPROVE_CODE_PROMPT_TEMPLATE = f.read()

with open(f"{cwd}/openai_server/browser/prompts/date_info.txt") as f:
    DATE_INFO_PROMPT_TEMPLATE = f.read()

class WebAgent:
    def __init__(self):
        # TODO: is max_tokens ok?
        # TODO: is streaming ok?
        # TODO: is request_timeout ok?
        self.llm = ChatOpenAI(model=MODEL, temperature=0.1, streaming=False, max_retries=5, api_key=API_KEY, base_url=API_BASE, max_tokens=2048, request_timeout=60)
        self.format_answer_chain = FORMAT_ANSWER_PROMPT | self.llm | StrOutputParser()

        self.tool_choice_output_parser = JsonOutputParser(pydantic_object=ToolChoice)
        choose_tool_prompt = PromptTemplate(
            template=CHOOSE_TOOL_PROMPT_TEMPLATE, 
            input_variables=['steps', 'question', 'date_info'], 
            partial_variables={"format_instructions": self.tool_choice_output_parser.get_format_instructions()}
        )
        self.choose_tool_chain = choose_tool_prompt | self.llm | self.tool_choice_output_parser

        self.improve_code_output_parser = JsonOutputParser(pydantic_object=ImproveCode)
        improve_code_prompt = PromptTemplate(
            template=IMPROVE_CODE_PROMPT_TEMPLATE, 
            input_variables=['steps', 'question', 'code'],
            partial_variables={"format_instructions": self.improve_code_output_parser.get_format_instructions()}
        )
        self.improve_code_chain = improve_code_prompt | self.llm | self.improve_code_output_parser

        self.summarize_tool_chain = SUMMARIZE_STEP_PROMPT_TEMPLATE | self.llm | StrOutputParser()

        browser_config={
            "bing_api_key": BING_API_KEY,
            "viewport_size": 1024 * 16,
            "downloads_folder": "coding",
            "request_kwargs": {
                "headers": {"User-Agent":  "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0"},
            },
            "bing_cache": None, # TODO: We don't want to cache the search results ?
        }
        self.browser = SimpleTextBrowser(**browser_config)
        self.llm_callback_handler = LLMCallbackHandler()

        # TODO: use H2OConversableAgent instead?
        final_answer_agent = autogen.ConversableAgent(
            name="Final Answer",
            system_message='''
You are a helpful assistant. When answering a question, you must explain your thought process step by step before answering the question. 
When others make suggestions about your answers, think carefully about whether or not to adopt the opinions of others. 
If provided, you have to mention websites or sources that you used to find the answer. 
If you are unable to solve the question, make a well-informed EDUCATED GUESS based on the information we have provided. 
If you think the provided web search steps  or findings are not enough to answer the question, 
you should let the user know that the current web search results are not enough to answer the question. 
DO NOT OUTPUT 'I don't know', 'Unable to determine', etc.
''',
            llm_config={"config_list": [{"model": MODEL, "temperature": 0.1, "api_key": API_KEY, "base_url": API_BASE}]},
            is_termination_msg=lambda x: x.get("content", "").find("TERMINATE") >= 0,
            human_input_mode="NEVER",
        )
        self.final_answer_agent = final_answer_agent

    def browser_state(self) -> Tuple[str, str]:
        header = f"Address: {self.browser.address}\n"
        if self.browser.page_title is not None:
            header += f"Title: {self.browser.page_title}\n"

        current_page = self.browser.viewport_current_page
        total_pages = len(self.browser.viewport_pages)

        header += f"Viewport position: Showing page {current_page+1} of {total_pages}.\n"
        return (header, self.browser.viewport)
    
    def informational_web_search(self, query: str) -> str:
        self.browser.visit_page(f"bing: {query}")
        header, content = self.browser_state()
        return header.strip() + "\n=======================\n" + content
    
    def navigational_web_search(self, query: str) -> str:
        self.browser.visit_page(f"bing: {query}")
        # Extract the first linl
        m = re.search(r"\[.*?\]\((http.*?)\)", self.browser.page_content)
        if m:
            self.browser.visit_page(m.group(1))

        # Return where we ended up
        header, content = self.browser_state()
        return header.strip() + "\n=======================\n" + content

    def visit_page(self, url: str) -> str:
        self.browser.visit_page(url)
        header, content = self.browser_state()
        return header.strip() + "\n=======================\n" + content

    def page_up(self) -> str:
        self.browser.page_up()
        header, content = self.browser_state()
        return header.strip() + "\n=======================\n" + content

    def page_down(self) -> str:
        self.browser.page_down()
        header, content = self.browser_state()
        return header.strip() + "\n=======================\n" + content

    def download_file(self, url: str) -> str:
        self.browser.visit_page(url)
        header, content = self.browser_state()
        return header.strip() + "\n=======================\n" + content

    def find_on_page_ctrl_f(self, search_string: str) -> str:
        find_result = self.browser.find_on_page(search_string)
        header, content = self.browser_state()

        if find_result is None:
            return (
                header.strip()
                + "\n=======================\nThe search string '"
                + search_string
                + "' was not found on this page."
            )
        else:
            return header.strip() + "\n=======================\n" + content

    def find_next(self) -> str:
        find_result = self.browser.find_next()
        header, content = self.browser_state()

        if find_result is None:
            return header.strip() + "\n=======================\nThe search string was not found on this page."
        else:
            return header.strip() + "\n=======================\n" + content

    def ask(self, raw_question: str, attachment_file_path: str = None) -> str:
        steps = []

        # TODO: make sure that attachment_file_path works ?
        if attachment_file_path is not None and attachment_file_path.strip() != "":
            question = f"{raw_question}\nAttachment file path: {attachment_file_path}"
        else:
            question = raw_question
        # pp(f"Question: {question}")

        try:
            date_info_prompt = PromptTemplate(
                template=DATE_INFO_PROMPT_TEMPLATE, 
                input_variables=['question'], 
            )
            date_info_fetcher = date_info_prompt | self.llm | StrOutputParser()
            date_info = date_info_fetcher.invoke({'question': question})
            print(f"\n\n Web search date info: {date_info}")
        except Exception as e:
            print(f"Error: {e}")
            date_info = None

        for i in range(20):
            # TODO: pass has_error info to the choose_tool_chain
            has_error = False
            for _ in range(3):
                try:
                    tool_choice = self.choose_tool_chain.invoke({'question': question, 'steps': '\n\n'.join(steps), 'date_info': date_info})
                    print(f"\n\nWebAgent {i+1} tool_choice: {tool_choice}")
                    # h2ogpt models may return with 'properties' key
                    if 'properties' in tool_choice:
                        tool_choice = tool_choice['properties']
                    if 'tool' not in tool_choice or 'tool_args' not in tool_choice:
                        has_error = True
                        break
                    else:
                        break
                except Exception as e:
                    print(f"Error: {e}")
                    has_error = True
                    continue
            tool = tool_choice['tool']
            args = tool_choice['tool_args']
            reason = tool_choice.get('reason', '')
            pp(f"\n\n * {i+1} - Tool: {tool}, Args: {args} Reason: {reason} ")
            if tool == "informational_web_search":
                tool_result = self.informational_web_search(**args)
            elif tool == "navigational_web_search":
                tool_result = self.navigational_web_search(**args)
            elif tool == "visit_page":
                tool_result = self.visit_page(**args)
            elif tool == "page_up":
                tool_result = self.page_up()
            elif tool == "page_down":
                tool_result = self.page_down()
            elif tool == "download_file":
                tool_result = self.download_file(**args)
            elif tool == "find_on_page_ctrl_f":
                tool_result = self.find_on_page_ctrl_f(**args)
            elif tool == "find_next":
                tool_result = self.find_next()
            elif tool == 'None':
                tool_result = None
            else:
                print(f"Unknown tool: {tool}")
                tool_result = f"ERROR: You provided an unknown tool: {tool} with the args: {args}."
                has_error = True
            
            if tool == 'None':
                print(f"No tool chosen, break")
                break
            # if tool_result:
            #     print(f"\n * Current tool result: {tool_result}")
            try:
                step_note = self.summarize_tool_chain.invoke({'question': question, 'steps': '\n\n'.join(steps), 'tool_result': tool_result, 'tool': tool, 'args': args})
            except Exception as e:
                print(f"Error: {e}")
                step_note = e
            steps.append(f"Step:{len(steps)+1}\nTool: {tool}, Args: {args}\n{step_note}\n\n")

        steps_prompt = '\n'.join(steps)
        answer = f"""
{question}\nTo answer the above question, I followed the steps below:
{steps_prompt}

Referring to the steps I followed and information I have obtained (which may not be accurate), you may find the answer to the web search query in the steps above.
"""
# TODO: If below agents used, include cost calculations from these agent interactions too (Or automatically added?)
#     if not steps_prompt:
#         message=f"""{question}\nIf you are unable to solve the question, make a well-informed EDUCATED GUESS based on the information we have provided.
# Your EDUCATED GUESS should be a number OR as few words as possible OR a comma separated list of numbers and/or strings. DO NOT OUTPUT 'I don't know', 'Unable to determine', etc.
# """
#     else:
#         message = f"""
# {question}\nTo answer the above question, I did the following:
# {steps_prompt}

# Referring to the information I have obtained (which may not be accurate), what do you think is the answer to the question?
# If provided, also mention websites or sources that you used to find the answer. Sharing sources is a mandatory step to ensure that the answer is reliable.
# If you are unable to solve the question, make a well-informed EDUCATED GUESS based on the information we have provided.
# If you think the provided web search steps or findings are not enough to answer the question, 
# you should let the user know that the current web search results are not enough to answer the question. 
# DO NOT OUTPUT 'I don't know', 'Unable to determine', etc.
# """
#     answer = self.final_answer_agent.generate_reply(messages=[{"content": message, "role": "user"}])
        # formatted_answer = self.format_answer_chain.invoke({'question': question, 'answer': answer})#.answer
        return answer


def main():
    parser = argparse.ArgumentParser(description="Do web search")
    parser.add_argument("--task", type=str, required=True, help="Web-related task to perform for the WebAgent")
    args = parser.parse_args()

    web_agent = WebAgent()
    # TODO: what about attachment_file_path? Will native agents handle them or should we pass them to the tool?
    answer = web_agent.ask(raw_question = args.task)
    print(f"For the task '{args.task}', the WebAgent result is:\n{answer}")


if __name__ == "__main__":
    main()