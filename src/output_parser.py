import re
from typing import Union

from langchain.agents.mrkl.output_parser import MRKLOutputParser
from langchain.schema import AgentAction, AgentFinish, OutputParserException

FORMAT_INSTRUCTIONS0 = """Use the following format and be sure to use new lines after each task.

Question: the input question you must answer

Thought: you should always think about what to do

Action: Exactly only one word out of: {tool_names}

Action Input: the input to the action

Observation: the result of the action

... (this Thought/Action/Action Input/Observation can repeat N times)

Thought: I now know the final answer

Final Answer: the final answer to the original input question"""

FORMAT_INSTRUCTIONS = """List of tools, use exactly one word when choosing Action: {tool_names}

Only user asks a question, not you.  For example user might ask: What is the latest news?

Here is an example sequence you can follow:
Thought: I should search online for the latest news.
Action: Search
Action Input: What is the latest news?
Observation: X is going away.  Z is again happening.
Thought: That is interesting, I should search for more information about X and Z and also search about Q.
Action: Search
Action Input: How is X impacting things.  Why is Z happening again, and what are the consequences?
Observation: X is causing Y.  Z may be caused by P and will lead to H.
Thought: I now know the final answer
Final Answer: The latest news is:
* X is going away, and this is caused by Y.
* Z is happening again, and the cause is P and will lead to H.
Overall, X and Z are important problems.
"""

FORMAT_INSTRUCTIONS_PYTHON = """List of tools, use exactly one word when choosing Action: {tool_names}

Only user asks a question, not you.  For example user might ask: How many rows are in the dataset?

Here is an example sequence you can follow.  You can repeat Thoughts, but as soon as possible you should try to answer the original user question.  Once you an answer the user question, just say: Thought: I now know the final answer
Thought: I should use python_repl_ast tool.
Action: python_repl_ast
Action Input: df.shape
Observation: (25, 10)
Thought: I now know the final answer
Final Answer: There are 25 rows in the dataset.
"""


FINAL_ANSWER_ACTION = "Final Answer:"
MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action:' after 'Thought:"
)
MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE = (
    "Invalid Format: Missing 'Action Input:' after 'Action:'"
)
FINAL_ANSWER_AND_PARSABLE_ACTION_ERROR_MESSAGE = (
    "Parsing LLM output produced both a final answer and a parse-able action:"
)


class H2OMRKLOutputParser(MRKLOutputParser):
    """MRKL Output parser for the chat agent."""

    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS

    def parse(self, text: str) -> Union[AgentAction, AgentFinish]:
        includes_answer = FINAL_ANSWER_ACTION in text
        regex = (
            r"Action\s*\d*\s*:[\s]*(.*?)[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        )
        action_match = re.search(regex, text, re.DOTALL)
        if includes_answer:
            return AgentFinish(
                {"output": text.split(FINAL_ANSWER_ACTION)[-1].strip()}, text
            )
        elif action_match:
            action = action_match.group(1).strip()
            action_input = action_match.group(2)
            tool_input = action_input.strip(" ")
            # ensure if its a well formed SQL query we don't remove any trailing " chars
            if tool_input.startswith("SELECT ") is False:
                tool_input = tool_input.strip('"')

            return AgentAction(action, tool_input, text)

        if not re.search(r"Action\s*\d*\s*:[\s]*(.*?)", text, re.DOTALL):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_AFTER_THOUGHT_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        elif not re.search(
            r"[\s]*Action\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)", text, re.DOTALL
        ):
            raise OutputParserException(
                f"Could not parse LLM output: `{text}`",
                observation=MISSING_ACTION_INPUT_AFTER_ACTION_ERROR_MESSAGE,
                llm_output=text,
                send_to_llm=True,
            )
        else:
            raise OutputParserException(f"Could not parse LLM output: `{text}`")

    @property
    def _type(self) -> str:
        return "mrkl"


class H2OPythonMRKLOutputParser(H2OMRKLOutputParser):
    def get_format_instructions(self) -> str:
        return FORMAT_INSTRUCTIONS_PYTHON
