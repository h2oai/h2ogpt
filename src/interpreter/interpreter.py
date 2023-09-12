import json

import requests
from interpreter.utils import parse_partial_json

# Function schema for gpt-4
function_schema = {
    "name": "run_code",
    "description":
        "Executes code on the user's machine and returns the output",
    "parameters": {
        "type": "object",
        "properties": {
            "language": {
                "type": "string",
                "description":
                    "The programming language",
                "enum": ["python", "R", "shell", "applescript", "javascript", "html"]
            },
            "code": {
                "type": "string",
                "description": "The code to execute"
            }
        },
        "required": ["language", "code"]
    },
}


def get_query(messages):
    # Use the last two messages' content or function call to semantically search
    query = []
    for message in messages[-2:]:
        message_for_semantic_search = {"role": message["role"]}
        if "content" in message:
            message_for_semantic_search["content"] = message["content"]
        if "function_call" in message and "parsed_arguments" in message["function_call"]:
            message_for_semantic_search["function_call"] = message["function_call"]["parsed_arguments"]
        query.append(message_for_semantic_search)


def get_relevant(query):
    # Use them to query Open Procedures
    url = "https://open-procedures.replit.app/search/"

    try:
        relevant_procedures = requests.get(url, data=json.dumps(query)).json()["procedures"]
        info = "\n\n# Recommended Procedures\n" + "\n---\n".join(
            relevant_procedures) + "\nIn your plan, include steps and, if present, **EXACT CODE SNIPPETS** (especially for depracation notices, **WRITE THEM INTO YOUR PLAN -- underneath each numbered step** as they will VANISH once you execute your first line of code, so WRITE THEM DOWN NOW if you need them) from the above procedures if they are relevant to the task. Again, include **VERBATIM CODE SNIPPETS** from the procedures above if they are relevent to the task **directly in your plan.**"
    except:
        # For someone, this failed for a super secure SSL reason.
        # Since it's not stricly necessary, let's worry about that another day. Should probably log this somehow though.
        inf = ''
    return info


def get_llama_code_help():
    # Tell Code-Llama how to run code.
    info = "\n\nTo run code, write a fenced code block (i.e ```python, R or ```shell) in markdown. When you close it with ```, it will be run. You'll then be given its output."
    # We make references in system_message.txt to the "function" it can call, "run_code".

    return info


#      self.system_message += "\nOnly do what the user asks you to do, then ask what they'd like to do next."
#      if messages[-1]["role"] != "function":
#        prompt += "Let's explore this. By the way, I can run code on your machine by writing the code in a markdown code block. This works for shell, javascript, python, R, and applescript. I'm going to try to do this for your task. Anyway, "
#      elif messages[-1]["role"] == "function" and messages[-1]["content"] != "No output":
#        prompt += "Given the output of the code I just ran, "
#      elif messages[-1]["role"] == "function" and messages[-1]["content"] == "No output":
#        prompt += "Given the fact that the code I just ran produced no output, "


def parse_openai_functioncall(messages):
    # gpt-4
    # Parse arguments and save to parsed_arguments, under function_call
    if "arguments" in messages[-1]["function_call"]:
        arguments = messages[-1]["function_call"]["arguments"]
        new_parsed_arguments = parse_partial_json(arguments)
        if new_parsed_arguments:
            # Only overwrite what we have if it's not None (which means it failed to parse)
            messages[-1]["function_call"][
                "parsed_arguments"] = new_parsed_arguments

        return messages


def parse_llama_functioncall(messages):
    # Code-Llama
    # Parse current code block and save to parsed_arguments, under function_call
    if "content" in messages[-1]:

        content = messages[-1]["content"]

        if "```" in content:
            # Split by "```" to get the last open code block
            blocks = content.split("```")

            current_code_block = blocks[-1]

            lines = current_code_block.split("\n")

            if content.strip() == "```":  # Hasn't outputted a language yet
                language = None
            else:
                if lines[0] != "":
                    language = lines[0].strip()
                else:
                    language = "python"
                    # In anticipation of its dumbassery let's check if "pip" is in there
                    if len(lines) > 1:
                        if lines[1].startswith("pip"):
                            language = "shell"

            # Join all lines except for the language line
            code = '\n'.join(lines[1:]).strip("` \n")

            arguments = {"code": code}
            if language:  # We only add this if we have it-- the second we have it, an interpreter gets fired up (I think? maybe I'm wrong)
                if language == "bash":
                    language = "shell"
                arguments["language"] = language

        # Code-Llama won't make a "function_call" property for us to store this under, so:
        if "function_call" not in messages[-1]:
            messages[-1]["function_call"] = {}

        messages[-1]["function_call"]["parsed_arguments"] = arguments


#              "content": """Your function call could not be parsed. Please use ONLY the `run_code` function, which takes two parameters: `code` and `language`. Your response should be formatted as a JSON."""

def code_result(messages):
    # Create or retrieve a Code Interpreter for this language
    language = messages[-1]["function_call"]["parsed_arguments"][
        "language"]
#    self.code_interpreters[language] = CodeInterpreter(language, self.debug_mode)
# code_interpreter.run()
