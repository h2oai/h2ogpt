import os
import tempfile
import typing
from datetime import datetime
from tempfile import TemporaryDirectory

from websockets.sync.client import connect as ws_connect

import autogen
from autogen.io.websockets import IOWebsockets


model_name = "gpt-4o"
api_key = os.getenv('OPENAI_API_KEY')
base_url = "https://api.openai.com/v1/"


def on_connect(iostream: IOWebsockets) -> typing.List:
    print(f" - on_connect(): Connected to client using IOWebsockets {iostream}", flush=True)

    print(" - on_connect(): Receiving message from client.", flush=True)

    # 1. Receive Initial Message
    query = iostream.input()

    # https://stackoverflow.com/questions/44761246/temporary-failure-in-name-resolution-errno-3-with-docker
    # sudo systemctl restart docker

    from autogen import ConversableAgent
    from autogen.coding import LocalCommandLineCodeExecutor

    # Create a temporary directory to store the code files.
    temp_dir = tempfile.TemporaryDirectory()

    use_docker = True

    if use_docker:
        from autogen.coding import DockerCommandLineCodeExecutor
        # Create a Docker command line code executor.
        executor = DockerCommandLineCodeExecutor(
            #image="python:3.12-slim",  # Execute code using the given docker image name.
            image="python:3.10-slim-bullseye",
            timeout=20,  # Timeout for each code execution in seconds.
            work_dir=temp_dir.name,  # Use the temporary directory to store the code files.
        )

        # Create an agent with code executor configuration that uses docker.
        code_executor_agent = ConversableAgent(
            "code_executor_agent_docker",
            llm_config=False,  # Turn off LLM for this agent.
            code_execution_config={"executor": executor},  # Use the docker command line code executor.
            human_input_mode="NEVER",  # Always take human input for this agent for safety.
        )
    else:
        # Create a local command line code executor.
        executor = LocalCommandLineCodeExecutor(
            timeout=20,  # Timeout for each code execution in seconds.
            work_dir=temp_dir.name,  # Use the temporary directory to store the code files.
        )

        # Create an agent with code executor configuration.
        code_executor_agent = ConversableAgent(
            "code_executor_agent",
            llm_config=False,  # Turn off LLM for this agent.
            code_execution_config={"executor": executor},  # Use the local command line code executor.
            human_input_mode="NEVER",  # Always take human input for this agent for safety.
        )

    # The code writer agent's system message is to instruct the LLM on how to use
    # the code executor in the code executor agent.
    code_writer_system_message = """You are a helpful AI assistant.
    Solve tasks using your coding and language skills.
    In the following cases, suggest python code (in a python coding block) or shell script (in a sh coding block) for the user to execute.
    1. When you need to collect info, use the code to output the info you need, for example, browse or search the web, download/read a file, print the content of a webpage or a file, get the current date/time, check the operating system. After sufficient info is printed and the task is ready to be solved based on your language skill, you can solve the task by yourself.
    2. When you need to perform some task with code, use the code to perform the task and output the result. Finish the task smartly.
    Solve the task step by step if you need to. If a plan is not provided, explain your plan first. Be clear which step uses code, and which step uses your language skill.
    When using code, you must indicate the script type in the code block. The user cannot provide any other feedback or perform any other action beyond executing the code you suggest. The user can't modify your code. So do not suggest incomplete code which requires users to modify. Don't use a code block if it's not intended to be executed by the user.
    If you want the user to save the code in a file before executing it, put # filename: <filename> inside the code block as the first line. Don't include multiple code blocks in one response. Do not ask users to copy and paste the result. Instead, use 'print' function for the output when relevant. Check the execution result returned by the user.
    If the result indicates there is an error, fix the error and output the code again. Suggest the full code instead of partial code or code changes. If the error can't be fixed or if the task is not solved even after the code is executed successfully, analyze the problem, revisit your assumption, collect additional info you need, and think of a different approach to try.
    When you find an answer, verify the answer carefully. Include verifiable evidence in your response if possible.
    Reply 'TERMINATE' in the end when everything is done.
    """

    code_writer_agent = ConversableAgent(
        "code_writer_agent",
        system_message=code_writer_system_message,
        llm_config={"config_list": [{"model": model_name, "api_key": api_key, "base_url": base_url}]},
        code_execution_config=False,  # Turn off code execution for this agent.
    )
    chat_result = code_executor_agent.initiate_chat(
        code_writer_agent,
        message=query,
    )

    print(os.listdir(temp_dir.name))
    # We can see the output scatter.png and the code file generated by the agent.

    # temp_dir.cleanup()
    executor.stop()  # Stop the docker command line code executor.

    return os.listdir(temp_dir.name)


from contextlib import asynccontextmanager  # noqa: E402
from pathlib import Path  # noqa: E402

from fastapi import FastAPI  # noqa: E402
from fastapi.responses import HTMLResponse  # noqa: E402

PORT = 8005

html = """
<!DOCTYPE html>
<html>
    <head>
        <title>Autogen websocket test</title>
    </head>
    <body>
        <h1>WebSocket Chat</h1>
        <form action="" onsubmit="sendMessage(event)">
            <input type="text" id="messageText" autocomplete="off"/>
            <button>Send</button>
        </form>
        <ul id='messages'>
        </ul>
        <script>
            var ws = new WebSocket("ws://localhost:8080/ws");
            ws.onmessage = function(event) {
                var messages = document.getElementById('messages')
                var message = document.createElement('li')
                var content = document.createTextNode(event.data)
                message.appendChild(content)
                messages.appendChild(message)
            };
            function sendMessage(event) {
                var input = document.getElementById("messageText")
                ws.send(input.value)
                input.value = ''
                event.preventDefault()
            }
        </script>
    </body>
</html>
"""


@asynccontextmanager
async def run_websocket_server(app):
    with IOWebsockets.run_server_in_thread(on_connect=on_connect, port=PORT) as uri:
        print(f"Websocket server started at {uri}.", flush=True)

        yield


app = FastAPI(lifespan=run_websocket_server)


@app.get("/")
async def get():
    return HTMLResponse(html)

import uvicorn  # noqa: E402

config = uvicorn.Config(app)
server = uvicorn.Server(config)
server.serve()  # noqa: F704
