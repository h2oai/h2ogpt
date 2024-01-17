import contextlib
import logging
import os
import sys
import ast
import json
from threading import Thread
import time
from traceback import print_exception
from typing import List
from pydantic import BaseModel, Field

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse
from starlette.responses import PlainTextResponse

from openai_server.log import logger

sys.path.append('openai_server')


# https://github.com/h2oai/h2ogpt/issues/1132
# https://github.com/jquesnelle/transformers-openai-api
# https://community.openai.com/t/trying-to-turn-this-into-an-automatic-web-search-engine/306383


class Generation(BaseModel):
    # put here things not supported by OpenAI but are by torch or vLLM
    # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
    top_k: int | None = 1
    repetition_penalty: float | None = 1
    min_p: float | None = 0.0
    max_time: float | None = 360


class Params(BaseModel):
    # https://platform.openai.com/docs/api-reference/completions/create
    user: str | None = Field(default=None, description="Track user")
    model: str | None = Field(default=None, description="Choose model")
    best_of: int | None = Field(default=1, description="Unused")
    frequency_penalty: float | None = 0.0
    max_tokens: int | None = 256
    n: int | None = Field(default=1, description="Unused")
    presence_penalty: float | None = 0.0
    stop: str | List[str] | None = None
    stop_token_ids: List[int] | None = None
    stream: bool | None = False
    temperature: float | None = 0.3
    top_p: float | None = 1.0
    seed: int | None = 1234


class CompletionParams(Params):
    prompt: str | List[str]
    logit_bias: dict | None = None
    logprobs: int | None = None


class TextRequest(Generation, CompletionParams):
    pass


class TextResponse(BaseModel):
    id: str
    choices: List[dict]
    created: int = int(time.time())
    model: str
    object: str = "text_completion"
    usage: dict


class ChatParams(Params):
    messages: List[dict]
    tools: list | None = Field(default=None, description="WIP")
    tool_choice: str | None = Field(default=None, description="WIP")


class ChatRequest(Generation, ChatParams):
    # https://platform.openai.com/docs/api-reference/chat/create
    pass


class ChatResponse(BaseModel):
    id: str
    choices: List[dict]
    created: int = int(time.time())
    model: str
    object: str = "chat.completion"
    usage: dict


class ModelInfoResponse(BaseModel):
    model_name: str


class ModelListResponse(BaseModel):
    model_names: List[str]


def verify_api_key(authorization: str = Header(None)) -> None:
    server_api_key = os.getenv('H2OGPT_OPENAI_API_KEY', 'EMPTY')
    if server_api_key == 'EMPTY':
        # dummy case since '' cannot be handled
        return
    if server_api_key and (authorization is None or authorization != f"Bearer {server_api_key}"):
        raise HTTPException(status_code=401, detail="Unauthorized")


app = FastAPI()
check_key = [Depends(verify_api_key)]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# https://platform.openai.com/docs/models/how-we-use-your-data


class InvalidRequestError(Exception):
    pass


@app.exception_handler(Exception)
async def validation_exception_handler(request, exc):
    print_exception(exc)
    exc2 = InvalidRequestError(str(exc))
    return PlainTextResponse(str(exc2), status_code=400)


@app.options("/", dependencies=check_key)
async def options_route():
    return JSONResponse(content="OK")


@app.post('/v1/completions', response_model=TextResponse, dependencies=check_key)
async def openai_completions(request: Request, request_data: TextRequest):
    if request_data.stream:
        async def generator():
            from openai_server.backend import stream_completions
            response = stream_completions(dict(request_data))
            for resp in response:
                disconnected = await request.is_disconnected()
                if disconnected:
                    break

                yield {"data": json.dumps(resp)}

        return EventSourceResponse(generator())

    else:
        from openai_server.backend import completions
        response = completions(dict(request_data))
        return JSONResponse(response)


@app.post('/v1/chat/completions', response_model=ChatResponse, dependencies=check_key)
async def openai_chat_completions(request: Request, request_data: ChatRequest):
    if request_data.stream:
        from openai_server.backend import stream_chat_completions

        async def generator():
            response = stream_chat_completions(dict(request_data))
            for resp in response:
                disconnected = await request.is_disconnected()
                if disconnected:
                    break

                yield {"data": json.dumps(resp)}

        return EventSourceResponse(generator())
    else:
        from openai_server.backend import chat_completions
        response = chat_completions(dict(request_data))
        return JSONResponse(response)


# https://platform.openai.com/docs/api-reference/models/list
@app.get("/v1/models", dependencies=check_key)
@app.get("/v1/models/{model}", dependencies=check_key)
@app.get("/v1/models/{repo}/{model}", dependencies=check_key)
async def handle_models(request: Request):
    path = request.url.path
    model_name = path[len('/v1/models/'):]

    from openai_server.backend import gradio_client
    model_dict = ast.literal_eval(gradio_client.predict(api_name='/model_names'))
    base_models = [x['base_model'] for x in model_dict]

    if not model_name:
        response = {
            "object": "list",
            "data": base_models,
        }
    else:
        model_index = base_models.index(model_name)
        if model_index >= 0:
            response = model_dict[model_index]
        else:
            response = dict(model_name='INVALID')

    return JSONResponse(response)


@app.get("/v1/internal/model/info", response_model=ModelInfoResponse, dependencies=check_key)
async def handle_model_info():
    from openai_server.backend import get_model_info
    return JSONResponse(content=get_model_info())


@app.get("/v1/internal/model/list", response_model=ModelListResponse, dependencies=check_key)
async def handle_list_models():
    from openai_server.backend import get_model_list
    return JSONResponse(content=get_model_list())


def run_server(host='0.0.0.0',
               port=5000,
               ssl_certfile=None,
               ssl_keyfile=None,
               gradio_prefix=None,
               gradio_host=None,
               gradio_port=None,
               h2ogpt_key=None,
               ):
    os.environ['GRADIO_PREFIX'] = gradio_prefix or 'http'
    os.environ['GRADIO_SERVER_HOST'] = gradio_host or 'localhost'
    os.environ['GRADIO_SERVER_PORT'] = gradio_port or '7860'
    os.environ['GRADIO_H2OGPT_H2OGPT_KEY'] = h2ogpt_key or ''  # don't use H2OGPT_H2OGPT_KEY, mixes things up
    # use h2ogpt_key if no server api key, so OpenAI inherits key by default if any keys set and enforced via API for h2oGPT
    # but OpenAI key cannot be '', so dummy value is EMPTY and if EMPTY we ignore the key in authorization
    server_api_key = os.getenv('H2OGPT_OPENAI_API_KEY', os.environ['GRADIO_H2OGPT_H2OGPT_KEY']) or 'EMPTY'
    os.environ['H2OGPT_OPENAI_API_KEY'] = server_api_key

    port = int(os.getenv('H2OGPT_OPENAI_PORT', port))
    ssl_certfile = os.getenv('H2OGPT_OPENAI_CERT_PATH', ssl_certfile)
    ssl_keyfile = os.getenv('H2OGPT_OPENAI_KEY_PATH', ssl_keyfile)

    prefix = 'https' if ssl_keyfile and ssl_certfile else 'http'
    logger.info(f'OpenAI API URL: {prefix}://{host}:{port}')
    logger.info(f'OpenAI API key: {server_api_key}')

    logging.getLogger("uvicorn.error").propagate = False
    uvicorn.run(app, host=host, port=port, ssl_certfile=ssl_certfile, ssl_keyfile=ssl_keyfile)


def run(wait=True, **kwargs):
    if wait:
        run_server(**kwargs)
    else:
        Thread(target=run_server, kwargs=kwargs, daemon=True).start()
