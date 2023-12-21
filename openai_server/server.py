import os
import sys
import ast
import json
from threading import Thread
import time
from typing import List
from pydantic import BaseModel, Field

import uvicorn
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import JSONResponse
from sse_starlette import EventSourceResponse

sys.path.append('openai_server')
from log import logger, logging


# https://github.com/h2oai/h2ogpt/issues/1132
# https://github.com/jquesnelle/transformers-openai-api
# https://community.openai.com/t/trying-to-turn-this-into-an-automatic-web-search-engine/306383


class Generation(BaseModel):
    top_k: int = 0
    top_p: int = 0
    repetition_penalty: float = 1
    typical_p: float = 1
    tfs: float = 1
    top_a: float = 0
    epsilon_cutoff: float = 0
    eta_cutoff: float = 0
    penalty_alpha: float = 0
    do_sample: bool = False
    seed: int = -1
    encoder_repetition_penalty: float = 1
    no_repeat_ngram_size: int = 0
    min_length: int = 0
    num_beams: int = 1
    length_penalty: float = 1
    early_stopping: bool = False


class Params(BaseModel):
    user: str | None = Field(default=None, description="Track user")
    model: str | None = Field(default=None, description="Choose model")
    best_of: int | None = Field(default=1, description="Unused")
    frequency_penalty: float | None = 0
    max_tokens: int | None = 16
    n: int | None = Field(default=1, description="Unused")
    presence_penalty: float | None = 0
    stop: str | List[str] | None = None
    stream: bool | None = False
    temperature: float | None = 1
    top_p: float | None = 1


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


server_api_key = os.getenv('H2OGPT_OPENAI_API_KEY', 'EMPTY')


def verify_api_key(authorization: str = Header(None)) -> None:
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


def get_gradio_client():
    from gradio_client import Client
    gradio_port = int(os.getenv('H2OGPT_OPENAI_HOST', '7860'))
    client = Client(os.getenv('H2OGPT_OPENAI_HOST', 'http://localhost:%d' % gradio_port))
    return client


gradio_client = get_gradio_client()


# https://platform.openai.com/docs/api-reference/models/list
@app.get("/v1/models", dependencies=check_key)
@app.get("/v1/models/{model}", dependencies=check_key)
@app.get("/v1/models/{repo}/{model}", dependencies=check_key)
async def handle_models(request: Request):
    path = request.url.path
    model_name = path[len('/v1/models/'):]

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
    model_dict = ast.literal_eval(gradio_client.predict(api_name='/model_names'))
    response = dict(model_names=model_dict[0])
    return JSONResponse(content=response)


@app.get("/v1/internal/model/list", response_model=ModelListResponse, dependencies=check_key)
async def handle_list_models():
    model_dict = ast.literal_eval(gradio_client.predict(api_name='/model_names'))
    base_models = [x['base_model'] for x in model_dict]
    response = dict(model_names=base_models)
    return JSONResponse(content=response)


def run_server(host='0.0.0.0',
               port=5000,
               ssl_certfile=None,
               ssl_keyfile=None,
               gradio_host=None,
               gradio_port=None,
               ):
    os.environ['H2OGPT_OPENAI_HOST'] = gradio_host or 'localhost'
    os.environ['H2OGPT_OPENAI_PORT'] = gradio_port or '7860'

    port = int(os.getenv('H2OGPT_OPENAIPORT', port))
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
