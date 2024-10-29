import copy
import io
import logging
import os
import sys
import ast
import json
import time
import traceback
import uuid
from traceback import print_exception
from typing import List, Dict, Optional, Literal, Union, Any

import filelock
import jsonschema
from pydantic import BaseModel, Field

from fastapi import FastAPI, Header, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request, Depends
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi import File, UploadFile
from sse_starlette import EventSourceResponse
from starlette.responses import PlainTextResponse

from openai_server.backend_utils import get_user_dir, run_upload_api, meta_ext

from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

sys.path.append('openai_server')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s: %(message)s')

# https://github.com/h2oai/h2ogpt/issues/1132
# https://github.com/jquesnelle/transformers-openai-api
# https://community.openai.com/t/trying-to-turn-this-into-an-automatic-web-search-engine/306383


class Generation(BaseModel):
    # put here things not supported by OpenAI but are by torch or vLLM
    # https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py
    top_k: int | None = 1
    min_p: float | None = 0.0


class ResponseFormat(BaseModel):
    # type must be "json_object" or "text"
    type: str = Literal["text", "json_object", "json_code", "json_schema"]
    json_schema: Optional[Dict[str, Any]] = None
    strict: Optional[bool] = True


# https://github.com/vllm-project/vllm/blob/a3c226e7eb19b976a937e745f3867eb05f809278/vllm/entrypoints/openai/protocol.py#L62
class H2oGPTParams(BaseModel):
    # keep in sync with evaluate()
    # handled by extra_body passed to OpenAI API
    enable_caching: bool | None = None
    prompt_type: str | None = None
    prompt_dict: Dict | str | None = None
    chat_template: str | None = None
    penalty_alpha: float | None = 0.0
    num_beams: int | None = 1
    min_new_tokens: int | None = 1
    early_stopping: bool | None = False
    max_time: float | None = 360
    repetition_penalty: float | None = 1
    num_return_sequences: int | None = 1
    do_sample: bool | None = None
    chat: bool | None = True
    langchain_mode: str | None = 'LLM'
    add_chat_history_to_context: bool | None = True
    langchain_action: str | None = 'Query'
    langchain_agents: List | None = []
    top_k_docs: int | None = 10
    chunk: bool | None = True
    chunk_size: int | None = 512
    document_subset: str | None = 'Relevant'
    document_choice: str | None = 'All'
    document_source_substrings: List | None = []
    document_source_substrings_op: str | None = 'and'
    document_content_substrings: List | None = []
    document_content_substrings_op: str | None = 'and'

    pre_prompt_query: str | None = None
    prompt_query: str | None = None
    pre_prompt_summary: str | None = None
    prompt_summary: str | None = None
    hyde_llm_prompt: str | None = None
    all_docs_start_prompt: str | None = None,
    all_docs_finish_prompt: str | None = None,

    user_prompt_for_fake_system_prompt: str | None = None
    json_object_prompt: str | None = None
    json_object_prompt_simpler: str | None = None
    json_code_prompt: str | None = None
    json_code_prompt_if_no_schema: str | None = None
    json_schema_instruction: str | None = None
    json_preserve_system_prompt: bool | None = False
    json_object_post_prompt_reminder: str | None = None
    json_code_post_prompt_reminder: str | None = None
    json_code2_post_prompt_reminder: str | None = None

    system_prompt: str | None = 'auto'

    image_audio_loaders: List | None = None
    pdf_loaders: List | None = None
    url_loaders: List | None = None
    jq_schema: str | None = None
    extract_frames: int | None = 10
    llava_prompt: str | None = 'auto'
    # visible_models
    # h2ogpt_key,
    add_search_to_context: bool | None = False

    chat_conversation: List | None = []
    text_context_list: List | None = []
    docs_ordering_type: str | None = None
    min_max_new_tokens: int | None = 512
    max_input_tokens: int | None = -1
    max_total_input_tokens: int | None = -1
    docs_token_handling: str | None = None
    docs_joiner: str | None = None
    hyde_level: int | None = 0
    hyde_template: str | None = 'auto'
    hyde_show_only_final: bool | None = False
    doc_json_mode: bool | None = False
    metadata_in_context: Union[str, list] | None = 'auto'

    chatbot_role: str | None = 'None'
    speaker: str | None = 'None'
    tts_language: str | None = 'autodetect'
    tts_speed: float | None = 1.0

    image_file: Union[str, list] | None = None
    image_control: str | None = None
    images_num_max: int | None = None
    image_resolution: tuple | None = None
    image_format: str | None = None
    rotate_align_resize_image: bool | None = None
    video_frame_period: int | None = None
    image_batch_image_prompt: str | None = None
    image_batch_final_prompt: str | None = None
    image_batch_stream: bool | None = None
    visible_vision_models: Union[str, int, list] | None = 'auto'
    video_file: Union[str, list] | None = None

    model_lock: dict | None = None
    client_metadata: str | None = ''

    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description=(
            "Similar to chat completion, this parameter specifies the format of "
            "output. Only {'type': 'text' } or {'type': 'json_object'} or {'type': 'json_code'} or {'type': 'json_schema'} are "
            "supported."
        ),
    )
    guided_json: Optional[Union[str, dict, BaseModel]] = Field(
        default=None,
        description="If specified, the output will follow the JSON schema.",
    )
    guided_regex: Optional[str] = Field(
        default=None,
        description=("If specified, the output will follow the regex pattern."),
    )
    guided_choice: Optional[List[str]] = Field(
        default=None,
        description="If specified, the output will be exactly one of the choices.",
    )
    guided_grammar: Optional[str] = Field(
        default=None,
        description="If specified, the output will follow the context free grammar.",
    )
    guided_whitespace_pattern: Optional[str] = Field(
        default=None,
        description="If specified, JSON white space will be restricted.",
    )


class AgentParams(BaseModel):
    use_agent: bool | None = False
    autogen_stop_docker_executor: bool | None = False
    autogen_run_code_in_docker: bool | None = False
    autogen_max_consecutive_auto_reply: int | None = 10
    autogen_max_turns: int | None = None
    autogen_timeout: int = 120
    agent_verbose: bool = False
    autogen_cache_seed: int | None = None
    agent_venv_dir: str | None = None
    agent_code_writer_system_message: str | None = None
    agent_system_site_packages: bool = True
    autogen_code_restrictions_level: int = 2
    autogen_silent_exchange: bool = True
    agent_type: str | None = 'auto'
    agent_accuracy: str | None = 'standard'
    agent_work_dir: str | None = None
    agent_chat_history: list | None = []
    agent_files: list | None = []


class Params(H2oGPTParams, AgentParams):
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
    seed: int | None = 0  # 0 means random seed if sampling


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


class Model(BaseModel):
    id: str
    object: str = 'model'
    created: str = 'na'
    owned_by: str = 'H2O.ai'


class ModelInfoResponse(BaseModel):
    model_info: str


class ModelListResponse(BaseModel):
    model_names: List[Model]


def verify_api_key(authorization: str = Header(None)) -> None:
    server_api_key = os.getenv('H2OGPT_OPENAI_API_KEY', 'EMPTY')
    if server_api_key:
        h2ogpt_api_keys = [server_api_key]
    else:
        h2ogpt_api_keys = []

    if server_api_key == 'EMPTY':
        # dummy case since '' cannot be handled
        # disables all auth
        return

    # assume if set file, shared keys for h2oGPT and OpenAI uses
    server_api_key_file = os.getenv('H2OGPT_H2OGPT_API_KEYS')

    # string of list case
    if isinstance(server_api_key_file, str) and not os.path.isfile(server_api_key_file):
        h2ogpt_api_keys.extend(ast.literal_eval(server_api_key_file))

    # file case
    if isinstance(server_api_key_file, str) and os.path.isfile(server_api_key_file):
        with filelock.FileLock(server_api_key_file + '.lock'):
            with open(server_api_key_file, 'rt') as f:
                h2ogpt_api_keys.extend(json.load(f))

    # no keys case
    if len(h2ogpt_api_keys) == 0:
        return

    if any([authorization is not None and authorization == f"Bearer {x}" for x in h2ogpt_api_keys]):
        return

    raise HTTPException(status_code=401, detail="Unauthorized")


# Dependency that extracts the model and stores it in request state
async def extract_model_from_request(request: Request, request_data: ChatRequest):
    request.state.model = request_data.model
    return request_data


limiter = Limiter(key_func=get_remote_address)
global_limiter = Limiter(key_func=lambda: "global")  # Global limiter with constant key


def model_rate_limit_key(request: Request):
    # Extract the model from request data, assuming it's in the JSON body
    # Since we are in FastAPI, we'll retrieve the model from the request object
    # FastAPI request's `state` can store request data parsed by dependency injection

    model = request.state.model  # Set by a dependency or manually within the route
    if not model:
        raise ValueError("Model not provided in request data")

    # Use the model name as the key for rate limiting
    return model


def api_key_rate_limit_key(request: Request):
    # Example: Extract user ID or API key for rate limiting
    return request.headers.get("X-User-ID", 'unknown')


app = FastAPI()
check_key = [Depends(verify_api_key)]
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Add SlowAPI middleware for rate limiting (without limiter argument)
app.add_middleware(SlowAPIMiddleware)

# Set limiter in the app state
app.state.limiter = limiter
app.state.global_limiter = global_limiter

# Exception handler for rate limit exceeded
app.add_exception_handler(RateLimitExceeded,
                          lambda request, exc: JSONResponse({"error": "rate limit exceeded"}, status_code=429))


# https://platform.openai.com/docs/models/how-we-use-your-data


class InvalidRequestError(Exception):
    pass


status_limiter_global = os.getenv('H2OGPT_STATUS_LIMITER_GLOBAL', '100/second')
status_limiter_user = os.getenv('H2OGPT_STATUS_LIMITER_USER', '3/second')

completion_limiter_global = os.getenv('H2OGPT_COMPLETION_LIMITER_GLOBAL', '30/second')
completion_limiter_user = os.getenv('H2OGPT_STATUS_LIMITER_USER', '5/second')
completion_limiter_model = os.getenv('H2OGPT_STATUS_LIMITER_MODEL', '1/second')

audio_limiter_global = os.getenv('H2OGPT_AUDIO_LIMITER_GLOBAL', '20/second')
audio_limiter_user = os.getenv('H2OGPT_AUDIO_LIMITER_USER', '5/second')

image_limiter_global = os.getenv('H2OGPT_IMAGE_LIMITER_GLOBAL', '5/second')
image_limiter_user = os.getenv('H2OGPT_IMAGE_LIMITER_USER', '1/second')

embedding_limiter_global = os.getenv('H2OGPT_EMBEDDING_LIMITER_GLOBAL', '30/second')
embedding_limiter_user = os.getenv('H2OGPT_EMBEDDING_LIMITER_USER', '1/second')

file_limiter_global = os.getenv('H2OGPT_FILE_LIMITER_GLOBAL', '50/second')
file_limiter_user = os.getenv('H2OGPT_FILE_LIMITER_USER', '20/second')


@app.get("/health")
@limiter.limit(status_limiter_user, key_func=api_key_rate_limit_key)
@global_limiter.limit(status_limiter_global)
async def health(request: Request) -> Response:
    """Health check."""
    return Response(status_code=200)


@app.get("/version")
@limiter.limit(status_limiter_user, key_func=api_key_rate_limit_key)
@global_limiter.limit(status_limiter_global)
async def show_version(request: Request):
    try:
        from ..src.version import __version__
        githash = __version__
    except:
        githash = 'unknown'
    ver = {"version": githash}
    return JSONResponse(content=ver)


@app.exception_handler(Exception)
async def validation_exception_handler(request, exc):
    print_exception(exc)
    exc2 = InvalidRequestError(str(exc))
    return PlainTextResponse(str(exc2), status_code=400)


@app.options("/", dependencies=check_key)
async def options_route():
    return JSONResponse(content="OK")


@app.post('/v1/completions', response_model=TextResponse, dependencies=check_key)
@global_limiter.limit(completion_limiter_global)
@limiter.limit(completion_limiter_user, key_func=api_key_rate_limit_key)
@limiter.limit(completion_limiter_model, key_func=model_rate_limit_key)
async def openai_completions(request: Request, request_data: TextRequest, authorization: str = Header(None)):
    try:
        request_data_dict = dict(request_data)
        request_data_dict['authorization'] = authorization

        if request_data.stream:
            async def generator():
                try:
                    from openai_server.backend import astream_completions
                    async for resp in astream_completions(request_data_dict, stream_output=True):
                        disconnected = await request.is_disconnected()
                        if disconnected:
                            return

                        yield {"data": json.dumps(resp)}
                except Exception as e1:
                    print(traceback.format_exc())
                    error_response = {
                        "error": {
                            "message": str(e1),
                            "type": "server_error",
                            "param": None,
                            "code": "500"
                        }
                    }
                    yield {"data": json.dumps(error_response)}
                    # After yielding the error, we'll close the connection
                    return
                    # raise e1

            return EventSourceResponse(generator())

        else:
            from openai_server.backend import astream_completions
            response = {}
            async for resp in astream_completions(request_data_dict, stream_output=False):
                if await request.is_disconnected():
                    return
                response = resp
            return JSONResponse(response)

    except Exception as e:
        # This will handle any exceptions that occur outside of the streaming context
        # or in the non-streaming case
        error_response = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "param": None,
                "code": 500
            }
        }
        raise HTTPException(status_code=500, detail=error_response)


def random_uuid() -> str:
    return str(uuid.uuid4().hex)


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ToolCall(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-tool-{random_uuid()}")
    type: Literal["function"] = "function"
    function: FunctionCall


async def get_tool(request: Request, request_data: ChatRequest, authorization: str = Header(None)):
    try:
        return _get_tool(request, request_data, authorization)
    except Exception as e1:
        # For non-streaming responses, we'll return a JSON error response
        raise HTTPException(status_code=500, detail={
            "error": {
                "message": str(e1),
                "type": "server_error",
                "param": None,
                "code": 500
            }
        })


async def _get_tool(request: Request, request_data: ChatRequest, authorization: str = Header(None)):
    request_data_dict = dict(request_data)
    request_data_dict = copy.deepcopy(request_data_dict)

    tools = request_data_dict.get('tools')
    model = request_data_dict.get('model')
    prompt = ""
    tool_names = []
    tool_dict = {}
    tool_dict['noop'] = None
    for tool in tools:
        assert tool['type'] == 'function'
        tool_name = tool['function']['name']
        tool_dict[tool_name] = tool
        tool_description = tool['function']['description']
        if 'claude' in model:
            prompt += f'<tool>\n<name>\n{tool_name}\n</name>\n<description>\n{tool_description}\n</description>\n</tool>\n'
        else:
            prompt += f'# Tool Name\n\n{tool_name}\n# Tool Description:\n\n{tool_description}\n\n'
        tool_names.append(tool_name)
    if not request_data_dict['messages']:
        raise ValueError("No messages in request, required for tool_choice='auto'")
    original_prompt = request_data_dict['messages'][0]['content']
    if 'claude' in model:
        prompt += f"<prompt>\n{original_prompt}\n</prompt>\n"
    else:
        prompt += f"# Prompt\n\n{original_prompt}\n\n"

    prompt += """
Choose the single tool that best solves the task inferred from the prompt.  Never choose more than one tool, i.e. act like parallel_tool_calls=False.  If no tool is a good fit, then only choose the noop tool.
"""
    request_data_dict['guided_json'] = {
        "type": "object",
        "properties": {
            "tool": {
                "type": "string",
                "description": "The name of the single best tool to use to solve the task inferred from the user prompt.  If no tool is a good fit, then only choose the noop tool.",
                "enum": tool_names + ['noop'],
            },
        },
        "required": ["tool"]
    }
    request_data_dict['response_format'] = dict(type='json_object')
    request_data_dict['text_context_list'] = []
    request_data_dict['use_agent'] = False
    request_data_dict['add_chat_history_to_context'] = False
    request_data_dict['chat_conversation'] = []
    request_data_dict['stream_output'] = False
    request_data_dict['stream'] = False
    request_data_dict['langchain_action'] = 'Query'
    request_data_dict['langchain_agents'] = []
    request_data_dict['system_prompt'] = "You are a JSON maker."
    request_data_dict['max_tokens'] = max(request_data_dict.get('max_tokens', 256), 256)
    request_data_dict['hyde_level'] = 0

    messages = [{'content': prompt, 'role': 'user'}]
    request_data_dict['messages'] = messages
    # avoid recursion
    request_data_dict['tools'] = None
    # recurse
    request_data = ChatRequest(**request_data_dict)

    trials = 3
    tool_name = None
    msgs = []
    for trial in range(trials):
        response_json = await openai_chat_completions(request, request_data, authorization)
        response_all = json.loads(response_json.body)
        json_answer = json.loads(response_all['choices'][0]['message']['content'])
        msgs.append(json_answer)
        print(json_answer)
        try:
            jsonschema.validate(instance=json_answer, schema=request_data_dict['guided_json'])
        except:
            continue
        if 'tool' not in json_answer:
            continue
        tool_name = json_answer['tool']
        break
    print(msgs)
    if tool_name is None:
        raise RuntimeError("Failed to get tool choice: %s" % msgs)
    return tool_name, tool_dict[tool_name]


def tool_to_guided_json(tool):
    guided_json = {
        "type": "object",
        "properties": tool,
    }
    return guided_json


@app.post('/v1/chat/completions', response_model=ChatResponse, dependencies=check_key)
@global_limiter.limit(completion_limiter_global)
@limiter.limit(completion_limiter_user, key_func=api_key_rate_limit_key)
@limiter.limit(completion_limiter_model, key_func=model_rate_limit_key)
async def openai_chat_completions(request: Request,
                                  request_data: ChatRequest = Depends(extract_model_from_request),
                                  authorization: str = Header(None)):
    request_data_dict = dict(request_data)
    request_data_dict['authorization'] = authorization

    str_uuid = str(uuid.uuid4())
    if 'client_metadata' in request_data_dict:
        logging.info(f"Chat Completions request {str_uuid}: {len(request_data_dict)} items client_metadata: {request_data_dict['client_metadata']}")
    else:
        logging.info(f"Chat Completions request {str_uuid}: {len(request_data_dict)} items")

    # don't allow tool use with guided_json for now
    if request_data_dict['guided_json'] and request_data_dict.get('tools'):
        raise NotImplementedError("Cannot use tools with guided_json, because guided_json used for tool use.")

    # extract tool or do auto
    if request_data_dict.get('tool_choice') == 'auto' and request_data_dict.get('tools'):
        tool_name_chosen, tool_chosen = await get_tool(request, request_data, authorization)
        request_data_dict['tools'] = []
        if tool_name_chosen != 'noop':
            request_data_dict['guided_json'] = tool_to_guided_json(tool_chosen)
            request_data_dict['tool_choice'] = tool_name_chosen
        else:
            request_data_dict['tool_choice'] = 'auto'

    # handle json_schema -> guided_json
    # https://platform.openai.com/docs/guides/structured-outputs/how-to-use?context=without_parse&lang=python
    if request_data_dict['response_format'] and request_data_dict['response_format'].type == 'json_schema':
        json_schema = request_data_dict['response_format'].json_schema
        if json_schema:
            # try to json.loads schema to ensure correct
            if not isinstance(json_schema, dict):
                json_schema_dict = json.loads(json_schema)
            else:
                json_schema_dict = json_schema
            assert 'schema' in json_schema_dict, "Schema should start by containing 'name' and 'schema' keys."
            schema = json_schema_dict['schema']
            assert schema, "Inner schema key should contain at least 'type: 'object' and 'properties' keys and can include 'required' or 'additionalProperties'"
            if not isinstance(schema, dict):
                schema_dict = json.loads(schema)
            else:
                schema_dict = schema
            assert schema_dict, "Inner schema key should contain at least 'type: 'object' and 'properties' keys and can include 'required' or 'additionalProperties'"
            request_data_dict['guided_json'] = schema_dict
        else:
            raise ValueError("Specified response_format type json_schema but no json_schema provided.")
        request_data_dict['response_format'] = ResponseFormat(type='json_object')

    if request_data.stream:
        from openai_server.backend import astream_chat_completions

        async def generator():
            try:
                async for resp1 in astream_chat_completions(request_data_dict, stream_output=True):
                    if await request.is_disconnected():
                        if 'client_metadata' in request_data_dict:
                            logging.info(f"Chat Completions disconnected {str_uuid}: client_metadata: {request_data_dict['client_metadata']}")
                        return

                    yield {"data": json.dumps(resp1)}
                if 'client_metadata' in request_data_dict:
                    logging.info(f"Chat Completions streaming finished {str_uuid}: client_metadata: {request_data_dict['client_metadata']}")
            except Exception as e1:
                print(traceback.format_exc())
                # Instead of raising an HTTPException, we'll yield a special error message
                error_response = {
                    "error": {
                        "message": str(e1),
                        "type": "server_error",
                        "param": None,
                        "code": "500"
                    }
                }
                print(error_response)
                if 'client_metadata' in request_data_dict:
                    logging.info(f"Chat Completions error {str_uuid}: client_metadata: {request_data_dict['client_metadata']}: {error_response}")
                yield {"data": json.dumps(error_response)}
                # After yielding the error, we'll close the connection
                return
                # avoid sending more data back as exception, just be done
                # raise e1

        return EventSourceResponse(generator())
    else:
        from openai_server.backend import astream_chat_completions
        try:
            response = {}
            async for resp in astream_chat_completions(request_data_dict, stream_output=False):
                if await request.is_disconnected():
                    return
                response = resp
            if 'client_metadata' in request_data_dict:
                logging.info(f"Chat Completions non-streaming finished {str_uuid}: client_metadata: {request_data_dict['client_metadata']}")
            return JSONResponse(response)
        except Exception as e:
            traceback.print_exc()
            # For non-streaming responses, we'll return a JSON error response
            error_response = {
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "param": None,
                    "code": 500
                }
            }
            print(error_response)
            raise HTTPException(status_code=500, detail=error_response)


# https://platform.openai.com/docs/api-reference/models/list
@app.get("/v1/models", dependencies=check_key)
@app.get("/v1/models/{model}", dependencies=check_key)
@app.get("/v1/models/{repo}/{model}", dependencies=check_key)
@limiter.limit(status_limiter_user, key_func=api_key_rate_limit_key)
@global_limiter.limit(status_limiter_global)
async def handle_models(request: Request):
    path = request.url.path
    model_name = path[len('/v1/models/'):]

    from openai_server.backend import get_client
    client = get_client()
    model_dict = ast.literal_eval(client.predict(api_name='/model_names'))
    for model_i, model in enumerate(model_dict):
        model_dict[model_i].update(dict(id=model.get('base_model'), object='model', created='NA', owned_by='H2O.ai'))

    if not model_name:
        response = {
            "object": "list",
            "data": model_dict,
        }
        return JSONResponse(response)
    else:
        model_info = [x for x in model_dict if x.get('base_model') == model_name]
        if model_info:
            model_info = model_info[0]
        response = model_info.copy() if model_info else {}
        if model_info is None:
            raise ValueError("No such model %s" % model_name)

        return JSONResponse(response)


@app.get("/v1/internal/model/info", response_model=ModelInfoResponse, dependencies=check_key)
@limiter.limit(status_limiter_user, key_func=api_key_rate_limit_key)
@global_limiter.limit(status_limiter_global)
async def handle_model_info(request: Request):
    from openai_server.backend import get_model_info
    return JSONResponse(content=get_model_info())


@app.get("/v1/internal/model/list", response_model=ModelListResponse, dependencies=check_key)
@limiter.limit(status_limiter_user, key_func=api_key_rate_limit_key)
@global_limiter.limit(status_limiter_global)
async def handle_list_models(request: Request):
    from openai_server.backend import get_model_list
    return JSONResponse(content=[dict(id=x) for x in get_model_list()])


# Define your request data model
class AudiotoTextRequest(BaseModel):
    model: str = ''
    file: str
    response_format: str = 'text'  # FIXME unused (https://platform.openai.com/docs/api-reference/audio/createTranscription#images/create-response_format)
    stream: bool = True  # NOTE: No effect on OpenAI API client, would have to use direct API
    timestamp_granularities: list = ["word"]  # FIXME unused
    chunk: Union[str, int] = 'silence'  # or 'interval'   No effect on OpenAI API client, would have to use direct API


@app.post('/v1/audio/transcriptions', dependencies=check_key)
@limiter.limit(audio_limiter_user, key_func=api_key_rate_limit_key)
@global_limiter.limit(audio_limiter_global)
async def handle_audio_transcription(request: Request):
    try:
        form = await request.form()
        audio_file = await form["file"].read()
        model = form["model"]
        stream = form.get("stream", False)
        response_format = form.get("response_format", 'text')
        chunk = form.get("chunk", 'interval')
        request_data = dict(model=model, stream=stream, audio_file=audio_file, response_format=response_format,
                            chunk=chunk)

        if stream:
            from openai_server.backend import audio_to_text

            async def generator():
                try:
                    async for resp in audio_to_text(**request_data):
                        disconnected = await request.is_disconnected()
                        if disconnected:
                            break

                        yield {"data": json.dumps(resp)}
                except Exception as e1:
                    error_response = {
                        "error": {
                            "message": str(e1),
                            "type": "server_error",
                            "param": None,
                            "code": "500"
                        }
                    }
                    yield {"data": json.dumps(error_response)}
                    # raise e1  # This will close the connection after sending the error
                    return

            return EventSourceResponse(generator())
        else:
            from openai_server.backend import _audio_to_text
            response = ''
            async for response1 in _audio_to_text(**request_data):
                response = response1
            return JSONResponse(response)

    except Exception as e:
        # This will handle any exceptions that occur outside of the streaming context
        # or in the non-streaming case
        error_response = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "param": None,
                "code": 500
            }
        }
        raise HTTPException(status_code=500, detail=error_response)


# Define your request data model
class AudioTextRequest(BaseModel):
    model: str = ''
    voice: str = ''  # overrides both chatbot_role and speaker if set
    input: str
    response_format: str = 'wav'  # "mp3", "opus", "aac", "flac", "wav", "pcm"
    stream: bool = True
    stream_strip: bool = True
    chatbot_role: str = "Female AI Assistant"  # Coqui TTS
    speaker: str = "SLT (female)"  # Microsoft TTS


def modify_wav_header(wav_bytes):
    # Ensure the bytes start with the 'RIFF' identifier
    if wav_bytes[:4] != b'RIFF':
        raise ValueError("This is not a valid WAV file.")

    # Get current size (which we will fake)
    original_size = int.from_bytes(wav_bytes[4:8], byteorder='little')
    # print("Original size:", original_size)

    # Calculate fake size (Maximum value for 32-bit unsigned int minus 8)
    fake_size = (2 ** 30 - 1) - 8
    modified_size_bytes = fake_size.to_bytes(4, byteorder='little')

    # Replace the original size with the fake size in the RIFF header
    modified_wav_bytes = wav_bytes[:4] + modified_size_bytes + wav_bytes[8:]

    # Find the 'data' chunk and modify its size too
    data_chunk_pos = modified_wav_bytes.find(b'data')
    if data_chunk_pos == -1:
        raise ValueError("Data chunk not found in WAV file.")

    # Set a large fake size for the data chunk as well
    modified_wav_bytes = (
            modified_wav_bytes[:data_chunk_pos + 4] +  # 'data' text
            modified_size_bytes +  # fake size for data chunk
            modified_wav_bytes[data_chunk_pos + 8:]  # rest of data
    )

    return modified_wav_bytes


@app.post('/v1/audio/speech', dependencies=check_key)
@limiter.limit(audio_limiter_user, key_func=api_key_rate_limit_key)
@global_limiter.limit(audio_limiter_global)
async def handle_audio_to_speech(request: Request):
    try:
        request_data = await request.json()
        audio_request = AudioTextRequest(**request_data)

        if audio_request.stream:
            from openai_server.backend import text_to_audio

            async def generator():
                try:
                    chunki = 0
                    async for chunk in text_to_audio(**dict(audio_request)):
                        disconnected = await request.is_disconnected()
                        if disconnected:
                            break

                        if chunki == 0 and audio_request.response_format == 'wav':
                            # pretend longer than is, like OpenAI does
                            chunk = modify_wav_header(chunk)
                        # h2oGPT sends each chunk as full object, we need rest to be raw data without header for real streaming
                        if chunki > 0 and audio_request.stream_strip:
                            from pydub import AudioSegment
                            chunk = AudioSegment.from_file(io.BytesIO(chunk),
                                                           format=audio_request.response_format).raw_data

                        yield chunk
                        chunki += 1
                except Exception as e:
                    # For streaming audio, we can't send a JSON error response in the middle of the stream
                    # Instead, we'll log the error and stop the stream
                    print(f"Error in audio streaming: {str(e)}")
                    return  # This will effectively close the stream

            return StreamingResponse(generator(), media_type=f"audio/{audio_request.response_format}")
        else:
            from openai_server.backend import text_to_audio
            response = b''
            async for response1 in text_to_audio(**dict(audio_request)):
                response = response1
            return Response(content=response, media_type=f"audio/{audio_request.response_format}")

    except Exception as e:
        # This will handle any exceptions that occur outside of the streaming context
        # or in the non-streaming case
        error_response = {
            "error": {
                "message": str(e),
                "type": "server_error",
                "param": None,
                "code": 500
            }
        }
        return JSONResponse(status_code=500, content=error_response)


class ImageGenerationRequest(BaseModel):
    model: str = ''
    prompt: str
    size: str = '1024x1024'
    quality: str = 'standard'
    n: int = 1
    response_format: str = 'url'  # FIXME: https://platform.openai.com/docs/api-reference/images/create#images/create-response_format
    style: str = 'vivid'
    user: str = None


@app.post('/v1/images/generations', dependencies=check_key)
@limiter.limit(image_limiter_user, key_func=api_key_rate_limit_key)
@global_limiter.limit(image_limiter_global)
async def handle_image_generation(request: Request):
    try:
        body = await request.json()
        model = body.get('model', '')  # will choose first if nothing passed
        prompt = body['prompt']
        size = body.get('size', '1024x1024')
        quality = body.get('quality', 'standard')
        guidance_scale = body.get('guidance_scale')
        num_inference_steps = body.get('num_inference_steps')
        n = body.get('n', 1)  # ignore the batch limits of max 10
        response_format = body.get('response_format', 'b64_json')  # or url

        # TODO: Why not using image_request? size, quality and stuff?
        image_request = dict(model=model, prompt=prompt, size=size, quality=quality, n=n,
                             response_format=response_format, guidance_scale=guidance_scale,
                             num_inference_steps=num_inference_steps)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key in request body: {str(e)}")

    # no streaming
    from openai_server.backend import astream_completions
    body_image = dict(prompt=prompt, langchain_action='ImageGen', visible_image_models=model,
                      image_size=size,
                      image_quality=quality,
                      image_guidance_scale=guidance_scale,
                      image_num_inference_steps=num_inference_steps)
    response = {}
    async for resp in astream_completions(body_image, stream_output=False):
        response = resp
    if 'choices' in response:
        image = response['choices'][0]['text'][0]
    else:
        image = b''
    resp = {
        'created': int(time.time()),
        'data': []
    }
    import base64
    if os.path.isfile(image):
        with open(image, 'rb') as f:
            image = f.read()
    encoded_image = base64.b64encode(image).decode('utf-8')
    if response_format == 'b64_json':
        resp['data'].extend([{'b64_json': encoded_image}])
        return JSONResponse(resp)
    else:
        # FIXME: jpg vs. others
        resp['data'].extend([{'url': f'data:image/jpg;base64,{encoded_image}'}])
        return JSONResponse(resp)


class EmbeddingsResponse(BaseModel):
    index: int
    embedding: List[float]
    object: str = "embedding"


class EmbeddingsRequest(BaseModel):
    input: str | List[str] | List[int] | List[List[int]]
    model: str | None = Field(default=None, description="Unused parameter.")
    encoding_format: str = Field(default="float", description="float or base64.")
    user: str | None = Field(default=None, description="Unused parameter.")


@app.post("/v1/embeddings", response_model=EmbeddingsResponse, dependencies=check_key)
@limiter.limit(embedding_limiter_user, key_func=api_key_rate_limit_key)
@global_limiter.limit(embedding_limiter_global)
async def handle_embeddings(request: Request, request_data: EmbeddingsRequest):
    # https://docs.portkey.ai/docs/api-reference/embeddings
    text = request_data.input
    model = request_data.model
    encoding_format = request_data.encoding_format

    str_uuid = str(uuid.uuid4())
    logging.info(
        f"Embeddings request {str_uuid}: {len(text)} items, model: {model}, encoding_format: {encoding_format}")

    from openai_server.backend import text_to_embedding
    response = text_to_embedding(model, text, encoding_format)

    try:
        return JSONResponse(response)
    except Exception as e:
        traceback.print_exc()
        print(str(e))
    finally:
        if response:
            logging.info(
                f"Done embeddings response {str_uuid}: {len(response['data'])} items, model: {model}, encoding_format: {encoding_format}")
        else:
            logging.error(f"No embeddings response {str_uuid}")


# https://platform.openai.com/docs/api-reference/files

class UploadFileResponse(BaseModel):
    id: str
    object: str
    bytes: int
    created_at: int
    filename: str
    purpose: str


@app.post("/v1/files", response_model=UploadFileResponse, dependencies=check_key)
@limiter.limit(file_limiter_user, key_func=api_key_rate_limit_key)
@global_limiter.limit(file_limiter_global)
async def upload_file(
        request: Request,
        file: UploadFile = File(...),
        purpose: str = Form(...),
        authorization: str = Header(None)
):
    content = await file.read()
    filename = file.filename
    response_dict = run_upload_api(content, filename, purpose, authorization)

    response = UploadFileResponse(**response_dict)
    return response


class FileData(BaseModel):
    id: str
    object: str
    bytes: int
    created_at: int
    filename: str
    purpose: str


class ListFilesResponse(BaseModel):
    data: List[FileData]


@app.get("/v1/files", response_model=ListFilesResponse, dependencies=check_key)
@limiter.limit(file_limiter_user, key_func=api_key_rate_limit_key)
@global_limiter.limit(file_limiter_global)
async def list_files(request: Request, authorization: str = Header(None)):
    user_dir = get_user_dir(authorization)

    if not user_dir:
        raise HTTPException(status_code=404, detail="No user_dir for authorization: %s" % authorization)

    if not os.path.isdir(user_dir):
        os.makedirs(user_dir, exist_ok=True)

    if not os.path.exists(user_dir):
        raise HTTPException(status_code=404, detail="Directory not found")

    files_list = []
    for file_id in os.listdir(user_dir):
        file_path = os.path.join(user_dir, file_id)
        if file_path.endswith(meta_ext):
            continue
        if os.path.isfile(file_path):
            file_stat = os.stat(file_path)
            file_path_meta = os.path.join(user_dir, file_id + meta_ext)
            if os.path.isfile(file_path_meta):
                with open(file_path_meta, "rt") as f:
                    meta = json.loads(f.read())
            else:
                meta = {}

            files_list.append(
                FileData(
                    id=file_id,
                    object="file",
                    bytes=meta.get('bytes', file_stat.st_size),
                    created_at=meta.get('created_at', int(file_stat.st_ctime)),
                    filename=meta.get('filename', file_id),
                    purpose=meta.get('purpose', "unknown"),
                )
            )

    return ListFilesResponse(data=files_list)


class RetrieveFileResponse(BaseModel):
    id: str
    object: str
    bytes: int
    created_at: int
    filename: str
    purpose: str


@app.get("/v1/files/{file_id}", response_model=RetrieveFileResponse, dependencies=check_key)
@limiter.limit(file_limiter_user, key_func=api_key_rate_limit_key)
@global_limiter.limit(file_limiter_global)
async def retrieve_file(request: Request, file_id: str, authorization: str = Header(None)):
    user_dir = get_user_dir(authorization)
    file_path = os.path.join(user_dir, file_id)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"retrieve_file: {file_id}: File not found")

    file_path_meta = os.path.join(user_dir, file_id + meta_ext)
    if os.path.isfile(file_path_meta):
        with open(file_path_meta, "rt") as f:
            meta = json.loads(f.read())
    else:
        meta = {}

    file_stat = os.stat(file_path)
    response = RetrieveFileResponse(
        id=file_id,
        object="file",
        bytes=meta.get('bytes', file_stat.st_size),
        created_at=meta.get('created_at', int(file_stat.st_ctime)),
        filename=meta.get('filename', file_id),
        purpose=meta.get('purpose', "unknown"),
    )

    return response


class DeleteFileResponse(BaseModel):
    id: str
    object: str
    deleted: bool


@app.delete("/v1/files/{file_id}", response_model=DeleteFileResponse, dependencies=check_key)
@limiter.limit(file_limiter_user, key_func=api_key_rate_limit_key)
@global_limiter.limit(file_limiter_global)
async def delete_file(request: Request, file_id: str, authorization: str = Header(None)):
    user_dir = get_user_dir(authorization)
    file_path = os.path.join(user_dir, file_id)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"delete_file {file_id}: File not found")

    try:
        os.remove(file_path)
        deleted = True
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred while deleting the file: {str(e)}")

    response = DeleteFileResponse(
        id=file_id,
        object="file",
        deleted=deleted
    )

    return response


@app.get("/v1/files/{file_id}/content", dependencies=check_key)
@limiter.limit(file_limiter_user, key_func=api_key_rate_limit_key)
@global_limiter.limit(file_limiter_global)
async def retrieve_file_content(request: Request, file_id: str, stream: bool = Query(False),
                                authorization: str = Header(None)):
    user_dir = get_user_dir(authorization)
    file_path = os.path.join(user_dir, file_id)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"retrieve_file_content: {file_id}: File not found")

    if stream:
        def iter_file():
            with open(file_path, mode="rb") as file_like:
                while chunk := file_like.read(1024):
                    yield chunk

        return StreamingResponse(iter_file(), media_type="application/octet-stream")
    else:
        with open(file_path, mode="rb") as file:
            content = file.read()
        return Response(content, media_type="application/octet-stream")
