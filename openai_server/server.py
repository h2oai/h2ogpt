import io
import os
import sys
import ast
import json
import time
from traceback import print_exception
from typing import List, Dict, Optional, Literal, Union
from pydantic import BaseModel, Field

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request, Depends
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sse_starlette import EventSourceResponse
from starlette.responses import PlainTextResponse

sys.path.append('openai_server')


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
    type: str = Literal["text", "json_object", "json_code"]


# https://github.com/vllm-project/vllm/blob/a3c226e7eb19b976a937e745f3867eb05f809278/vllm/entrypoints/openai/protocol.py#L62
class H2oGPTParams(BaseModel):
    # keep in sync with evaluate()
    # handled by extra_body passed to OpenAI API
    prompt_type: str | None = None
    prompt_dict: Dict | str | None = None
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

    user_prompt_for_fake_system_prompt: str | None = None
    json_object_prompt: str | None = None
    json_object_prompt_simpler: str | None = None
    json_code_prompt: str | None = None
    json_code_prompt_if_no_schema: str | None = None
    json_schema_instruction: str | None = None

    system_prompt: str | None = 'auto'

    image_audio_loaders: List | None = None
    pdf_loaders: List | None = None
    url_loaders: List | None = None
    jq_schema: List | None = None
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
    metadata_in_context: str | None = 'auto'

    chatbot_role: str | None = 'None'
    speaker: str | None = 'None'
    tts_language: str | None = 'autodetect'
    tts_speed: float | None = 1.0

    image_file: str | None = None
    image_control: str | None = None

    response_format: Optional[ResponseFormat] = Field(
        default=None,
        description=
        ("Similar to chat completion, this parameter specifies the format of "
         "output. Only {'type': 'text' } or {'type': 'json_object'} or {'type': 'json_code'} are "
         "supported."),
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


class Params(H2oGPTParams):
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


@app.get("/health")
async def health() -> Response:
    """Health check."""
    return Response(status_code=200)


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

    from openai_server.backend import get_client
    client = get_client()
    model_dict = ast.literal_eval(client.predict(api_name='/model_names'))
    base_models = [x['base_model'] for x in model_dict]

    if not model_name:
        response = {
            "object": "list",
            "data": [dict(id=x, object='model', created='NA', owned_by='H2O.ai') for x in base_models],
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
async def handle_audio_transcription(request: Request):
    form = await request.form()
    audio_file = await form["file"].read()
    model = form["model"]
    stream = form.get("stream", False)
    response_format = form.get("response_format", 'text')
    chunk = form.get("chunk", 'interval')
    request_data = dict(model=model, stream=stream, audio_file=audio_file, response_format=response_format, chunk=chunk)

    if stream:
        from openai_server.backend import audio_to_text

        async def generator():
            response = audio_to_text(**request_data)
            for resp in response:
                disconnected = await request.is_disconnected()
                if disconnected:
                    break

                yield {"data": json.dumps(resp)}

        return EventSourceResponse(generator())
    else:
        from openai_server.backend import _audio_to_text
        response = ''
        for response1 in _audio_to_text(**request_data):
            response = response1
        return JSONResponse(response)


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
    fake_size = (2**30 - 1) - 8
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
async def handle_audio_to_speech(
        request: Request,
):
    request_data = await request.json()
    audio_request = AudioTextRequest(**request_data)

    if audio_request.stream:
        from openai_server.backend import text_to_audio

        async def generator():
            chunki = 0
            for chunk in text_to_audio(**dict(audio_request)):
                disconnected = await request.is_disconnected()
                if disconnected:
                    break

                if chunki == 0 and audio_request.response_format == 'wav':
                    # pretend longer than is, like OpenAI does
                    chunk = modify_wav_header(chunk)
                # h2oGPT sends each chunk as full object, we need rest to be raw data without header for real streaming
                if chunki > 0 and audio_request.stream_strip:
                    from pydub import AudioSegment
                    chunk = AudioSegment.from_file(io.BytesIO(chunk), format=audio_request.response_format).raw_data

                yield chunk
                chunki += 1
        return StreamingResponse(generator(), media_type="audio/%s" % audio_request.response_format)
    else:
        from openai_server.backend import text_to_audio
        response = ''
        for response1 in text_to_audio(**dict(audio_request)):
            response = response1
        return Response(content=response, media_type="audio/%s" % audio_request.response_format)


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
async def handle_image_generation(request: Request):
    try:
        body = await request.json()
        model = body.get('model', '')  # will choose first if nothing passed
        prompt = body['prompt']
        size = body.get('size', '1024x1024')
        quality = body.get('quality', 'standard')
        n = body.get('n', 1)  # ignore the batch limits of max 10
        response_format = body.get('response_format', 'b64_json')  # or url

        image_request = dict(model=model, prompt=prompt, size=size, quality=quality, n=n,
                             response_format=response_format)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Missing key in request body: {str(e)}")

    # no streaming
    from openai_server.backend import completions
    body_image = dict(prompt=prompt, langchain_action='ImageGen', visible_image_models=model)
    response = completions(body_image)
    image = response['choices'][0]['text'][0]
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
async def handle_embeddings(request: Request, request_data: EmbeddingsRequest):
    # https://docs.portkey.ai/docs/api-reference/embeddings
    text = request_data.input
    model = request_data.model
    encoding_format = request_data.encoding_format

    from openai_server.backend import text_to_embedding
    response = text_to_embedding(model, text, encoding_format)
    return JSONResponse(response)
