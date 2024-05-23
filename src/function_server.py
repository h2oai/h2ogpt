import os
import sys
import json
import inspect
import typing
from traceback import print_exception
from typing import Union

from pydantic import BaseModel

from fastapi import FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request, Depends
from fastapi.responses import JSONResponse, Response, StreamingResponse
from sse_starlette import EventSourceResponse
from starlette.responses import PlainTextResponse

sys.path.append('src')


# similar to openai_server/server.py
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


class InvalidRequestError(Exception):
    pass


class FunctionRequest(BaseModel):
    function_name: str
    args: list
    kwargs: dict
    use_disk: bool = False


# Example functions
def example_function1(x, y):
    return x + y


def example_function2(path: str):
    if not os.path.exists(path):
        raise ValueError("Path does not exist")
    if not os.path.isdir(path):
        raise ValueError("Path is not a directory")
    docs = [f for f in os.listdir(path) if f.endswith('.doc') or f.endswith('.docx')]
    return {"documents": docs}


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


@app.post("/execute_function/")
def execute_function(request: FunctionRequest):
    # Mapping of function names to function objects
    FUNCTIONS = {
        'example_function1': example_function1,
        'example_function2': example_function2,
    }
    try:
        # Fetch the function from the function map
        func = FUNCTIONS.get(request.function_name)
        if not func:
            raise ValueError("Function not found")

        # Call the function with args and kwargs
        result = func(*request.args, **request.kwargs)

        if request.use_disk:
            # Save the result to a file on the shared disk
            file_path = "/path/to/shared/disk/function_result.json"
            with open(file_path, "w") as f:
                json.dump(result, f)
            return {"status": "success", "file_path": file_path}
        else:
            # Return the result directly
            return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
