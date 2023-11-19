import os
import typing
import json
from langchain.llms import SagemakerEndpoint
from langchain.llms.sagemaker_endpoint import LLMContentHandler
from pydantic.v1 import root_validator

from src.utils import FakeTokenizer


class ChatContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: typing.Dict) -> bytes:
        messages0 = []
        openai_system_prompt = "You are a helpful assistant."
        if openai_system_prompt:
            messages0.append({"role": "system", "content": openai_system_prompt})
        messages0.append({'role': 'user', 'content': prompt})
        input_dict = {'inputs': [messages0], "parameters": model_kwargs}
        return json.dumps(input_dict).encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generation"]['content']


class BaseContentHandler(LLMContentHandler):
    content_type = "application/json"
    accepts = "application/json"

    def transform_input(self, prompt: str, model_kwargs: typing.Dict) -> bytes:
        input_dict = {'inputs': prompt, "parameters": model_kwargs}
        return json.dumps(input_dict).encode("utf-8")

    def transform_output(self, output: bytes) -> str:
        response_json = json.loads(output.read().decode("utf-8"))
        return response_json[0]["generation"]


class H2OSagemakerEndpoint(SagemakerEndpoint):
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""
    tokenizer: typing.Any = None

    @root_validator()
    def validate_environment(cls, values: typing.Dict) -> typing.Dict:
        """Validate that AWS credentials to and python package exists in environment."""
        try:
            import boto3

            try:
                if values["credentials_profile_name"] is not None:
                    session = boto3.Session(
                        profile_name=values["credentials_profile_name"]
                    )
                else:
                    # use default credentials
                    session = boto3.Session()

                values["client"] = session.client(
                    "sagemaker-runtime",
                    region_name=values['region_name'],
                    aws_access_key_id=values['aws_access_key_id'],
                    aws_secret_access_key=values['aws_secret_access_key'],
                )

            except Exception as e:
                raise ValueError(
                    "Could not load credentials to authenticate with AWS client. "
                    "Please check that credentials in the specified "
                    "profile name are valid."
                ) from e

        except ImportError:
            raise ImportError(
                "Could not import boto3 python package. "
                "Please install it with `pip install boto3`."
            )
        return values

    def get_token_ids(self, text: str) -> typing.List[int]:
        tokenizer = self.tokenizer
        if tokenizer is not None:
            return tokenizer.encode(text)
        else:
            return FakeTokenizer().encode(text)['input_ids']

