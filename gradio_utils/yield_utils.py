from pydantic import BaseModel


class ReturnType(BaseModel):
    reply: str | list[str] | None
    prompt_raw: str | None = None
    actual_llm: str | None = None
    text_context_list: list[str] | None = []
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0
    time_to_first_token: float = 0.0
    vision_visible_model: str | None = None
    vision_batch_input_tokens: int = 0
    vision_batch_output_tokens: int = 0
    vision_batch_tokens_per_second: float = 0.0
    files: list[str] | None = []
    chat_history: list[dict[str, str]] | None = []
