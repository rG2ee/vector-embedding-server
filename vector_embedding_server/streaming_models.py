from typing import Optional

from pydantic import BaseModel

from .openai_like_api_models import CompletionUsage, MessageRole


class StreamingMessage(BaseModel):
    role: Optional[MessageRole]
    content: str


class StreamingChoice(BaseModel):
    index: int
    message: StreamingMessage
    finish_reason: Optional[str]
    delta: StreamingMessage


class ChatCompletionStreamingResponse(BaseModel):
    id: str
    object: str
    created: int

    choices: list[StreamingChoice]
    usage: Optional[CompletionUsage]
