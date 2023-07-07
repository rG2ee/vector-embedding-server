from enum import Enum
from typing import Optional

from pydantic import BaseModel


class EmbeddingData(BaseModel):
    object: str
    embedding: list[float]
    index: int


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class CompletionUsage(Usage):
    completion_tokens: int


class EmbeddingResponse(BaseModel):
    object: str
    data: list[EmbeddingData]
    model: str
    usage: Usage


class EmbeddingInput(BaseModel):
    model: str
    input: str | list[str]

    class Config:
        schema_extra = {
            "example": {
                "model": "e5-large",
                "input": [
                    "Query1",
                    "Query2",
                ],
            },
        }


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"


class Message(BaseModel):
    role: MessageRole
    content: str


class Choice(BaseModel):
    index: int
    message: Message
    finish_reason: str


class ChatCompletionInput(BaseModel):
    model: str
    messages: list[Message]

    temperature: Optional[int]
    top_p: Optional[int]
    n: Optional[int]
    stream: Optional[bool]
    stop: Optional[str | list[str]]
    max_tokens: int = 2048

    class Config:
        schema_extra = {
            "example": {
                "model": "mymodel",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": "Hello!"},
                ],
            },
        }


class ChatCompletionResponse(BaseModel):
    id: str
    object: str
    created: int

    choices: list[Choice]
    usage: CompletionUsage
