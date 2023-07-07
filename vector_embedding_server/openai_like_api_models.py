from enum import Enum

from pydantic import BaseModel


class EmbeddingData(BaseModel):
    object: str
    embedding: list[float]
    index: int


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str
    data: list[EmbeddingData]
    model: str
    usage: Usage


class EmbeddingInput(BaseModel):
    model: str
    input: str | list[str]


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
    usage: Usage
