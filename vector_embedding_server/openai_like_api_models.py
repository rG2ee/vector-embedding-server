from typing import List

from pydantic import BaseModel


class EmbeddingData(BaseModel):
    object: str
    embedding: List[float]
    index: int


class Usage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str
    data: List[EmbeddingData]
    model: str
    usage: Usage


class EmbeddingInput(BaseModel):
    model: str
    input: str
