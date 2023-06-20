from enum import Enum
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


class ModelName(str, Enum):
    e5_large_v2 = "e5-large-v2"


class EmbeddingInput(BaseModel):
    model: ModelName
    input: str
