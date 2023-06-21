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
