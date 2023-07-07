import json
import os
from pathlib import Path
from typing import Iterator, cast

import openai
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from vector_embedding_server.auth import (
    Credentials,
    User,
    get_current_user_wrapper,
    login_user,
)
from vector_embedding_server.e5_large_v2 import predict as e5_large_v2_predict
from vector_embedding_server.openai_like_api_models import (
    ChatCompletionInput,
    ChatCompletionResponse,
    EmbeddingData,
    EmbeddingInput,
    EmbeddingResponse,
    Usage,
)
from vector_embedding_server.streaming_models import ChatCompletionStreamingResponse

load_dotenv()


USERNAME = os.environ["USERNAME"]
HASHED_PASSWORD = os.environ["HASHED_PASSWORD"]
LANGUAGE_MODEL_SERVER = os.environ["LANGUAGE_MODEL_SERVER"]

openai.api_base = f"{LANGUAGE_MODEL_SERVER}/v1"
openai.api_key = "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"


FAKE_USERS_DB = {
    USERNAME: User(
        username=USERNAME,
        hashed_password=HASHED_PASSWORD,
        disabled=False,
    )
}

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Vector Embedding Server", docs_url=None)


@app.post("/token")
def login(credentials: Credentials) -> dict[str, str]:
    return login_user(FAKE_USERS_DB, credentials)


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
    embedding_input: EmbeddingInput,
    current_user: str = Depends(get_current_user_wrapper(FAKE_USERS_DB)),
) -> EmbeddingResponse:
    request_input: list[str]
    if isinstance(embedding_input.input, str):
        request_input = [embedding_input.input]
    else:
        request_input = embedding_input.input
    embeddings, prompt_tokens = e5_large_v2_predict(request_input)

    embeddings_data = [
        EmbeddingData(
            object="embedding",
            embedding=embedding,
            index=idx,
        )
        for (idx, embedding) in enumerate(embeddings)
    ]

    usage = Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens)

    embedding_response = EmbeddingResponse(
        model="text-embedding-ada-002",
        object="list",
        data=embeddings_data,
        usage=usage,
    )
    return embedding_response


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion_proxy(
    chat_completion_input: ChatCompletionInput,
    current_user: str = Depends(get_current_user_wrapper(FAKE_USERS_DB)),
) -> ChatCompletionResponse:
    response = openai.ChatCompletion.create(  # type: ignore
        **json.loads(chat_completion_input.json())
    )
    if not chat_completion_input.stream:
        return ChatCompletionResponse(**response)

    def event_stream() -> Iterator[bytes]:
        for chunk in response:
            resp = ChatCompletionStreamingResponse(**chunk)
            if resp.choices[0].finish_reason is None:
                yield ("data: " + resp.json() + "\r\n\r\n").encode("utf-8")
            else:
                yield ("data: " + resp.json() + "\r\n\r\ndata: [DONE]\r\n\r\n").encode(
                    "utf-8"
                )

    return cast(
        ChatCompletionResponse,
        StreamingResponse(event_stream(), media_type="text/event-stream"),
    )


@app.get("/docs", response_class=HTMLResponse)
async def docs(request: Request):  # type: ignore
    return templates.TemplateResponse(
        "stoplight-element-api-doc.html", {"request": request}
    )
