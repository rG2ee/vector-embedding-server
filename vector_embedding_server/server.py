from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from vector_embedding_server.auth import (
    Credentials,
    User,
    get_current_user_wrapper,
    login_user,
)
from vector_embedding_server.e5_large_v2 import predict as e5_large_v2_predict
from vector_embedding_server.openai_like_api_models import (
    EmbeddingData,
    EmbeddingInput,
    EmbeddingResponse,
    Usage,
)

FAKE_USERS_DB = {
    "BCH": User(
        username="BCH",
        hashed_password="$2b$12$YP6UgESiJ6.3c0EwnxNEnu9Ts075Jz82AcqawG7fxvFiMSUgs6cWK",
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


@app.get("/docs", response_class=HTMLResponse)
async def docs(request: Request):  # type: ignore
    return templates.TemplateResponse(
        "stoplight-element-api-doc.html", {"request": request}
    )
