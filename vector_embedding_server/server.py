from datetime import timedelta
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import HTMLResponse
from fastapi.security import OAuth2PasswordBearer
from fastapi.templating import Jinja2Templates
from jose import JWTError, jwt
from pydantic import BaseModel

from vector_embedding_server.auth import (
    FAKE_USERS_DB,
    authenticate_user,
    create_access_token,
    get_user,
)
from vector_embedding_server.e5_large_v2 import predict as e5_large_v2_predict
from vector_embedding_server.openai_like_api_models import (
    EmbeddingData,
    EmbeddingInput,
    EmbeddingResponse,
    Usage,
)

BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Vector Embedding Server", docs_url=None)


class Credentials(BaseModel):
    username: str
    password: str


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_current_user(token: str = Depends(oauth2_scheme)) -> dict[str, Any]:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, "SECRET_KEY", algorithms=["HS256"])
        print(payload)
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    print("username:udirtae", username)
    user = get_user(username)
    if user is None:
        raise credentials_exception
    return user


@app.post("/token")
def login(credentials: Credentials) -> dict[str, str]:
    user = authenticate_user(FAKE_USERS_DB, credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=15)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires  # type: ignore
    )
    return {"access_token": access_token}


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(
    embedding_input: EmbeddingInput, current_user: str = Depends(get_current_user)
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
