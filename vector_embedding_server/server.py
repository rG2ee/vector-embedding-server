import os
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Query, Request, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime, timedelta
from vector_embedding_server.auth import authenticate_user, create_access_token, FAKE_USERS_DB, OAuth2PasswordRequestForm, get_user

from vector_embedding_server.openai_like_api_models import (
    EmbeddingResponse,
    EmbeddingInput,
    ModelName,
    EmbeddingData,
    Usage,
)
from vector_embedding_server import e5_large_v2
from pydantic import BaseModel
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt


BASE_DIR = Path(__file__).resolve().parent

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

app = FastAPI(title="Vector Embedding Server", docs_url=None)


class Credentials(BaseModel):

    username: str
    password: str


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, "SECRET_KEY", algorithms=["HS256"])
        print(payload)
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    print("username:udirtae", username)
    user = get_user(username)  # Hier sollten Sie die Funktion zur Abrufung des Benutzers aus Ihrer Benutzerdatenbank aufrufen
    if user is None:
        raise credentials_exception
    return user

@app.post("/token")
def login(credentials: Credentials):
    user = authenticate_user(FAKE_USERS_DB, credentials.username, credentials.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=15)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    return {"access_token": access_token}


@app.post("/v1/embeddings", response_model=EmbeddingResponse)
async def create_embedding(embedding_input: EmbeddingInput, current_user: str = Depends(get_current_user)):
    if embedding_input.model == ModelName.e5_large_v2:
        embedding, prompt_tokens = e5_large_v2.predict(embedding_input.input)
    else:
        raise NotImplemented

    embedding_data = EmbeddingData(
        object="embedding",
        embedding=embedding,
        index=0,
    )

    usage = Usage(prompt_tokens=prompt_tokens, total_tokens=prompt_tokens)

    embedding_response = EmbeddingResponse(
        model=embedding_input.model.name,
        object="list",
        data=[embedding_data],
        usage=usage,
    )
    return embedding_response


@app.get("/docs", response_class=HTMLResponse)
async def docs(request: Request):  # type: ignore
    return templates.TemplateResponse(
        "stoplight-element-api-doc.html", {"request": request}
    )
