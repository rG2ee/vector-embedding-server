import os
from datetime import datetime, timedelta
from typing import Any, Callable, Optional

from dotenv import load_dotenv
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel

load_dotenv()

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

JWT_SECRET_KEY = os.environ["JWT_SECRET_KEY"]


class User(BaseModel):
    username: str
    hashed_password: str
    disabled: bool


class Credentials(BaseModel):
    username: str
    password: str


def authenticate_user(db: dict[str, User], username: str, password: str) -> User:
    user = get_user(db, username)
    if not user:
        raise ValueError("no user found")
    if not pwd_context.verify(password, user.hashed_password):
        raise ValueError("invalid password")
    return user


def get_user(db: dict[str, User], username: str) -> Optional[User]:
    if username in db:
        user = db[username]
        return user
    return None


def create_access_token(
    data: dict[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(days=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm="HS256")
    return encoded_jwt


def get_current_user(db: dict[str, User], token: str = Depends(oauth2_scheme)) -> User:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=["HS256"])
        username: str | None = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(db, username)
    if user is None:
        raise credentials_exception
    return user


def login_user(db: dict[str, User], credentials: Credentials) -> dict[str, str]:
    try:
        user = authenticate_user(db, credentials.username, credentials.password)
    except ValueError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(minutes=15)
    access_token = create_access_token(
        data={"sub": user.username},
        expires_delta=access_token_expires,
    )
    return {"access_token": access_token}


def get_current_user_wrapper(db: dict[str, User]) -> Callable[[str], User]:
    def _get_current_user(token: str = Depends(oauth2_scheme)) -> User:
        return get_current_user(db, token)

    return _get_current_user
