from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from passlib.context import CryptContext
from pydantic import BaseModel

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


class User(BaseModel):
    username: str
    hashed_password: str
    disabled: bool


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
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(
        to_encode, "SECRET_KEY", algorithm="HS256"
    )  # Ersetzen Sie SECRET_KEY durch Ihren geheimen Schlüssel
    return encoded_jwt
