from datetime import datetime, timedelta
from typing import Any, Optional

from fastapi.security import OAuth2PasswordBearer
from jose import jwt
from passlib.context import CryptContext

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

FAKE_USERS_DB = {
    "BCH": {
        "username": "BCH",
        "hashed_password": "$2b$12$YP6UgESiJ6.3c0EwnxNEnu9Ts075Jz82AcqawG7fxvFiMSUgs6cWK",
        "disabled": False,
    }
}


def authenticate_user(
    fake_db: dict[str, Any], username: str, password: str
) -> dict[str, Any] | bool:
    user = get_user(username)
    print(user)
    print(password)
    if not user:
        return False
    if not pwd_context.verify(password, user["hashed_password"]):
        return False
    return user


def get_user(username: str) -> Optional[dict[str, Any]]:
    if username in FAKE_USERS_DB:
        user_dict = FAKE_USERS_DB[username]
        return user_dict
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
