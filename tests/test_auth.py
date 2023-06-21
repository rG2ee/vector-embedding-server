import pytest

from vector_embedding_server.auth import (
    User,
    authenticate_user,
    create_access_token,
    get_current_user,
    get_user,
)

FAKE_USERS_DB = {
    "BCH": User(
        username="BCH",
        hashed_password="$2b$12$YP6UgESiJ6.3c0EwnxNEnu9Ts075Jz82AcqawG7fxvFiMSUgs6cWK",
        disabled=False,
    )
}


class TestAuthenticateUser:
    def test_valid_user(self):
        username = "BCH"
        password = "dainty-dumpling-charger-unruffled-hardy"
        user = authenticate_user(FAKE_USERS_DB, username, password)
        assert user == FAKE_USERS_DB[username]

    def test_invalid_user(self):
        with pytest.raises(ValueError):
            authenticate_user(FAKE_USERS_DB, "UNKNOWN", "password")

    def test_invalid_password(self):
        with pytest.raises(ValueError):
            authenticate_user(FAKE_USERS_DB, "BCH", "wrong_password")


class TestGetUser:
    def test_existing_user(self):
        user = get_user(FAKE_USERS_DB, "BCH")
        assert user == FAKE_USERS_DB["BCH"]

    def test_non_existing_user(self):
        user = get_user(FAKE_USERS_DB, "UNKNOWN")
        assert user is None


class TestCreateAccessToken:
    def test_create_access_token(self):
        data = {"sub": "BCH"}
        access_token = create_access_token(data)
        assert isinstance(access_token, str)


class TestGetCurrentUser:
    def test_valid_token(self):
        data = {"sub": "BCH"}
        access_token = create_access_token(data)
        user = get_current_user(FAKE_USERS_DB, access_token)
        assert user == FAKE_USERS_DB["BCH"]

    def test_invalid_token(self):
        with pytest.raises(Exception):
            get_current_user(FAKE_USERS_DB, "invalid_token")
