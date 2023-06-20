import pytest
from fastapi import Depends
from fastapi.security import HTTPBasicCredentials
from fastapi.testclient import TestClient

from vector_embedding_server.server import app


@pytest.fixture
def fastapi_client():
    client = TestClient(app)
    yield client


"""

@pytest.fixture()
def disable_authentication(monkeypatch):
    async def fake_has_access(
            credentials: HTTPBasicCredentials = Depends(security),
    ) -> None:
        pass

    app.dependency_overrides[has_access] = fake_has_access



@pytest.fixture
def fastapi_client_disabled_authentication(monkeypatch):
    client = TestClient(app)

    async def fake_has_access(
            credentials: HTTPBasicCredentials = Depends(security),
    ) -> None:
        pass

    app.dependency_overrides[has_access] = fake_has_access
    client.headers = {
        "Authorization": (
            "Bearer This is an invalid code. Use `disable_authentication`"
            " or `fastapi_client_disabled_authentication` fixture."
        )
    }
    yield client
"""
