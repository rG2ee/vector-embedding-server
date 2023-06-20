import json

import pytest
from fastapi.testclient import TestClient

from vector_embedding_server.server import app


@pytest.fixture
def fastapi_client():
    client = TestClient(app)
    yield client


@pytest.fixture
def authenticated_fastapi_client(fastapi_client):
    # Replace with valid credentials for your application
    credentials = {
        "username": "BCH",  # valid username
        "password": "dainty-dumpling-charger-unruffled-hardy",  # valid password
    }

    response = fastapi_client.post("/token", data=json.dumps(credentials))
    access_token = response.json()["access_token"]
    fastapi_client.headers.update({"Authorization": f"Bearer {access_token}"})
    yield fastapi_client
