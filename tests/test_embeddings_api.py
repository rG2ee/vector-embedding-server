import json


def test_embeddings_api_string_input(authenticated_fastapi_client):
    response = authenticated_fastapi_client.post(
        url="/v1/embeddings",
        content=json.dumps({"model": "text-embedding-ada-002", "input": "hello world"}),
    )

    response_json = response.json()
    response_json["data"][0]["embedding"] = [
        round(value, 4) for value in response_json["data"][0]["embedding"][:3]
    ]

    expected_response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [
                    0.40520,
                    -0.8627,
                    0.7137,
                ],
                "index": 0,
            }
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 4, "total_tokens": 4},
    }
    assert expected_response == response_json


def test_embeddings_api_list_input(authenticated_fastapi_client):
    response = authenticated_fastapi_client.post(
        url="/v1/embeddings",
        content=json.dumps(
            {
                "model": "text-embedding-ada-002",
                "input": [
                    "hello world",
                    "how are you doing?",
                ],
            }
        ),
    )

    response_json = response.json()
    for i in range(2):
        response_json["data"][i]["embedding"] = [
            round(value, 4) for value in response_json["data"][i]["embedding"][:3]
        ]

    expected_response = {
        "object": "list",
        "data": [
            {
                "object": "embedding",
                "embedding": [
                    0.40520,
                    -0.8627,
                    0.7137,
                ],
                "index": 0,
            },
            {
                "object": "embedding",
                "embedding": [
                    0.4635,
                    -1.3566,
                    0.4186,
                ],
                "index": 1,
            },
        ],
        "model": "text-embedding-ada-002",
        "usage": {"prompt_tokens": 11, "total_tokens": 11},
    }
    assert expected_response == response_json


def test_embeddings_api_list_input_padding_and_dimension(authenticated_fastapi_client):
    response = authenticated_fastapi_client.post(
        url="/v1/embeddings",
        content=json.dumps(
            {
                "model": "text-embedding-ada-002",
                "input": [
                    "hello world",
                    "how are you doing?",
                ],
            }
        ),
    )
    response_json = response.json()

    for i in range(2):
        embedding = response_json["data"][i]["embedding"]
        assert len(embedding) == 1536
        assert all(x != 0 for x in embedding[:1024])
        assert all(x == 0 for x in embedding[1024:])
