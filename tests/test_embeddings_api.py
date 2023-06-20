import json


def test_embeddings_api(authenticated_fastapi_client):
    response = authenticated_fastapi_client.post(
        url="/v1/embeddings",
        data=json.dumps({"model": "text-embedding-ada-002", "input": "hello world"}),
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
