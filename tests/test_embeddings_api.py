import json


def test_embeddings_api(fastapi_client):
    response = fastapi_client.post(
        url="/v1/embeddings",
        data=json.dumps({"model": "e5-large-v2", "input": "hello world"}),
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
        "model": "e5_large_v2",
        "usage": {"prompt_tokens": 5, "total_tokens": 5},
    }
    assert expected_response ==  response_json
