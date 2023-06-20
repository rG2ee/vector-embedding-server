from vector_embedding_server.e5_large_v2 import predict


def test_predict():
    input_text = "query: how much protein should a female eat"

    embedding = predict(input_text=input_text)
    rounded_embedding = [round(value, 4) for value in embedding]
    assert rounded_embedding[:3] == [0.134, -1.4772, 0.5544]
