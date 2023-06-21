from vector_embedding_server.e5_large_v2 import predict


def test_predict():
    input_texts = ["query: how much protein should a female eat"]

    embeddings, token_count = predict(input_texts=input_texts)
    rounded_embedding = [round(value, 4) for value in embeddings[0]]
    assert rounded_embedding[:3] == [0.134, -1.4772, 0.5544]
    assert token_count == 11
