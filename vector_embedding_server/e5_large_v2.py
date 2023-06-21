# copied mostly from https://huggingface.co/intfloat/e5-large-v2/blob/main/README.md

import numpy as np
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def predict(input_texts: list[str]) -> tuple[list[list[float]], int]:
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
    model = AutoModel.from_pretrained("intfloat/e5-large-v2")

    # Tokenize the input texts
    batch_dict = tokenizer(
        input_texts, max_length=512, padding=True, truncation=True, return_tensors="pt"
    )

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])

    target_dimension = 1536

    # Pad zeros if necessary
    current_length = embeddings.shape[1]
    if current_length < target_dimension:
        pad_width = ((0, 0), (0, target_dimension - current_length))
        padded_embeddings = np.pad(
            embeddings.detach().numpy(), pad_width, mode="constant"
        )

    return padded_embeddings.tolist(), int(batch_dict["attention_mask"].sum())
