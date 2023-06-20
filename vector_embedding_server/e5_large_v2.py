# copied mostly from https://huggingface.co/intfloat/e5-large-v2/blob/main/README.md

import torch.nn.functional as F
from torch import Tensor
from transformers import AutoModel, AutoTokenizer


def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def predict(input_text: str) -> tuple[list[float], int]:
    tokenizer = AutoTokenizer.from_pretrained("intfloat/e5-large-v2")
    model = AutoModel.from_pretrained("intfloat/e5-large-v2")

    # Tokenize the input texts
    batch_dict = tokenizer(
        [input_text], max_length=512, padding=True, truncation=True, return_tensors="pt"
    )

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict["attention_mask"])
    return embeddings.tolist()[0], len(tokenizer.all_special_ids)
