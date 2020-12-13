import sentiment.src.config as config
import torch
from utils import pad


def encode(text):
    text = " ".join(str(text).split())

    inputs = config.TOKENIZER.encode_plus(text, None, add_special_tokens=True, max_length=config.MAX_LENGTH,
                                          pad_to_max_length=True, truncation=True)

    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    return ids, mask, token_type_ids


def encode_single(text):
    device = config.DEVICE
    ids, mask, token_type_ids = encode(text)
    max_length = config.MAX_LENGTH

    ids = torch.tensor(pad(ids, max_length), dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(pad(mask, max_length), dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(pad(token_type_ids, max_length), dtype=torch.long).unsqueeze(0)

    ids = ids.to(device, dtype=torch.long)
    mask = mask.to(device, dtype=torch.long)
    token_type_ids = token_type_ids.to(device, dtype=torch.long)

    return ids, mask, token_type_ids



