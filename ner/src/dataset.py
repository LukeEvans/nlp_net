import ner.src.config as config
import torch
from utils import pad


class EntityDataset:
    def __init__(self, texts, pos, tags):
        self.texts = texts
        self.pos = pos
        self.tags = tags

    @staticmethod
    def predict_dataset(texts):
        empty_pos = [[0] * len(texts)]
        empty_tags = [[0] * len(texts)]
        return EntityDataset([texts], empty_pos, empty_tags)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, item):
        max_length = config.MAX_LENGTH
        text = self.texts[item]
        pos = self.pos[item]
        tags = self.tags[item]

        ids = []
        target_pos = []
        target_tag = []

        for index, sentence in enumerate(text):
            inputs = config.TOKENIZER.encode(sentence, add_special_tokens=False)

            input_len = len(inputs)
            ids.extend(inputs)
            target_pos.extend([pos[index]] * input_len)
            target_tag.extend([tags[index]] * input_len)

        ids = ids[:max_length - 2]
        target_pos = target_pos[:max_length - 2]
        target_tag = target_tag[:max_length - 2]

        ids = [101] + ids + [102]
        target_pos = [0] + target_pos + [0]
        target_tag = [0] + target_tag + [0]

        mask = [1] * len(ids)
        token_type_ids = [0] * len(ids)

        return {
            "ids": to_tensor(ids, max_length),
            "mask": to_tensor(mask, max_length),
            "token_type_ids": to_tensor(token_type_ids, max_length),
            "target_pos": to_tensor(target_pos, max_length),
            "target_tag": to_tensor(target_tag, max_length)
        }


def prediction_dataset(texts):
    return EntityDataset(texts, [0] * len(texts), [0] * len(texts))


def to_tensor(data, max_length):
    return torch.tensor(pad(data, max_length), dtype=torch.long)
