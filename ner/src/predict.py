import ner.src.config as config

import joblib
import torch
from ner.src.dataset import EntityDataset
from collections import OrderedDict
from ner.src.model import EntityModel
from os import path


def filter_non_flags(data, ignore_items=None):
    if ignore_items is None:
        ignore_items = ["O"]
    return {key: value for (key, value) in data.items() if value["tag"] not in ignore_items}


class NERPredictor:
    def __init__(self):
        self.ready = False
        if path.exists(config.MODEL_PATH) and path.exists(config.META_PATH):
            meta_data = joblib.load(config.META_PATH)
            self.tokenizer = config.TOKENIZER
            self.pos_encoder = meta_data["pos_encoder"]
            self.tag_encoder = meta_data["tag_encoder"]

            self.num_pos = len(list(self.pos_encoder.classes_))
            self.num_tag = len(list(self.tag_encoder.classes_))

            self.model = EntityModel(num_pos=self.num_pos, num_tag=self.num_tag)
            self.model.load_state_dict(torch.load(config.MODEL_PATH))
            self.model.to(config.DEVICE)

            self.ready = True

        else:
            print("Unable to load model")

    def predict(self, text):
        if not self.ready:
            print("No model to predict with")
            return None

        split_text = text.split()
        predict_dataset = EntityDataset.predict_dataset(split_text)

        with torch.no_grad():
            data = predict_dataset[0]
            for k, v in data.items():
                data[k] = v.to(config.DEVICE).unsqueeze(0)

            pos, tag, _ = self.model(**data)

            collected = self.collect_encodings(text, pos, tag)
            return filter_non_flags(collected)

    def collect_encodings(self, text, pos, tag):
        tokenized_text = self.tokenizer.encode(text)
        num_tokens = len(tokenized_text)
        split_text = text.split()

        pos_items = self.pos_encoder.inverse_transform(pos.argmax(2).cpu().numpy().reshape(-1))[1:num_tokens-1]
        tag_items = self.tag_encoder.inverse_transform(tag.argmax(2).cpu().numpy().reshape(-1))[1:num_tokens-1]

        trimmed_tokens = tokenized_text[1:num_tokens-1]

        word_encodings = OrderedDict()
        for word in split_text:
            word_encodings[word] = self.tokenizer.encode(word, add_special_tokens=False)

        token_map = {}
        for index, token_id in enumerate(trimmed_tokens):
            token_map[token_id] = {
                "pos": pos_items[index] if index < len(pos_items) else "O",
                "tag": tag_items[index] if index < len(tag_items) else "O"
            }

        output = OrderedDict()
        for word, tokens in word_encodings.items():
            output[word] = {
                "pos":  token_map[tokens[0]]["pos"],
                "tag":  token_map[tokens[0]]["tag"],
            }

        return output


if __name__ == "__main__":
    predictor = NERPredictor()
    predictor.predict("abhishek is going to india")
