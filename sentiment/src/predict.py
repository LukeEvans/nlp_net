import sentiment.src.config as config
import torch
from sentiment.src.encoder import encode_single
from sentiment.src.model import SentimentModel
from os import path


class SentimentPredictor:
    def __init__(self):
        self.ready = False
        if path.exists(config.MODEL_PATH):
            self.model = SentimentModel()
            self.model.load_state_dict(torch.load(config.MODEL_PATH))
            self.model.to(config.DEVICE)
            self.model.eval()
            self.ready = True

        else:
            print("Unable to load model")

    def predict(self, text):
        if not self.ready:
            print("No model to predict with")
            return None

        ids, mask, token_type_ids = encode_single(text)

        outputs = self.model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        outputs = torch.sigmoid(outputs).cpu().detach().numpy()
        return outputs[0][0]

    def predict_inverse(self, text):
        prediction = self.predict(text)

        if not prediction:
            return None

        return 1 - prediction
