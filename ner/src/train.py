import os

import config
import engine

import pandas as pd
import numpy as np
import joblib
import torch

from model import EntityModel
from dataset import EntityDataset
from utils import build_optimizer, build_scheduler
from sklearn import preprocessing
from sklearn import model_selection


def process_data(data_path):
    data = pd.read_csv(data_path, encoding="latin-1")
    data.loc[:, "Sentence #"] = data["Sentence #"].fillna(method="ffill")

    pos_encoder = preprocessing.LabelEncoder()
    tag_encoder = preprocessing.LabelEncoder()

    data.loc[:, "POS"] = pos_encoder.fit_transform(data["POS"])
    data.loc[:, "Tag"] = tag_encoder.fit_transform(data["Tag"])

    sentences = data.groupby("Sentence #")["Word"].apply(list).values
    pos = data.groupby("Sentence #")["POS"].apply(list).values
    tag = data.groupby("Sentence #")["Tag"].apply(list).values

    return sentences, pos, tag, pos_encoder, tag_encoder


def build_loader(sentences, pos, tag, batch_size, workers):
    data_set = EntityDataset(texts=sentences, pos=pos, tags=tag)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, num_workers=workers)
    return data_loader


def run():
    sentences, pos, tag, pos_encoder, tag_encoder = process_data(config.TRAINING_FILE)

    meta_data = {
        "pos_encoder": pos_encoder,
        "tag_encoder": tag_encoder
    }

    joblib.dump(meta_data, config.META_PATH)

    (
        train_sentences,
        test_sentences,
        train_pos,
        test_pos,
        train_tag,
        test_tag
    ) = model_selection.train_test_split(sentences, pos, tag,
                                         random_state=42,
                                         train_size=config.TRAIN_SIZE,
                                         test_size=config.VALID_SIZE)

    train_data_loader = build_loader(train_sentences, train_pos, train_tag, config.TRAIN_BATCH_SIZE, 4)
    valid_data_loader = build_loader(test_sentences, test_pos, test_tag, config.VALID_BATCH_SIZE, 1)

    num_pos = len(list(pos_encoder.classes_))
    num_tag = len(list(tag_encoder.classes_))

    model = EntityModel(num_pos, num_tag)
    model.to(config.DEVICE)
    optimizer = build_optimizer(model, config.LEARNING_RATE)
    scheduler = build_scheduler(optimizer, train_sentences, config.TRAIN_BATCH_SIZE, config.EPOCHS)

    best_loss = np.inf
    for epoch in range(config.EPOCHS):
        _ = engine.train(train_data_loader, model, optimizer, scheduler)
        test_loss = engine.evaluate(valid_data_loader, model)

        if test_loss < best_loss:
            print(f"Loss improved from [{best_loss}] to [{test_loss}]. Saving model..")
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_loss = test_loss
        print(f"Finished Epoch: [{epoch}] - Test Loss: [{test_loss}] - Best Loss: [{best_loss}]")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    run()
