import config
import dataset
import engine
import torch
import pandas as pd
import numpy as np
import os

from model import SentimentModel
from utils import build_optimizer
from sklearn import model_selection
from sklearn import metrics
from transformers import get_linear_schedule_with_warmup


def build_loader(data_frame, batch_size, workers):
    data_set = dataset.SentimentDataset(review=data_frame.review.values, target=data_frame.sentiment.values)
    data_loader = torch.utils.data.DataLoader(data_set, batch_size=batch_size, num_workers=workers)
    return data_set, data_loader


def run():
    data = pd.read_csv(config.TRAINING_FILE).fillna("none")
    data.sentiment = data.sentiment.apply(lambda x: 1 if x == "positive" else 0)

    train_data, eval_data = model_selection.train_test_split(data,
                                                             test_size=config.VALID_SIZE,
                                                             train_size=config.TRAIN_SIZE,
                                                             random_state=42,
                                                             stratify=data.sentiment.values)

    train_data_set, train_data_loader = build_loader(train_data, config.TRAIN_BATCH_SIZE, 4)
    eval_data_set, eval_data_loader = build_loader(eval_data, config.VALID_BATCH_SIZE, 1)

    model = SentimentModel()
    model.to(config.DEVICE)
    optimizer = build_optimizer(model, config.LEARNING_RATE)

    training_steps = int(len(train_data_set) / config.TRAIN_BATCH_SIZE * config.EPOCHS)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=training_steps
    )

    best_accuracy = 0
    for epoch in range(config.EPOCHS):
        engine.train(train_data_loader, model, optimizer, scheduler)
        outputs, targets = engine.evaluate(eval_data_loader, model)
        outputs = np.array(outputs) >= 0.5
        accuracy = metrics.accuracy_score(targets, outputs)
        print(f"\nAccuracy Score = {accuracy}")

        if accuracy > best_accuracy:
            print(f"Accuracy improved from [{best_accuracy}] to [{accuracy}]. Saving model..")
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy

        print(f"Finished epoch: [{epoch}] Accuracy: [{accuracy}] - Best Accuracy: [{best_accuracy}]")


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    run()
