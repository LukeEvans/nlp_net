import transformers
import torch

DEVICE = torch.device("cpu")

EPOCHS = 1

BERT_PATH = "/Users/lukeevans/nlp/nlp/bert"
MODEL_PATH = "/Users/lukeevans/nlp/nlp/sentiment/model.bin"

TRAIN_SIZE = 0.1
TRAINING_FILE = "/Users/lukeevans/nlp/nlp/sentiment/input/imdb.csv"
TRAIN_BATCH_SIZE = 8
LEARNING_RATE = 3e-5

VALID_BATCH_SIZE = 4
VALID_SIZE = TRAIN_SIZE

MAX_LENGTH = 64
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
