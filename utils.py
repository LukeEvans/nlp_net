from transformers import AdamW
from transformers import get_linear_schedule_with_warmup


def pad(value, max_length, pad_length=None):
    filled = pad_length if pad_length else len(value)
    padding_length = max_length - filled
    return value + ([0] * padding_length)


def build_optimizer(model, learning_rate):
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.001,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    return AdamW(optimizer_parameters, lr=learning_rate)


def build_scheduler(optimizer, text, batch_size, epochs):
    training_steps = int(len(text) / batch_size * epochs)
    return get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=training_steps)