import config
from tqdm import tqdm


def train(data_loader, model, optimizer, scheduler):
    model.train()

    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        data = data_to_device(data)

        optimizer.zero_grad()
        _, _, loss = model(**data)
        loss.backward()
        optimizer.step()
        scheduler.step()

        final_loss += loss.item()

    return final_loss / len(data_loader)


def evaluate(data_loader, model):
    model.eval()

    final_loss = 0
    for data in tqdm(data_loader, total=len(data_loader)):
        data = data_to_device(data)
        _, _, loss = model(**data)

        final_loss += loss.item()

    return final_loss / len(data_loader)


def data_to_device(data):
    for k, v in data.items():
        data[k] = v.to(config.DEVICE)

    return data
