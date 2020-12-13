import config
import torch
import torch.nn as nn

from tqdm import tqdm


def loss_function(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))


def train(data_loader, model, optimizer, scheduler):
    model.train()
    device = config.DEVICE

    for _, data_set in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = data_set["ids"]
        mask = data_set["mask"]
        token_type_ids = data_set["token_type_ids"]
        targets = data_set["targets"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()
        scheduler.step()


def evaluate(data_loader, model):
    model.eval()
    device = config.DEVICE

    final_outputs = []
    final_targets = []

    for batch_index, data_set in tqdm(enumerate(data_loader), total=len(data_loader)):
        ids = data_set["ids"]
        mask = data_set["mask"]
        token_type_ids = data_set["token_type_ids"]
        targets = data_set["targets"]

        ids = ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        outputs = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        final_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        final_targets.extend(targets.cpu().detach().numpy().tolist())

    return final_outputs, final_targets
