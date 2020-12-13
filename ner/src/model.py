import ner.src.config as config
import torch
import torch.nn as nn
import transformers


def loss_function(output, target, mask, num_labels):
    loss_fn = nn.CrossEntropyLoss()
    active_loss = mask.view(-1) == 1
    active_logits = output.view(-1, num_labels)
    active_labels = torch.where(
        active_loss,
        target.view(-1),
        torch.tensor(loss_fn.ignore_index).type_as(target)
    )

    return loss_fn(active_logits, active_labels)


class EntityModel(nn.Module):
    def __init__(self, num_pos, num_tag):
        super(EntityModel, self).__init__()
        self.num_tag = num_tag
        self.num_pos = num_pos
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop_1 = nn.Dropout(0.3)
        self.bert_drop_2 = nn.Dropout(0.3)
        self.out_tag = nn.Linear(768, self.num_tag)
        self.out_pos = nn.Linear(768, self.num_pos)

    def forward(self, ids, mask, token_type_ids, target_pos, target_tag):
        o1, _ = self.bert(ids, attention_mask=mask, token_type_ids=token_type_ids)

        bo_pos = self.bert_drop_2(o1)
        bo_tag = self.bert_drop_1(o1)

        pos = self.out_pos(bo_pos)
        tag = self.out_tag(bo_tag)

        pos_loss = loss_function(pos, target_pos, mask, self.num_pos)
        tag_loss = loss_function(tag, target_tag, mask, self.num_tag)

        loss = (pos_loss + tag_loss) / 2

        return pos, tag, loss
