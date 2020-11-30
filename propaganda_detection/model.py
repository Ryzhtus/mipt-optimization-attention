from transformers import BertPreTrainedModel, BertModel
from torch import nn
import torch
from torch.nn import CrossEntropyLoss


class BertPropaganda(nn.Module):
    def __init__(self, num_labels):
        super(BertPropaganda, self).__init__()
        self.num_labels = num_labels
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropouts = nn.ModuleList([nn.Dropout(0.5) for _ in range(5)])
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None):

        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, position_ids=position_ids,
                            head_mask=head_mask, inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        for i, dropout in enumerate(self.dropouts):
            if i == 0:
                logits = self.classifier(dropout(sequence_output))
            else:
                logits += self.classifier(dropout(sequence_output))
        logits = logits / len(self.dropouts)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs