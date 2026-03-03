import torch
from torch import nn
import numpy as np
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from dataclasses import dataclass

from transformers import PreTrainedModel, PretrainedConfig, AutoModel
import torch.nn as nn
import torch

class CarmaniaSequenceClassificationConfig(PretrainedConfig):
    model_type = "carmania-seq-cls"

    def __init__(
        self,
        encoder_name: str | None = None,
        num_labels: int = 2,
        pad_token_id: int = 4,
        dropout_prob: float = 0.1,
        id2label=None,
        label2id=None,
        cache_dir=None,
        trust_remote_code: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        # Provide sensible defaults so self.__class__() works.
        self.encoder_name = encoder_name
        self.num_labels = num_labels
        self.pad_token_id = pad_token_id
        self.dropout_prob = dropout_prob
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code

        self.id2label = id2label or {i: str(i) for i in range(num_labels)}
        self.label2id = label2id or {v: k for k, v in self.id2label.items()}


class CarmaniaForSequenceClassification(PreTrainedModel):
    config_class = CarmaniaSequenceClassificationConfig

    def __init__(self, config):
        super().__init__(config)
        if config.encoder_name is None:
            raise ValueError("config.encoder_name must be set to load the base encoder.")

        self.encoder = AutoModel.from_pretrained(
            config.encoder_name,
            trust_remote_code=config.trust_remote_code,
            cache_dir=config.cache_dir,
        )

        hidden_size = getattr(self.encoder.config, "hidden_size",
                              getattr(self.encoder.config, "d_model", None))
        if hidden_size is None:
            raise ValueError("Couldn't infer hidden size from encoder config.")

        self.dropout = nn.Dropout(config.dropout_prob)
        self.classifier = nn.Linear(hidden_size, config.num_labels)


    def forward(self, input_ids=None, attention_mask=None, num_items_in_batch=None,
                labels=None, **kwargs):

        if attention_mask is None:
            attention_mask = (input_ids != self.config.pad_token_id).long()

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden = outputs.hidden_states[-1]
        else:
            last_hidden = outputs[0]

        mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
        pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=getattr(outputs, "hidden_states", None),
            attentions=getattr(outputs, "attentions", None),
        )
    # def forward(self, input_ids=None, attention_mask=None, num_items_in_batch=None,
    #             labels=None, **kwargs):

    #     if attention_mask is None:
    #         attention_mask = (input_ids != self.config.pad_token_id).long()

    #     outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    #     if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
    #         last_hidden = outputs.hidden_states[-1]
    #     else:
    #         last_hidden = outputs[0]

    #     if attention_mask is not None:
    #         mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    #         summed = (last_hidden * mask).sum(dim=1)
    #         denom = mask.sum(dim=1).clamp(min=1e-6)
    #         pooled = summed / denom
    #     else:
    #         pooled = last_hidden.mean(dim=1)

    #     pooled = self.dropout(pooled)
    #     logits = self.classifier(pooled)

    #     loss = None
    #     if labels is not None:
    #         loss = nn.CrossEntropyLoss()(logits, labels)

    #     return {"loss": loss, "logits": logits}



# # A classification model with a dropout layer on top of the carmania encoder
# class CarmaniaForSequenceClassification(nn.Module):
#     def __init__(self, model_name: str, num_labels: int, pad_token_id: int=4 , dropout_prob: float = 0.1):
#         super().__init__()
#         self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
#         hidden_size = getattr(self.encoder.config, "hidden_size", getattr(self.encoder.config, "d_model", None))
# 
#         self.pad_token_id = pad_token_id
#         self.dropout = nn.Dropout(dropout_prob)
#         self.classifier = nn.Linear(hidden_size, num_labels)
#         self.num_labels = num_labels
# 
#     def forward(self, input_ids=None, attention_mask=None,num_items_in_batch=None, labels=None, **kwargs):
# 
#         if attention_mask is None:
#             attention_mask = (input_ids != self.pad_token_id).long()
# 
#         outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
# 
#         if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
#             last_hidden = outputs.hidden_states[-1]
#         else:
#             last_hidden = outputs[0]
# 
#         if attention_mask is not None:
#             # attention_mask: (B, L) with 1 for real tokens, 0 for pads
#             mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)      # (B, L, 1)
#             summed = (last_hidden * mask).sum(dim=1)                       # (B, H)
#             denom = mask.sum(dim=1).clamp(min=1e-6)                        # (B, 1)
#             pooled = summed / denom                                        # (B, H)
#         else:
#             pooled = last_hidden.mean(dim=1)                               # (B, H)
# 
#         pooled = self.dropout(pooled)
#         logits = self.classifier(pooled)
# 
#         loss = None
#         if labels is not None:
#             loss = nn.CrossEntropyLoss()(logits, labels)
# 
#         return {"loss": loss, "logits": logits}