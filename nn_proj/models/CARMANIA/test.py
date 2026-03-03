import torch
from torch import nn
import numpy as np
from datasets import load_dataset, ClassLabel
from transformers import AutoTokenizer, AutoModel, Trainer, TrainingArguments, DataCollatorWithPadding
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from dataclasses import dataclass

# Configuration parameters
TASK_NAME = 'enhancers'  #  'enhancers', 'splice_sites_acceptor'
MODEL_NAME = 'MsAlEhR/carmania-160k-seqlen-human'
NUM_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 0.00001
WEIGHT_DECAY = 0.01
MAX_LENGTH = 400


# A classification model with a dropout layer on top of the carmania encoder
class CarmaniaForSequenceClassification(nn.Module):
    def __init__(self, model_name: str, num_labels: int, pad_token_id: int=4 , dropout_prob: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        hidden_size = getattr(self.encoder.config, "hidden_size", getattr(self.encoder.config, "d_model", None))

        self.pad_token_id = pad_token_id
        self.dropout = nn.Dropout(dropout_prob)
        self.classifier = nn.Linear(hidden_size, num_labels)
        self.num_labels = num_labels

    def forward(self, input_ids=None, attention_mask=None,num_items_in_batch=None, labels=None, **kwargs):

        if attention_mask is None:
            attention_mask = (input_ids != self.pad_token_id).long()

        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            last_hidden = outputs.hidden_states[-1]
        else:
            last_hidden = outputs[0]

        if attention_mask is not None:
            # attention_mask: (B, L) with 1 for real tokens, 0 for pads
            mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)      # (B, L, 1)
            summed = (last_hidden * mask).sum(dim=1)                       # (B, H)
            denom = mask.sum(dim=1).clamp(min=1e-6)                        # (B, 1)
            pooled = summed / denom                                        # (B, H)
        else:
            pooled = last_hidden.mean(dim=1)                               # (B, H)

        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)

        return {"loss": loss, "logits": logits}


# Metric computation function for classification
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    accuracy = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='weighted')
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
    }

# Function to load and prepare a downstream task
def load_and_prepare_dataset(task_name: str, tokenizer, model_max_length: int = 600, test_size: float = 0.1, seed: int = 42):

    # Load the entire dataset and filter by task
    dataset_all = load_dataset('InstaDeepAI/nucleotide_transformer_downstream_tasks_revised', split='train')
    dataset = dataset_all.filter(lambda x: x['task'] == task_name)

    # Ensure labels are integer‑encoded
    if not isinstance(dataset.features['label'], ClassLabel):
        dataset = dataset.class_encode_column('label')

    # Create label mappings
    label_feature = dataset.features['label']
    id2label = {i: name for i, name in enumerate(label_feature.names)}
    label2id = {name: i for i, name in id2label.items()}

    # Tokenisation function
    def tokenize_fn(examples):
        return tokenizer(examples['sequence'], truncation=True, max_length=model_max_length)

    # Tokenise and format the dataset
    tokenised_dataset = dataset.map(tokenize_fn, batched=True)
    tokenised_dataset.set_format(type='torch', columns=['input_ids', 'label'])

    # Train/validation split
    split_dataset = tokenised_dataset.train_test_split(test_size=test_size, seed=seed, stratify_by_column='label')
    return split_dataset['train'], split_dataset['test'], id2label, label2id


def main():
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, model_max_length=MAX_LENGTH)
    tokenizer.padding_side = 'right'
    # Some models need an EOS token as the pad token when doing classification
    if tokenizer.eos_token is None and tokenizer.pad_token is not None:
        tokenizer.eos_token = tokenizer.pad_token

    # Load and prepare the dataset
    train_dataset, eval_dataset, id2label, label2id = load_and_prepare_dataset(
        task_name=TASK_NAME,
        tokenizer=tokenizer,
        model_max_length=MAX_LENGTH,
        test_size=0.1,
        seed=42,
    )

    # Instantiate model
    num_labels = len(id2label)
    model = CarmaniaForSequenceClassification(
        model_name=MODEL_NAME,
        num_labels=num_labels,
        dropout_prob=0.1,
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='carmania_finetune_results',
        eval_strategy='epoch',
        save_strategy='epoch',
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        logging_dir='carmania_logs',
        logging_steps=50,
        report_to="none",
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
    )

    # Define data collator for dynamic padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    # Start training
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print('Evaluation metrics:', eval_results)


if __name__ == "__main__":
    main()