from pathlib import Path

import numpy as np
import torch

from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.utils import compute_class_weight
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from data_split import ID2LABEL, LABEL2ID, load_and_split


ROOT_DIR = Path(__file__).parent.parent
IN_CSV = ROOT_DIR / "csv" / "epic_tragedy_sentences.csv"
OUTPUT_DIR = ROOT_DIR / "model_output"

if not OUTPUT_DIR.exists():
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)


class CustomTrainer(Trainer):
    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        if class_weights is not None:
            self.class_weights = class_weights.to(self.args.device)
        else:
            self.class_weights = None

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss_fct = torch.nn.CrossEntropyLoss(weight=self.class_weights)
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    preds = np.argmax(predictions, axis=1)

    accuracy = accuracy_score(labels, preds)
    score = f1_score(labels, preds, average="weighted", zero_division=0)

    return {
        "accuracy": accuracy,
        "f1": float(score),
    }


label2id = LABEL2ID
id2label = ID2LABEL

model_path = "pnadel/ancient-greek-bert"

tokenizer = AutoTokenizer.from_pretrained(model_path)

device = torch.device("mps") if torch.backends.mps.is_available() else "cpu"

print(f"Device: {device}")

num_epochs = 5
lr = 5e-5
batch_size = 16

train_df, eval_df, test_df = load_and_split(IN_CSV)

print(
    f"Train size = {len(train_df)}; eval size = {len(eval_df)}; test size = {len(test_df)}"
)

train_dataset = Dataset.from_pandas(train_df[["text", "label"]])
eval_dataset = Dataset.from_pandas(eval_df[["text", "label"]])
test_dataset = Dataset.from_pandas(test_df[["text", "label"]])


def tokenize_fn(batch):
    return tokenizer(
        batch["text"], padding="max_length", truncation=True, max_length=512
    )  # ty:ignore[call-non-callable]


train_dataset = train_dataset.map(tokenize_fn, batched=True)
eval_dataset = eval_dataset.map(tokenize_fn, batched=True)
test_dataset = test_dataset.map(tokenize_fn, batched=True)


class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.unique(train_dataset["label"]),
    y=np.array(train_dataset["label"]),
)

class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

config = AutoConfig.from_pretrained(
    model_path,
    attention_probs_dropout_prob=0.3,
    hidden_dropout_prob=0.3,
    id2label=id2label,
    label2id=label2id,
    num_labels=2,
)

model = AutoModelForSequenceClassification.from_pretrained(model_path, config=config)

model.to(device)

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR),
    bf16=torch.cuda.is_available(),
    eval_strategy="epoch",
    greater_is_better=True,
    learning_rate=lr,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    num_train_epochs=num_epochs,
    per_device_eval_batch_size=batch_size,
    per_device_train_batch_size=batch_size,
    save_strategy="epoch",
    save_total_limit=2,
    seed=42,
    warmup_steps=50,
    weight_decay=0.1,
)

trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    class_weights=class_weights_tensor,
    processing_class=tokenizer,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    compute_metrics=compute_metrics,
)

trainer.train()

test_results = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")

print(test_results)

trainer.save_model(
    str(
        OUTPUT_DIR
        / "grc-epic-tragedy-sentence-classification-no-nestor-no-apologoi-no-soph-electra"
    )
)
