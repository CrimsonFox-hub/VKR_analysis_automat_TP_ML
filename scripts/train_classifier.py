import json
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoTokenizer,
    ModelSeqClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
import torch
import os
import sys

sys.path.append(str(Path(__file__).parent.parent))
from shared.config import PROCESSED_DATA_DIR, MODELS_DIR
from shared.logger import get_logger

log = get_logger(__name__)

LABEL2ID = {
    "incident": 0,
    "routine": 1
}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}

def load_data(json_path: Path) -> DatasetDict:
    with open(json_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    texts = []
    labels = []
    for rec in records:
        cat = rec.get("category")
        if cat not in LABEL2ID:
            log.warning("Неизвестная категория", category=cat, id=rec.get("id"))
            continue
        texts.append(rec["text_clean"])
        labels.append(LABEL2ID[cat])

    # Разделение на train/val
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = Dataset.from_dict({"text": train_texts, "label": train_labels})
    val_dataset = Dataset.from_dict({"text": val_texts, "label": val_labels})

    return DatasetDict({"train": train_dataset, "validation": val_dataset})

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

def main():
    # Путь к обработанным данным
    records_path = PROCESSED_DATA_DIR / "records.json"
    if not records_path.exists():
        log.error("Запустите сначала prepare_data.py.")
        return

    # Загрузка датасета
    dataset = load_data(records_path)
    log.info("Dataset loaded", train_size=len(dataset["train"]), val_size=len(dataset["validation"]))

    # Инициализация модели и токенизатора
    model_name = "DeepPavlov/rubert-base-cased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,
            max_length=256
        )

    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = ModelSeqClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL2ID),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    # Настройки обучения
    output_dir = MODELS_DIR / "classifier"
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=5,
        weight_decay=0.01,
        logging_dir=str(output_dir / "logs"),
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    # Обучение
    log.info("Starting training...")
    trainer.train()

    # Сохранение финальной модели
    final_model_path = output_dir / "final"
    trainer.save_model(str(final_model_path))
    tokenizer.save_pretrained(str(final_model_path))
    log.info("Model saved", path=str(final_model_path))

if __name__ == "__main__":
    main()