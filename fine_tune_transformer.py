import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

DATASET_NAME = "deepset/prompt-injections"
MODEL_NAME = "distilbert-base-uncased" 
OUTPUT_DIR = "security/transformer_model"

def normalize_label(raw):
    if isinstance(raw, str):
        return 1 if raw.strip().upper() in ("INJECTION", "1", "TRUE") else 0
    return int(raw)

def main():
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
    )
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}" + (" (GPU detected - Kaggle T4 or similar)" if device == "cuda" else " (no GPU - this will be slow, consider Kaggle)"))

    train_ds = load_dataset(DATASET_NAME, split="train")
    test_ds = load_dataset(DATASET_NAME, split="test")

    train_ds = train_ds.map(lambda x: {"labels": normalize_label(x["label"])})
    test_ds = test_ds.map(lambda x: {"labels": normalize_label(x["label"])})

    print(f"Train: {len(train_ds)} examples | Test: {len(test_ds)} examples")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    train_ds = train_ds.map(tokenize, batched=True)
    test_ds = test_ds.map(tokenize, batched=True)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2, id2label={0: "legit", 1: "injection"}, label2id={"legit": 0, "injection": 1}
    )

    training_args = TrainingArguments(
        output_dir="./training_checkpoints",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=20,
        report_to="none", 
    )

    def compute_metrics(eval_pred):
        from sklearn.metrics import precision_recall_fscore_support, accuracy_score
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()

    predictions = trainer.predict(test_ds)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    print("\n" + "=" * 60)
    print(f"FINAL RESULTS on {DATASET_NAME} [test split]")
    print("=" * 60)
    print(classification_report(labels, preds, target_names=["legit", "injection"]))
    print("Confusion matrix:")
    print(confusion_matrix(labels, preds))

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"\nModel + tokenizer saved to {OUTPUT_DIR}/")
    print("On Kaggle: download this folder from the notebook's Output panel,")
    print("then copy it into your project at security/transformer_model/")


if __name__ == "__main__":
    main()
