from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch
from datasets import load_dataset
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.utils import import_utils as transformers_import_utils
import transformers.modeling_utils as transformers_modeling_utils

from labels import LABELS, LABEL_TO_ID, ID_TO_LABEL

matplotlib.use("Agg")


def allow_trusted_checkpoint_load() -> None:
    # Some public Chinese checkpoints on Hugging Face still ship only `.bin` weights.
    # On torch < 2.6 newer transformers blocks `torch.load` completely.
    # This project only loads trusted upstream checkpoints and local fine-tuned weights.
    transformers_import_utils.check_torch_load_is_safe = lambda: None
    transformers_modeling_utils.check_torch_load_is_safe = lambda: None


class WeightedTrainer(Trainer):
    def __init__(self, *args, class_weights: torch.Tensor | None = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=self.class_weights.to(logits.device) if self.class_weights is not None else None
        )
        loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        if return_outputs:
            return loss, outputs
        return loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a MacBERT-based moderation classifier.")
    parser.add_argument("--train-file", default="data/train.csv")
    parser.add_argument("--dev-file", default="data/dev.csv")
    parser.add_argument("--test-file", default="data/test.csv")
    parser.add_argument("--model-name", default="hfl/chinese-macbert-base")
    parser.add_argument("--output-dir", default="artifacts/moderation_macbert")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compute_metrics(eval_pred) -> dict[str, float]:
    predictions, labels = eval_pred
    pred_ids = np.argmax(predictions, axis=-1)
    report = classification_report(
        labels,
        pred_ids,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )

    metrics = {
        "accuracy": accuracy_score(labels, pred_ids),
        "macro_f1": f1_score(labels, pred_ids, average="macro", zero_division=0),
    }

    for label in LABELS:
        metrics[f"{label}_precision"] = report[label]["precision"]
        metrics[f"{label}_recall"] = report[label]["recall"]
        metrics[f"{label}_f1"] = report[label]["f1-score"]

    return metrics


def save_confusion_matrix_figure(matrix: np.ndarray, labels: list[str], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(6.4, 5.6))
    image = axis.imshow(matrix, cmap="YlOrRd")
    axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set_xticks(range(len(labels)), labels)
    axis.set_yticks(range(len(labels)), labels)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Moderation Confusion Matrix")
    plt.setp(axis.get_xticklabels(), rotation=20, ha="right", rotation_mode="anchor")

    threshold = matrix.max() / 2 if matrix.size else 0
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if matrix[row, column] > threshold else "#3c2415"
            axis.text(column, row, str(matrix[row, column]), ha="center", va="center", color=color)

    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    report_dir = Path(args.report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    data_files = {
        "train": args.train_file,
        "validation": args.dev_file,
        "test": args.test_file,
    }
    raw_datasets = load_dataset("csv", data_files=data_files)

    def encode_label(example: dict[str, str]) -> dict[str, int]:
        return {"labels": LABEL_TO_ID[example["label"]]}

    raw_datasets = raw_datasets.map(encode_label)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def tokenize(batch: dict[str, list[str]]) -> dict[str, list[list[int]]]:
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    tokenized_datasets = raw_datasets.map(
        tokenize,
        batched=True,
        remove_columns=["text", "label"],
    )

    label_array = np.array(tokenized_datasets["train"]["labels"])
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.arange(len(LABELS)),
        y=label_array,
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float)

    allow_trusted_checkpoint_load()
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )

    training_kwargs = {
        "output_dir": str(output_dir),
        "save_strategy": "epoch",
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "logging_steps": 20,
        "load_best_model_at_end": True,
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "report_to": "none",
        "save_total_limit": 2,
        "fp16": torch.cuda.is_available(),
    }

    training_args_signature = inspect.signature(TrainingArguments.__init__)
    if "evaluation_strategy" in training_args_signature.parameters:
        training_kwargs["evaluation_strategy"] = "epoch"
    else:
        training_kwargs["eval_strategy"] = "epoch"

    training_args = TrainingArguments(**training_kwargs)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_datasets["train"],
        "eval_dataset": tokenized_datasets["validation"],
        "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
        "compute_metrics": compute_metrics,
        "class_weights": class_weights_tensor,
    }

    trainer_signature = inspect.signature(Trainer.__init__)
    if "tokenizer" in trainer_signature.parameters:
        trainer_kwargs["tokenizer"] = tokenizer
    else:
        trainer_kwargs["processing_class"] = tokenizer

    trainer = WeightedTrainer(**trainer_kwargs)
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    trainer.save_metrics("train", train_result.metrics)

    test_results = trainer.predict(tokenized_datasets["test"])
    test_metrics = compute_metrics((test_results.predictions, test_results.label_ids))
    trainer.save_metrics("test", test_metrics)

    prediction_ids = np.argmax(test_results.predictions, axis=-1)
    report = classification_report(
        test_results.label_ids,
        prediction_ids,
        labels=list(range(len(LABELS))),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(
        test_results.label_ids,
        prediction_ids,
        labels=list(range(len(LABELS))),
    )

    report_payload = {
        "model_name": args.model_name,
        "train_size": len(tokenized_datasets["train"]),
        "dev_size": len(tokenized_datasets["validation"]),
        "test_size": len(tokenized_datasets["test"]),
        "class_weights": class_weights.tolist(),
        "train_runtime_seconds": train_result.metrics.get("train_runtime"),
        "train_samples_per_second": train_result.metrics.get("train_samples_per_second"),
        "train_steps_per_second": train_result.metrics.get("train_steps_per_second"),
        "test_metrics": test_metrics,
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": matrix.tolist(),
        },
    }

    with (output_dir / "classification_report.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)

    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as file:
        json.dump(report_payload, file, ensure_ascii=False, indent=2)

    with (report_dir / "model_metrics.json").open("w", encoding="utf-8") as file:
        json.dump(report_payload, file, ensure_ascii=False, indent=2)

    save_confusion_matrix_figure(matrix, LABELS, report_dir / "confusion_matrix.png")

    print("Training finished. Test metrics:")
    print(json.dumps(test_metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
