#!/usr/bin/env python3
"""
train_roberta.py  —  5-fold CV, tiny-data friendly, version-agnostic Trainer usage.

CSV columns (required): id, text, labels, split
- labels: pipe-separated labels, e.g. "T1566.002|APT29" or empty
- split: one of train/val/test (case-insensitive; trimmed)

Install:
  pip install torch transformers numpy

Examples:
  # CV only
  python Data/train_roberta.py --csv Data/dataset.csv --labels labels.txt --auto_build_labels \
      --model_name roberta-base --output_dir runs/roberta --epochs 5 --fp16 --save_best

  # CV and then train on full (train+val) and evaluate 'test' with avg CV threshold
  python Data/train_roberta.py --csv Data/dataset.csv --labels labels.txt --auto_build_labels \
      --model_name roberta-base --output_dir runs/roberta --epochs 5 --fp16 --save_best \
      --train_full_after_cv
"""

import argparse
import csv
import json
import pathlib
from inspect import signature
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

# EarlyStopping is optional
try:
    from transformers import EarlyStoppingCallback  # type: ignore
    HAS_EARLY_STOP = True
except Exception:
    EarlyStoppingCallback = None  # type: ignore
    HAS_EARLY_STOP = False


# ---------------- CLI ----------------

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=pathlib.Path, help="Path to dataset.csv (with split column)")
    ap.add_argument("--labels", type=pathlib.Path, default=pathlib.Path("labels.txt"),
                    help="Path to labels.txt (one label per line)")
    ap.add_argument("--auto_build_labels", action="store_true",
                    help="Build labels.txt from CSV before training (overwrite if exists)")

    ap.add_argument("--model_name", default="roberta-base", help="HF model name or local path")
    ap.add_argument("--output_dir", default="runs/roberta", help="Root output (fold subdirs will be created here)")

    ap.add_argument("--max_len", type=int, default=256, help="Max token length")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--train_batch", type=int, default=8)
    ap.add_argument("--eval_batch", type=int, default=16)
    ap.add_argument("--warmup_ratio", type=float, default=0.06)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--fp16", action="store_true", help="Use mixed precision (AMP) if supported")
    ap.add_argument("--seed", type=int, default=42)

    # in-training eval knobs (used if supported by your transformers build)
    ap.add_argument("--eval_steps", type=int, default=200, help="Eval/save cadence when in-training eval is supported")
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--save_best", action="store_true", help="Track best validation and load it at end (if supported)")
    ap.add_argument("--patience", type=int, default=5, help="Early stopping patience (eval steps, if supported)")

    ap.add_argument("--threshold", type=float, default=0.5, help="Default sigmoid threshold (CV tunes per-fold)")
    ap.add_argument("--save_total_limit", type=int, default=None, help="Max checkpoints to keep (if supported)")

    ap.add_argument("--folds", type=int, default=5, help="Number of CV folds")
    ap.add_argument("--train_full_after_cv", action="store_true",
                    help="Retrain on all (train+val) and evaluate 'test' with average CV threshold")

    return ap.parse_args()


# -------------- Utils / Labels --------------

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def read_labels_from_csv(csv_path: pathlib.Path) -> List[str]:
    uniq = set()
    with csv_path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            lab = (r.get("labels") or "").strip()
            if not lab:
                continue
            for x in lab.split("|"):
                x = x.strip()
                if x:
                    uniq.add(x)
    tech = sorted([x for x in uniq if x.startswith("T")])
    grp  = sorted([x for x in uniq if not x.startswith("T")])
    return tech + grp


def ensure_labels_file(labels_path: pathlib.Path, csv_path: pathlib.Path, auto_build: bool) -> List[str]:
    if auto_build or not labels_path.exists():
        labels = read_labels_from_csv(csv_path)
        labels_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
        print(f"[OK] Built {labels_path} with {len(labels)} labels.")
        return labels
    labels = [l.strip() for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"[OK] Loaded {labels_path} with {len(labels)} labels.")
    return labels


def scan_split_counts(csv_path: pathlib.Path) -> Dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    uniques = set()
    with csv_path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            s = (r.get("split") or "").strip().lower()
            if s in counts:
                counts[s] += 1
            if s:
                uniques.add(s)
    print(f"[INFO] Split counts: train={counts['train']}  val={counts['val']}  test={counts['test']}")
    if uniques - set(counts.keys()):
        print(f"[WARN] Found unexpected split values: {sorted(uniques - set(counts.keys()))}")
    return counts


# ---------------- Dataset ----------------

class MultiLabelArrayDataset(Dataset):
    """Simple array-backed dataset for a fixed (texts, Y) subset."""
    def __init__(self, texts: List[str], Y: np.ndarray, tokenizer, max_len: int = 256):
        self.texts = texts
        self.Y = Y.astype(np.float32)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        y = self.Y[idx]
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_len,
            padding=False,
        )
        enc["labels"] = y
        return enc


def load_pool_and_test(csv_path: pathlib.Path, label2id: Dict[str, int]) -> Tuple[List[str], np.ndarray, List[str], np.ndarray]:
    """Return (pool_texts, pool_Y) where split in {train,val}, and (test_texts, test_Y) for split == test (or empty)."""
    pool_texts, pool_Y = [], []
    test_texts, test_Y = [], []
    with csv_path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            s = (r.get("split") or "").strip().lower()
            text = (r.get("text") or "").strip()
            labs = (r.get("labels") or "").strip()
            y = np.zeros(len(label2id), dtype=np.float32)
            if labs:
                for l in labs.split("|"):
                    l = l.strip()
                    if l in label2id:
                        y[label2id[l]] = 1.0
            if s in {"train", "val"}:
                pool_texts.append(text)
                pool_Y.append(y)
            elif s == "test":
                test_texts.append(text)
                test_Y.append(y)
    pool_Y = np.stack(pool_Y) if pool_Y else np.zeros((0, len(label2id)), dtype=np.float32)
    test_Y = np.stack(test_Y) if test_Y else np.zeros((0, len(label2id)), dtype=np.float32)
    return pool_texts, pool_Y, test_texts, test_Y


# ---------------- Metrics ----------------

def _precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-9):
    # micro
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()
    prec_micro = tp / (tp + fp + eps)
    rec_micro = tp / (tp + fn + eps)
    f1_micro = 2 * prec_micro * rec_micro / (prec_micro + rec_micro + eps)

    # macro
    per_label = []
    for j in range(y_true.shape[1]):
        yt = y_true[:, j]
        yp = y_pred[:, j]
        tp = ((yt == 1) & (yp == 1)).sum()
        fp = ((yt == 0) & (yp == 1)).sum()
        fn = ((yt == 1) & (yp == 0)).sum()
        denom = (tp + fp + fn)
        if denom == 0:
            per_label.append((0.0, 0.0, 0.0))
            continue
        p = tp / (tp + fp + eps)
        r = tp / (tp + fn + eps)
        f1 = 2 * p * r / (p + r + eps)
        per_label.append((p, r, f1))
    prec_macro = float(np.mean([x[0] for x in per_label]))
    rec_macro  = float(np.mean([x[1] for x in per_label]))
    f1_macro   = float(np.mean([x[2] for x in per_label]))

    return {
        "precision_micro": float(prec_micro),
        "recall_micro": float(rec_micro),
        "f1_micro": float(f1_micro),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
    }


def evaluate_from_logits(logits: np.ndarray, labels_np: np.ndarray, threshold: float) -> Dict[str, float]:
    probs = 1 / (1 + np.exp(-logits))
    preds = (probs >= threshold).astype(np.int32)
    return _precision_recall_f1(labels_np.astype(np.int32), preds)


def tune_threshold_on_val(logits: np.ndarray, labels_np: np.ndarray) -> Tuple[float, Dict[str, float]]:
    """Grid-search threshold in [0.1, 0.9] to maximize f1_micro."""
    probs = 1 / (1 + np.exp(-logits))
    best_t, best_f1, best_metrics = 0.5, -1.0, {}
    for t in np.linspace(0.1, 0.9, 81):
        preds = (probs >= t).astype(np.int32)
        m = _precision_recall_f1(labels_np.astype(np.int32), preds)
        if m["f1_micro"] > best_f1:
            best_f1 = m["f1_micro"]
            best_t = float(t)
            best_metrics = m
    return best_t, best_metrics


# ---------------- TrainingArgs + callbacks ----------------

def build_training_args(args, do_eval: bool):
    """
    Build TrainingArguments using only kwargs supported by the installed transformers version.
    Ensures save/eval strategies match when in-training eval is supported.
    Returns (training_args, supports_in_training_eval: bool).
    """
    sig = signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())

    supports_eval_strategy = "evaluation_strategy" in allowed
    supports_save_strategy = "save_strategy" in allowed
    supports_eval_steps    = "eval_steps" in allowed
    supports_save_steps    = "save_steps" in allowed
    supports_load_best     = "load_best_model_at_end" in allowed and "metric_for_best_model" in allowed

    kw = {
        "output_dir": args.output_dir,  # replaced per-fold
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.train_batch,
        "per_device_eval_batch_size": args.eval_batch,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "fp16": args.fp16,
        "logging_steps": args.logging_steps,
        "seed": args.seed,
        "report_to": [],
        "save_total_limit": args.save_total_limit,
    }

    supports_in_training_eval = False
    if do_eval and supports_eval_strategy and supports_save_strategy:
        kw["evaluation_strategy"] = "steps"
        kw["save_strategy"] = "steps"
        if supports_eval_steps: kw["eval_steps"] = args.eval_steps
        if supports_save_steps: kw["save_steps"] = args.eval_steps
        if args.save_best and supports_load_best:
            kw["load_best_model_at_end"] = True
            kw["metric_for_best_model"] = "f1_micro"
            if "greater_is_better" in allowed:
                kw["greater_is_better"] = True
        supports_in_training_eval = True
    else:
        if supports_save_strategy:
            kw["save_strategy"] = "no"
        if do_eval and not supports_eval_strategy:
            print("[WARN] transformers build lacks in-training eval; will evaluate after training.")

    kw = {k: v for k, v in kw.items() if k in allowed}
    return TrainingArguments(**kw), supports_in_training_eval


def maybe_make_callbacks(do_eval_in_training: bool, save_best: bool, patience: int):
    if not (do_eval_in_training and save_best and HAS_EARLY_STOP):
        return None
    try:
        return [EarlyStoppingCallback(early_stopping_patience=patience, early_stopping_threshold=0.0)]
    except Exception:
        return None


# ---------------- Class-weighted loss ----------------

def compute_pos_weight(Y: np.ndarray) -> torch.Tensor:
    # Y shape: [N, C]
    counts_pos = Y.sum(axis=0).astype(np.float32)  # [C]
    counts_all = Y.shape[0]
    counts_neg = counts_all - counts_pos
    counts_neg[counts_neg == 0] = 1.0
    pos = np.maximum(counts_pos, 1.0)
    w = counts_neg / pos
    return torch.tensor(w, dtype=torch.float)


import torch.nn as nn
class WeightedTrainer(Trainer):
    def __init__(self, *args, pos_weight: Optional[torch.Tensor] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._pos_weight = pos_weight

    # IMPORTANT FIX: accept **kwargs (e.g., num_items_in_batch from newer transformers)
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        if self._pos_weight is None:
            loss_fct = nn.BCEWithLogitsLoss()
        else:
            loss_fct = nn.BCEWithLogitsLoss(pos_weight=self._pos_weight.to(logits.device))
        loss = loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss


# ---------------- CV helper ----------------

def kfold_indices(n: int, k: int, seed: int):
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    folds = np.array_split(idx, k)
    for i in range(k):
        val_idx = folds[i]
        train_idx = np.concatenate([folds[j] for j in range(k) if j != i]) if k > 1 else idx
        yield train_idx, val_idx


# ---------------- Main ----------------

def main():
    args = parse_args()
    set_seed(args.seed)

    counts = scan_split_counts(args.csv)
    labels = ensure_labels_file(args.labels, args.csv, args.auto_build_labels)
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    num_labels = len(labels)
    print(f"[INFO] num_labels={num_labels}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

    # Load pool (train+val) and test
    pool_texts, pool_Y, test_texts, test_Y = load_pool_and_test(args.csv, label2id)
    print(f"[INFO] Pool size (train+val rows): {len(pool_texts)} | Test size: {len(test_texts)}")

    if len(pool_texts) < args.folds:
        raise SystemExit(f"Not enough pool rows ({len(pool_texts)}) for {args.folds}-fold CV.")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    cv_metrics = []
    thresholds = []

    for fold, (tr_idx, va_idx) in enumerate(kfold_indices(len(pool_texts), args.folds, args.seed), start=1):
        print(f"\n===== Fold {fold}/{args.folds} =====")
        fold_dir = pathlib.Path(args.output_dir) / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        config = AutoConfig.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            id2label=id2label,
            label2id=label2id,
        )
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

        train_ds = MultiLabelArrayDataset([pool_texts[i] for i in tr_idx], pool_Y[tr_idx], tokenizer, max_len=args.max_len)
        val_ds   = MultiLabelArrayDataset([pool_texts[i] for i in va_idx], pool_Y[va_idx], tokenizer, max_len=args.max_len)

        pos_weight = compute_pos_weight(pool_Y[tr_idx])

        args_per_fold = argparse.Namespace(**vars(args))
        args_per_fold.output_dir = str(fold_dir)
        training_args, supports_in_training_eval = build_training_args(args_per_fold, do_eval=True)
        callbacks = maybe_make_callbacks(supports_in_training_eval, args.save_best, args.patience)

        trainer = WeightedTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            tokenizer=tokenizer,          # ok; transformers warns about v5 rename
            data_collator=data_collator,
            compute_metrics=None,         # we’ll evaluate manually and tune threshold
            callbacks=callbacks,
            pos_weight=pos_weight,
        )

        print("[INFO] Training…")
        trainer.train()

        print("[INFO] Validating & tuning threshold…")
        logits, labels_np, _ = trainer.predict(val_ds)
        best_t, best_metrics = tune_threshold_on_val(logits, labels_np)
        thresholds.append(best_t)
        cv_metrics.append(best_metrics)

        (fold_dir / "best_threshold.txt").write_text(f"{best_t}", encoding="utf-8")
        (fold_dir / "val_metrics.json").write_text(json.dumps(best_metrics, indent=2), encoding="utf-8")
        (fold_dir / "label2id.json").write_text(json.dumps({k: int(v) for k, v in label2id.items()}, indent=2), encoding="utf-8")
        (fold_dir / "id2label.json").write_text(json.dumps({int(k): v for k, v in id2label.items()}, indent=2), encoding="utf-8")
        (fold_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

        print(f"[OK] Fold {fold} best_t={best_t:.2f}  f1_micro={best_metrics['f1_micro']:.4f}")

    # Aggregate CV metrics
    def agg(metric_name: str):
        vals = [m[metric_name] for m in cv_metrics]
        return float(np.mean(vals)), float(np.std(vals))

    cv_summary = {
        "folds": args.folds,
        "avg_threshold": float(np.mean(thresholds)) if thresholds else args.threshold,
        "metrics_mean_std": {
            k: {"mean": agg(k)[0], "std": agg(k)[1]}
            for k in ["precision_micro", "recall_micro", "f1_micro", "precision_macro", "recall_macro", "f1_macro"]
        }
    }
    root = pathlib.Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "cv_metrics.json").write_text(json.dumps(cv_summary, indent=2), encoding="utf-8")
    print("\n===== CV summary =====")
    print(json.dumps(cv_summary, indent=2))

    # Optional: train on full pool and evaluate test
    if args.train_full_after_cv and len(test_texts) > 0:
        print("\n===== Final train on full pool + test evaluation =====")
        final_dir = root / "final_full_pool"
        final_dir.mkdir(parents=True, exist_ok=True)

        config = AutoConfig.from_pretrained(
            args.model_name,
            num_labels=num_labels,
            problem_type="multi_label_classification",
            id2label=id2label,
            label2id=label2id,
        )
        model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

        full_train_ds = MultiLabelArrayDataset(pool_texts, pool_Y, tokenizer, max_len=args.max_len)
        test_ds = MultiLabelArrayDataset(test_texts, test_Y, tokenizer, max_len=args.max_len)

        pos_weight_full = compute_pos_weight(pool_Y)

        args_full = argparse.Namespace(**vars(args))
        args_full.output_dir = str(final_dir)
        training_args_full, _ = build_training_args(args_full, do_eval=False)

        trainer_full = WeightedTrainer(
            model=model,
            args=training_args_full,
            train_dataset=full_train_ds,
            eval_dataset=None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=None,
            callbacks=None,
            pos_weight=pos_weight_full,
        )

        print("[INFO] Training on full pool…")
        trainer_full.train()

        print("[INFO] Evaluating on test with avg CV threshold…")
        logits, labels_np, _ = trainer_full.predict(test_ds)
        avg_t = float(np.mean(thresholds)) if thresholds else args.threshold
        test_metrics = evaluate_from_logits(logits, labels_np, avg_t)

        (final_dir / "avg_threshold.txt").write_text(f"{avg_t}", encoding="utf-8")
        (final_dir / "test_metrics.json").write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")
        (final_dir / "label2id.json").write_text(json.dumps({k: int(v) for k, v in label2id.items()}, indent=2), encoding="utf-8")
        (final_dir / "id2label.json").write_text(json.dumps({int(k): v for k, v in id2label.items()}, indent=2), encoding="utf-8")
        (final_dir / "labels.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
        print(f"[OK] Test f1_micro={test_metrics['f1_micro']:.4f}  (threshold={avg_t:.2f})")

    print(f"\n[OK] Finished. Artifacts in: {args.output_dir}")


if __name__ == "__main__":
    main()
