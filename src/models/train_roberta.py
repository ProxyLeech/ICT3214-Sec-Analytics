#!/usr/bin/env python3
"""
train_roberta.py — zero-CLI version with baked-in config (just run: python train_roberta.py)

Dataset CSV schema (required columns): id, text, labels, split
- labels: pipe-separated, e.g. "T1566.002|APT29" (can be empty)
- split: one of train / val / test (case-insensitive; script respects your splits)

"""

from __future__ import annotations
import csv
import json
import pathlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from inspect import signature
from typing import Optional, List, Tuple, Dict, Set
import numpy as np
from torch.utils.data import Dataset
import numpy as np
import torch
import shutil
import torch.nn.functional as F

from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from paths.paths import (
    PROJECT_ROOT, DATA_ROOT, PROCESSED_DIR, MODELS_ROOT, output_dir_for_folds, EXPERIMENTS_ROOT  ,
)
from pathlib import Path
# EarlyStopping is optional; present on most recent transformers
try:
    from transformers import EarlyStoppingCallback  # type: ignore
    HAS_EARLY_STOP = True
except Exception:
    EarlyStoppingCallback = None  # type: ignore
    HAS_EARLY_STOP = False


# =====================================================================
#                         EDIT ME
# =====================================================================

@dataclass
class Config:      
    MODEL_NAME: str = "roberta-base"  # e.g., "roberta-base", "roberta-large", "microsoft/deberta-v3-base", local path, etc.

    # --- Training hyperparams ---
    EPOCHS: int = 5            # More epochs can overfit small data; try 3–10
    LR: float = 3e-5           # Typical range: 1e-5 to 5e-5
    MAX_LEN: int = 384         # 128/256/384; longer -> slower but might help
    TRAIN_BATCH: int = 8       # If OOM, lower it (e.g., 4). If GPU RAM is big, try 16
    EVAL_BATCH: int = 16       # Can be larger than train batch
    WARMUP_RATIO: float = 0.06 # 0.0–0.1 usually fine
    WEIGHT_DECAY: float = 0.01 # 0.0–0.1; helps regularize
    FP16: bool = True          # True if you have a CUDA GPU; ignored on CPU

    # --- Reproducibility ---
    SEED: int = 42

    # --- Logging / checkpointing ---
    SAVE_BEST: bool = True         # Track best model on validation / folds
    SAVE_TOTAL_LIMIT: Optional[int] = 3  # None to keep all; small number to prune
    LOGGING_STEPS: int = 50        # Console log frequency (steps)
    EVAL_STEPS: int = 200          # If in-training eval supported; otherwise ignored
    # when you build TrainingArguments
    save_strategy="epoch",                 # or "steps"
    load_best_model_at_end=True            # if you evaluate during training
    save_total_limit=3

    # --- Metrics ---
    THRESHOLD: float = 0.5         # Sigmoid threshold for multi-label prediction

    # --- Cross-validation (uses your CSV's train+val pool; keeps test fixed) ---
    USE_KFOLD: bool = True     # <- disable k-fold when using random split
    USE_RANDOM_SPLIT: bool = True
    SPLIT_RATIOS: tuple[float, float, float] = (0.8, 0.1, 0.1)
    N_FOLDS: int = 0               # 5-fold by default
    SHUFFLE_POOL: bool = True      # Shuffle train+val pool before folding

    OUTPUT_DIR = output_dir_for_folds(N_FOLDS, model_slug="roberta_base_v1")
    CSV_PATH   = PROCESSED_DIR / "dataset.csv"
    LABELS_PATH = PROCESSED_DIR / "labels.txt"


    # ensure dirs exist
    for d in [DATA_ROOT, PROCESSED_DIR, MODELS_ROOT, OUTPUT_DIR,]:
        d.mkdir(parents=True, exist_ok=True)

CFG = Config()

# =====================================================================
#                         Helper functions
# =====================================================================

def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

def scan_split_counts(csv_path: pathlib.Path) -> Dict[str, int]:
    counts = {"train": 0, "val": 0, "test": 0}
    with csv_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = (r.get("split") or "").strip().lower()
            if s in counts:
                counts[s] += 1
    print(f"[INFO] Split counts: train={counts['train']}  val={counts['val']}  test={counts['test']}")
    return counts

def read_labels_from_csv(csv_path: pathlib.Path) -> List[str]:
    uniq = set()
    with csv_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            labs = (r.get("labels") or "").strip()
            if not labs:
                continue
            for x in labs.split("|"):
                x = x.strip()
                if x:
                    uniq.add(x)
    # keep a stable order: techniques first, then groups/others
    tech = sorted([x for x in uniq if x.startswith("T")])
    grp  = sorted([x for x in uniq if not x.startswith("T")])
    return tech + grp

def ensure_labels_file(labels_path: pathlib.Path, csv_path: pathlib.Path) -> List[str]:
    if not labels_path.exists():
        labels = read_labels_from_csv(csv_path)
        labels_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
        return labels
    return [l.strip() for l in labels_path.read_text(encoding="utf-8").splitlines() if l.strip()]


def print_run_banner(labels: List[str], cfg: Config, out_dir: pathlib.Path):
    # Console banner with folds + hyperparameters
    print(
        "[OK] Labels loaded: {:d} | Folds: {} | Model: {} | "
        "epochs={}, lr={}, max_len={}, train/eval batch={}/{} | warmup={}, weight_decay={}, fp16={}\n[OUT] {}".format(
            len(labels),
            (cfg.N_FOLDS if cfg.USE_KFOLD else "none"),
            cfg.MODEL_NAME,
            cfg.EPOCHS, cfg.LR, cfg.MAX_LEN, cfg.TRAIN_BATCH, cfg.EVAL_BATCH,
            cfg.WARMUP_RATIO, cfg.WEIGHT_DECAY, bool(cfg.FP16 and torch.cuda.is_available()),
            out_dir
        )
    )

def write_readme(out_dir: pathlib.Path, labels: List[str], cfg: Config):
    readme = (
        f"Model: {cfg.MODEL_NAME}\n"
        f"Labels: {len(labels)}\n"
        f"K-Fold: {cfg.N_FOLDS if cfg.USE_KFOLD else 'None (fixed splits)'}\n"
        f"Hyperparameters:\n"
        f"  epochs={cfg.EPOCHS}\n"
        f"  learning_rate={cfg.LR}\n"
        f"  max_len={cfg.MAX_LEN}\n"
        f"  train_batch={cfg.TRAIN_BATCH}\n"
        f"  eval_batch={cfg.EVAL_BATCH}\n"
        f"  warmup_ratio={cfg.WARMUP_RATIO}\n"
        f"  weight_decay={cfg.WEIGHT_DECAY}\n"
        f"  fp16={bool(cfg.FP16 and torch.cuda.is_available())}\n"
        f"  threshold={cfg.THRESHOLD}\n"
    )
    (out_dir / "README.txt").write_text(readme, encoding="utf-8")

def compute_output_dir(cfg: Config) -> pathlib.Path:
    """
    When k-fold is enabled, place outputs under '{N}foldruns/roberta'.
    Otherwise respect cfg.OUTPUT_DIR.
    """
    if cfg.USE_KFOLD:
        return pathlib.Path(f"{cfg.N_FOLDS}foldruns/roberta")
    return cfg.OUTPUT_DIR

def prune_experiments(exp_root: Path, keep_names=("metrics.json", "README.txt")):
    """
    In EXPERIMENTS_ROOT, remove everything except metrics.json & README.txt
    for each run directory. Also removes now-empty subfolders.
    """
    if not exp_root.exists():
        return

    # 1) Delete all files that are not in keep_names
    for p in exp_root.rglob("*"):
        try:
            if p.is_file() and p.name not in keep_names:
                p.unlink()
        except Exception as e:
            print(f"[WARN] Failed to delete {p}: {e}")

    # 2) Remove empty directories bottom-up (but keep run dirs that still
    #    contain the kept files)
    for d in sorted([d for d in exp_root.rglob("*") if d.is_dir()], key=lambda x: len(str(x)), reverse=True):
        try:
            # If directory is empty now, remove it
            next(d.iterdir())
        except StopIteration:
            try:
                d.rmdir()
            except Exception as e:
                print(f"[WARN] Failed to remove empty dir {d}: {e}")

class MultiLabelCSVDataset(Dataset):
    """
    CSV-backed multi-label dataset.
    """
    def __init__(
        self,
        csv_path,
        split: Optional[str],
        tokenizer,
        label2id: Dict[str, int],
        max_len: int = 256,
        index_subset: Optional[List[int]] = None
    ):
        self.tokenizer = tokenizer
        self.max_len = int(max_len)
        self.label2id = dict(label2id)
        self.records: List[Tuple[str, np.ndarray]] = []

        # ---- load rows and build records ----
        import csv, pathlib
        with pathlib.Path(csv_path).open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        def add_row(r):
            text = (r.get("text") or "").strip()
            labs = (r.get("labels") or "").strip()
            w = float(r.get("weight") or 1.0)    
            effective_labels = [x.strip() for x in labs.split("|") if x.strip()]

            y = np.zeros(len(self.label2id), dtype=np.float32)
            for l in effective_labels:
                if l in self.label2id:
                    y[self.label2id[l]] = 1.0
            self.records.append((text, y, w))

        if index_subset is not None:
            for i in list(index_subset):
                add_row(rows[int(i)])
        else:
            want = (split or "").strip().lower()
            for r in rows:
                s = (r.get("split") or "").strip().lower()
                if s == want:
                    add_row(r)

        if not self.records:
            raise ValueError("No records found for the specified selection.")

        self._length = len(self.records)  # cache so __len__ is bulletproof

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, idx: int):
        text, y, w = self.records[idx]
        enc = self.tokenizer(text, truncation=True, max_length=self.max_len, padding=False)
        enc["labels"] = y
        enc["sample_weight"] = np.float32(w) 
        return enc

# --- Multi-label metrics ---
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

def make_compute_metrics(threshold: float):
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        probs = 1 / (1 + np.exp(-logits))
        preds = (probs >= threshold).astype(np.int32)
        return _precision_recall_f1(labels.astype(np.int32), preds)
    return compute_metrics

# --- Version-agnostic TrainingArguments builder ---
def build_training_args(cfg: Config, do_eval_in_training: bool, out_dir: pathlib.Path) -> TrainingArguments:
    sig = signature(TrainingArguments.__init__)
    allowed = set(sig.parameters.keys())

    kw = {
        "output_dir": str(out_dir),
        "learning_rate": cfg.LR,
        "per_device_train_batch_size": cfg.TRAIN_BATCH,
        "per_device_eval_batch_size": cfg.EVAL_BATCH,
        "num_train_epochs": cfg.EPOCHS,
        "weight_decay": cfg.WEIGHT_DECAY,
        "warmup_ratio": cfg.WARMUP_RATIO,
        "fp16": bool(cfg.FP16 and torch.cuda.is_available()),
        "logging_steps": cfg.LOGGING_STEPS,
        "seed": cfg.SEED,
        "report_to": [],  # disable wandb/etc
        "save_total_limit": cfg.SAVE_TOTAL_LIMIT,
    }

    if do_eval_in_training and "evaluation_strategy" in allowed and "save_strategy" in allowed:
        kw["evaluation_strategy"] = "steps"
        kw["save_strategy"] = "steps"
        if "eval_steps" in allowed:
            kw["eval_steps"] = cfg.EVAL_STEPS
        if "save_steps" in allowed:
            kw["save_steps"] = cfg.EVAL_STEPS

        # Always set a metric name when we plan to evaluate in training
        if "metric_for_best_model" in allowed:
            kw["metric_for_best_model"] = "f1_micro"
        if cfg.SAVE_BEST and "load_best_model_at_end" in allowed:
            kw["load_best_model_at_end"] = True
            if "greater_is_better" in allowed:
                kw["greater_is_better"] = True
    else:
        # No in-training eval: avoid mismatched strategies
        if "save_strategy" in allowed:
            kw["save_strategy"] = "no"

    kw = {k: v for k, v in kw.items() if k in allowed}
    return TrainingArguments(**kw)

def maybe_early_stopping(use_eval_in_training: bool, cfg: Config, targs: TrainingArguments):
    """
    Only attach EarlyStoppingCallback if:
      - we will evaluate during training,
      - transformers provides EarlyStoppingCallback,
      - and a metric_for_best_model is defined on TrainingArguments.
    """
    if not (use_eval_in_training and cfg.SAVE_BEST and HAS_EARLY_STOP):
        return None
    metric_name = getattr(targs, "metric_for_best_model", None)
    eval_strategy = getattr(targs, "evaluation_strategy", "no")
    if not metric_name or eval_strategy == "no":
        return None
    try:
        # Patience counts eval calls, not epochs.
        return [EarlyStoppingCallback(early_stopping_patience=cfg.EVAL_STEPS,
                                      early_stopping_threshold=0.0)]
    except Exception:
        return None

def main():
    cfg = CFG
    set_seed(cfg.SEED)

    # --- Build/Load labels (once) ---
    labels = ensure_labels_file(cfg.LABELS_PATH, cfg.CSV_PATH)

    if len(labels) == 0:
        print("[ERROR] Label space is empty. Ensure ti_groups_techniques.csv maps your techniques to groups.")
        return

    # --- Static artifacts for model init (shared across runs) ---
    label2id = {l: i for i, l in enumerate(labels)}
    id2label = {i: l for l, i in label2id.items()}
    tokenizer_base = AutoTokenizer.from_pretrained(cfg.MODEL_NAME, use_fast=True)
    config_base = AutoConfig.from_pretrained(
        cfg.MODEL_NAME,
        num_labels=len(labels),
        problem_type="multi_label_classification",
        id2label=id2label,
        label2id=label2id,
    )
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer_base)

    # --- Where we keep a single best model for inference ---
    BEST_DIR = MODELS_ROOT / "best_roberta_for_predict"
    BEST_DIR.mkdir(parents=True, exist_ok=True)
    best_score = -1.0
    best_src_dir: Path | None = None

    # --- Try k = 0..10 (skip k=1) ---
    for k in range(0, 11):
        if k == 1:
            continue

        # Decide split mode & run directory
        use_random_split = (k == 0)
        use_kfold = (k >= 2)
        if use_kfold:
            cfg.USE_KFOLD = True
            cfg.USE_RANDOM_SPLIT = False
            cfg.N_FOLDS = k
        else:
            cfg.USE_KFOLD = False
            cfg.USE_RANDOM_SPLIT = True

        run_dir = output_dir_for_folds(k if use_kfold else 0, model_slug="roberta_base_v1")
        run_dir.mkdir(parents=True, exist_ok=True)
        print_run_banner(labels, cfg, run_dir)

        # Write static artifacts for this run
        (run_dir / "label2id.json").write_text(json.dumps(label2id, indent=2), encoding="utf-8")
        (run_dir / "id2label.json").write_text(json.dumps({i: l for i, l in enumerate(labels)}, indent=2), encoding="utf-8")
        (run_dir / "run_config.json").write_text(json.dumps(CFG.__dict__, indent=2, default=str), encoding="utf-8")
        write_readme(run_dir, labels, cfg)

        metrics_json: Dict[str, dict] = {}

        # Fresh base model/tokenizer per run
        tokenizer = tokenizer_base 
        base_model = AutoModelForSequenceClassification.from_pretrained(cfg.MODEL_NAME, config=config_base)

        if use_random_split:
            # ---- Random 80/10/10 split (seeded) ----
            with cfg.CSV_PATH.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

            n = len(rows)
            idx = np.arange(n)
            rng = np.random.default_rng(cfg.SEED)
            rng.shuffle(idx)

            r_train, r_val, r_test = cfg.SPLIT_RATIOS
            total = r_train + r_val + r_test
            r_train, r_val, r_test = [x / total for x in (r_train, r_val, r_test)]

            n_train = int(round(r_train * n))
            n_val   = int(round(r_val   * n))
            n_test  = max(0, n - n_train - n_val)
            if n_test == 0 and n >= 3:
                n_train = max(1, n_train - 1)
                n_test  = n - n_train - n_val

            train_idx = idx[:n_train].tolist()
            val_idx   = idx[n_train:n_train + n_val].tolist()
            test_idx  = idx[n_train + n_val:].tolist()

            print(f"[INFO][k=0] Random split sizes: train={len(train_idx)}  val={len(val_idx)}  test={len(test_idx)}")

            train_ds = MultiLabelCSVDataset(cfg.CSV_PATH, split=None, tokenizer=tokenizer,
                                            label2id=label2id, max_len=cfg.MAX_LEN, index_subset=train_idx)
            val_ds   = MultiLabelCSVDataset(cfg.CSV_PATH, split=None, tokenizer=tokenizer,
                                            label2id=label2id, max_len=cfg.MAX_LEN, index_subset=val_idx) if len(val_idx) else None

            use_eval_in_training = len(val_idx) > 0
            targs = build_training_args(cfg, do_eval_in_training=use_eval_in_training, out_dir=run_dir)
            callbacks = maybe_early_stopping(use_eval_in_training, cfg, targs) or []

            class WeightedBCETrainer(Trainer):
                def compute_loss(
                    self,
                    model,
                    inputs,
                    return_outputs: bool = False,
                    num_items_in_batch: Optional[int] = None,  # <-- accept the new kwarg
                ):
                    labels = inputs.pop("labels")
                    weights = inputs.pop("sample_weight", None)

                    # forward
                    outputs = model(**inputs)
                    logits = outputs.logits

                    # ensure float dtype
                    if not torch.is_floating_point(labels):
                        labels = labels.float()
                    loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
                    loss = loss.mean(dim=1)  # per-example

                    if weights is not None:
                        if not torch.is_floating_point(weights):
                            weights = weights.float()
                        # match device/shape
                        weights = weights.to(loss.device)
                        loss = loss * weights

                    loss = loss.mean()
                    return (loss, outputs) if return_outputs else loss
            trainer = WeightedBCETrainer(
            model=base_model,
            args=targs,
            train_dataset=train_ds,
            eval_dataset=val_ds if use_eval_in_training else None,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=make_compute_metrics(cfg.THRESHOLD) if use_eval_in_training else None,
            callbacks=callbacks,

        )

            print("[INFO] Training…")
            trainer.train()
            trainer.save_model(str(run_dir))
            tokenizer.save_pretrained(str(run_dir))

            val_f1 = -1.0
            if val_ds is not None:
                print("[INFO] Evaluating on val…")
                logits, labels_np, _ = trainer.predict(val_ds)
                probs = 1 / (1 + np.exp(-logits))
                preds = (probs >= cfg.THRESHOLD).astype(np.int32)
                mval = _precision_recall_f1(labels_np.astype(np.int32), preds)
                metrics_json["val"] = mval
                val_f1 = float(mval.get("f1_micro", -1.0))

            if len(test_idx) > 0:
                print("[INFO] Evaluating on test…")
                test_ds = MultiLabelCSVDataset(cfg.CSV_PATH, split=None, tokenizer=tokenizer,
                                               label2id=label2id, max_len=cfg.MAX_LEN, index_subset=test_idx)
                targs_noeval = build_training_args(cfg, do_eval_in_training=False, out_dir=run_dir)
                trainer_test = WeightedBCETrainer(
                    model=base_model,
                    args=targs_noeval,
                    data_collator=data_collator,
                    tokenizer=tokenizer,
                )
                logits, labels_np, _ = trainer_test.predict(test_ds)

                probs = 1 / (1 + np.exp(-logits))
                preds = (probs >= cfg.THRESHOLD).astype(np.int32)
                metrics_json["test"] = _precision_recall_f1(labels_np.astype(np.int32), preds)

            score_k = val_f1

        else:
            with cfg.CSV_PATH.open("r", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            pool_idx = [i for i, r in enumerate(rows) if (r.get("split") or "").strip().lower() in ("train", "val")]
            test_idx = [i for i, r in enumerate(rows) if (r.get("split") or "").strip().lower() == "test"]
            print(f"[INFO][k={k}] Pool size (train+val rows): {len(pool_idx)} | Test size: {len(test_idx)}")

            rng = np.random.default_rng(cfg.SEED)
            if cfg.SHUFFLE_POOL:
                rng.shuffle(pool_idx)

            folds = np.array_split(pool_idx, cfg.N_FOLDS)
            fold_metrics = []

            # Train per fold with a fresh head
            for i_fold in range(cfg.N_FOLDS):
                print(f"\n===== Fold {i_fold+1}/{cfg.N_FOLDS} =====")
                val_idx = folds[i_fold].tolist()
                train_idx = [x for j, part in enumerate(folds) if j != i_fold for x in part.tolist()]

                train_ds = MultiLabelCSVDataset(cfg.CSV_PATH, split=None, tokenizer=tokenizer,
                                                label2id=label2id, max_len=cfg.MAX_LEN, index_subset=train_idx)
                val_ds   = MultiLabelCSVDataset(cfg.CSV_PATH, split=None, tokenizer=tokenizer,
                                                label2id=label2id, max_len=cfg.MAX_LEN, index_subset=val_idx)

                model_fold = base_model.__class__.from_pretrained(cfg.MODEL_NAME, config=config_base)
                use_eval_in_training = len(val_idx) > 0
                targs = build_training_args(cfg, do_eval_in_training=use_eval_in_training, out_dir=run_dir)
                callbacks = maybe_early_stopping(use_eval_in_training, cfg, targs) or []

                trainer = Trainer(
                    model=model_fold,
                    args=targs,
                    train_dataset=train_ds,
                    eval_dataset=val_ds if use_eval_in_training else None,
                    tokenizer=tokenizer,
                    data_collator=data_collator,
                    compute_metrics=make_compute_metrics(cfg.THRESHOLD) if use_eval_in_training else None,
                    callbacks=callbacks,
                )

                if getattr(targs, "evaluation_strategy", "no") == "no":
                    print("[WARN] transformers build lacks in-training eval; will evaluate after training.")

                print("[INFO] Training…")
                trainer.train()

                print("[INFO] Evaluating fold on its validation split…")
                logits, labels_np, _ = trainer.predict(val_ds)
                probs = 1 / (1 + np.exp(-logits))
                preds = (probs >= cfg.THRESHOLD).astype(np.int32)
                m = _precision_recall_f1(labels_np.astype(np.int32), preds)
                print("[OK] Fold metrics:", m)
                fold_metrics.append(m)

            # Average fold metrics
            avg = {key: float(np.mean([fm[key] for fm in fold_metrics])) for key in fold_metrics[0].keys()}
            metrics_json[f"cv{cfg.N_FOLDS}_avg"] = avg
            metrics_json[f"cv{cfg.N_FOLDS}_folds"] = fold_metrics

            # Save the last trained fold model + tokenizer (representative snapshot for this k)
            trainer.save_model(str(run_dir))
            tokenizer.save_pretrained(str(run_dir))

            if len(test_idx) > 0:
                print("\n[INFO] Evaluating on held-out test split…")
                test_ds = MultiLabelCSVDataset(cfg.CSV_PATH, split=None, tokenizer=tokenizer,
                                               label2id=label2id, max_len=cfg.MAX_LEN, index_subset=test_idx)
                targs_noeval = build_training_args(cfg, do_eval_in_training=False, out_dir=run_dir)
                trainer_test = Trainer(model=trainer.model, args=targs_noeval, data_collator=data_collator, tokenizer=tokenizer)
                logits, labels_np, _ = trainer_test.predict(test_ds)
                probs = 1 / (1 + np.exp(-logits))
                preds = (probs >= cfg.THRESHOLD).astype(np.int32)
                metrics_json["test"] = _precision_recall_f1(labels_np.astype(np.int32), preds)

            # Choose score for model selection (avg val f1_micro)
            score_k = float(metrics_json.get(f"cv{cfg.N_FOLDS}_avg", {}).get("f1_micro", -1.0))

        # Save metrics for this run & maybe update global best
        if metrics_json:
            (run_dir / "metrics.json").write_text(json.dumps(metrics_json, indent=2), encoding="utf-8")
            print(f"[OK] Wrote metrics.json -> {run_dir}")

        if score_k > best_score:
            best_score = score_k
            best_src_dir = run_dir
            try:
                for p in BEST_DIR.iterdir():
                    if p.is_file():
                        p.unlink()
                    elif p.is_dir():
                        shutil.rmtree(p)
                shutil.copytree(best_src_dir, BEST_DIR, dirs_exist_ok=True)
                print(f"[OK] New best model: {best_src_dir} (f1_micro={best_score:.3f}) → {BEST_DIR}")
            except Exception as e:
                print(f"[WARN] Failed to copy best run to {BEST_DIR}: {e}")

    if best_src_dir is None:
        print("[WARN] No best model selected (scores missing?).")
    else:
        print(f"[OK] Finished. Best model from: {best_src_dir}  →  {BEST_DIR}")
    # Finally, prune experiments (keep only metrics.json and README.txt)
    try:
        prune_experiments(EXPERIMENTS_ROOT)
        print(f"[OK] Pruned experiments under {EXPERIMENTS_ROOT} (kept metrics.json & README.txt per run).")
    except Exception as e:
        print(f"[WARN] Pruning experiments failed: {e}")

if __name__ == "__main__":
    main()
