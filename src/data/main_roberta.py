#!/usr/bin/env python3
# Orchestrator: build dataset → train RoBERTa
from __future__ import annotations
import argparse, pathlib, sys
from datetime import datetime, UTC
from pathlib import Path

# Make sure imports resolve when running from /app
ROOT = Path(__file__).resolve().parents[1]  # repo root
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from common.paths import PROCESSED_DIR
from data.build_dataset import run_build
from models.train_roberta import run_train, Config as TrainCfg

def parse_args():
    ap = argparse.ArgumentParser(description="End-to-end: dataset build + RoBERTa training")
    ap.add_argument("--skip-build", action="store_true", help="Skip dataset build")
    ap.add_argument("--skip-train", action="store_true", help="Skip training")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--include-groups", action="store_true", help="Include groups in labels")
    ap.add_argument("--weak-rules", type=pathlib.Path, help="Optional JSON of extra weak rules")
    return ap.parse_args()

def main():
    args = parse_args()
    print("=== Pipeline start:", datetime.now(UTC).isoformat(), "===")

    dataset_csv = PROCESSED_DIR / "dataset.csv"
    labels_txt  = PROCESSED_DIR / "labels.txt"

    if not args.skip_build:
        print("[STEP] Build dataset")
        run_build(
            ti_csv=PROCESSED_DIR / "ti_groups_techniques.csv",
            out_csv=dataset_csv,
            out_labels=labels_txt,
            include_groups=args.include_groups,
            auto_split=True,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            weak_rules_json=args.weak_rules,
        )

    if not args.skip_train:
        print("[STEP] Train RoBERTa")
        cfg = TrainCfg(CSV_PATH=dataset_csv, LABELS_PATH=labels_txt)
        run_train(cfg)

    print("=== Pipeline done ===")

if __name__ == "__main__":
    main()
