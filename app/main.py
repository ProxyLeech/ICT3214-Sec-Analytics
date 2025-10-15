#!/usr/bin/env python3
"""

Flow:
  1) extract_pdfs.py           -> Data/extracted_pdfs/extracted_iocs.csv
  2) enterprise_attack.py      -> Data/attack_stix/processed/ti_groups_techniques.csv
  3) map_iocs_to_attack.py     -> Data/mapped/{ranked_groups.csv, group_ttps_detail.csv}

Usage:
  python main.py
  python main.py --force              # re-run all steps even if outputs exist
  python main.py --skip-extract       # skip PDF extraction
  python main.py --skip-attack        # skip ATT&CK normalization
  python main.py --skip-map           # skip mapping step
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import sys

ROOT = Path(__file__).resolve().parents[1]   
sys.path.insert(0, str(ROOT / "src"))        # make common/, data/, models/ importable
from common.paths import (
    PROJECT_ROOT, DATA_ROOT, RAW_DIR,
    EXTRACTED_PDFS_DIR, 
    PROCESSED_DIR, 
    DATASCRIPT_ROOT,
    MODELS_ROOT,
)


# ---- Expected inputs/outputs per step ----
PDFS_IN_DIR            = RAW_DIR / "pdfs"
EXTRACTED_IOCS_CSV     = EXTRACTED_PDFS_DIR / "extracted_iocs.csv"
TI_GROUPS_TECHS_CSV    = PROCESSED_DIR / "ti_groups_techniques.csv"
DATASET_CSV   = PROCESSED_DIR / "dataset.csv"
LABELS_TXT    = PROCESSED_DIR / "labels.txt"
# RANKED_GROUPS_CSV      = PROCESSED_DIR / "ranked_groups.csv"
# GROUP_TTPS_DETAIL_CSV  = PROCESSED_DIR / "group_ttps_detail.csv"

# Scripts (relative to repo root)
EXTRACT_SCRIPT         = DATASCRIPT_ROOT  / "extract_pdfs.py"
ATTACK_SCRIPT          = DATASCRIPT_ROOT / "enterprise_attack.py"
# MAP_SCRIPT             = DATASCRIPT_ROOT / "map_iocs_to_attack.py"
BUILD_DATASET_SCRIPT = DATASCRIPT_ROOT / "build_dataset.py"
TRAIN_ROBERTA_SCRIPT = MODELS_ROOT  / "train_roberta.py"

#Trained model
BEST_MODEL_DIR = MODELS_ROOT / "best_roberta_for_predict"
BEST_REQUIRED  = [
    BEST_MODEL_DIR / "config.json",
    BEST_MODEL_DIR / "tokenizer.json",     
    BEST_MODEL_DIR / "id2label.json",
]

def needs_run_verbose(outputs: list[Path], force: bool, label: str = "") -> bool:
    print(f"\n[DEBUG] needs_run('{label}'): force={force}")
    all_exist = True
    for p in outputs:
        exists = p.exists()
        print(f"  - {p}  exists={exists}")
        if not exists:
            all_exist = False
    print(f"[DEBUG] -> all_exist={all_exist}  => needs_run={force or not all_exist}")
    return force or not all_exist

def run(cmd: list[str], cwd: Path | None = None) -> None:
    """Run a subprocess with nice printing & error checking."""
    print(f"\n$ {' '.join(cmd)}")
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if res.returncode != 0:
        raise SystemExit(res.returncode)


def needs_run(outputs: list[Path], force: bool) -> bool:
    if force:
        return True
    return not all(p.exists() for p in outputs)


def step_extract_pdfs(force: bool, in_dir: Path = PDFS_IN_DIR, out_dir: Path = EXTRACTED_PDFS_DIR):
    """Step 1: Extract text/metadata/IOCs from PDFs using PyMuPDF."""
    out_dir.mkdir(parents=True, exist_ok=True)
    if not needs_run([EXTRACTED_IOCS_CSV], force):
        print(f"[SKIP] extract_pdfs.py — up to date: {EXTRACTED_IOCS_CSV}")
        return
    if not in_dir.exists():
        print(f"[WARN] Input folder not found: {in_dir}. Continuing anyway.")
    run([sys.executable, str(EXTRACT_SCRIPT),
         "--in", str(in_dir),
         "--out", str(out_dir)])


def step_enterprise_attack(force: bool):
    """Step 2: Normalize local ATT&CK bundle(s) into CSV edges."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not needs_run([TI_GROUPS_TECHS_CSV], force):
        print(f"[SKIP] enterprise_attack.py — up to date: {TI_GROUPS_TECHS_CSV}")
        return
    run([sys.executable, str(ATTACK_SCRIPT)])


# # def step_map_iocs_to_attack(force: bool):
#     """Step 3: Score & rank groups based on observed techniques."""
#     MAPPED_DIR.mkdir(parents=True, exist_ok=True)
#     if not needs_run([RANKED_GROUPS_CSV, GROUP_TTPS_DETAIL_CSV], force):
#         print(f"[SKIP] map_iocs_to_attack.py — up to date: {RANKED_GROUPS_CSV}, {GROUP_TTPS_DETAIL_CSV}")
#         return
#     # map_iocs_to_attack.py uses common.paths defaults; no args needed
#     run([sys.executable, str(MAP_SCRIPT)])

def step_build_dataset(force: bool) -> None:
    """
    Runs src/data/build_dataset.py which:
      - reads extracted_iocs + ti_groups_techniques
      - writes dataset.csv + labels.txt into PROCESSED_DIR
      - assigns splits deterministically (inside the builder)
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    if not needs_run([DATASET_CSV, LABELS_TXT], force):
        print(f"[SKIP] build_dataset.py — up to date: {DATASET_CSV}, {LABELS_TXT}")
        return
    run([sys.executable, str(BUILD_DATASET_SCRIPT)])

def step_train_roberta(force: bool) -> None:
    """
    Runs src/models/train_roberta.py which:
      - loops k = 0..10 (skips k=1), trains, saves each run under Experiments/<k>foldruns/roberta_base_v1
      - copies the best run into Experiments/best_roberta_for_predict/ (used by predict script)
    """
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"\n[DEBUG] BEST_MODEL_DIR: {BEST_MODEL_DIR}")
    if BEST_MODEL_DIR.exists():
        try:
            print("[DEBUG] best dir contents:", sorted([p.name for p in BEST_MODEL_DIR.iterdir()]))
        except Exception as e:
            print("[DEBUG] failed to list best dir:", e)
    if not needs_run(BEST_REQUIRED, force):
        print(f"[SKIP] train_roberta.py — best model already present: {BEST_MODEL_DIR}")
        return

    run([sys.executable, str(TRAIN_ROBERTA_SCRIPT)])

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="End-to-end APTnotes → ATT&CK mapping pipeline.")
    ap.add_argument("--force", action="store_true", help="Re-run steps even if outputs already exist.")
    ap.add_argument("--skip-extract", action="store_true", help="Skip PDF extraction step.")
    ap.add_argument("--skip-attack", action="store_true", help="Skip ATT&CK normalization step.")
    ap.add_argument("--skip-map", action="store_true", help="Skip mapping/ranking step.")
    ap = argparse.ArgumentParser(description="Build dataset + train RoBERTa (skip if already built).")
    ap.add_argument("--force", action="store_true", help="Re-run steps even if outputs exist.")
    ap.add_argument("--skip-build", action="store_true", help="Skip dataset build step.")
    ap.add_argument("--skip-train", action="store_true", help="Skip training step.")
    return ap.parse_args()


def main():
    args = parse_args()

    print("=== Pipeline start:", datetime.utcnow().isoformat() + "Z", "===")
    print(f"[ROOT] {PROJECT_ROOT}")
    print(f"[DATA] {DATA_ROOT}")
    print(f"[RAW ] {RAW_DIR}")
    print(f"[OUT ] extracted → {EXTRACTED_PDFS_DIR}")
    print(f"[OUT ] processed → {PROCESSED_DIR}")
    if not args.skip_build:
            step_build_dataset(force=args.force)
    else:
            print("[SKIP] Step: build dataset")

    if not args.skip_train:
            if not DATASET_CSV.exists():
                print(f"[WARN] {DATASET_CSV} not found; training may fail. Run without --skip-build or with --force.")
            step_train_roberta(force=args.force)
    else:
            print("[SKIP] Step: train roberta")

    print("\n=== Pipeline complete ===")
    print(f"- IOCs CSV:        {EXTRACTED_IOCS_CSV if EXTRACTED_IOCS_CSV.exists() else '(missing)'}")
    print(f"- ATT&CK edges:    {TI_GROUPS_TECHS_CSV if TI_GROUPS_TECHS_CSV.exists() else '(missing)'}")
    # print(f"- Ranked groups:   {RANKED_GROUPS_CSV if RANKED_GROUPS_CSV.exists() else '(missing)'}")
    # print(f"- Group details:   {GROUP_TTPS_DETAIL_CSV if GROUP_TTPS_DETAIL_CSV.exists() else '(missing)'}")
    print(f"- dataset.csv:  {DATASET_CSV if DATASET_CSV.exists() else '(missing)'}")
    print(f"- labels.txt:   {LABELS_TXT if LABELS_TXT.exists() else '(missing)'}")
    if BEST_MODEL_DIR.exists():
        ok = all(p.exists() for p in BEST_REQUIRED)
        print(f"- best model:   {BEST_MODEL_DIR} {'(OK)' if ok else '(incomplete)'}")
    else:
        print("- best model:   (missing)")


if __name__ == "__main__":
    main()
