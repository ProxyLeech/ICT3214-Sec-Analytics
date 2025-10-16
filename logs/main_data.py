#!/usr/bin/env python3
"""
main.py — end-to-end pipeline runner

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

from common.paths import (
    PROJECT_ROOT, DATA_ROOT, RAW_DIR,
    EXTRACTED_PDFS_DIR, EXTRACTED_IOCS_CSV,
    PROCESSED_DIR, 
    MAPPED_DIR,DATASCRIPT_ROOT,
)


# ---- Expected inputs/outputs per step ----
PDFS_IN_DIR            = RAW_DIR / "pdfs"
EXTRACTED_IOCS_CSV     = EXTRACTED_PDFS_DIR / "extracted_iocs.csv"


TI_GROUPS_TECHS_CSV    = PROCESSED_DIR / "ti_groups_techniques.csv"

RANKED_GROUPS_CSV      = PROCESSED_DIR / "ranked_groups.csv"
GROUP_TTPS_DETAIL_CSV  = PROCESSED_DIR / "group_ttps_detail.csv"

# Scripts (relative to repo root)
EXTRACT_SCRIPT         = DATASCRIPT_ROOT  / "extract_pdfs.py"
ATTACK_SCRIPT          = DATASCRIPT_ROOT / "enterprise_attack.py"
MAP_SCRIPT             = DATASCRIPT_ROOT / "map_iocs_to_attack.py"


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
    # enterprise_attack.py discovers the latest bundle automatically
    run([sys.executable, str(ATTACK_SCRIPT)])


def step_map_iocs_to_attack(force: bool):
    """Step 3: Score & rank groups based on observed techniques."""
    MAPPED_DIR.mkdir(parents=True, exist_ok=True)
    if not needs_run([RANKED_GROUPS_CSV, GROUP_TTPS_DETAIL_CSV], force):
        print(f"[SKIP] map_iocs_to_attack.py — up to date: {RANKED_GROUPS_CSV}, {GROUP_TTPS_DETAIL_CSV}")
        return
    # map_iocs_to_attack.py uses common.paths defaults; no args needed
    run([sys.executable, str(MAP_SCRIPT)])


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="End-to-end APTnotes → ATT&CK mapping pipeline.")
    ap.add_argument("--force", action="store_true", help="Re-run steps even if outputs already exist.")
    ap.add_argument("--skip-extract", action="store_true", help="Skip PDF extraction step.")
    ap.add_argument("--skip-attack", action="store_true", help="Skip ATT&CK normalization step.")
    ap.add_argument("--skip-map", action="store_true", help="Skip mapping/ranking step.")
    return ap.parse_args()


def main():
    args = parse_args()

    print("=== Pipeline start:", datetime.utcnow().isoformat() + "Z", "===")
    print(f"[ROOT] {PROJECT_ROOT}")
    print(f"[DATA] {DATA_ROOT}")
    print(f"[RAW ] {RAW_DIR}")
    print(f"[OUT ] extracted → {EXTRACTED_PDFS_DIR}")
    print(f"[OUT ] processed → {PROCESSED_DIR}")
    print(f"[OUT ] mapped    → {MAPPED_DIR}")

    if not args.skip_extract:
        step_extract_pdfs(force=args.force)
    else:
        print("[SKIP] Step 1: extract PDFs")

    if not args.skip_attack:
        step_enterprise_attack(force=args.force)
    else:
        print("[SKIP] Step 2: enterprise ATT&CK normalize")

    if not args.skip_map:
        step_map_iocs_to_attack(force=args.force)
    else:
        print("[SKIP] Step 3: map IOCs to ATT&CK")

    print("\n=== Pipeline complete ===")
    print(f"- IOCs CSV:        {EXTRACTED_IOCS_CSV if EXTRACTED_IOCS_CSV.exists() else '(missing)'}")
    print(f"- ATT&CK edges:    {TI_GROUPS_TECHS_CSV if TI_GROUPS_TECHS_CSV.exists() else '(missing)'}")
    print(f"- Ranked groups:   {RANKED_GROUPS_CSV if RANKED_GROUPS_CSV.exists() else '(missing)'}")
    print(f"- Group details:   {GROUP_TTPS_DETAIL_CSV if GROUP_TTPS_DETAIL_CSV.exists() else '(missing)'}")


if __name__ == "__main__":
    main()
