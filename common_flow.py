# src/flows/common_flow.py
from __future__ import annotations
import os, tempfile, subprocess, sys, re, json
from pathlib import Path
from typing import Iterable
import pandas as pd
BASE_DIR = Path(__file__).resolve().parent
# -------------------------
# Project paths (same logic)
# -------------------------
# BASE_DIR = Path(__file__).resolve().parents[2]  # .../ICT3214-Sec-Analytics/src/flows/common_flow.py -> repo root
DATA_ROOT       = BASE_DIR / "data"
RAW_DIR         = DATA_ROOT / "raw"
PROCESSED_DIR   = DATA_ROOT / "processed"
SRC_ROOT        = BASE_DIR / "src"
MODELS_ROOT     = SRC_ROOT / "models"
DATASCRIPT_ROOT = SRC_ROOT / "data"
EXPERIMENTS_ROOT= BASE_DIR / "experiments"

PDFS_IN_DIR          = RAW_DIR / "pdfs"
EXTRACTED_IOCS_CSV   = PROCESSED_DIR / "extracted_iocs.csv"
TI_GROUPS_TECHS_CSV  = PROCESSED_DIR / "ti_groups_techniques.csv"
DATASET_CSV          = PROCESSED_DIR / "dataset.csv"
LABELS_TXT           = PROCESSED_DIR / "labels.txt"
EXTRACTED_PDFS_DIR   = DATA_ROOT / "extracted_pdfs"

EXTRACT_SCRIPT       = DATASCRIPT_ROOT / "extract_pdfs.py"
ATTACK_SCRIPT        = DATASCRIPT_ROOT / "enterprise_attack.py"
BUILD_DATASET_SCRIPT = DATASCRIPT_ROOT / "build_dataset.py"
TRAIN_ROBERTA_SCRIPT = MODELS_ROOT / "train_roberta.py"
PREDICT_SCRIPT       = MODELS_ROOT / "predict_roberta.py"

BEST_MODEL_DIR = MODELS_ROOT / "best_roberta_for_predict"

# -------------
# Small helpers
# -------------
def _atomic_to_csv(df: pd.DataFrame, path: str):
    d = Path(path).parent
    d.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=d, newline="", suffix=".tmp") as tmp:
        tmp_name = tmp.name
        df.to_csv(tmp, index=False)
    os.replace(tmp_name, path)

def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(map(str, cmd))}")
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if res.returncode != 0:
        raise SystemExit(res.returncode)

def needs_run(outputs: Iterable[Path], inputs: Iterable[Path] = ()) -> bool:
    outs = list(outputs)
    if not outs or any(not p.exists() for p in outs):
        return True
    out_mtime = min(p.stat().st_mtime for p in outs)
    ins = [p for p in inputs if p is not None and Path(p).exists()]
    if not ins:
        return False
    return max(Path(p).stat().st_mtime for p in ins) > out_mtime

def _run_mitigations_and_get_csv() -> Path:
    """
    Run mitigations.py once, return Data/mitigations/mitigations.csv
    """
    script = BASE_DIR / "mitigations.py"
    out_csv = BASE_DIR / "Data" / "mitigations" / "mitigations.csv"
    if out_csv.exists():
        print(f"[SKIP] mitigations.py — up to date: {out_csv}")
        return out_csv
    print(f"[RUN] mitigations.py — generating: {out_csv}")
    res = subprocess.run([sys.executable, str(script)], cwd=str(BASE_DIR))
    if res.returncode != 0:
        raise RuntimeError(f"mitigations.py failed with exit code {res.returncode}")
    if not out_csv.exists():
        raise FileNotFoundError(f"Expected mitigations CSV not found at: {out_csv}")
    return out_csv

# -----------------
# Score/rank guards
# -----------------
def _ensure_score_and_rank_rule(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    if df is None or df.empty:
        return df
    candidates = ["score", "prob", "probability", "confidence", "logit", "logprob"]
    src = next((c for c in candidates if c in df.columns), None)
    if src is None:
        if "rank" in df.columns:
            df["score"] = pd.to_numeric(df["rank"], errors="coerce")
            df["score"] = 1.0 / (1.0 + df["score"].fillna(df["score"].max() or 1))
        else:
            n = len(df)
            df["score"] = np.linspace(1.0, 0.0, n, endpoint=False)
    else:
        df["score"] = pd.to_numeric(df[src], errors="coerce").fillna(0.0)
    if "rank" not in df.columns or df["rank"].isna().all():
        df["rank"] = (-df["score"]).rank(method="first").astype(int)
    return df

def _ensure_score_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    if df is None or df.empty:
        return df
    candidates = ["score", "group_score", "prob", "probability", "confidence", "logit", "logprob"]
    src = next((c for c in candidates if c in df.columns), None)
    if src is None:
        if "rank" in df.columns:
            df["score"] = pd.to_numeric(df["rank"], errors="coerce")
            df["score"] = 1.0 / (1.0 + df["score"].fillna(df["score"].max() or 1))
        else:
            n = len(df)
            df["score"] = np.linspace(1.0, 0.0, n, endpoint=False)
    else:
        df["score"] = pd.to_numeric(df[src], errors="coerce").fillna(0.0)
    if "rank" not in df.columns or df["rank"].isna().all():
        df["rank"] = (-df["score"]).rank(method="first").astype(int)
    return df

# ------------------------------
# Group→TTP mapping (TOP actor!)
# ------------------------------
def _collect_top_group_ttps(matched_df: pd.DataFrame) -> list[str]:
    """
    Return technique IDs ONLY for the top matched group
    using Data/mapped/group_ttps_detail.csv (ID→name→aliases fallback).
    """
    map_path = BASE_DIR / "Data" / "mapped" / "group_ttps_detail.csv"
    if matched_df is None or matched_df.empty or not map_path.exists():
        return []
    df = matched_df.copy()
    if "rank" in df.columns and df["rank"].notna().any():
        df = df.sort_values(["rank", "score"], ascending=[True, False])
    elif "score" in df.columns:
        df = df.sort_values("score", ascending=False)
    top = df.iloc[0]

    group_name = None
    for c in ("group_name", "group", "actor", "name"):
        if c in df.columns and pd.notna(top.get(c)):
            group_name = str(top[c]).strip().lower()
            break

    group_id = None
    for c in ("group_id", "id", "mitre_id"):
        if c in df.columns and pd.notna(top.get(c)):
            group_id = str(top[c]).strip().lower()
            break

    try:
        g = pd.read_csv(map_path)
    except Exception:
        return []
    g.columns = [c.strip().lower() for c in g.columns]

    def norm(s): return str(s).strip().lower() if pd.notna(s) else ""
    hit = pd.Series([False] * len(g))
    if group_id and "group_id" in g.columns:
        hit = g["group_id"].astype(str).str.strip().str.lower() == group_id
    if not hit.any() and group_name and "group_name" in g.columns:
        hit = g["group_name"].astype(str).str.strip().str.lower() == group_name
    if not hit.any() and group_name:
        alias_cols = [c for c in ("aliases", "aka", "synonyms") if c in g.columns]
        alias_match = pd.Series([False] * len(g))
        for c in alias_cols:
            col = g[c].fillna("").astype(str).str.lower()
            alias_match |= col.str.contains(fr"\b{re.escape(group_name)}\b", regex=True)
        hit = alias_match

    gsel = g.loc[hit].copy()
    if gsel.empty:
        return []
    id_re = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
    ttps = []
    for col in ("matched_exact", "matched_root_only", "ttp_list", "techniques", "technique_ids"):
        if col in gsel.columns:
            for entry in gsel[col].fillna("").tolist():
                ttps.extend([m.upper() for m in id_re.findall(entry)])

    def _key(tid: str):
        m = re.match(r"T(\d{4})(?:\.(\d{3}))?$", tid)
        return (int(m.group(1)), int(m.group(2) or 999)) if m else (9999, 999)
    return sorted(set(ttps), key=_key)
