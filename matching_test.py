#!/usr/bin/env python3
from __future__ import annotations
import csv
import logging
from pathlib import Path
import re
import sys
from typing import Iterable, Set, Tuple

import pandas as pd

TTP_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")

PREFERRED_KEYS = [
    "matched_exact", "matched_root_only", "ttps", "ttp",
    "techniques", "technique", "attack"
]

def validate_ttps(ttps: Iterable[str]) -> Tuple[str, ...]:
    ttps = tuple(t.strip().upper() for t in ttps if t.strip())
    if not ttps:
        raise ValueError("No TTPs entered.")
    if len(ttps) > 5:
        raise ValueError("Maximum of 5 TTPs allowed.")
    for t in ttps:
        if not TTP_RE.match(t):
            raise ValueError(f"Invalid TTP format: {t}")
    return ttps

def find_candidate_dataset(data_dir: Path) -> Path:
    for name in ("group_ttps_detail.csv", "ranked_groups.csv"):
        p = data_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(f"No dataset found in {data_dir}")

def score_column(col: str) -> Tuple[int, int]:
    """Higher is better: exact match first, then keyword count."""
    cl = col.lower()
    exact = any(cl == k for k in PREFERRED_KEYS)
    hits = sum(1 for k in PREFERRED_KEYS if k in cl)
    return (1 if exact else 0, hits)

def find_ttp_column(df: pd.DataFrame) -> str:
    ranked = sorted(df.columns, key=lambda c: score_column(c), reverse=True)
    for c in ranked:
        cl = c.lower()
        if any(k in cl for k in PREFERRED_KEYS):
            return c
    for c in df.columns:
        cl = c.lower()
        if any(x in cl for x in ["ttp", "technique", "attack"]):
            return c
    raise KeyError("Could not find a TTP-related column.")

def split_tokens(cell) -> Set[str]:
    if pd.isna(cell):
        return set()
    s = str(cell)
    for sep in (",", ";", "|"):
        if sep in s:
            toks = [p.strip().upper() for p in s.split(sep) if p.strip()]
            return set(toks)
    return set(p.strip().upper() for p in s.split() if p.strip())

def with_roots(tts: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for t in tts:
        out.add(t)
        if "." in t:
            out.add(t.split(".", 1)[0])  # add root (e.g., T1110.001 → T1110)
    return out

def match_ttps(ttps: Tuple[str, ...], dataset_path: Path) -> pd.DataFrame:
    df = pd.read_csv(dataset_path)
    ttp_col = find_ttp_column(df)

    df["_ttp_set"] = df[ttp_col].map(split_tokens)
    df["_ttp_root_set"] = df["_ttp_set"].map(with_roots)

    input_full = set(ttps)
    input_plus_roots = with_roots(input_full)

    mask = df["_ttp_root_set"].apply(lambda s: bool(input_plus_roots & s))
    matched = df.loc[mask].drop(columns=["_ttp_set", "_ttp_root_set"])
    return matched

def write_outputs(matched: pd.DataFrame, ttps: Tuple[str, ...], out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    matched_out = out_dir / "matched_groups.csv"
    ttps_out = out_dir / "inputted_ttps.csv"

    matched.to_csv(matched_out, index=False)
    with ttps_out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["TTP"])
        for t in ttps:
            writer.writerow([t])
    return matched_out, ttps_out

def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    try:
        data_dir = Path("Data/mapped")
        out_dir = Path(".")
        user_input = input("Enter up to 5 TTPs (e.g. T1110 T1110.001 ...): ").strip()
        ttps = validate_ttps(user_input.split())

        dataset = find_candidate_dataset(data_dir)
        logging.info(f"Using dataset: {dataset}")

        matched = match_ttps(ttps, dataset)
        m_out, t_out = write_outputs(matched, ttps, out_dir)

        logging.info(f"Matched {len(matched)} rows -> {m_out}")
        logging.info(f"Saved inputted TTPs -> {t_out}")

        if not matched.empty and "group_name" in matched.columns:
            print("\nTop matched groups:")
            for g in matched["group_name"].head(10):
                print("-", g)
        else:
            print("No matches found.")
        return 0

    except Exception as e:
        logging.error(str(e))
        return 1

if __name__ == "__main__":
    sys.exit(main())
