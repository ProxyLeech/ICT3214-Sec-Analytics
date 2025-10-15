from __future__ import annotations
import csv
import logging
from pathlib import Path
import re
import sys
from typing import Iterable, Set, Tuple, Dict
import pandas as pd
from collections import defaultdict

# ============================================
# Regex patterns
# ============================================
TTP_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")
TTP_PATTERN = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)

PREFERRED_KEYS = [
    "matched_exact", "matched_root_only", "ttps", "ttp",
    "techniques", "technique", "attack"
]

# ============================================
# Validation
# ============================================
def validate_ttps(ttps: Iterable[str]) -> Tuple[str, ...]:
    """
    Validate that user-input TTPs follow MITRE format (T#### or T####.###)
    and that there are at most 5.
    """
    ttps = tuple(t.strip().upper() for t in ttps if t.strip())
    if not ttps:
        raise ValueError("No TTPs entered.")
    if len(ttps) > 5:
        raise ValueError("Maximum of 5 TTPs allowed.")
    for t in ttps:
        if not TTP_RE.match(t):
            raise ValueError(f"Invalid TTP format: {t}")
    return ttps

# ============================================
# Dataset loading
# ============================================
def load_combined_dataset(data_dir: Path) -> pd.DataFrame:
    """
    Load and merge 'group_ttps_detail.csv' and 'ranked_groups.csv' if both exist.
    Removes duplicates and returns a unified dataframe.
    """
    group_path = data_dir / "group_ttps_detail.csv"
    ranked_path = data_dir / "ranked_groups.csv"

    dfs = []
    if group_path.exists():
        dfs.append(pd.read_csv(group_path))
        logging.info(f"Loaded {group_path.name} ({len(dfs[-1])} rows)")
    if ranked_path.exists():
        dfs.append(pd.read_csv(ranked_path))
        logging.info(f"Loaded {ranked_path.name} ({len(dfs[-1])} rows)")

    if not dfs:
        raise FileNotFoundError(f"No datasets found in {data_dir}")

    combined = pd.concat(dfs, ignore_index=True).drop_duplicates()
    logging.info(f"Combined dataset size: {len(combined)} rows")
    return combined

# ============================================
# Column identification
# ============================================
def score_column(col: str) -> Tuple[int, int]:
    cl = col.lower()
    exact = any(cl == k for k in PREFERRED_KEYS)
    hits = sum(1 for k in PREFERRED_KEYS if k in cl)
    return (1 if exact else 0, hits)

def find_ttp_column(df: pd.DataFrame) -> str:
    """
    Automatically detect which column in the dataset contains TTPs.
    """
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

# ============================================
# Token extraction
# ============================================
def split_tokens(cell) -> Set[str]:
    """
    Extract only the TTP IDs (T#### or T####.###) from dataset cells that may
    include technique names, e.g., 'T1110.001 (Password Guessing)'.
    """
    if pd.isna(cell):
        return set()
    text = str(cell).upper()
    matches = re.findall(TTP_PATTERN, text)
    return set(matches)

# ============================================
# Strict matching logic
# ============================================
def match_ttps(ttps: Tuple[str, ...], data_dir: Path) -> pd.DataFrame:
    """
    Perform strict matching: returns only rows containing the exact TTP IDs
    provided by the user. No root/sub-technique inference.
    """
    df = load_combined_dataset(data_dir)
    ttp_col = find_ttp_column(df)

    df["_ttp_set"] = df[ttp_col].map(split_tokens)
    input_ttps = set(ttps)

    mask = df["_ttp_set"].apply(lambda s: bool(input_ttps & s))
    matched = df.loc[mask].drop(columns=["_ttp_set"])
    return matched

# ============================================
# Extract mapping of ID -> full string (with name)
# ============================================
def extract_ttp_pairs(df: pd.DataFrame, ttp_col: str) -> Dict[str, str]:
    """
    Extract a mapping of 'T1110.001' -> 'T1110.001 (PASSWORD GUESSING)'
    from the dataset for dropdown display.
    """
    mapping = {}
    for val in df[ttp_col].dropna().unique():
        val_str = str(val).strip()
        matches = re.findall(TTP_PATTERN, val_str.upper())
        for m in matches:
            if m not in mapping:
                mapping[m] = val_str
    return mapping

# ============================================
# CSV output
# ============================================
def write_outputs(matched: pd.DataFrame, ttps: Tuple[str, ...], out_dir: Path) -> Tuple[Path, Path]:
    """
    Save matched groups and inputted TTPs to CSV for traceability.
    """
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

# ============================================
# CLI entry point
# ============================================
def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        data_dir = Path("Data/mapped")
        out_dir = Path(".")
        user_input = input("Enter up to 5 TTPs (e.g. T1110 T1110.001 ...): ").strip()
        ttps = validate_ttps(user_input.split())

        logging.info("Loading and merging datasets...")
        matched = match_ttps(ttps, data_dir)

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
