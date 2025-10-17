from __future__ import annotations
import csv
import logging
from pathlib import Path
import re
import sys
from typing import Iterable, Set, Tuple
import pandas as pd

# ============================================
# Regex definitions
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
# Dataset handling — now merges both CSVs
# ============================================
def load_combined_dataset(data_dir: Path) -> pd.DataFrame:
    """
    Load and merge both 'group_ttps_detail.csv' and 'ranked_groups.csv'
    if they exist. This ensures that main and sub-techniques like
    T1110 / T1110.002 are both represented.
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
    if pd.isna(cell):
        return set()
    return set(m.upper() for m in TTP_PATTERN.findall(str(cell)))

def with_roots(tts: Iterable[str]) -> Set[str]:
    out: Set[str] = set()
    for t in tts:
        out.add(t)
        if "." in t:
            out.add(t.split(".", 1)[0])  # e.g., T1110.001 → T1110
    return out

# ============================================
# Matching logic
# ============================================
def match_ttps(ttps: Tuple[str, ...], data_dir: Path) -> pd.DataFrame:
    """
    Match input TTPs against the combined dataset.
    """
    df = load_combined_dataset(data_dir)
    ttp_col = find_ttp_column(df)

    df["_ttp_set"] = df[ttp_col].map(split_tokens)
    df["_ttp_root_set"] = df["_ttp_set"].map(with_roots)

    input_full = set(ttps)
    input_plus_roots = with_roots(input_full)

    mask = df["_ttp_root_set"].apply(lambda s: bool(input_plus_roots & s))
    matched = df.loc[mask].drop(columns=["_ttp_set", "_ttp_root_set"])
    return matched

# ============================================
# CSV output
# ============================================
def write_outputs(matched: pd.DataFrame, ttps: Tuple[str, ...], out_dir: Path) -> Tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    matched_out = out_dir / "matched_groups_rule.csv"
    ttps_out = out_dir / "inputted_ttps_rule.csv"

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
