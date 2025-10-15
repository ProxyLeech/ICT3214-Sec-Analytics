from __future__ import annotations

import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

REQUIRED_SHEET = "associated mitigations"
DEFAULT_REQUIRED_COLUMNS = ["target id", "target name", "mapping description"]

COLUMN_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "target id": ("target id", "technique id", "targetid", "technique_id"),
    "target name": ("target name", "technique name", "target", "name"),
    "mapping description": (
        "mapping description",
        "relationship description",
        "description",
        "mapping",
    ),
}


def _normalize(s: str) -> str:
    return s.strip().lower()


def choose_sheet(excel_path: str, requested: str) -> str | None:
    """Return a best-effort match for the desired sheet name (case-insensitive, startswith)."""
    xls = pd.ExcelFile(excel_path)
    wanted = _normalize(requested)
    lowers = {name: _normalize(name) for name in xls.sheet_names}
    for name, low in lowers.items():
        if low == wanted:
            return name
    for name, low in lowers.items():
        if low.startswith(wanted):
            return name
    for name, low in lowers.items():
        if wanted in low:
            return name
    return None


def pick_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Return dataframe with required columns renamed to canonical names."""
    norm_to_original = {_normalize(col): col for col in df.columns}
    selected = {}
    missing = []

    for canon in required_columns:
        candidates: Iterable[str] = COLUMN_SYNONYMS.get(_normalize(canon), (canon,))
        found_col = None
        for cand in candidates:
            key = _normalize(cand)
            if key in norm_to_original:
                found_col = norm_to_original[key]
                break
        if found_col is None:
            missing.append(canon)
        else:
            selected[canon] = found_col

    if missing:
        return df, missing

    out = df[[selected[c] for c in required_columns]].copy()
    out.columns = list(required_columns)
    return out, []


def clean_text_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Trim whitespace and replace NaN with empty strings."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else ("" if pd.isna(v) else str(v)))
    return df


def main() -> None:
    # Input Excel path
    excel_path = r"Data/excel/enterprise-attack-v17.1-techniques.xlsx"

    # Ensure output folder Data/mapped/ exists
    output_dir = os.path.join(os.path.dirname(excel_path), "..", "mapped")
    os.makedirs(output_dir, exist_ok=True)

    # Output file in Data/mapped/
    out_path = os.path.join(output_dir, "mitigations.csv")

    required_columns = list(DEFAULT_REQUIRED_COLUMNS)
    sheet_name = choose_sheet(excel_path, REQUIRED_SHEET)
    if sheet_name is None:
        available = pd.ExcelFile(excel_path).sheet_names
        print(f"Error: required sheet '{REQUIRED_SHEET}' not found. Available sheets: {available}")
        sys.exit(2)

    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]

    out_df, missing = pick_columns(df, required_columns)
    if missing:
        print(f"Error: missing required columns {missing}. Found columns: {list(df.columns)}")
        sys.exit(3)

    out_df = clean_text_columns(out_df, required_columns)
    before = len(out_df)
    out_df = out_df.drop_duplicates().reset_index(drop=True)
    after = len(out_df)
    if before != after:
        print(f"Dropped {before - after} duplicate rows.")

    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {len(out_df):,} rows to {out_path}")


if __name__ == "__main__":
    main()
