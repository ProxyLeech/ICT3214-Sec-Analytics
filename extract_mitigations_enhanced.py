#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

REQUIRED_SHEET = "techniques addressed"
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


def infer_output_path(input_path: str, out_arg: str | None, tsv: bool) -> str:
    if out_arg:
        return out_arg
    base, _ext = os.path.splitext(os.path.basename(input_path))
    suffix = "_mappings.tsv" if tsv else "_mappings.csv"
    return os.path.join(os.path.dirname(input_path), f"{base}{suffix}")


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
    """
    Return a dataframe with the desired columns (renamed to canonical names),
    searching with synonyms in a case-insensitive way.
    Returns (out_df, missing) where missing is a list of canonical names not found.
    """
    # Build lookup from normalized header -> original header
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
    out.columns = list(required_columns)  # rename to canonical order
    return out, []


def clean_text_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Trim whitespace and replace NaN with empty strings without turning NaN into 'nan' strings."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else ("" if pd.isna(v) else str(v)))
    return df


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Extract target/mapping fields from a MITRE ATT&CK mitigations workbook",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("excel_path", help="Path to the enterprise-attack-*-mitigations.xlsx file")
    ap.add_argument("--out", help="Optional output file path (CSV by default; TSV if --tsv is set)")
    ap.add_argument("--stdout", action="store_true", help="Write output to stdout instead of a file")
    ap.add_argument("--tsv", action="store_true", help="Use tab-separated output format")
    ap.add_argument("--dedupe", action="store_true", help="Drop exact duplicate rows")
    ap.add_argument("--list-sheets", action="store_true", help="List available sheets and exit")
    ap.add_argument("--sheet", default=REQUIRED_SHEET, help="Name of the sheet that contains the mappings")
    ap.add_argument(
        "--columns",
        help="Override the required columns (comma-separated, in order)",
    )
    args = ap.parse_args()

    if not os.path.exists(args.excel_path):
        print(f"Error: file not found: {args.excel_path}", file=sys.stderr)
        sys.exit(1)

    if args.list_sheets:
        try:
            xls = pd.ExcelFile(args.excel_path)
        except Exception as e:  # noqa: BLE001
            print(f"Error: cannot open workbook: {e}", file=sys.stderr)
            sys.exit(1)
        print("Sheets:")
        for name in xls.sheet_names:
            print(f"  - {name}")
        return

    # Resolve columns to use
    required_columns: List[str] = (
        [c.strip() for c in args.columns.split(",")] if args.columns else list(DEFAULT_REQUIRED_COLUMNS)
    )

    # Pick a suitable sheet (case-insensitive, fuzzy)
    sheet_name = choose_sheet(args.excel_path, args.sheet)
    if sheet_name is None:
        try:
            available = pd.ExcelFile(args.excel_path).sheet_names
        except Exception:
            available = []
        print(
            f"Error: required sheet '{args.sheet}' not found. Available sheets: {available}",
            file=sys.stderr,
        )
        sys.exit(2)

    try:
        # Read only the chosen sheet for performance
        df = pd.read_excel(args.excel_path, sheet_name=sheet_name, engine="openpyxl")
    except ValueError as e:
        print(f"Error: failed to read sheet '{sheet_name}': {e}", file=sys.stderr)
        sys.exit(2)

    # Normalize headers for robust selection (we preserve original df for values)
    df.columns = [c.strip() for c in df.columns]

    out_df, missing = pick_columns(df, required_columns)
    if missing:
        print(
            f"Error: missing required columns {missing}. Found columns: {list(df.columns)}",
            file=sys.stderr,
        )
        sys.exit(3)

    # Clean up text cells safely
    out_df = clean_text_columns(out_df, required_columns)

    if args.dedupe:
        before = len(out_df)
        out_df = out_df.drop_duplicates().reset_index(drop=True)
        after = len(out_df)
        if before != after:
            print(f"Info: dropped {before - after} duplicate rows.", file=sys.stderr)

    # Decide output target
    sep = "\t" if args.tsv else ","
    if args.stdout:
        out_df.to_csv(sys.stdout, index=False, encoding="utf-8", sep=sep, lineterminator="\n")
        return

    out_path = infer_output_path(args.excel_path, args.out, args.tsv)
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8", sep=sep, lineterminator="\n")
    print(f"Wrote {len(out_df):,} rows to {out_path}")


if __name__ == "__main__":
    main()
