#!/usr/bin/env python3
"""
Extract "target id", "target name", and "mapping description" from the
"techniques addressed" sheet in a MITRE ATT&CK mitigations Excel export.

Usage:
    python extract_attack_mappings.py /path/to/enterprise-attack-*-mitigations.xlsx [--out /path/to/output.csv]

Notes:
- Defaults to writing a CSV next to the input named: <input stem>_mappings.csv
- If the sheet or columns are missing, the script prints a helpful error.
"""
import argparse
import os
import sys
import pandas as pd

REQUIRED_SHEET = "techniques addressed"
REQUIRED_COLUMNS = ["target id", "target name", "mapping description"]

def infer_output_path(input_path: str, out_arg: str | None) -> str:
    if out_arg:
        return out_arg
    base, ext = os.path.splitext(os.path.basename(input_path))
    return os.path.join(os.path.dirname(input_path), f"{base}_mappings.csv")

def main():
    ap = argparse.ArgumentParser(description="Extract target/mapping fields from MITRE ATT&CK mitigations workbook")
    ap.add_argument("excel_path", help="Path to the enterprise-attack-*-mitigations.xlsx file")
    ap.add_argument("--out", help="Optional output CSV path")
    args = ap.parse_args()

    if not os.path.exists(args.excel_path):
        print(f"Error: file not found: {args.excel_path}", file=sys.stderr)
        sys.exit(1)

    try:
        # Read only the required sheet for performance
        df = pd.read_excel(args.excel_path, sheet_name=REQUIRED_SHEET)
    except ValueError as e:
        print(f"Error: required sheet '{REQUIRED_SHEET}' not found. Available sheets: {pd.ExcelFile(args.excel_path).sheet_names}", file=sys.stderr)
        sys.exit(2)

    # Normalize columns to lower-case for robust selection
    df.columns = [c.strip().lower() for c in df.columns]

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        print(f"Error: missing required columns {missing}. Found columns: {list(df.columns)}", file=sys.stderr)
        sys.exit(3)

    out_df = df[REQUIRED_COLUMNS].copy()

    # Clean up whitespace and NaNs
    for col in REQUIRED_COLUMNS:
        if out_df[col].dtype == object:
            out_df[col] = out_df[col].astype(str).str.strip()
    out_df = out_df.fillna("")

    out_path = infer_output_path(args.excel_path, args.out)
    # Ensure directory exists
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {len(out_df):,} rows to {out_path}")

if __name__ == "__main__":
    main()
