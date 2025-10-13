#!/usr/bin/env python3
import os
import re
import sys
import pandas as pd
import csv

TTP_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")

def validate_ttps(ttps):
    if not ttps:
        print("No TTPs entered.")
        sys.exit(1)
    if len(ttps) > 5:
        print("Maximum of 5 TTPs allowed.")
        sys.exit(1)
    for t in ttps:
        if not TTP_RE.match(t):
            print(f"Invalid TTP format: {t}")
            sys.exit(1)

def find_ttp_column(df):
    preferred = ["matched_exact", "matched_root_only", "ttps", "ttp", "techniques", "technique", "attack"]
    for p in preferred:
        for c in df.columns:
            if p in c.lower():
                return c
    for c in df.columns:
        if any(x in c.lower() for x in ["ttp", "technique", "attack"]):
            return c
    return None

def normalize_ttp_cell(cell):
    if pd.isna(cell):
        return set()
    s = str(cell)
    for sep in [",", ";", "|"]:
        if sep in s:
            return set(p.strip() for p in s.split(sep) if p.strip())
    return set(p.strip() for p in s.split() if p.strip())

def match_ttps(ttps):
    mapped_dir = os.path.join(os.getcwd(), "Data", "mapped")
    candidates = [
        os.path.join(mapped_dir, "group_ttps_detail.csv"),
        os.path.join(mapped_dir, "ranked_groups.csv")
    ]
    dataset_path = next((p for p in candidates if os.path.exists(p)), None)
    if not dataset_path:
        print("No dataset found in mapped folder.")
        sys.exit(1)
    print(f"Using dataset: {dataset_path}")

    df = pd.read_csv(dataset_path)
    ttp_col = find_ttp_column(df)
    if not ttp_col:
        print("Could not find a TTP-related column.")
        sys.exit(1)

    df["_ttp_set"] = df[ttp_col].apply(normalize_ttp_cell)
    input_set = set(t.strip() for t in ttps)
    matched = df[df["_ttp_set"].apply(lambda s: len(input_set & s) > 0)].drop(columns=["_ttp_set"])

    root_path = os.getcwd()
    matched_out = os.path.join(root_path, "matched_groups.csv")
    ttps_out = os.path.join(root_path, "inputted_ttps.csv")

    matched.to_csv(matched_out, index=False)
    print(f"Matched {len(matched)} rows -> {matched_out}")

    with open(ttps_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["TTP"])
        for t in ttps:
            writer.writerow([t])
    print(f"Saved inputted TTPs -> {ttps_out}")

    if not matched.empty and "group_name" in matched.columns:
        print("\nTop matched groups:")
        for g in matched["group_name"].head(10):
            print("-", g)

if __name__ == "__main__":
    user_input = input("Enter up to 5 TTPs (e.g. T1110 T1110.001 ...): ").strip()
    ttps = [t.strip() for t in user_input.split() if t.strip()]
    validate_ttps(ttps)
    match_ttps(ttps)
