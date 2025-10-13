#!/usr/bin/env python3
import sys, os, re
from pathlib import Path
import pandas as pd

TTP_RE = re.compile(r"^T\d{4}(?:\.\d{3})?$")

def validate_ttps(ttps):
    if not ttps: print("no ttps entered"); sys.exit(1)
    if len(ttps) > 5: print("maximum 5 ttps allowed"); sys.exit(1)
    for t in ttps:
        if not TTP_RE.match(t): print(f"invalid ttp format: {t}"); sys.exit(1)

def find_ttp_column(df):
    prefs = ["matched_exact","matched_root_only","ttps","ttp","techniques","technique","attack"]
    for p in prefs:
        for c in df.columns:
            if p in c.lower(): return c
    for c in df.columns:
        if any(x in c.lower() for x in ["ttp","technique","attack"]): return c
    return None

def normalize_ttp_cell(cell):
    if pd.isna(cell): return set()
    s = str(cell)
    seps = [",",";","|"]
    if any(sep in s for sep in seps):
        out = set()
        for sep in seps:
            if sep in s:
                out.update(p.strip() for p in s.split(sep) if p.strip())
                s = "|".join(p.strip() for p in s.split(sep) if p.strip())
        return out
    return set(p.strip() for p in s.split() if p.strip())

def candidate_dirs(script_dir: Path, cwd: Path):
    c = []
    c.append(Path(os.getenv("MAPPED_DIR", "")) if os.getenv("MAPPED_DIR") else None)
    c += [
        cwd/"mapped",
        cwd/"Data"/"mapped",
        script_dir/"mapped",
        script_dir/"Data"/"mapped",
        script_dir.parent/"mapped",
        script_dir.parent/"Data"/"mapped",
        script_dir.parent.parent/"mapped",
        script_dir.parent.parent/"Data"/"mapped",
    ]
    return [p for p in c if p]

def locate_mapped_dir():
    script_dir = Path(__file__).resolve().parent
    cwd = Path.cwd()
    tried = []
    for p in candidate_dirs(script_dir, cwd):
        tried.append(str(p))
        if p.exists() and p.is_dir(): return p, tried
    return None, tried

def load_dataset(mapped_dir: Path):
    for name in ["group_ttps_detail.csv","ranked_groups.csv"]:
        p = mapped_dir/name
        if p.exists(): return p
    return None

def main():
    mapped_dir, tried = locate_mapped_dir()
    if not mapped_dir:
        print("no mapped directory found. tried:")
        for t in tried: print(" -", t)
        sys.exit(1)
    dataset_path = load_dataset(mapped_dir)
    if not dataset_path:
        print("no dataset found in:", mapped_dir)
        print("expected one of: group_ttps_detail.csv, ranked_groups.csv")
        sys.exit(1)
    print("using dataset:", dataset_path)

    df = pd.read_csv(dataset_path)
    ttp_col = find_ttp_column(df)
    if not ttp_col:
        print("could not find a ttp column in dataset"); sys.exit(1)

    df["_ttp_set"] = df[ttp_col].apply(normalize_ttp_cell)

    user_input = input("Enter up to 5 TTPs (e.g. T1110 T1110.001 ...): ").strip()
    ttps = [t.strip() for t in user_input.split() if t.strip()]
    validate_ttps(ttps)
    input_set = set(ttps)

    matched = df[df["_ttp_set"].apply(lambda s: len(input_set & s) > 0)].drop(columns=["_ttp_set"])
    if matched.empty:
        print("no matches found"); sys.exit(0)

    out_path = mapped_dir/"matched_groups.csv"
    matched.to_csv(out_path, index=False)
    print(f"matched {len(matched)} rows -> {out_path}")
    if "group_name" in matched.columns:
        print("matched groups (top 10):")
        for g in matched["group_name"].head(10): print(" -", g)

if __name__ == "__main__":
    main()
