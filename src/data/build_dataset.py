"""
build_dataset.py

Build a multi-label text classification dataset from:
- extracted IOCs:      <DATA_ROOT>/extracted_pdfs/extracted_iocs.csv
- ATT&CK group->tech:  <DATA_ROOT>/attack_stix/processed/ti_groups_techniques.csv

Outputs:
- <DATA_ROOT>/processed/dataset.csv
- <DATA_ROOT>/processed/labels.txt

"""

from __future__ import annotations
import csv
import json
import re
import hashlib
import random
import pathlib
from typing import Dict, List, Set
from collections import defaultdict

from common.paths import (
    EXTRACTED_PDFS_DIR,   # <DATA_ROOT>/extracted_pdfs/extracted_iocs.csv
    PROCESSED_DIR,        # <DATA_ROOT>/processed
)
EXTRACTED_IOCS_CSV =EXTRACTED_PDFS_DIR / "extracted_iocs.csv"

DEFAULT_WEAK_RULES: List[dict] = [
    {"when": {"kind": "url"},    "add_techniques": ["T1566.002"]},  # Spearphishing Link
    {"when": {"kind": "domain"}, "add_techniques": ["T1566.002"]},
    {"when": {"kind": "md5"},    "add_techniques": ["T1204.002"]},  # User Execution: Malicious File
    {"when": {"kind": "sha256"}, "add_techniques": ["T1204.002"]},
    # You can add more patterns via a JSON file passed with --weak-rules
]

def load_weak_rules(extra_json: pathlib.Path | None) -> List[dict]:
    rules = list(DEFAULT_WEAK_RULES)
    if extra_json and extra_json.exists():
        try:
            rules.extend(json.loads(extra_json.read_text(encoding="utf-8")))
            print(f"[OK] Loaded extra weak rules from {extra_json}")
        except Exception as e:
            print(f"[WARN] Failed to read extra weak rules {extra_json}: {e}")
    return rules

def load_tech_to_groups(ti_csv: pathlib.Path) -> Dict[str, Set[str]]:
    # Build mapping technique_id -> {group_names}
    # Also map technique roots (e.g., T1059) so T1059.001 hits still expand to groups.
    m: Dict[str, Set[str]] = defaultdict(set)
    with ti_csv.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = (r.get("technique_id") or "").strip()
            g = (r.get("group_name")   or "").strip()
            if not t or not g:
                continue
            m[t].add(g)
            # root mapping, e.g., T1566.002 -> T1566
            root = t.split(".", 1)[0]
            m[root].add(g)
    return m

ATTACK_ID_RX = re.compile(r"^T\d{4}(?:\.\d{3})?$", re.I)

def apply_rules(kind: str, value: str, rules: List[dict]) -> Set[str]:
    k = (kind or "").lower().strip()
    v = value or ""
    out: Set[str] = set()

    # Pass-through: if the IOC already is an ATT&CK ID, keep it
    if k == "attack_id" and ATTACK_ID_RX.fullmatch(v or ""):
        out.add(v.upper())

    for rule in rules:
        when = rule.get("when", {}) or {}
        want_kind = (when.get("kind") or "").lower().strip()
        if want_kind and want_kind != k:
            continue

        contains = when.get("contains") or []
        if contains and not any(str(c).lower() in v.lower() for c in contains):
            continue

        regexes = when.get("regex") or []
        if regexes:
            try:
                if not any(re.search(rx, v) for rx in regexes):
                    continue
            except re.error:
                # ignore bad regex entry
                continue

        for t in rule.get("add_techniques", []) or []:
            t = (t or "").strip()
            if ATTACK_ID_RX.fullmatch(t):
                out.add(t.upper())

    return out

def _join(vals: List[str], n: int) -> str:
    return " ".join(str(x) for x in vals[:n])

def build_rows_from_iocs(
    iocs_csv: pathlib.Path,
    ti_csv: pathlib.Path,
    include_groups: bool = True,
    max_per_kind: int = 10,
    rules_json: pathlib.Path | None = None,
) -> List[dict]:
    rules = load_weak_rules(rules_json)
    tech2groups = load_tech_to_groups(ti_csv)

    per_file: Dict[str, dict] = defaultdict(
        lambda: {"iocs": [], "tech_labels": set(), "group_labels": set()}
    )

    with iocs_csv.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fid  = (r.get("file")  or "").strip()
            kind = (r.get("kind")  or "").strip().lower()
            val  = (r.get("value") or "").strip()
            if not fid or not kind:
                continue

            per_file[fid]["iocs"].append((kind, val))
            per_file[fid]["tech_labels"].update(apply_rules(kind, val, rules))

    # Expand techniques -> groups
    if include_groups:
        for rec in per_file.values():
            for t in list(rec["tech_labels"]):
                # exact or root expansion (root handled by load_tech_to_groups)
                rec["group_labels"].update(tech2groups.get(t, set()))
                root = t.split(".", 1)[0]
                rec["group_labels"].update(tech2groups.get(root, set()))

    # Build dataset rows
    rows: List[dict] = []
    for fid, rec in per_file.items():
        by_kind: Dict[str, List[str]] = defaultdict(list)
        for k, v in rec["iocs"]:
            by_kind[k].append(v)

        parts = [f"Report:{fid}"]
        if by_kind.get("url"):    parts.append(f"URLs: {_join(by_kind['url'],    max_per_kind)}")
        if by_kind.get("domain"): parts.append(f"Domains: {_join(by_kind['domain'], max_per_kind)}")
        if by_kind.get("ipv4"):   parts.append(f"IPs: {_join(by_kind['ipv4'],     max_per_kind)}")
        if by_kind.get("md5"):    parts.append(f"MD5: {_join(by_kind['md5'],     max_per_kind)}")
        if by_kind.get("sha256"): parts.append(f"SHA256: {_join(by_kind['sha256'], max_per_kind)}")

        labels: Set[str] = set(rec["tech_labels"])
        if include_groups:
            labels |= set(rec["group_labels"])

        rows.append({
            "id": fid,
            "text": " | ".join(parts),
            "labels": "|".join(sorted(labels)) if labels else "",
            "split": "train",  # will be overwritten by assign_splits_inplace if enabled
        })

    return rows

def assign_splits_inplace(rows: List[dict], train: float, val: float, test: float, seed: int) -> None:
    """
    Deterministic split by 'id' with a seeded hash+shuffle. Ensures all instances
    of the same id stay in the same split.
    """
    tot = train + val + test
    if abs(tot - 1.0) > 1e-6:
        raise ValueError(f"Ratios must sum to 1.0 (got {tot})")

    ids = sorted(set(r["id"] for r in rows))
    # hash sort then shuffle for stability
    ids.sort(key=lambda x: hashlib.md5((str(x) + str(seed)).encode("utf-8")).hexdigest())
    rng = random.Random(seed)
    rng.shuffle(ids)

    n = len(ids)
    n_tr = int(round(train * n))
    n_val = int(round(val * n))
    train_ids = set(ids[:n_tr])
    val_ids   = set(ids[n_tr:n_tr + n_val])
    test_ids  = set(ids[n_tr + n_val:])

    for r in rows:
        fid = r["id"]
        r["split"] = "train" if fid in train_ids else "val" if fid in val_ids else "test"

def write_dataset_csv(path: pathlib.Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "labels", "split"])
        w.writeheader()
        w.writerows(rows)
    labeled = sum(1 for r in rows if r["labels"])
    print(f"[OK] Wrote {path} ({len(rows)} rows; labeled={labeled}, unlabeled={len(rows) - labeled})")

def autobuild_labels_from_csv(csv_path: pathlib.Path, out_labels: pathlib.Path) -> None:
    uniq: Set[str] = set()
    with csv_path.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            s = (r.get("labels") or "").strip()
            if s:
                uniq.update(x for x in s.split("|") if x)

    # Techniques first, then groups
    tech = sorted([x for x in uniq if x.startswith("T")])
    grp  = sorted([x for x in uniq if not x.startswith("T")])
    labels = tech + grp

    out_labels.write_text("\n".join(labels) + "\n", encoding="utf-8")
    print(f"[OK] Wrote {out_labels} with {len(labels)} labels.")

def run_build(
    ti_csv: pathlib.Path,
    out_csv: pathlib.Path = PROCESSED_DIR / "dataset.csv",
    out_labels: pathlib.Path = PROCESSED_DIR / "labels.txt",
    include_groups: bool = True,
    auto_split: bool = True,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    max_per_kind: int = 10,
    weak_rules_json: pathlib.Path | None = None,
    ioc_csv: pathlib.Path = EXTRACTED_IOCS_CSV,
) -> tuple[pathlib.Path, pathlib.Path]:
    rows = build_rows_from_iocs(
        iocs_csv=ioc_csv,
        ti_csv=ti_csv,
        include_groups=include_groups,
        max_per_kind=max_per_kind,
        rules_json=weak_rules_json,
    )
    if auto_split:
        assign_splits_inplace(rows, train_ratio, val_ratio, test_ratio, seed)
    write_dataset_csv(out_csv, rows)
    autobuild_labels_from_csv(out_csv, out_labels)
    return out_csv, out_labels

if __name__ == "__main__":
    import argparse
    from common.paths import DATA_ROOT

    default_ti_csv = PROCESSED_DIR / "ti_groups_techniques.csv"
    ap = argparse.ArgumentParser(description="Build dataset.csv and labels.txt from extracted IOCs and ATT&CK mapping.")
    ap.add_argument("--iocs", type=pathlib.Path, default=EXTRACTED_IOCS_CSV,
                    help=f"Path to extracted_iocs.csv (default: {EXTRACTED_IOCS_CSV})")
    ap.add_argument("--ti", type=pathlib.Path, default=default_ti_csv,
                    help=f"Path to ti_groups_techniques.csv (default: {default_ti_csv})")
    ap.add_argument("--out-csv", type=pathlib.Path, default=PROCESSED_DIR / "dataset.csv",
                    help=f"Output dataset.csv (default: {PROCESSED_DIR / 'dataset.csv'})")
    ap.add_argument("--out-labels", type=pathlib.Path, default=PROCESSED_DIR / "labels.txt",
                    help=f"Output labels.txt (default: {PROCESSED_DIR / 'labels.txt'})")
    ap.add_argument("--include-groups", action="store_true", default=True,
                    help="Include threat group names as labels (default: True)")
    ap.add_argument("--no-include-groups", dest="include_groups", action="store_false")
    ap.add_argument("--weak-rules", type=pathlib.Path,
                    help="Optional JSON file with extra weak rules")
    ap.add_argument("--max-per-kind", type=int, default=10,
                    help="Max values to include per IOC kind in the text field")
    ap.add_argument("--auto-split", action="store_true", default=True,
                    help="Automatically split rows into train/val/test (default: True)")
    ap.add_argument("--no-auto-split", dest="auto_split", action="store_false")
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = build_rows_from_iocs(
        iocs_csv=args.iocs,
        ti_csv=args.ti,
        include_groups=args.include_groups,
        max_per_kind=args.max_per_kind,
        rules_json=args.weak_rules,
    )
    if args.auto_split:
        assign_splits_inplace(rows, args.train_ratio, args.val_ratio, args.test_ratio, args.seed)
    write_dataset_csv(args.out_csv, rows)
    autobuild_labels_from_csv(args.out_csv, args.out_labels)
    print("[DONE]")
