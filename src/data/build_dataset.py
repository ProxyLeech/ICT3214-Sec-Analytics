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
from typing import Dict, List, Set, Tuple
from collections import defaultdict

# =================== CONFIG  ===================

from paths.paths import (
    PROCESSED_DIR,        
    RULES_DIR,
    DATA_ROOT, ATTACK_STIX_DIR,
)

# Paths
BUNDLES_DIR = ATTACK_STIX_DIR / "enterprise-attack"
RULES_JSON  = RULES_DIR / "attack_rules_auto.json"
INDEX_JSON  = ATTACK_STIX_DIR / "index.json"
ATTACK_DIR  = DATA_ROOT / "attack_stix"
EXTRACTED_IOCS_CSV = PROCESSED_DIR / "extracted_iocs.csv"
DEFAULT_TI_CSV = PROCESSED_DIR / "ti_groups_techniques.csv"
OUT_CSV    = PROCESSED_DIR / "dataset.csv"
OUT_LABELS = PROCESSED_DIR / "labels.txt"

# Build options
INCLUDE_GROUPS  = True
SEED            = 42
MAX_PER_KIND    = 10

# Rule options
REBUILD_RULES_FROM_BUNDLE = False  # set True to regenerate RULES_JSON from local ATT&CK bundle
WEAK_RULES_JSON_PATH: pathlib.Path | None = RULES_JSON  # set None to skip weak rules

# =======================================================================

ATTACK_ID_RX = re.compile(r"^T\d{4}(?:\.\d{3})?$", re.I)

DEFAULT_WEAK_RULES: list[dict] = []


def load_weak_rules(json_path: pathlib.Path | None) -> list[dict]:
    """
    Args:
        json_path: Path to a JSON file containing a list of rule dicts. If None
                   or missing/invalid, returns an empty list.

    Returns:
        A list of rules (each rule is a dict with "when" and "add_techniques" fields).
    """
    if json_path and json_path.exists():
        try:
            rules = json.loads(json_path.read_text(encoding="utf-8"))
            if isinstance(rules, list):
                print(f"[OK] Loaded {len(rules)} rules from {json_path}")
                return rules
            else:
                print(f"[WARN] {json_path} is not a JSON list; ignoring.")
        except Exception as e:
            print(f"[WARN] Failed to read rules from {json_path}: {e}")
    else:
        print(f"[WARN] Rules file not found: {json_path} — proceeding with NO rules.")
    return []


def load_tech_to_groups(ti_csv: pathlib.Path) -> Dict[str, Set[str]]:
    """
    Build a mapping from technique_id -> set of group_names.

    Also maps technique roots (e.g., T1059.001 contributes to T1059) so that
    either the exact or root form will resolve to group sets.

    Args:
        ti_csv: Path to ti_groups_techniques.csv with columns: technique_id, group_name.

    Returns:
        Dict mapping technique or root technique to a set of group names.
    """
    m: Dict[str, Set[str]] = defaultdict(set)
    with ti_csv.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            t = (r.get("technique_id") or "").strip()
            g = (r.get("group_name")   or "").strip()
            if not t or not g:
                continue
            m[t].add(g)
            root = t.split(".", 1)[0]
            m[root].add(g)
    return m


def apply_rules(kind: str, value: str, rules: List[dict]) -> Set[str]:
    """
    Apply weak rules to an IOC (kind, value) to produce ATT&CK technique IDs.

    Args:
        kind: IOC kind (e.g., 'url', 'domain', 'md5', 'sha256', 'ipv4', or 'attack_id').
        value: IOC value string.
        rules: List of rule dicts as loaded by load_weak_rules().

    Returns:
        Set of ATT&CK technique IDs inferred (e.g., {'T1566.002', 'T1204.002'}).
    """
    k = (kind or "").lower().strip()
    v = value or ""
    out: Set[str] = set()

    # Pass-through: if IOC itself is an ATT&CK ID, keep it
    if k == "attack_id" and ATTACK_ID_RX.fullmatch(v or ""):
        out.add(v.upper())

    for rule in rules:
        when = rule.get("when", {}) or {}
        want_kind = (when.get("kind") or "").lower().strip()

        # Support wildcard kind
        if want_kind and want_kind not in ("*", k):
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
    """
    Join up to n values into a single space-separated string.

    Args:
        vals: List of strings.
        n: Maximum number of items to include.

    Returns:
        A single string with up to n values separated by spaces.
    """
    return " ".join(str(x) for x in vals[:n])


def build_rows_from_iocs(
    iocs_csv: pathlib.Path,
    ti_csv: pathlib.Path,
    include_groups: bool = True,
    max_per_kind: int = 10,
    rules_json: pathlib.Path | None = None,
) -> List[dict]:
    """
    Construct dataset rows by reading IOC records and mapping to techniques/groups.

    Args:
        iocs_csv: Path to extracted_iocs.csv with columns: file, kind, value.
        ti_csv:   Path to ti_groups_techniques.csv for technique->group expansion.
        include_groups: If True, include group names as labels (in addition to techniques).
        max_per_kind: Max number of values to include per IOC kind in the 'text' field.
        rules_json: Optional path to a weak-rules JSON to infer techniques from IOC text.

    Returns:
        A list of row dicts with keys: id, text, labels, split (split may be overwritten later).
    """
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
            "split": "train",  # overwritten by assign_splits_inplace if enabled
        })

    return rows



def write_dataset_csv(path: pathlib.Path, rows: List[dict]) -> None:
    """
    Write dataset rows to CSV with header: id,text,labels,split.

    Args:
        path: Output CSV path.
        rows: List of row dicts as produced by build_rows_from_iocs().

    Returns:
        None. Writes CSV to disk and prints a short summary.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "labels", "split"])
        w.writeheader()
        w.writerows(rows)
    labeled = sum(1 for r in rows if r["labels"])
    print(f"[OK] Wrote {path} ({len(rows)} rows; labeled={labeled}, unlabeled={len(rows) - labeled})")


def autobuild_labels_from_csv(csv_path: pathlib.Path, out_labels: pathlib.Path) -> None:
    """
    Derive unique labels from the dataset CSV and write them to labels.txt.

    Args:
        csv_path: Path to the dataset CSV written by write_dataset_csv().
        out_labels: Destination path for labels.txt (one label per line).

    Returns:
        None. Writes labels file and prints a summary.
    """
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
    out_csv: pathlib.Path = OUT_CSV,
    out_labels: pathlib.Path = OUT_LABELS,
    include_groups: bool = INCLUDE_GROUPS,
    seed: int = SEED,
    max_per_kind: int = MAX_PER_KIND,
    weak_rules_json: pathlib.Path | None = WEAK_RULES_JSON_PATH,
    ioc_csv: pathlib.Path = EXTRACTED_IOCS_CSV,
) -> tuple[pathlib.Path, pathlib.Path]:
    """
    One-call dataset builder. Reads IOCs, applies weak rules, expands to groups,
    writes dataset.csv and labels.txt.

    Args:
        ti_csv: Path to ti_groups_techniques.csv.
        out_csv: Destination dataset.csv path.
        out_labels: Destination labels.txt path.
        include_groups: Include group names as labels (in addition to techniques).
        seed:        Random seed for deterministic splitting.
        max_per_kind: Max number of values per IOC kind included in the text.
        weak_rules_json: Optional path to a weak-rules JSON.
        ioc_csv: Path to extracted_iocs.csv.

    Returns:
        Tuple of (out_csv_path, out_labels_path).
    """
    rows = build_rows_from_iocs(
        iocs_csv=ioc_csv,
        ti_csv=ti_csv,
        include_groups=include_groups,
        max_per_kind=max_per_kind,
        rules_json=weak_rules_json,
    )
    return out_csv, out_labels


# =================== RULE GENERATION (from ATT&CK bundle) ===================

def norm(s): 
    """Trim whitespace from a string; returns empty string if None."""
    return (s or "").strip()


def mk_regex(words):
    """
    Build a single case-insensitive word-boundary regex that matches any of the input words.

    Args:
        words: Iterable of strings.

    Returns:
        Regex string or None if no eligible words.
    """
    w = sorted({w.lower() for w in words if w and len(w) >= 3})
    if not w: 
        return None
    body = "|".join(re.escape(x) for x in w)
    return f"(?i)\\b({body})\\b"


def extract_terms(obj):
    """
    Extract token-like terms from an ATT&CK attack-pattern object.

    Args:
        obj: A dict for one STIX 'attack-pattern' object.

    Returns:
        A set of candidate terms (strings) for regex construction.
    """
    terms = set()
    terms.add(norm(obj.get("name")))
    blob = " ".join([obj.get("description") or "",
                     " ".join(obj.get("x_mitre_detection") or [])])
    for t in re.findall(r"[A-Za-z][A-Za-z0-9\-\._]{2,}", blob):
        terms.add(t)
    for r in obj.get("external_references", []) or []:
        if r.get("source_name") == "mitre-attack":
            continue
        al = norm(r.get("description"))
        if al:
            for t in re.findall(r"[A-Za-z][A-Za-z0-9\-\._]{2,}", al):
                terms.add(t)
    return terms


def _latest_local_bundle() -> pathlib.Path | None:
    """
    Find the latest enterprise-attack-<ver>.json in BUNDLES_DIR, or fallback.

    Returns:
        Path to the newest ATT&CK bundle if present, else fallback path, else None.
    """
    candidates = list(BUNDLES_DIR.glob("enterprise-attack-*.json"))
    if not candidates:
        fallback = ATTACK_DIR / "enterprise-attack.json"
        return fallback if fallback.exists() else None

    def vernum(p: pathlib.Path) -> Tuple[int, int]:
        m = re.search(r"(\d+)\.(\d+)", p.name)  # e.g., 17.1
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    return sorted(candidates, key=vernum, reverse=True)[0]


def _find_bundle() -> pathlib.Path:
    """
    Locate an ATT&CK bundle to use for rule generation.

    Returns:
        Path to the selected ATT&CK bundle JSON.

    Raises:
        FileNotFoundError if no appropriate bundle can be located.
    """
    path = _latest_local_bundle()
    if path:
        return path

    if INDEX_JSON.exists():
        try:
            idx = json.loads(INDEX_JSON.read_text(encoding="utf-8"))
            entries = idx.get("objects") or idx.get("collections") or []
            for e in entries:
                if "enterprise-attack" in (e.get("name", "") or e.get("id", "")):
                    local = e.get("path") or e.get("file") or "enterprise-attack/enterprise-attack.json"
                    candidate = BUNDLES_DIR / local
                    if candidate.exists():
                        return candidate
        except Exception:
            pass

    raise FileNotFoundError(
        "No ATT&CK bundle found.\n"
        f"Expected under: {BUNDLES_DIR} (e.g., enterprise-attack-<ver>.json)\n"
        f"or fallback: {BUNDLES_DIR / 'enterprise-attack.json'}\n"
        f"Optional index file: {INDEX_JSON}"
    )


def buildrules() -> None:
    """
    Generate weak rules from the local ATT&CK bundle and write to RULES_JSON.

    Inputs:
        Uses the ATT&CK bundle found by _find_bundle().

    Outputs:
        Writes RULES_JSON containing a list of rules:
        [
          {"when": {"kind": "*", "regex": ["(?i)\\b(term1|term2|...)\\b"]},
           "add_techniques": ["T####(.###)"]},
          ...
        ]

    Returns:
        None. Prints a summary and writes the JSON file.
    """
    bundle = _find_bundle()
    print(f"[INFO] Using ATT&CK bundle: {bundle}")
    data = json.loads(bundle.read_text(encoding="utf-8"))
    rules = []
    for o in data.get("objects", []):
        if o.get("type") != "attack-pattern":
            continue
        # external_id
        ext = None
        for ref in o.get("external_references") or []:
            if ref.get("source_name") == "mitre-attack":
                ext = ref.get("external_id")
                break
        if not ext:
            continue
        terms = extract_terms(o)
        rx = mk_regex(terms)
        if not rx:
            continue
        rules.append({
            "when": {"kind": "*", "regex": [rx]},
            "add_techniques": [ext]
        })
    RULES_JSON.parent.mkdir(parents=True, exist_ok=True)
    RULES_JSON.write_text(json.dumps(rules, indent=2), encoding="utf-8")
    print(f"[OK] wrote {RULES_JSON} with {len(rules)} rules")


# =================== ENTRY POINT (hard-coded run) ===================

if __name__ == "__main__":
    # Optionally regenerate weak rules from local ATT&CK bundle
    if REBUILD_RULES_FROM_BUNDLE:
        buildrules()

    # Build dataset (no CLI args — uses CONFIG above)
    out_csv_path, out_labels_path = run_build(
        ti_csv=DEFAULT_TI_CSV,
        out_csv=OUT_CSV,
        out_labels=OUT_LABELS,
        include_groups=INCLUDE_GROUPS,
        seed=SEED,
        max_per_kind=MAX_PER_KIND,
        weak_rules_json=WEAK_RULES_JSON_PATH,
        ioc_csv=EXTRACTED_IOCS_CSV,
    )
    print(f"[DONE] dataset={out_csv_path} labels={out_labels_path}")
