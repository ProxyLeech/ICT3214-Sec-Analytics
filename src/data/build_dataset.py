"""
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
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))
from project_paths import (
    PROJECT_ROOT, DATA_ROOT, SRC_ROOT, MODELS_ROOT, EXPERIMENTS_ROOT,
    RAW_DIR, PROCESSED_DIR, EXTRACTED_PDFS_DIR,
    MAPPED_DIR, EXCEL_DIR, MITIGATIONS_DIR,
    output_dir_for_folds, project_path, ATTACK_STIX_DIR, RULES_DIR,
)

EXTRACTED_IOCS_CSV = PROCESSED_DIR / "extracted_iocs.csv"
DEFAULT_TI_CSV     = PROCESSED_DIR / "ti_groups_techniques.csv"
OUT_CSV            = PROCESSED_DIR / "dataset.csv"
OUT_LABELS         = PROCESSED_DIR / "labels.txt"
BUNDLES_DIR = ATTACK_STIX_DIR / "enterprise-attack"  
OUT = RULES_DIR / "attack_rules_auto.json"
INDEX_JSON   = ATTACK_STIX_DIR / "index.json"
ATTACK_DIR   = DATA_ROOT / "attack_stix"


# =================== HARD-CODED SETTINGS ===================

INCLUDE_GROUPS: bool = True
MAX_PER_KIND: int = 10
SEED: int = 42
TRAIN_RATIO: float = 0.8
VAL_RATIO: float   = 0.1
TEST_RATIO: float  = 0.1

# If you have extra weak rules, point this to the JSON file; else set to None
WEAK_RULES_JSON: pathlib.Path | None = None  # e.g., PROCESSED_DIR / "rules" / "attack_rules_auto.json"

import time

def dbg(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def assert_really_csv(path: pathlib.Path):
    """Detects common 'xlsx renamed to .csv' mistake."""
    with path.open("rb") as f:
        sig = f.read(4)
    if sig[:2] == b"PK":  # ZIP header = likely XLSX
        raise RuntimeError(f"{path} appears to be an Excel .xlsx (ZIP). Export a real CSV.")

# =================== BUILD RULES ===================
def norm(s): return (s or "").strip()

def mk_regex(words):
    w = sorted({w.lower() for w in words if w and len(w) >= 3})
    if not w: return None
    body = "|".join(re.escape(x) for x in w)
    return f"(?i)\\b({body})\\b"

def extract_terms(obj):
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
    candidates = list(BUNDLES_DIR.glob("enterprise-attack-*.json"))
    if not candidates:
        fallback = ATTACK_DIR / "enterprise-attack.json"
        return fallback if fallback.exists() else None

    def vernum(p: pathlib.Path) -> Tuple[int, int]:
        m = re.search(r"(\d+)\.(\d+)", p.name)  # e.g., 17.1
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    return sorted(candidates, key=vernum, reverse=True)[0]

def _find_bundle() -> pathlib.Path:
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


# =================== RULES & MAPPERS ===================
ATTACK_ID_RX = re.compile(r"^T\d{4}(?:\.\d{3})?$", re.I)

def load_weak_rules(extra_json: pathlib.Path | None) -> List[dict]:
    """
    Load rules; convert wildcard kind '*' to 'text'; precompile regex.
    """
    t0 = time.perf_counter()
    rules: List[dict] = []

    if extra_json and extra_json.exists():
        raw = json.loads(extra_json.read_text(encoding="utf-8"))
        if not isinstance(raw, list):
            print(f"[WARN] rules file is not a list: {extra_json}")
            raw = []

        for r in raw:
            when = r.get("when", {}) or {}
            k = when.get("kind", "*")

            # ---- normalize kind: treat '*' as 'text'
            if isinstance(k, str):
                k = k.lower().strip()
                k = "text" if k == "*" else k
            elif isinstance(k, (list, tuple)):
                k = [("text" if str(x).lower().strip() == "*" else str(x).lower().strip()) for x in k]
            else:
                k = "text"

            # ---- pre-compile
            contains = [str(c).lower() for c in (when.get("contains") or [])]
            compiled = []
            for rx in (when.get("regex") or []):
                try:
                    compiled.append(re.compile(rx))
                except re.error:
                    pass

            rules.append({
                "kind": k,
                "_contains": contains,
                "_regex": compiled,
                "add_techniques": r.get("add_techniques", []) or [],
            })

        dt = time.perf_counter() - t0
        kinds = sorted({kk for rr in rules for kk in (rr["kind"] if isinstance(rr["kind"], list) else [rr["kind"]])})
        print(f"[OK] Loaded {len(rules)} rules from {extra_json} (kinds={kinds}) in {dt:.2f}s")
    else:
        print(f"[INFO] No weak rules file found: {extra_json} (0 rules)")

    return rules

def index_rules_by_kind(rules: List[dict]) -> Dict[str, List[dict]]:
    by_kind: Dict[str, List[dict]] = defaultdict(list)
    for r in rules:
        k = r.get("kind", "text")
        if isinstance(k, list):
            for kk in k:
                by_kind[kk].append(r)
        else:
            by_kind[k].append(r)
    return by_kind

def apply_rules_prepared(kind: str, value: str, candidate_rules: List[dict]) -> Set[str]:
    out: Set[str] = set()
    v = value or ""
    vl = v.lower()

    # pass-through if IOC already labeled as an ATT&CK ID
    if kind == "attack_id" and ATTACK_ID_RX.fullmatch(v):
        out.add(v.upper())

    for r in candidate_rules:
        cs = r.get("_contains") or []
        if cs and not any(c in vl for c in cs):
            continue
        rxs = r.get("_regex") or []
        if rxs and not any(rx.search(v) for rx in rxs):
            continue
        for t in r.get("add_techniques", []) or []:
            t = (t or "").strip().upper()
            if ATTACK_ID_RX.fullmatch(t):
                out.add(t)
    return out

def load_tech_to_groups(ti_csv: pathlib.Path) -> Dict[str, Set[str]]:
    """Build mapping technique_id -> {group_names}, plus root technique mapping."""
    dbg(f"Reading technique→group map from {ti_csv} …")
    assert_really_csv(ti_csv)
    t0 = time.perf_counter()
    m: Dict[str, Set[str]] = defaultdict(set)
    rows = 0
    # try a few encodings defensively
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            with ti_csv.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                for rows, r in enumerate(reader, start=1):
                    if rows % 100000 == 0:
                        dbg(f"…tech2groups parsed {rows:,} rows (encoding={enc})")
                    t = (r.get("technique_id") or "").strip()
                    g = (r.get("group_name")   or "").strip()
                    if not t or not g:
                        continue
                    m[t].add(g)
            break
        except UnicodeDecodeError:
            dbg(f"[WARN] Unicode decode error with {enc}, trying next …")
    dbg(f"[OK] technique→group map: {len(m)} technique keys from {rows:,} rows in {(time.perf_counter()-t0):.2f}s")
    return m


# =================== DATASET BUILD HELPERS ===================

def _join(vals: List[str], n: int) -> str:
    return " ".join(str(x) for x in vals[:n])
def normalize_kind(k: str) -> str:
    k = (k or "").lower().strip()
    aliases = {
        "ip": "ipv4",
        "ipv6": "ipv6",
        "sha-256": "sha256",
        "sha256sum": "sha256",
        "md5sum": "md5",
        "fqdn": "domain",
        "host": "domain",
        "hostname": "domain",
        "link": "url",
        "uri": "url",
        "mail": "email",
        "email_address": "email",
        "hash": "md5",  # if you want a default
    }
    return aliases.get(k, k)
def build_rows_from_iocs(iocs_csv: pathlib.Path, ti_csv: pathlib.Path,
                         include_groups: bool = True, max_per_kind: int = 10,
                         rules_json: pathlib.Path | None = None) -> List[dict]:

    # 1) load + index rules
    rules = load_weak_rules(rules_json)
    rule_index = index_rules_by_kind(rules)

    tech2groups = load_tech_to_groups(ti_csv)

    per_file: Dict[str, dict] = defaultdict(
        lambda: {"iocs": [], "tech_labels": set(), "group_labels": set()}
    )

    # 2) read IOCs
    with iocs_csv.open("r", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            fid  = (r.get("file")  or "").strip()
            kind = normalize_kind(r.get("kind") or "")
            val  = (r.get("value") or "").strip()
            if not fid or not kind:
                continue
            per_file[fid]["iocs"].append((kind, val))

    # 3) synthesize ONE "text" IOC per file & apply TEXT rules to it
    from collections import defaultdict as _dd
    for fid, rec in per_file.items():
        by_kind = _dd(list)
        for k, v in rec["iocs"]:
            by_kind[k].append(v)

        text_blob = " ".join(
            by_kind.get("url", []) +
            by_kind.get("domain", []) +
            by_kind.get("ipv4", []) +
            by_kind.get("email", []) +
            by_kind.get("md5", []) +
            by_kind.get("sha256", [])
        )
        if text_blob:
            text_hits = apply_rules_prepared("text", text_blob, rule_index.get("text", []))
            if text_hits:
                rec["tech_labels"].update(text_hits)

        # 4) apply any per-kind rules (if you add such rules later)
        for k, vals in by_kind.items():
            if k == "text":  # skip the synthetic one
                continue
            cand = rule_index.get(k, [])
            if not cand:
                continue
            for v in vals:
                hits = apply_rules_prepared(k, v, cand)
                if hits:
                    rec["tech_labels"].update(hits)

        # 5) expand techniques → groups
        if include_groups and rec["tech_labels"]:
            for t in list(rec["tech_labels"]):
                rec["group_labels"].update(tech2groups.get(t, set()))
                rec["group_labels"].update(tech2groups.get(t.split(".", 1)[0], set()))

    # 6) build final rows (unchanged from your version)
    rows: List[dict] = []
    for fid, rec in per_file.items():
        by_kind = _dd(list)
        for k, v in rec["iocs"]:
            by_kind[k].append(v)

        parts = [f"Report:{fid}"]
        if by_kind.get("url"):    parts.append(f"URLs: {' '.join(by_kind['url'][:max_per_kind])}")
        if by_kind.get("domain"): parts.append(f"Domains: {' '.join(by_kind['domain'][:max_per_kind])}")
        if by_kind.get("ipv4"):   parts.append(f"IPs: {' '.join(by_kind['ipv4'][:max_per_kind])}")
        if by_kind.get("email"):  parts.append(f"Emails: {' '.join(by_kind['email'][:max_per_kind])}")
        if by_kind.get("md5"):    parts.append(f"MD5: {' '.join(by_kind['md5'][:max_per_kind])}")
        if by_kind.get("sha256"): parts.append(f"SHA256: {' '.join(by_kind['sha256'][:max_per_kind])}")

        labels: Set[str] = set(rec["tech_labels"])
        if include_groups:
            labels |= set(rec["group_labels"])
        # after you've computed labels / rule hits per file
        rule_hits = sorted(rec["tech_labels"])                  # techniques inferred by rules
        if rule_hits:
            parts.append("RULES: " + " ".join(rule_hits))       # <-- prepend heuristic cues
        has_attack_id = any(k == "attack_id" for k, _ in rec["iocs"])
        weight = 1.0 if has_attack_id else 0.6      # tune as you like
        rows.append({
            "id": fid,
            "text": " | ".join(parts),
            "labels": "|".join(sorted(labels)) if labels else "",
            "weight": weight,                        
        })

    return rows


def write_dataset_csv(path: pathlib.Path, rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "text", "labels", "weight"])
        w.writeheader()
        w.writerows(rows)
    num_unlabeled = sum(1 for r in rows if not r["labels"])
    print(f"[OK] Wrote {path} (total={len(rows)} labeled={len(rows)-num_unlabeled} unlabeled={num_unlabeled})")

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

# =================== ONE-CALL BUILD ===================

def run_build(
    ti_csv: pathlib.Path = DEFAULT_TI_CSV,
    out_csv: pathlib.Path = OUT_CSV,
    out_labels: pathlib.Path = OUT_LABELS,
) -> tuple[pathlib.Path, pathlib.Path]:
    rows = build_rows_from_iocs(
        iocs_csv=EXTRACTED_IOCS_CSV,
        ti_csv=ti_csv,
        include_groups=INCLUDE_GROUPS,
        max_per_kind=MAX_PER_KIND,
        rules_json=OUT,
    )
    write_dataset_csv(out_csv, rows)
    autobuild_labels_from_csv(out_csv, out_labels)
    return out_csv, out_labels

def main():
    # pick the newest bundle
    bundle = _find_bundle()
    print(f"[INFO] Using ATT&CK bundle: {bundle}")
    # bundle = sorted(ATTACK.glob("enterprise-attack-*.json"))[-1]
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
        # rules.append({
        #     "when": {"kind": "*", "regex": [rx]},
        #     "add_techniques": [ext]
        rules.append({
    "when": {"kind": "text", "regex": [rx]},
    "add_techniques": [ext]
    })

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rules, indent=2), encoding="utf-8")
    print(f"[OK] wrote {OUT} with {len(rules)} rules")
# =================== ENTRY POINT ===================

if __name__ == "__main__":
    main()
    out_csv_path, out_labels_path = run_build()
    print(f"[DONE] dataset={out_csv_path} labels={out_labels_path}")
