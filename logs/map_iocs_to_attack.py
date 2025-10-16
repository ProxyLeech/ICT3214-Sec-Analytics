import csv
import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path
from common.paths import (
    PROJECT_ROOT, DATA_ROOT,
    EXTRACTED_PDFS_DIR, EXTRACTED_IOCS_CSV,   # read extracted_iocs.csv
    ATTACK_STIX_DIR, PROCESSED_DIR,    # read ti_groups_techniques.csv
    MAPPED_DIR, ENTERPRISE_DIR,                               # write ranked_groups.csv, group_ttps_detail.csv, bundles
)


# Inputs (all dynamic now, derived from the repo root via common.paths):
# - EXTRACTED_IOCS_CSV  -> Data/extracted_pdfs/extracted_iocs.csv
# - PROCESSED_ATTACK_DIR/ti_groups_techniques.csv
# - ENTERPRISE_DIR/enterprise-attack-*.json (optional, for tactics)

# Outputs:
# - MAPPED_DIR/ranked_groups.csv
# - MAPPED_DIR/group_ttps_detail.csv
# - MAPPED_DIR/<file>-bundle.json (mini STIX bundle)

IOC_CSV   = EXTRACTED_IOCS_CSV
TI_EDGES  = PROCESSED_DIR / "ti_groups_techniques.csv"
OUT_DIR   = MAPPED_DIR
ATTACK_DIR = ENTERPRISE_DIR  # holds enterprise-attack-*.json files
OUT_DIR_CSV = PROCESSED_DIR

OUT_DIR.mkdir(parents=True, exist_ok=True)
# put this in a shared util or top of both extract_pdfs.py and the mapper
import re
ATTACK_STD_RX = re.compile(r'(?i)T\s*?(\d{4})(?:\s*[\.\-_/]?\s*(\d{1,3}))?')

def normalize_attack_id(s: str) -> str | None:
    """Return canonical 'Txxxx' or 'Txxxx.yyy' (zero-padded subtechnique) or None."""
    if not s:
        return None
    m = ATTACK_STD_RX.search(s)
    if not m:
        return None
    if m.group(2):
        return f"T{m.group(1)}.{int(m.group(2)):03d}"
    return f"T{m.group(1)}"


def read_csv(path: Path):
    with path.open("r", encoding="utf-8", newline="") as f:
        r = csv.DictReader(f)
        return list(r)

def write_csv(path: Path, rows, headers):
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        w.writerows(rows)

def technique_root(tid: str) -> str:
    return tid.split(".", 1)[0]

def load_attack_tactics():
    """
    Loads tactics for each technique external_id (e.g., T1566.002 -> 'initial-access, …')
    by scanning the latest enterprise-attack-*.json in ENTERPRISE_DIR.
    Falls back to Data/attack_stix/enterprise-attack.json if present.
    """
    bundle = None
    # Pick the latest versioned file if available
    candidates = sorted(ATTACK_DIR.glob("enterprise-attack-*.json"), reverse=True)
    if candidates:
        bundle = candidates[0]
    else:
        fallback = ATTACK_DIR.parent / "enterprise-attack.json"
        if fallback.exists():
            bundle = fallback

    if bundle is None:
        return {}

    data = json.loads(bundle.read_text(encoding="utf-8"))
    t2tactic = defaultdict(set)  # technique external_id -> {tactic phases}
    for o in data.get("objects", []):
        if not isinstance(o, dict): 
            continue
        if o.get("type") != "attack-pattern": 
            continue

        # external ATT&CK ID (e.g., T1566.002)
        ext_id = None
        for ref in o.get("external_references", []) or []:
            if ref.get("source_name") == "mitre-attack":
                ext_id = ref.get("external_id")
                break
        if not ext_id:
            continue

        for ph in o.get("kill_chain_phases") or []:
            if ph.get("kill_chain_name") == "mitre-attack":
                t2tactic[ext_id].add(ph.get("phase_name"))

    return {k: ", ".join(sorted(v)) for k, v in t2tactic.items()}

def score_groups(observed_tids, group_edges):
    """
    - Direct match on technique_id (exact sub-technique included) = +2
    - Root-only match (T1059 == T1059.001) when no exact match = +1
    Returns dict: group_name -> dict(score, exact Counter, root Counter, ids...)
    """
    by_group = defaultdict(lambda: {
        "score": 0,
        "exact": Counter(),
        "root": Counter(),
        "group_id": None,
        "group_sid": None,
        "group_name": None
    })

    obs_exact = set(observed_tids)
    obs_roots = {technique_root(t) for t in observed_tids}

    for row in group_edges:
        gname = row["group_name"]
        gid   = row.get("group_id") or ""
        gsid  = row.get("group_sid") or ""
        tid   = row["technique_id"]
        troot = technique_root(tid)

        rec = by_group[gname]
        rec["group_id"]  = gid
        rec["group_sid"] = gsid
        rec["group_name"] = gname

        if tid in obs_exact:
            rec["score"] += 2
            rec["exact"][tid] += 1
        elif troot in obs_roots:
            rec["score"] += 1
            rec["root"][tid] += 1

    return {g: v for g, v in by_group.items() if v["score"] > 0}

def build_mini_stix_bundle(file_id, iocs, matched_attack_ids):
    import uuid, datetime as dt
    now = dt.datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    def sid(prefix):
        return f"{prefix}--{uuid.uuid4()}"

    objs = []

    for tid in sorted(matched_attack_ids):
        objs.append({
            "type": "x-attack-pattern-ref",
            "spec_version": "2.1",
            "id": sid("x-attack-pattern-ref"),
            "created": now,
            "modified": now,
            "name": tid,
            "external_references": [{
                "source_name": "mitre-attack",
                "external_id": tid
            }]
        })

    # Indicators from IOCs
    for kind, val in iocs:
        pattern = None
        if kind == "url":
            pattern = f"[url:value = '{val}']"
        elif kind == "domain":
            pattern = f"[domain-name:value = '{val}']"
        elif kind == "ipv4":
            pattern = f"[ipv4-addr:value = '{val}']"
        elif kind in ("md5", "sha1", "sha256"):
            algo = kind.upper()
            pattern = f"[file:hashes.'{algo}' = '{val}']"
        elif kind == "email":
            pattern = f"[email-addr:value = '{val}']"
        else:
            continue

        objs.append({
            "type": "indicator",
            "spec_version": "2.1",
            "id": sid("indicator"),
            "created": now,
            "modified": now,
            "name": f"{kind}:{val}",
            "pattern": pattern,
            "pattern_type": "stix",
            "valid_from": now,
            "labels": [kind, "extracted-from-report", file_id],
        })

    return {"type": "bundle", "id": sid("bundle"), "objects": objs}

def main():
    if not IOC_CSV.exists():
        print(f"Missing {IOC_CSV}")
        sys.exit(1)
    if not TI_EDGES.exists():
        print(f"Missing {TI_EDGES}")
        sys.exit(1)

    iocs  = read_csv(IOC_CSV)
    edges = read_csv(TI_EDGES)
    tactics_map = load_attack_tactics()

    by_file_attack = defaultdict(set)
    by_file_iocs   = defaultdict(list)

    for row in iocs:
        file_id = row["file"]
        kind = (row["kind"] or "").lower()
        val  = row["value"]
        if kind == "attack_id":
            norm = normalize_attack_id(val)
            if norm:
                by_file_attack[file_id].add(norm)
        else:
            by_file_iocs[file_id].append((kind, val))

    ranked_rows, detail_rows = [], []

    for file_id, tids in by_file_attack.items():
        if not tids:
            continue

        scores = score_groups(tids, edges)
        if not scores:
            ranked_rows.append({
                "file": file_id, "rank": "", "group_name": "(no match)",
                "group_id": "", "score": 0, "matched_exact": "", "matched_root_only": ""
            })
            continue

        scored_list = sorted(
            scores.values(),
            key=lambda r: (r["score"], sum(r["exact"].values())),
            reverse=True
        )

        # Detailed rows per group (with tactics text)
        for g in scored_list:
            matched_exact = sorted(g["exact"].keys())
            matched_root  = sorted(g["root"].keys())
            matched_exact_tactics = "; ".join(
                f"{t} ({tactics_map.get(t, '')})".strip() for t in matched_exact
            )
            matched_root_tactics = "; ".join(
                f"{t} ({tactics_map.get(t, '')})".strip() for t in matched_root
            )
            detail_rows.append({
                "file": file_id,
                "group_name": g["group_name"],
                "group_id": g["group_id"],
                "score": g["score"],
                "matched_exact": matched_exact_tactics,
                "matched_root_only": matched_root_tactics
            })

        # Top-N (ranked) rows
        for rank, g in enumerate(scored_list[:10], start=1):
            ranked_rows.append({
                "file": file_id,
                "rank": rank,
                "group_name": g["group_name"],
                "group_id": g["group_id"],
                "score": g["score"],
                "matched_exact": ", ".join(sorted(g["exact"].keys())),
                "matched_root_only": ", ".join(sorted(g["root"].keys()))
            })

        # Optional mini-bundle per file
        bundle = build_mini_stix_bundle(file_id, by_file_iocs[file_id], tids)
        (OUT_DIR / f"{file_id}-bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    # Write CSVs
    if ranked_rows:
        write_csv(
            OUT_DIR_CSV / "ranked_groups.csv",
            ranked_rows,
            ["file","rank","group_name","group_id","score","matched_exact","matched_root_only"]
        )
    if detail_rows:
        write_csv(
            OUT_DIR_CSV / "group_ttps_detail.csv",
            detail_rows,
            ["file","group_name","group_id","score","matched_exact","matched_root_only"]
        )
    total_files = len({r["file"] for r in iocs})
    with_attack = sum(1 for _ in by_file_attack if by_file_attack[_])
    print(f"[INFO] Files in IOC CSV: {total_files} | with attack_id: {with_attack} | without: {total_files - with_attack}")

    print(f"Done → {OUT_DIR_CSV.resolve()}")

if __name__ == "__main__":
    main()
