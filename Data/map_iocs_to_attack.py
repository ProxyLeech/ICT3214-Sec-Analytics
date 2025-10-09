import csv
import json
import re
import sys
from collections import defaultdict, Counter
from pathlib import Path

#Inputs:
# Data/extracted_pdfs/extracted_iocs.csv
#  columns: file,page,kind,value
#  kind includes: url, domain, ipv4, md5, sha1, sha256, email, attack_id

# Data/attack_stix/processed/ti_groups_techniques.csv
#  columns: group_sid,group_id,group_name,technique_id,technique_name,is_subtechnique

#Data/attack_stix/enterprise-attack/enterprise-attack-*.json (optional, to fetch tactics)

#Outputs:
# Data/mapped/ranked_groups.csv
# Data/mapped/group_ttps_detail.csv
# (optional) Data/mapped/<file>-bundle.json (mini STIX bundle)
IOC_CSV = Path("Data/extracted_pdfs/extracted_iocs.csv")
TI_EDGES = Path("Data/attack_stix/processed/ti_groups_techniques.csv")
ATTACK_DIR = Path("Data/attack_stix/enterprise-attack")
OUT_DIR = Path("Data/mapped")

OUT_DIR.mkdir(parents=True, exist_ok=True)


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
    bundle = None
    for p in sorted(ATTACK_DIR.glob("enterprise-attack-*.json"), reverse=True):
        bundle = p
        break
    if bundle is None:
        p = Path("Data/attack_stix/enterprise-attack.json")
        if p.exists():
            bundle = p
    if bundle is None:
        return {}

    data = json.loads(bundle.read_text(encoding="utf-8"))
    # map technique external_id to set(tactics)
    t2tactic = defaultdict(set)
    for o in data.get("objects", []):
        if not isinstance(o, dict): 
            continue
        if o.get("type") != "attack-pattern": 
            continue
        # external ATT&CK ID
        ext_id = None
        for ref in o.get("external_references", []) or []:
            if ref.get("source_name") == "mitre-attack":
                ext_id = ref.get("external_id")
                break
        if not ext_id:
            continue
        phases = o.get("kill_chain_phases") or []
        for ph in phases:
            if ph.get("kill_chain_name") == "mitre-attack":
                t2tactic[ext_id].add(ph.get("phase_name"))
    return {k: ", ".join(sorted(v)) for k, v in t2tactic.items()}

def score_groups(observed_tids, group_edges):
    """
      - direct match on technique_id (exact ID, including sub-techniques)
      - if no exact matches, also consider root technique matches (T1059 == T1059.001)
      - score = #exact*2 + #root_only*1
    Returns dict: group_name -> (score, counters, matched_exact, matched_root)
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
        gid = row.get("group_id") or ""
        gsid = row.get("group_sid") or ""
        tid = row["technique_id"]
        troot = technique_root(tid)

        rec = by_group[gname]
        rec["group_id"] = gid
        rec["group_sid"] = gsid
        rec["group_name"] = gname

        if tid in obs_exact:
            rec["score"] += 2
            rec["exact"][tid] += 1
        elif troot in obs_roots:
            rec["score"] += 1
            rec["root"][tid] += 1

    out = {g: v for g, v in by_group.items() if v["score"] > 0}
    return out

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

    iocs = read_csv(IOC_CSV)
    edges = read_csv(TI_EDGES)
    tactics_map = load_attack_tactics()  

    by_file_attack = defaultdict(set)
    by_file_iocs = defaultdict(list)

    for row in iocs:
        file_id = row["file"]     #shorten file name
        kind = row["kind"].lower()
        val  = row["value"]
        if kind == "attack_id":
            by_file_attack[file_id].add(val)
        else:
            by_file_iocs[file_id].append((kind, val))

    ranked_rows = []
    detail_rows = []

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

        # top-N ranking rows 
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

        bundle = build_mini_stix_bundle(file_id, by_file_iocs[file_id], tids)
        (OUT_DIR / f"{file_id}-bundle.json").write_text(json.dumps(bundle, indent=2), encoding="utf-8")

    # write CSVs
    if ranked_rows:
        write_csv(OUT_DIR / "ranked_groups.csv",
                  ranked_rows,
                  ["file","rank","group_name","group_id","score","matched_exact","matched_root_only"])
    if detail_rows:
        write_csv(OUT_DIR / "group_ttps_detail.csv",
                  detail_rows,
                  ["file","group_name","group_id","score","matched_exact","matched_root_only"])

    print(f"Done → {OUT_DIR.resolve()}")

if __name__ == "__main__":
    main()
