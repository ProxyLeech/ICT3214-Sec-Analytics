
"""
Parses the local MITRE ATT&CK Enterprise bundle and writes a mapping:
  intrusion-set (group)  --uses-->  attack-pattern (technique/sub-technique)

Output:
  <DATA_ROOT>/attack_stix/processed/ti_groups_techniques.csv

Paths are repo-relative and environment-overridable via common.paths:
  - DATA_DIR env var can redirect <DATA_ROOT> if you want data elsewhere.
"""

from __future__ import annotations
import csv
import json
import pathlib
import re
from typing import Dict, List, Tuple
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))
from project_paths import (
    PROJECT_ROOT, DATA_ROOT, EXPERIMENTS_ROOT, SRC_ROOT, MODELS_ROOT, EXPERIMENTS_ROOT,SCRIPTS_DIR,
    RAW_DIR, PROCESSED_DIR, EXTRACTED_PDFS_DIR,
    MAPPED_DIR, EXCEL_DIR, MITIGATIONS_DIR,
    ATTACK_STIX_DIR,PDFS_DIR,RULES_DIR,EXTRACT_SCRIPT,ATTACK_SCRIPT,MAP_IOCS_SCRIPT,
    BUILD_DATASET_SCRIPT,MITIGATIONS_SCRIPT,
    GROUP_TTPS_DETAIL_CSV,MATCHING_SCRIPT,REPORT_GENERATION_SCRIPT,TECHNIQUE_LABELS_SCRIPT,
    TRAIN_ROBERTA_SCRIPT,PREDICT_SCRIPT,BEST_MODEL_DIR,
    MAPPING_CSV,MITIGATIONS_CSV,EXCEL_ATTACK_TECHS,
    EXTRACTED_IOCS_CSV,TI_GROUPS_TECHS_CSV,DATASET_CSV,LABELS_TXT,GROUP_TTPS_DETAIL_CSV,RANKED_GROUPS_CSV,
    output_dir_for_folds, project_path,ensure_dir_tree,add_src_to_syspath
)
INDEX_JSON   = ATTACK_STIX_DIR / "index.json"
OUT_CSV      = TI_GROUPS_TECHS_CSV


def _latest_local_bundle() -> pathlib.Path | None:
    candidates = list(ATTACK_STIX_DIR.glob("enterprise-attack-*.json"))
    if not candidates:
        fallback = ATTACK_STIX_DIR / "enterprise-attack.json"
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
                    candidate = ATTACK_STIX_DIR / local
                    if candidate.exists():
                        return candidate
        except Exception:
            pass

    raise FileNotFoundError(
        "No ATT&CK bundle found.\n"
        f"Expected under: {ATTACK_STIX_DIR} (e.g., enterprise-attack-<ver>.json)\n"
        f"or fallback: {ATTACK_STIX_DIR / 'enterprise-attack.json'}\n"
        f"Optional index file: {INDEX_JSON}"
    )


def _group_row(g: Dict) -> Tuple[str, str | None, str | None]:
    name = g.get("name")
    gid = next(
        (ref.get("external_id")
         for ref in g.get("external_references", [])
         if ref.get("source_name") == "mitre-attack"),
        None,
    )
    return g["id"], gid, name


def main() -> None:
    bundle_path = _find_bundle()
    print(f"[INFO] Using ATT&CK bundle: {bundle_path}")

    data = json.loads(bundle_path.read_text(encoding="utf-8"))
    objects = [o for o in data.get("objects", []) if isinstance(o, dict)]
    objs: Dict[str, Dict] = {o["id"]: o for o in objects if "id" in o}

    # Intrusion sets (groups)
    groups = [
        o for o in objs.values()
        if o.get("type") == "intrusion-set"
        and not o.get("revoked")
        and not o.get("x_mitre_deprecated")
    ]

    # Attack patterns (techniques & sub-techniques)
    techs = [
        o for o in objs.values()
        if o.get("type") == "attack-pattern"
        and not o.get("revoked")
        and not o.get("x_mitre_deprecated")
    ]
    tech_by_id = {t["id"]: t for t in techs}

    # Map STIX attack-pattern -> MITRE technique ID (e.g., T1566.002)
    techid_by_id: Dict[str, str | None] = {
        t["id"]: next(
            (ref.get("external_id")
             for ref in t.get("external_references", [])
             if ref.get("source_name") == "mitre-attack"),
            None,
        )
        for t in techs
    }

    # Relationships: intrusion-set --uses--> attack-pattern
    edges = [
        o for o in objs.values()
        if o.get("type") == "relationship"
        and o.get("relationship_type") == "uses"
    ]

    rows: List[Dict[str, object]] = []
    for r in edges:
        src, tgt = r.get("source_ref"), r.get("target_ref")
        if src in objs and tgt in tech_by_id and objs[src].get("type") == "intrusion-set":
            g_sid, g_external, g_name = _group_row(objs[src])
            t_tid = techid_by_id.get(tgt)
            t_obj = tech_by_id[tgt]
            t_name = t_obj.get("name")
            is_sub = bool(t_obj.get("x_mitre_is_subtechnique", False))

            if g_name and t_tid:
                rows.append(
                    {
                        "group_sid": g_sid,
                        "group_id": g_external,
                        "group_name": g_name,
                        "technique_id": t_tid,
                        "technique_name": t_name,
                        "is_subtechnique": is_sub,
                    }
                )

    # Write CSV
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError("No intrusion-set → technique relationships found to write.")

    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote {OUT_CSV} with {len(rows)} rows")


if __name__ == "__main__":
    main()
