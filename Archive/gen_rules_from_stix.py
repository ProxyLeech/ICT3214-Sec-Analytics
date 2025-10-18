#!/usr/bin/env python3
import json, re
import pathlib
from typing import Dict, List, Tuple

from paths.paths import (
    PROJECT_ROOT, DATA_ROOT, ATTACK_STIX_DIR,  PROCESSED_DIR,RULES_DIR,
)
BUNDLES_DIR = ATTACK_STIX_DIR / "enterprise-attack"  
OUT = RULES_DIR / "attack_rules_auto.json"
INDEX_JSON   = ATTACK_STIX_DIR / "index.json"
ATTACK_DIR   = DATA_ROOT / "attack_stix"

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
        rules.append({
            "when": {"kind": "*", "regex": [rx]},
            "add_techniques": [ext]
        })
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rules, indent=2), encoding="utf-8")
    print(f"[OK] wrote {OUT} with {len(rules)} rules")

if __name__ == "__main__":
    main()
