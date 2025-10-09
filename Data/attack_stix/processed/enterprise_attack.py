import json, csv, pathlib, re

ATTACK_DIR   = pathlib.Path("Data/attack_stix")
BUNDLES_DIR  = ATTACK_DIR / "enterprise-attack"
INDEX        = ATTACK_DIR / "index.json"

def latest_local_bundle():
    #Use the latest file
    candidates = list(BUNDLES_DIR.glob("enterprise-attack-*.json"))
    if not candidates:
        # fallback to unversioned file 
        fallback = ATTACK_DIR / "enterprise-attack.json"
        return fallback if fallback.exists() else None

    def vernum(p: pathlib.Path):
        m = re.search(r"(\d+)\.(\d+)", p.name)  # e.g., 17.1
        return (int(m.group(1)), int(m.group(2))) if m else (0, 0)

    return sorted(candidates, key=vernum, reverse=True)[0]

def find_bundle():
    path = latest_local_bundle()
    if path:
        return path
    if INDEX.exists():
        idx = json.loads(INDEX.read_text(encoding="utf-8"))
        entries = idx.get("objects") or idx.get("collections") or []
        for e in entries:
            if "enterprise-attack" in (e.get("name","") or e.get("id","")):
                local = e.get("path") or e.get("file") or "enterprise-attack/enterprise-attack.json"
                candidate = ATTACK_DIR / local
                if candidate.exists():
                    return candidate
    raise FileNotFoundError("No ATT&CK bundle found. Put files in Data/attack_stix/enterprise-attack/")

BUNDLE_PATH = find_bundle()
print(f"Using bundle: {BUNDLE_PATH}")

data = json.loads(BUNDLE_PATH.read_text(encoding="utf-8"))
objs = {o["id"]: o for o in data.get("objects", []) if isinstance(o, dict)}

# Index intrusion-sets (groups)
groups = [o for o in objs.values()
          if o.get("type")=="intrusion-set"
          and not o.get("revoked")
          and not o.get("x_mitre_deprecated")]

def group_row(g):
    name = g.get("name")
    gid  = next((ref.get("external_id") for ref in g.get("external_references", [])
                 if ref.get("source_name")=="mitre-attack"), None)
    return g["id"], gid, name

# Index attack-patterns (techniques & sub-techniques)
techs = [o for o in objs.values()
         if o.get("type")=="attack-pattern"
         and not o.get("revoked")
         and not o.get("x_mitre_deprecated")]

tech_by_id   = {t["id"]: t for t in techs}
techid_by_id = {
    t["id"]: next((ref.get("external_id") for ref in t.get("external_references", [])
                   if ref.get("source_name")=="mitre-attack"), None)
    for t in techs
}

# Traverse relationships: intrusion-set --uses--> attack-pattern
edges = [o for o in objs.values()
         if o.get("type")=="relationship" and o.get("relationship_type")=="uses"]

rows = []
for r in edges:
    src, tgt = r.get("source_ref"), r.get("target_ref")
    if src in objs and tgt in tech_by_id and objs[src]["type"]=="intrusion-set":
        g_sid, g_external, g_name = group_row(objs[src])
        t_tid  = techid_by_id.get(tgt)          
        t_name = tech_by_id[tgt].get("name")
        is_sub = tech_by_id[tgt].get("x_mitre_is_subtechnique", False)
        if g_name and t_tid:
            rows.append({
                "group_sid": g_sid,
                "group_id": g_external,
                "group_name": g_name,
                "technique_id": t_tid,
                "technique_name": t_name,
                "is_subtechnique": is_sub
            })

# 5) Write CSV 
OUT_DIR = ATTACK_DIR / "processed"
OUT_DIR.mkdir(parents=True, exist_ok=True)
out = OUT_DIR / "ti_groups_techniques.csv"
with out.open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    writer.writeheader(); writer.writerows(rows)

print(f"Wrote {out} with {len(rows)} rows")
