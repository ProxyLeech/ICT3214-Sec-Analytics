#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re, csv
from pathlib import Path
from typing import List, Dict, Tuple, Iterable
import sys
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


CURRENT_DIR = Path(__file__).resolve().parent
SRC_ROOT = CURRENT_DIR.parent  # goes up from models → src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
from paths.paths import MODELS_ROOT, PROCESSED_DIR
EXTRACTED_IOCS_CSV = PROCESSED_DIR / "extracted_iocs.csv"
DEFAULT_MODEL_DIR = MODELS_ROOT / "best_roberta_for_predict"  
DEFAULT_THRESHOLD = 0.5

ATTACK_ID_RX = re.compile(r"^T\d{4}(?:\.\d{1,3})?$", re.I)

def join_some(vals: List[str], n: int = 10) -> str:
    vals = [str(x).strip() for x in vals if str(x).strip()]
    return " ".join(vals[:n])

def build_text(
    report_id: str = "adhoc",
    urls: List[str] | None = None,
    domains: List[str] | None = None,
    ips: List[str] | None = None,
    md5s: List[str] | None = None,
    sha256s: List[str] | None = None,
    attack_ids: List[str] | None = None,
    free_text: str | None = None,
) -> str:
    urls, domains, ips = urls or [], domains or [], ips or []
    md5s, sha256s = md5s or [], sha256s or []
    attack_ids = [a.strip().upper() for a in (attack_ids or []) if ATTACK_ID_RX.match(a.strip())]
    parts = [f"Report:{report_id}"]
    if urls:    parts.append(f"URLs: {join_some(urls)}")
    if domains: parts.append(f"Domains: {join_some(domains)}")
    if ips:     parts.append(f"IPs: {join_some(ips)}")
    if md5s:    parts.append(f"MD5: {join_some(md5s)}")
    if sha256s: parts.append(f"SHA256: {join_some(sha256s)}")
    if attack_ids: parts.append(f"ATTACK: {join_some(attack_ids)}")
    if free_text:  parts.append(f"Notes: {free_text.strip()}")
    return " | ".join(parts)

def load_labels_from_dir(run_dir: Path) -> list[str]:
    id2label_path = run_dir / "id2label.json"
    if not id2label_path.exists():
        raise FileNotFoundError(f"Missing {id2label_path}")
    id2label = json.loads(id2label_path.read_text(encoding="utf-8"))
    return [id2label[str(i)] for i in range(len(id2label))]

def load_attack_index() -> List[dict]:
    rows: List[dict] = []
    path = PROCESSED_DIR / "ti_groups_techniques.csv"
    with path.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append(r)
    return rows

def predict(
    text: str,
    threshold: float = DEFAULT_THRESHOLD,
    top_k: int | None = 10,
    device: str | None = None,
) -> list[tuple[str, float]]:
    labels = load_labels_from_dir(DEFAULT_MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL_DIR, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(DEFAULT_MODEL_DIR)


    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    enc = tokenizer(text, truncation=True, max_length=384, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.sigmoid(logits).cpu().numpy()[0]  # multi-label

    # gather predictions
    indexed = list(enumerate(probs))
    # filter by threshold first
    above = [(labels[i], float(p)) for i, p in indexed if p >= threshold]
    # if nothing above threshold, show best top_k anyway
    if not above:
        best = sorted(indexed, key=lambda x: x[1], reverse=True)
        if top_k is not None:
            best = best[:top_k]
        above = [(labels[i], float(p)) for i, p in best]

    # sort descending by score
    above.sort(key=lambda x: x[1], reverse=True)
    if top_k is not None:
        above = above[:top_k]
    return above

def expand_to_attack_rows(
    predicted: Iterable[Tuple[str, float]],
    attack_rows: List[dict],
    attack_ids_in_input: set[str] | None = None,
    origin_doc: str | None = None,
) -> List[dict]:
    """
    For each predicted label:
      - If label is a technique (T####(.###)), emit mappings for that technique.
        origin includes 'model_pred' and also 'attack_input' if that technique
        was present in the user's ATT&CK inputs.
      - If label is a group name, emit that group's techniques.
        origin is 'group_pred'.
    Also carries 'doc_source' (provenance of the input doc/PDF) on each row.
    """
    attack_ids_in_input = attack_ids_in_input or set()

    by_tid: Dict[str, List[dict]] = {}
    by_group: Dict[str, List[dict]] = {}
    for r in attack_rows:
        by_tid.setdefault(r["technique_id"], []).append(r)
        by_group.setdefault(r["group_name"], []).append(r)

    out: List[dict] = []
    for label, score in predicted:
        if ATTACK_ID_RX.match(label or ""):  # technique predicted
            # provenance of output
            src_bits = ["model_pred"]
            if label.upper() in attack_ids_in_input:
                src_bits.append("attack_input")
            origin = "+".join(sorted(set(src_bits)))
            for r in by_tid.get(label, []):
                out.append({
                    "group_id": r.get("group_id") or "",
                    "group_name": r.get("group_name") or "",
                    "technique_id": r.get("technique_id") or "",
                    "technique_name": r.get("technique_name") or "",
                    "score": round(float(score), 3),
                    "origin": origin,
                    "doc_source": origin_doc or "",
                })
        else:  # group predicted
            origin = "group_pred"
            for r in by_group.get(label, []):
                out.append({
                    "group_id": r.get("group_id") or "",
                    "group_name": r.get("group_name") or "",
                    "technique_id": r.get("technique_id") or "",
                    "technique_name": r.get("technique_name") or "",
                    "score": round(float(score), 3),
                    "origin": origin,
                    "doc_source": origin_doc or "",
                })

    # de-dup by (group, technique), keep best score
    dedup: Dict[Tuple[str, str, str], dict] = {}
    for r in out:
        k = (r["group_id"], r["group_name"], r["technique_id"])
        if k not in dedup or r["score"] > dedup[k]["score"]:
            dedup[k] = r
    return list(dedup.values())

def aggregate_by_group(rows: List[dict]) -> List[dict]:
    """
    Collapse to one row per group.
    - technique_id_list / technique_name_list
    - group_score: max(score)
    - origin: joined unique output provenance tags (attack_input, model_pred, group_pred)
    - doc_source: input document provenance (from resolve_sources... or --id)
    """
    agg: Dict[Tuple[str, str], dict] = {}
    for r in rows:
        gkey = (r["group_id"], r["group_name"])
        if gkey not in agg:
            agg[gkey] = {
                "group_id": r["group_id"],
                "group_name": r["group_name"],
                "technique_id_list": [],
                "technique_name_list": [],
                "group_score": 0.0,
                "origin_set": set(),
                "doc_source": r.get("doc_source", ""),
            }
        agg[gkey]["technique_id_list"].append(r["technique_id"])
        agg[gkey]["technique_name_list"].append(r["technique_name"])
        agg[gkey]["group_score"] = max(agg[gkey]["group_score"], float(r["score"]))
        if r.get("origin"):
            agg[gkey]["origin_set"].add(r["origin"])
        # keep first non-empty doc_source
        if not agg[gkey]["doc_source"] and r.get("doc_source"):
            agg[gkey]["doc_source"] = r["doc_source"]

    out: List[dict] = []
    for _, rec in agg.items():
        paired = sorted(zip(rec["technique_id_list"], rec["technique_name_list"]), key=lambda x: (x[0], x[1]))
        tids   = [p[0] for p in paired]
        tnames = [p[1] for p in paired]
        out.append({
            "group_id": rec["group_id"],
            "group_name": rec["group_name"],
            "technique_id_list": ", ".join(dict.fromkeys(tids)),
            "technique_name_list": ", ".join(dict.fromkeys(tnames)),
            "group_score": round(rec["group_score"], 3),
            "origin": "+".join(sorted(rec["origin_set"])) if rec["origin_set"] else "",
            "doc_source": rec["doc_source"],
        })
    out.sort(key=lambda r: (-r["group_score"], r["group_name"]))
    return out

def print_group_table(groups: List[dict]) -> None:
    if not groups:
        print("No ATT&CK mappings matched your predictions.")
        return
    print("\nPREDICTED GROUPS (one row per group)")
    print("-" * 160)
    print(f"{'group_id':10s}  {'group_name':20s}  {'group_score':11s}  {'origin':20s}  {'doc_source':30s}  {'technique_id_list':s}")
    print("-" * 160)
    for g in groups:
        left = (
            f"{g['group_id']:10s}  "
            f"{g['group_name'][:20]:20s}  "
            f"{g['group_score']:<11.3f}  "
            f"{(g.get('origin') or '-')[:20]:20s}  "
            f"{(g.get('doc_source') or '-')[:30]:30s}  "
        )
        print(left + (g.get("technique_id_list") or ""))
    print("\n(technique_name_list)")
    for g in groups:
        print(f"- {g['group_name']}: {g.get('technique_name_list','')}")


def resolve_sources_from_inputs(
    urls: List[str], domains: List[str], ips: List[str], md5s: List[str], sha256s: List[str], attacks: List[str]
) -> List[str]:
    want: Dict[str, set[str]] = {
        "url": set(x.strip() for x in (urls or []) if x.strip()),
        "domain": set(x.strip().lower() for x in (domains or []) if x.strip()),
        "ipv4": set(x.strip() for x in (ips or []) if x.strip()),
        "md5": set(x.strip().lower() for x in (md5s or []) if x.strip()),
        "sha256": set(x.strip().lower() for x in (sha256s or []) if x.strip()),
        "attack_id": set(a.strip().upper() for a in (attacks or []) if ATTACK_ID_RX.match(a or "")),
    }
    hits: set[str] = set()
    if not EXTRACTED_IOCS_CSV.exists():
        return []

    with EXTRACTED_IOCS_CSV.open("r", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            kind = (r.get("kind") or "").strip().lower()
            val  = (r.get("value") or "").strip()
            file_folder = (r.get("file") or "").strip()  # this is the extracted folder name
            if not kind or not val or not file_folder:
                continue
            # normalize to match how we collected targets above
            if kind in ("domain", "md5", "sha256"):
                key = val.lower()
            elif kind == "attack_id":
                key = val.upper()
            else:
                key = val

            if key in want.get(kind, set()):
                hits.add(file_folder)

    # return a sorted, stable list of the matching extracted PDF folders
    return sorted(hits)

def main():
    p = argparse.ArgumentParser(description="Predict and expand to ATT&CK rows.")
    p.add_argument("--id", default="adhoc")
    # p.add_argument("--source", help="Source label (e.g., PDF filename). Defaults to --id.")
    p.add_argument("--text")
    p.add_argument("--url", action="append", default=[])
    p.add_argument("--domain", action="append", default=[])
    p.add_argument("--ip", action="append", default=[])
    p.add_argument("--md5", action="append", default=[])
    p.add_argument("--sha256", action="append", default=[])
    p.add_argument("--attack", action="append", default=[])
    p.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    p.add_argument("--top-k", type=int, default=10)
    args = p.parse_args()


    text = build_text(
        report_id=args.id,
        urls=args.url,
        domains=args.domain,
        ips=args.ip,
        md5s=args.md5,
        sha256s=args.sha256,
        attack_ids=args.attack,
        free_text=args.text,
    )

    # Infer document provenance (PDF folders or fallback to id)
    sources = resolve_sources_from_inputs(args.url, args.domain, args.ip, args.md5, args.sha256, args.attack)
    doc_src = ", ".join(sources) if sources else args.id

    # Run model
    preds = predict(text=text, threshold=args.threshold, top_k=args.top_k)

    # Expand + aggregate (now with output origin + doc_source)
    attack_rows = load_attack_index()
    attack_ids_from_input = {a.strip().upper() for a in args.attack if ATTACK_ID_RX.match(a or "")}
    flat_rows = expand_to_attack_rows(preds, attack_rows, attack_ids_in_input=attack_ids_from_input, origin_doc=doc_src)
    group_rows = aggregate_by_group(flat_rows)

    print("\nINPUT TEXT:")
    print(text)
    print_group_table(group_rows)


if __name__ == "__main__":
    main()