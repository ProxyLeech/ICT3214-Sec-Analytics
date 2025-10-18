from flask import Flask, render_template, request, make_response, jsonify
import pandas as pd

from datetime import datetime
from collections import defaultdict
import importlib.util
import io
import re
from typing import Iterable
from pathlib import Path
import subprocess, sys
import os, tempfile
import json
# =======================================================
# Project-relative paths
# =======================================================
from project_paths import (
    PROJECT_ROOT, DATA_ROOT, EXPERIMENTS_ROOT, SRC_ROOT, MODELS_ROOT,SCRIPTS_DIR,
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
app = Flask(__name__)
sys.path.insert(0, str(SRC_ROOT))  

# from scripts.technique_labels import extract_techniques  # import the extractor
# from scripts.report_generator import (
#     analyze_TTP,
#     parse_ai_response,
#     generate_word_report,
#     load_filtered_mitigations,
#     summarize_mitigations
# )

def _import_from_path(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

# load the three script modules by file path
matching           = _import_from_path(MATCHING_SCRIPT, "matching")
technique_labels   = _import_from_path(TECHNIQUE_LABELS_SCRIPT, "technique_labels")
report_generator   = _import_from_path(REPORT_GENERATION_SCRIPT, "report_generator")

# expose the functions you need
validate_ttps         = matching.validate_ttps
match_ttps            = matching.match_ttps

extract_techniques    = technique_labels.extract_techniques

analyze_TTP           = report_generator.analyze_TTP
parse_ai_response     = report_generator.parse_ai_response
generate_word_report  = report_generator.generate_word_report
load_filtered_mitigations = report_generator.load_filtered_mitigations
summarize_mitigations     = report_generator.summarize_mitigations

# Cache
LAST_RESULTS = {} #Global
LAST_RESULTS_RULE = {}
LAST_RESULTS_ROBERTA = {}

def _ensure_score_and_rank_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has numeric 'score' and 'rank' columns.
    """
    import numpy as np

    if df is None or df.empty:
        return df

    candidates = ["score", "prob", "probability", "confidence", "logit", "logprob"]
    src = next((c for c in candidates if c in df.columns), None)

    if src is None:
        if "rank" in df.columns:
            df["score"] = pd.to_numeric(df["rank"], errors="coerce")
            df["score"] = 1.0 / (1.0 + df["score"].fillna(df["score"].max() or 1))
        else:
            n = len(df)
            df["score"] = np.linspace(1.0, 0.0, n, endpoint=False)
    else:
        df["score"] = pd.to_numeric(df[src], errors="coerce").fillna(0.0)

    if "rank" not in df.columns or df["rank"].isna().all():
        df["rank"] = (-df["score"]).rank(method="first").astype(int)

    return df
#For ROBERTA
def _ensure_score_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    if df is None or df.empty:
        return df

    # include group_score here
    candidates = ["score", "group_score", "prob", "probability", "confidence", "logit", "logprob"]
    src = next((c for c in candidates if c in df.columns), None)

    if src is None:
        if "rank" in df.columns:
            df["score"] = pd.to_numeric(df["rank"], errors="coerce")
            df["score"] = 1.0 / (1.0 + df["score"].fillna(df["score"].max() or 1))
        else:
            n = len(df)
            df["score"] = np.linspace(1.0, 0.0, n, endpoint=False)
    else:
        df["score"] = pd.to_numeric(df[src], errors="coerce").fillna(0.0)

    if "rank" not in df.columns or df["rank"].isna().all():
        df["rank"] = (-df["score"]).rank(method="first").astype(int)

    return df

def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(map(str, cmd))}")
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if res.returncode != 0:
        raise SystemExit(res.returncode)

def needs_run(outputs: Iterable[Path], inputs: Iterable[Path] = ()) -> bool:
    outs = list(outputs)
    if not outs or any(not p.exists() for p in outs):
        return True  # missing outputs => run

    # If any input (or the script itself) is newer than any output => run
    out_mtime = min(p.stat().st_mtime for p in outs)
    ins = [p for p in inputs if p is not None and Path(p).exists()]
    if not ins:
        return False
    return max(Path(p).stat().st_mtime for p in ins) > out_mtime



def _atomic_to_csv(df, path: str):
    d = Path(path).parent
    d.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=d, newline="", suffix=".tmp") as tmp:
        tmp_name = tmp.name
        df.to_csv(tmp, index=False)
    os.replace(tmp_name, path)  # atomic on POSIX & Windows

#For mitigations
def _collect_top_group_ttps(df_ml: pd.DataFrame) -> list[str]:
    """
    Return technique IDs ONLY for the top (first/highest-score) matched group
    by looking them up in Data/mapped/group_ttps_detail.csv.
    """
    map_path = GROUP_TTPS_DETAIL_CSV
    if df_ml is None or df_ml.empty or not map_path.exists():
        return []

    df = df_ml.copy()
    # Prefer highest score when present
    if "score" in df.columns:
        df = df.sort_values("score", ascending=False)
    top = df.iloc[0]

    # find group name/id columns robustly
    group_name = None
    for c in ("group_name", "group", "actor", "name"):
        if c in df.columns and pd.notna(top[c]):
            group_name = str(top[c]).strip().lower()
            break

    group_id = None
    for c in ("group_id", "id", "mitre_id"):
        if c in df.columns and pd.notna(top[c]):
            group_id = str(top[c]).strip().lower()
            break
    try:
            g = pd.read_csv(map_path)
    except Exception:
            return []

    # normalize columns + values
    g.columns = [c.strip().lower() for c in g.columns]
    for col in ("group_id", "group_name", "aliases", "matched_exact", "matched_root_only",
                "ttp_list", "techniques", "technique_ids"):
        if col in g.columns:
            g[col] = g[col].astype(str).fillna("").str.strip()

    # --------- primary: exact match on id or name ----------
    sel = pd.Series([True] * len(g))
    matched = pd.Series([True] * len(g))

    if group_id and "group_id" in g.columns:
        matched &= g["group_id"].str.lower() == group_id
    elif group_name and "group_name" in g.columns:
        matched &= g["group_name"].str.lower() == group_name
    else:
        matched &= False  # force empty

    gsel = g[matched]
    # --------- secondary: alias/contains fallback ----------
    if gsel.empty:
        if group_name:
            alias_mask = pd.Series([False] * len(g))
            if "aliases" in g.columns:
                alias_mask |= g["aliases"].str.lower().str.contains(group_name, na=False)
            if "group_name" in g.columns:
                alias_mask |= g["group_name"].str.lower().str.contains(group_name, na=False)
            gsel = g[alias_mask]

    if gsel.empty:
        return []

    # safe copy
    gsel = gsel.copy()

    import re
    id_re = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)

    ttps = []
    for col in ("matched_exact", "matched_root_only", "ttp_list", "techniques", "technique_ids"):
        if col in gsel.columns:
            gsel.loc[:, col] = gsel[col].fillna("")
            for entry in gsel[col].tolist():
                ttps.extend([m.upper() for m in id_re.findall(entry)])

    # dedupe + numeric sort
    def _key(tid: str):
        m = re.match(r"T(\d{4})(?:\.(\d{3}))?$", tid)
        return (int(m.group(1)), int(m.group(2) or 999)) if m else (9999, 999)
    return sorted(set(ttps), key=_key)
# =========================
# STRICT group → TTP lookup
# =========================
def _collect_group_ttps(matched_df: pd.DataFrame) -> list[str]:
    """
    Extract unique MITRE technique IDs (e.g. T1110, T1110.003) from
    Data/mapped/group_ttps_detail.csv for the matched groups.
    Falls back gracefully if columns differ between datasets.
    """
    map_path = GROUP_TTPS_DETAIL_CSV
    if not map_path.exists():
        print(f"[ERROR] {map_path} not found.")
        return []

    try:
        g = pd.read_csv(map_path)
    except Exception as e:
        print(f"[ERROR] Failed reading {map_path}: {e}")
        return []

    # Validate required minimal columns
    expected_cols = {"group_name", "group_id", "matched_exact", "matched_root_only"}
    missing = expected_cols - set(g.columns.str.lower())
    if missing:
        print(f"[WARN] group_ttps_detail.csv missing columns: {missing}; using best-effort extraction.")

    # Normalization helpers
    import re
    id_re = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)

    def _extract_ttps(text: str) -> list[str]:
        if not isinstance(text, str):
            return []
        found = id_re.findall(text)
        return [f"T{f[1:]}" if not f.startswith("T") else f.upper() for f in found]

    # Collect TTPs from relevant columns
    all_ttps = []
    for col in ["matched_exact", "matched_root_only"]:
        if col in g.columns:
            g[col] = g[col].fillna("")
            for entry in g[col].tolist():
                all_ttps.extend(_extract_ttps(entry))

    # Remove duplicates + sort by numeric order
    def _sort_key(tid: str):
        m = re.match(r"T(\d{4})(?:\.(\d{3}))?", tid)
        return (int(m.group(1)), int(m.group(2) or 999)) if m else (9999, 999)

    uniq_ttps = sorted(set(all_ttps), key=_sort_key)
    print(f"[DEBUG] Extracted {len(uniq_ttps)} unique technique IDs from group_ttps_detail.csv")
    return uniq_ttps


# ============================================
# Rule-based flow helper
# ============================================
def _run_rule_match_flow(ttps: list[str]) -> dict:
    matched_df = match_ttps(ttps, MAPPED_DIR).copy()
    matched_df = _ensure_score_and_rank_rule(matched_df)
    if "rank" in matched_df.columns and matched_df["rank"].notna().any():
        matched_df = matched_df.sort_values(by=["rank", "score"], ascending=[True, False])
    else:
        matched_df = matched_df.sort_values(by="score", ascending=False)
    top3_df = matched_df.head(3)

    matched_df.to_csv("matched_groups_rule.csv", index=False)
    pd.DataFrame({"TTP": ttps}).to_csv("inputted_ttps.csv", index=False)
    # Mitigations (idempotent) + GPT analysis
    mit_csv_path = MITIGATIONS_CSV
    gpt_response = analyze_TTP(ttps, matched_df, mitigations_csv=str(mit_csv_path))
    parsed = parse_ai_response(gpt_response)

    # --- NEW: limit mitigations to the TOP group only ---
    top_group_ttps = _collect_group_ttps(matched_df)
    if not top_group_ttps:
        print("[INFO] No top-group TTPs resolved for RULES; skipping mitigations (no group mapping).")


    mit_filtered = load_filtered_mitigations(str(mit_csv_path), top_group_ttps)

    # Remove duplicate mitigation descriptions
    if not mit_filtered.empty:
        mit_filtered = mit_filtered.drop_duplicates(
            subset=["target id", "target name", "mapping description"], keep="first"
        )
        # summarize just the top group’s mitigations for the HTML view
        parsed["mitigation"] = summarize_mitigations(mit_filtered.to_dict(orient="records"))
        # write a scoped CSV and pass it to the Word generator
        _atomic_to_csv(mit_filtered, "mitigations_rule_top.csv")
        mit_for_docx = "mitigations_rule_top.csv"
    else:
        parsed["mitigation"] = "No mitigations found for these techniques."
        mit_for_docx = None

    # Generate DOCX (non-fatal)
    try:
        out_path = generate_word_report(
            gpt_response,
            ttps,
            mitigations_csv=mit_for_docx  # <-- pass the scoped CSV
        )
        if not out_path:
            out_path = "threat_report_rule.docx"
    except Exception as e:
        print("[WARN] Rule-based DOCX generation failed:", e)
        out_path = None
    
    return {
        "ttps": ttps,
        "matched_full_df": matched_df,
        "matched_top3": top3_df.to_dict(orient="records"),
        "analysis": parsed,
        "doc_path": out_path,
    }

# ============================================
# RoBERTa flow helper
# ============================================
def _save_roberta_traces(df_ml):
    _atomic_to_csv(df_ml, "matched_groups_roberta.csv")

def _predict_with_module(payload: dict):
    """
    Try importing predict_roberta.py and using its functions directly.
    Fallback to subprocess if import fails.
    """
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("predict_roberta", str(PREDICT_SCRIPT))
        pr = importlib.util.module_from_spec(spec)
        assert spec and spec.loader
        spec.loader.exec_module(pr)  # type: ignore

        text = pr.build_text(
            report_id=payload.get("id") or "adhoc",
            urls=payload.get("urls") or [],
            domains=payload.get("domains") or [],
            ips=payload.get("ips") or [],
            md5s=payload.get("md5s") or [],
            sha256s=payload.get("sha256s") or [],
            attack_ids=payload.get("attacks") or [],
            free_text=payload.get("text") or None,
        )
        preds = pr.predict(text=text, threshold=float(payload.get("threshold", 0.5)), top_k=int(payload.get("top_k", 10)))
        attack_rows = pr.load_attack_index()

        srcs = pr.resolve_sources_from_inputs(
            payload.get("urls") or [], payload.get("domains") or [], payload.get("ips") or [],
            payload.get("md5s") or [], payload.get("sha256s") or [], payload.get("attacks") or []
        )
        origin_doc = ", ".join(srcs) if srcs else (payload.get("id") or "adhoc")

        attack_ids_in_input = {a.strip().upper() for a in (payload.get("attacks") or []) if a}

        flat = pr.expand_to_attack_rows(
            preds,
            attack_rows,
            attack_ids_in_input=attack_ids_in_input,
            origin_doc=origin_doc,
        )
        group_rows = pr.aggregate_by_group(flat)

        return {"text": text, "groups": group_rows}

    except Exception as e:
        print(f"[WARN] Direct import predict_roberta failed: {e}; falling back to subprocess.")
        args = [
            sys.executable, str(PREDICT_SCRIPT),
            "--id", payload.get("id") or "adhoc",
        ]
        for k, flag in [
            ("urls", "--url"), ("domains", "--domain"), ("ips", "--ip"),
            ("md5s", "--md5"), ("sha256s", "--sha256"), ("attacks", "--attack")
        ]:
            for v in payload.get(k) or []:
                args += [flag, str(v)]
        if payload.get("text"):
            args += ["--text", payload["text"]]
        args += ["--threshold", str(payload.get("threshold", 0.5)),
                 "--top-k", str(payload.get("top_k", 10))]

        res = subprocess.run(args, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout)
            print(res.stderr)
            raise RuntimeError(res.stderr or "predict_roberta failed")

        # NEW: try to parse JSON from CLI and normalize the shape
        try:
            payload = json.loads(res.stdout)
            if isinstance(payload, dict) and "groups" in payload:
                return payload
            # allow older CLIs: maybe the top-level is a list
            if isinstance(payload, list):
                return {"groups": payload}
            # last resort: empty groups
            return {"groups": []}
        except Exception:
            print("[WARN] CLI output was not JSON; returning empty groups.")
            return {"groups": []}
        
def _run_roberta_flow(ttps: list[str]) -> dict:
    ml = _predict_with_module({
        "id": "from_ttps",
        "attacks": ttps,
        "top_k": 50,
        "threshold": 0.0
    })
    group_rows = ml.get("groups", [])
    if not group_rows:
        raise RuntimeError("RoBERTa returned no groups. Check model artifacts and labels.")
    df_ml = pd.DataFrame(group_rows)

    if "group" in df_ml.columns and "group_name" not in df_ml.columns:
        df_ml.rename(columns={"group": "group_name"}, inplace=True)

    df_ml = _ensure_score_and_rank(df_ml)
    df_ml.sort_values("score", ascending=False, inplace=True)
    # after df_ml.sort_values("score", ascending=False, inplace=True)
    # Persist files for the report script
    _save_roberta_traces(df_ml)
    top3_df = df_ml.head(3).drop(columns=["origin"], errors="ignore")

        # Mitigations (idempotent) + GPT analysis
    mit_csv_path = MITIGATIONS_CSV
    gpt_response = analyze_TTP(ttps, df_ml, mitigations_csv=str(mit_csv_path))
    parsed = parse_ai_response(gpt_response)

    top_group_ttps = _collect_top_group_ttps(df_ml)
    if not top_group_ttps:
        print("[INFO] No top-group TTPs resolved for ROBERTA; skipping mitigations (no group mapping).")


    mit_filtered = load_filtered_mitigations(str(mit_csv_path), top_group_ttps)

    if not mit_filtered.empty:
        mit_filtered = mit_filtered.drop_duplicates(
            subset=["target id", "target name", "mapping description"], keep="first"
        )
        parsed["mitigation"] = summarize_mitigations(mit_filtered.to_dict(orient="records"))
        _atomic_to_csv(mit_filtered, "mitigations_roberta_top.csv")
        mit_for_docx = "mitigations_roberta_top.csv"
    else:
        parsed["mitigation"] = "No mitigations found for these techniques."
        mit_for_docx = None

    # Generate DOCX (non-fatal)
    try:
        out_path = generate_word_report(
            gpt_response,
            ttps,
            mitigations_csv=mit_for_docx  # <-- pass the scoped CSV
        )
        if not out_path:
            out_path = "threat_report_roberta.docx"
    except Exception as e:
        print("[WARN] RoBERTa DOCX generation failed:", e)
        out_path = None
    return {
        "ttps": ttps,
        "matched_full_df": df_ml,
        "matched_top3": top3_df.to_dict(orient="records"),
        "analysis": parsed,
        "doc_path": out_path,
    }

# ============================================
# Small helpers
# ============================================
#For Roberta 0-10 fold runs. 
# def output_dir_for_folds(n_folds: int, model_slug: str = "roberta_base"):
#     return EXPERIMENTS_ROOT / f"{n_folds}foldruns" / model_slug
   
# =======================================================
# Pipeline steps (extract pdf → extract stix → build → train)
# =======================================================
def step_extract_pdfs(in_dir: Path = PDFS_DIR, out_dir: Path = EXTRACTED_PDFS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = EXTRACTED_IOCS_CSV

    # Treat the extractor script and input folder as inputs
    inputs = [EXTRACT_SCRIPT] + list(in_dir.glob("*.pdf"))
    if not needs_run([out_csv], inputs=inputs):
        print(f"[SKIP] extract_pdfs.py — up to date: {out_csv}")
        return

    run([sys.executable, str(EXTRACT_SCRIPT), "--in", str(in_dir), "--out", str(out_dir)])


def step_enterprise_attack():
    outputs = [TI_GROUPS_TECHS_CSV]
    inputs  = [ATTACK_SCRIPT]  # add STIX source dirs/files if you have them
    if not needs_run(outputs, inputs=inputs):
        print(f"[SKIP] enterprise_attack.py — up to date: {TI_GROUPS_TECHS_CSV}")
        return
    run([sys.executable, str(ATTACK_SCRIPT)])


def step_build_dataset():
    outputs = [DATASET_CSV, LABELS_TXT]
    inputs  = [BUILD_DATASET_SCRIPT, EXTRACTED_IOCS_CSV, TI_GROUPS_TECHS_CSV]
    if not needs_run(outputs, inputs=inputs):
        print(f"[SKIP] build_dataset.py — up to date: {DATASET_CSV}, {LABELS_TXT}")
        return
    run([sys.executable, str(BUILD_DATASET_SCRIPT)])

def step_mitigations(force: bool = False):
    """
    Generate mitigations CSV (Data/mitigations/mitigations.csv).
    Runs mitigations.py if inputs changed or file missing.
    """
    out_csv = MITIGATIONS_CSV
    if not needs_run([out_csv], inputs=[MITIGATIONS_SCRIPT, GROUP_TTPS_DETAIL_CSV, MAPPING_CSV, EXCEL_ATTACK_TECHS]) and not force:
        print(f"[SKIP] mitigations.py — up to date: {out_csv}")
        return

    print(f"[RUN] mitigations.py — generating: {out_csv}")
    res = subprocess.run([sys.executable, str(MITIGATIONS_SCRIPT)], cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"mitigations.py failed with exit code {res.returncode}")
    if not out_csv.exists():
        raise FileNotFoundError(f"Expected mitigations CSV not found at: {out_csv}")
    print(f"[OK] mitigations.csv generated: {out_csv}")

def step_map_iocs_to_attack(force: bool = False):
    """
    Run map_iocs_to_attack.py to produce:
      - ranked_groups.csv
      - group_ttps_detail.csv
    """
    outputs = [RANKED_GROUPS_CSV, GROUP_TTPS_DETAIL_CSV]
    inputs = [MAP_IOCS_SCRIPT, EXTRACTED_IOCS_CSV, TI_GROUPS_TECHS_CSV]
    if not needs_run(outputs, inputs=inputs) and not force:
        print(f"[SKIP] map_iocs_to_attack.py — up to date: {RANKED_GROUPS_CSV}, {GROUP_TTPS_DETAIL_CSV}")
        return

    print(f"[RUN] map_iocs_to_attack.py — generating IOC→ATT&CK mappings…")
    res = subprocess.run([sys.executable, str(MAP_IOCS_SCRIPT)], cwd=str(PROJECT_ROOT))
    if res.returncode != 0:
        raise RuntimeError(f"map_iocs_to_attack.py failed with exit code {res.returncode}")

    for out in outputs:
        if not out.exists():
            raise FileNotFoundError(f"Expected output missing: {out}")

    print(f"[OK] Generated: {RANKED_GROUPS_CSV.name}, {GROUP_TTPS_DETAIL_CSV.name}")

def _is_empty_dir(p: Path) -> bool:
    return (not p.exists()) or (next(p.iterdir(), None) is None)

def step_train_roberta():
    if not _is_empty_dir(BEST_MODEL_DIR):
        print(f"[SKIP] train_roberta.py — best model exists in {BEST_MODEL_DIR}")
        return
    run([sys.executable, str(TRAIN_ROBERTA_SCRIPT)])

# ============================================
# Index & Workflow
#index, workflow, roberta, submit_both, predict with module, predict api, match, export
# ============================================
@app.route('/')
def index():
    ensure_dir_tree()
    try:
        # Always regenerate latest mapping from Excel
        extract_techniques(EXCEL_ATTACK_TECHS, MAPPING_CSV)

        # Load mapping: id, name, label
        df_map = pd.read_csv(MAPPING_CSV)

        # Group main + sub-techniques
        ttp_dict = defaultdict(list)
        for tid, label in zip(df_map["id"], df_map["label"]):
            root = tid.split(".")[0]
            if "." in tid:
                ttp_dict[root].append((tid, label))
            else:
                ttp_dict.setdefault(tid, [])

        # Build grouped list for dropdown
        id_to_label = dict(zip(df_map["id"], df_map["label"]))
        grouped_ttps = []
        for root in sorted(ttp_dict.keys()):
            root_label = id_to_label.get(root, root)
            subs = [lbl for _, lbl in sorted(ttp_dict[root], key=lambda x: x[0])]
            grouped_ttps.append((root_label, subs))

        # Debug info
        print("\n================ TECHNIQUE SUMMARY ================")
        print(f"Total techniques loaded: {len(df_map)}")
        print(f"Total root techniques: {len(ttp_dict)}")
        print("Example entries:")
        print(df_map.head(5))
        print("===================================================\n")
        
        print(f"[ROOT] {PROJECT_ROOT}")
        print(f"[DATA] {DATA_ROOT}")
        print(f"[PROC] {PROCESSED_DIR}")
        print(f"[MODELS] {MODELS_ROOT}")
        #END of debug
        step_extract_pdfs()
        step_enterprise_attack()
        step_build_dataset()
        step_mitigations()
        step_map_iocs_to_attack()
        step_train_roberta()

        return render_template('index.html', grouped_ttps=grouped_ttps)

    except Exception as e:
        return render_template('error.html', error=str(e))
# =======================================================
# WORKFLOW ROUTE
# =======================================================
@app.route('/workflow')
def workflow():
    return render_template('workflow.html')
# =======================================================
# ROBERTA ROUTE 
# =======================================================
@app.route('/roberta', methods=['POST'])
def roberta():
    try:
        ttps_input = [t.split()[0].upper() for t in request.form.getlist('ttps[]')]
        ttps = validate_ttps(ttps_input)

        # ✅ Single source of truth
        res = _run_roberta_flow(ttps)

        # cache for export
        global LAST_RESULTS, LAST_RESULTS_ROBERTA
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        LAST_RESULTS = {
            "ttps": res["ttps"],
            "matched": res["matched_top3"],
            "analysis": res["analysis"],
            "timestamp": timestamp,
        }
        LAST_RESULTS_ROBERTA = LAST_RESULTS.copy()

        # render using roberta keys
        return render_template(
            'results.html',
            ttps=res["ttps"],
            rob_matched=res["matched_top3"],
            rob_analysis=res["analysis"],
            export_mode=False,
            timestamp=timestamp,
        )
    except Exception as e:
        return render_template('error.html', error=str(e))

    
@app.route('/results', methods=['POST'])
def results():
    try:
        ttps_input = [t.split()[0].upper() for t in request.form.getlist('ttps[]')]
        ttps = validate_ttps(ttps_input)

        rule_res = _run_rule_match_flow(ttps)
        rob_res  = _run_roberta_flow(ttps)

        global LAST_RESULTS_RULE, LAST_RESULTS_ROBERTA
        LAST_RESULTS_RULE = {
            "ttps": rule_res["ttps"],
            "matched": rule_res["matched_top3"],
            "analysis": rule_res["analysis"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        LAST_RESULTS_ROBERTA = {
            "ttps": rob_res["ttps"],
            "matched": rob_res["matched_top3"],
            "analysis": rob_res["analysis"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        return render_template(
            "results.html",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ttps=ttps,
            rule_matched=rule_res["matched_top3"],
            rule_analysis=rule_res["analysis"],
            rob_matched=rob_res["matched_top3"],
            rob_analysis=rob_res["analysis"],
            export_mode=False,
        )

    except Exception as e:
        return render_template('error.html', error=str(e)), 500

@app.route('/predict', methods=['POST'])
def predict_api():
    """
    JSON body:
    {
      "id": "sample1",
      "urls": [...], "domains": [...], "ips": [...],
      "md5s": [...], "sha256s": [...], "attacks": ["T1566.002", ...],
      "text": "optional free text",
      "threshold": 0.5, "top_k": 10
    }
    """
    try:
        payload = request.get_json(force=True, silent=False) or {}
        result = _predict_with_module(payload)
        return jsonify({"status": "ok", **result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route('/match', methods=['POST'])
def match():
    try:
        # Collect only clean Technique IDs
        ttps_input = [t.split()[0].upper() for t in request.form.getlist('ttps[]')]
        ttps = validate_ttps(ttps_input)

        matched_df = match_ttps(ttps, MAPPED_DIR)

        # Convert numeric, sort, top3
        matched_df["rank"]  = pd.to_numeric(matched_df.get("rank", float("nan")), errors="coerce")
        matched_df["score"] = pd.to_numeric(matched_df.get("score", float("nan")), errors="coerce")
        if matched_df["rank"].notna().any():
            matched_df = matched_df.sort_values(by="rank", ascending=True)
        else:
            matched_df = matched_df.sort_values(by="score", ascending=False)
        top3_df = matched_df.head(3)

        # Save traceability outputs
        matched_df.to_csv("matched_groups_rule.csv", index=False)

        # GPT Analysis
        mit_csv_path = MITIGATIONS_CSV
        gpt_response = analyze_TTP(ttps, matched_df, mitigations_csv=str(mit_csv_path))
        parsed = parse_ai_response(gpt_response)

        # Filter mitigations for matched group techniques
        # group_ttps = list({t.strip().upper() for t in _collect_group_ttps(matched_df, ttps)})
        # Only show mitigations for the TOP matched actor in /match as well
        group_ttps = _collect_top_group_ttps(matched_df) or list({t.strip().upper() for t in ttps if t})

        mit_filtered = load_filtered_mitigations(str(mit_csv_path), group_ttps)

        # Remove duplicate mitigation descriptions
        if not mit_filtered.empty:
         mit_filtered = mit_filtered.drop_duplicates(
         subset=["target id", "target name", "mapping description"], keep="first"
        )

        parsed["mitigation"] = mit_filtered.to_dict(orient="records")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        global LAST_RESULTS
        LAST_RESULTS = {
            "ttps": ttps,
            "matched": top3_df.to_dict(orient="records"),
            "analysis": parsed,
            "timestamp": timestamp
        }

        return render_template(
            'results.html',
            ttps=ttps,
            matched=top3_df.to_dict(orient='records'),
            analysis=parsed,
            timestamp=timestamp
        )

    except Exception as e:
        return render_template('error.html', error=str(e))
# =======================================================
# EXPORT ROUTE - For when printing results to document
# =======================================================

@app.route('/export')
def export():
    try:
        which = (request.args.get("which") or "rule").lower()
        if which == "roberta":
            ctx = LAST_RESULTS_ROBERTA
        else:
            ctx = LAST_RESULTS_RULE if LAST_RESULTS_RULE else LAST_RESULTS

        if not ctx:
            return "No results available to export. Please generate a report first."

        # Always provide BOTH sets so the template never complains.
        rendered = render_template(
            "results.html",
            ttps=ctx.get("ttps", []),

            # Rule side
            rule_matched=(ctx.get("matched", []) if which == "rule" else []),
            rule_analysis=(ctx.get("analysis", {}) if which == "rule" else {}),

            # RoBERTa side
            rob_matched=(ctx.get("matched", []) if which == "roberta" else []),
            rob_analysis=(ctx.get("analysis", {}) if which == "roberta" else {}),

            export_mode=True,
            timestamp=ctx.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        )

        # strip CSS link if needed
        rendered = re.sub(r'<link rel="stylesheet" href="[^"]*attribution\.css">', "", rendered, flags=re.IGNORECASE)

        response = make_response(rendered)
        response.headers["Content-Type"] = "application/msword"
        fname = "threat_report_roberta.doc" if which == "roberta" else "threat_report_rule.doc"
        response.headers["Content-Disposition"] = f"attachment; filename={fname}"
        return response

    except Exception as e:
        return f"Error exporting to .doc: {e}"

if __name__ == '__main__':
    app.run(debug=True, threaded=True)