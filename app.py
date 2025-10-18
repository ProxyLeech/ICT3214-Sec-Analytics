from flask import Flask, render_template, request, make_response, jsonify
import pandas as pd
from matching import (
    validate_ttps,
    match_ttps,
)
from report_generator import (
    analyze_TTP,
    load_mitigations_summary,
    parse_ai_response,
    generate_word_report,
    load_filtered_mitigations,
    summarize_mitigations
)
from datetime import datetime
from collections import defaultdict
from technique_labels import extract_techniques  # import the extractor
import io
import re
from typing import Iterable
import subprocess, sys
import sys
import os, tempfile
from pathlib import Path

# =======================================================
# Project-relative paths
# =======================================================

BASE_DIR = Path(__file__).resolve().parent
SRC_PATH = BASE_DIR / "src"  # ✅ src is inside ICT3214-Sec-Analytics
sys.path.insert(0, str(SRC_PATH))
app = Flask(__name__)

DATA_DIR = BASE_DIR / "Data" / "mapped"
EXCEL_PATH = BASE_DIR / "Data" / "excel" / "enterprise-attack-v17.1-techniques.xlsx"
MAPPING_CSV = BASE_DIR / "techniques_mapping.csv"
EXPERIMENTS_ROOT = BASE_DIR / "experiments"
DATA_ROOT      = BASE_DIR / "data"
RAW_DIR        = DATA_ROOT / "raw"
PROCESSED_DIR  = DATA_ROOT / "processed"
SRC_ROOT       = BASE_DIR / "src"
MODELS_ROOT    = SRC_ROOT / "models"
DATASCRIPT_ROOT= SRC_ROOT / "data"
#Trained model
BEST_MODEL_DIR = MODELS_ROOT / "best_roberta_for_predict"
DATA_DIR = BASE_DIR / "Data" / "mapped"
EXCEL_PATH = BASE_DIR / "Data" / "excel" / "enterprise-attack-v17.1-techniques.xlsx"
MAPPING_CSV = BASE_DIR / "techniques_mapping.csv"

# Idempotent mitigations runner
def _run_mitigations_and_get_csv() -> Path:
    """
    Run mitigations.py synchronously ONCE and return the output CSV path:
      Data/mitigations/mitigations.csv
    """
    script = BASE_DIR / "mitigations.py"
    out_csv = BASE_DIR / "Data" / "mitigations" / "mitigations.csv"

    # Only run if the CSV doesn't exist (idempotent)
    if out_csv.exists():
        print(f"[SKIP] mitigations.py — up to date: {out_csv}")
        return out_csv

    print(f"[RUN] mitigations.py — generating: {out_csv}")
    res = subprocess.run([sys.executable, str(script)], cwd=str(BASE_DIR))
    if res.returncode != 0:
        raise RuntimeError(f"mitigations.py failed with exit code {res.returncode}")
    if not out_csv.exists():
        raise FileNotFoundError(f"Expected mitigations CSV not found at: {out_csv}")
    return out_csv

# ---- Extra project roots (as in your original)
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))  # make common/, data/, models/ importable
from paths.paths import (  # type: ignore  # noqa: E402
    DATA_ROOT, RAW_DIR,
    EXTRACTED_PDFS_DIR,
    PROCESSED_DIR,
    DATASCRIPT_ROOT,
    MODELS_ROOT,
    EXPERIMENTS_ROOT,
)

# ---- Expected inputs/outputs per step ----
PDFS_IN_DIR            = RAW_DIR / "pdfs"
EXTRACTED_IOCS_CSV     = PROCESSED_DIR / "extracted_iocs.csv"
TI_GROUPS_TECHS_CSV    = PROCESSED_DIR / "ti_groups_techniques.csv"
DATASET_CSV   = PROCESSED_DIR / "dataset.csv"
LABELS_TXT    = PROCESSED_DIR / "labels.txt"
EXTRACTED_PDFS_DIR   = DATA_ROOT / "extracted_pdfs"

# Scripts (relative to repo root)
EXTRACT_SCRIPT       = DATASCRIPT_ROOT / "extract_pdfs.py"
ATTACK_SCRIPT        = DATASCRIPT_ROOT / "enterprise_attack.py"
BUILD_DATASET_SCRIPT = DATASCRIPT_ROOT / "build_dataset.py"
TRAIN_ROBERTA_SCRIPT = MODELS_ROOT     / "train_roberta.py"
PREDICT_SCRIPT       = MODELS_ROOT     / "predict_roberta.py"


#=================================================

# Cache
LAST_RESULTS = {} #Global
LAST_RESULTS_RULE = {}
LAST_RESULTS_ROBERTA = {}


# -------------------------------------------------------
# Small helpers
# -------------------------------------------------------
#For Roberta 0-10 fold runs. 
def output_dir_for_folds(n_folds: int, model_slug: str = "roberta_base"):
    return EXPERIMENTS_ROOT / f"{n_folds}foldruns" / model_slug

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

def ensure_dirs():
    for p in [DATA_ROOT, RAW_DIR, EXTRACTED_PDFS_DIR, PROCESSED_DIR, MODELS_ROOT, EXPERIMENTS_ROOT]:
        p.mkdir(parents=True, exist_ok=True)

# =========================
# STRICT group → TTP lookup
# =========================
def _collect_group_ttps(matched_df: pd.DataFrame) -> list[str]:
    """
    Extract unique MITRE technique IDs (e.g. T1110, T1110.003) from
    Data/mapped/group_ttps_detail.csv for the matched groups.
    Falls back gracefully if columns differ between datasets.
    """
    map_path = BASE_DIR / "Data" / "mapped" / "group_ttps_detail.csv"
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
def _atomic_to_csv(df, path: str):
    d = Path(path).parent
    d.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=d, newline="", suffix=".tmp") as tmp:
        tmp_name = tmp.name
        df.to_csv(tmp, index=False)
    os.replace(tmp_name, path)  # atomic on POSIX & Windows

# =======================================================
# Matching Flow
# =======================================================
def _run_rule_match_flow(ttps: list[str]) -> dict:
    matched_df = match_ttps(ttps, DATA_DIR).copy()
    matched_df = _ensure_score_and_rank(matched_df)
    if "rank" in matched_df.columns and matched_df["rank"].notna().any():
        matched_df = matched_df.sort_values(by=["rank", "score"], ascending=[True, False])
    else:
        matched_df = matched_df.sort_values(by="score", ascending=False)
    top3_df = matched_df.head(3)
    pd.DataFrame({"TTP": ttps}).to_csv("inputted_ttps.csv", index=False)

    # Mitigations (idempotent) + GPT analysis
    mit_csv_path = _run_mitigations_and_get_csv()
    gpt_response = analyze_TTP(ttps, matched_df, mitigations_csv=str(mit_csv_path))
    parsed = parse_ai_response(gpt_response)

    # Filter mitigations for the matched groups' techniques
    # 1) get group-based TTPs (strict CSV mapping)
    group_ttps = _collect_group_ttps(matched_df)

    # 2) fallback ONLY if none found
    if not group_ttps:
      print("[INFO] No group-mapped TTPs found; falling back to inputted TTPs.")
      group_ttps = list({t.strip().upper() for t in ttps if t})

    # 3) filter mitigations using those TTPs (includes sub-techniques)
    mit_filtered = load_filtered_mitigations(str(mit_csv_path), group_ttps)
    if mit_filtered.empty:
        parsed["mitigation"] = "No mitigations found for these techniques."
    else:
        mit_dicts = mit_filtered.to_dict(orient="records")
        parsed["mitigation"] = summarize_mitigations(mit_dicts)
    
    # Try generating docx (non-fatal)
    try:
        out_path = generate_word_report(gpt_response, ttps, mitigations_csv=str(mit_csv_path))
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

# =======================================================
# RoBERTa Flow
# =======================================================
def _save_roberta_traces(df_ml):
    _atomic_to_csv(df_ml, "matched_groups_roberta.csv")

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
    df_ml.drop(columns=["score","origin"], errors="ignore", inplace=True)
    # Persist files for the report script
    _save_roberta_traces(df_ml)
    top3_df = df_ml.head(3).drop(columns=["origin"], errors="ignore")
    # GPT narrative for roberta
    gpt_response = analyze_TTP(ttps, df_ml)
    parsed = parse_ai_response(gpt_response)
    # NEW: run mitigations and include in OpenAI context
    mit_csv_path = _run_mitigations_and_get_csv()
    
    from report_generator import load_mitigations_summary
    parsed["mitigation"] = load_mitigations_summary(str(mit_csv_path))

    # Filter mitigations for the matched groups' techniques
    # 1) get group-based TTPs (strict CSV mapping)
    group_ttps = _collect_group_ttps(df_ml)

    # 2) fallback ONLY if none found
    if not group_ttps:
      print("[INFO] No group-mapped TTPs found; falling back to inputted TTPs.")
      group_ttps = list({t.strip().upper() for t in ttps if t})

    # 3) filter mitigations using those TTPs (includes sub-techniques)
    mit_filtered = load_filtered_mitigations(str(mit_csv_path), group_ttps)
    if mit_filtered.empty:
        parsed["mitigation"] = "No mitigations found for these techniques."
    else:
        mit_dicts = mit_filtered.to_dict(orient="records")
        parsed["mitigation"] = summarize_mitigations(mit_dicts)

    # Try generating docx (non-fatal)
    try:
        out_path = generate_word_report(gpt_response, ttps, mitigations_csv=str(mit_csv_path))
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
# =======================================================
# Pipeline steps (extract pdf → extract stix → build → train)
# =======================================================
def step_extract_pdfs(in_dir: Path = PDFS_IN_DIR, out_dir: Path = EXTRACTED_PDFS_DIR):
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

def _is_empty_dir(p: Path) -> bool:
    return (not p.exists()) or (next(p.iterdir(), None) is None)

def step_train_roberta():
    if not _is_empty_dir(BEST_MODEL_DIR):
        print(f"[SKIP] train_roberta.py — best model exists in {BEST_MODEL_DIR}")
        return
    run([sys.executable, str(TRAIN_ROBERTA_SCRIPT)])


# =======================================================
# Flask Routes
#index, workflow, roberta, submit_both, predict with module, predict api, match, export
# =======================================================
@app.route('/')
def index():
    try:
        # Always regenerate latest mapping from Excel
        extract_techniques(EXCEL_PATH, MAPPING_CSV)

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
        print(f"[ROOT] {BASE_DIR}")
        print(f"[DATA] {DATA_ROOT}")
        print(f"[PROC] {PROCESSED_DIR}")
        print(f"[MODELS] {MODELS_ROOT}")

        step_extract_pdfs()
        step_enterprise_attack()
        step_build_dataset()
        step_train_roberta()
        return render_template('0_index.html', grouped_ttps=grouped_ttps)

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
        # 1) Grab user selections 
        ttps_input = [t.split()[0].upper() for t in request.form.getlist('ttps[]')]
        ttps = validate_ttps(ttps_input)

        # 2) Call RoBERTa prediction using ATT&CK IDs as inputs
        #    This uses the predictor functions from predict_roberta.py via the helper.
        ml = _predict_with_module({
            "id": "from_ttps",
            "attacks": ttps,        # <- IMPORTANT: pass the selected TTPs
            "top_k": 50,
            "threshold": 0.0
        })

        # 3) Normalize the predictor output into a DataFrame
        #    Expecting a list[dict] like [{"group_name": "...", "score": 0.87, ...}, ...]
        group_rows = ml.get("groups", [])
        if not group_rows:
            raise RuntimeError("ML predictor returned no groups. Check model artifacts and inputs.")

        df_ml = pd.DataFrame(group_rows)
        if "group" in df_ml.columns and "group_name" not in df_ml.columns:
            df_ml.rename(columns={"group": "group_name"}, inplace=True)

        df_ml = _ensure_score_and_rank(df_ml)
        df_ml.sort_values("score", ascending=False, inplace=True)

        if "prob" in df_ml.columns and "score" not in df_ml.columns:
            df_ml.rename(columns={"prob": "score"}, inplace=True)


        # Rank by score if no rank provided
        if "rank" not in df_ml.columns:
            df_ml["rank"] = df_ml["score"].rank(ascending=False, method="first")

        # 4) Persist files for the report generator (keeps your current contract)
        df_ml.sort_values("score", ascending=False, inplace=True)

        # 5)  Show top 3 on the web page, like before
        top3_df = df_ml.head(3)

        # 6) Produce the AI narrative and the DOCX report
        #    If you don't have OPENAI_API_KEY set, wrap this with your earlier "disable AI" flag.
        mit_csv_path = _run_mitigations_and_get_csv()
        gpt_response = analyze_TTP(ttps, df_ml, mitigations_csv=str(mit_csv_path))
        parsed = parse_ai_response(gpt_response)

        # If you want the .docx to be generated immediately on submit:
        try:
            generate_word_report(gpt_response, ttps)
        except Exception as e:
            # Non-fatal: still render HTML even if DOCX write fails
            print("[WARN] DOCX generation failed:", e)

        # 7) Render your HTML results page as before
        from datetime import datetime
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
# Submit to HTML
# =======================================================
@app.route('/submit_both', methods=['POST'])
def submit_both():
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
            "results_compare.html",
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

# -------------------------------------------------------
# PREDICT ROUTE – JSON in, ranked groups out
# -------------------------------------------------------
import json

def _predict_with_module(payload: dict):
    try:
        # Late import to avoid circulars during app startup
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

        # Build doc_source (provenance of the input document) just like your CLI does
        srcs = pr.resolve_sources_from_inputs(
            payload.get("urls") or [], payload.get("domains") or [], payload.get("ips") or [],
            payload.get("md5s") or [], payload.get("sha256s") or [], payload.get("attacks") or []
        )
        origin_doc = ", ".join(srcs) if srcs else (payload.get("id") or "adhoc")

        # Pass the ATT&CK IDs you received from the user so techniques that were also input
        # are tagged with 'attack_input' in the origin field.
        attack_ids_in_input = {a.strip().upper() for a in (payload.get("attacks") or []) if a}

        # Expand & aggregate with the NEW signatures
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

        matched_df = match_ttps(ttps, DATA_DIR)

        # Convert numeric, sort, top3
        matched_df["rank"]  = pd.to_numeric(matched_df.get("rank", float("nan")), errors="coerce")
        matched_df["score"] = pd.to_numeric(matched_df.get("score", float("nan")), errors="coerce")
        if matched_df["rank"].notna().any():
            matched_df = matched_df.sort_values(by="rank", ascending=True)
        else:
            matched_df = matched_df.sort_values(by="score", ascending=False)
        top3_df = matched_df.head(3)
        pd.DataFrame({"TTP": ttps}).to_csv("inputted_ttps.csv", index=False)

        # GPT Analysis
        mit_csv_path = _run_mitigations_and_get_csv()
        gpt_response = analyze_TTP(ttps, matched_df, mitigations_csv=str(mit_csv_path))
        parsed = parse_ai_response(gpt_response)

        # Filter mitigations for matched group techniques
        group_ttps = list({t.strip().upper() for t in _collect_group_ttps(matched_df, ttps)})

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
# Export

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

        rendered = render_template(
            "results.html",
            ttps=ctx["ttps"],
            matched=ctx["matched"],
            analysis=ctx["analysis"],
            timestamp=ctx["timestamp"],
            export_mode=True
        )

        # strip CSS link
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