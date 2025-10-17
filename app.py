from flask import Flask, render_template, request, make_response, jsonify
import pandas as pd
from matching import (
    validate_ttps,
    match_ttps,
)
from report_generator import analyze_TTP, parse_ai_response, generate_word_report
from datetime import datetime
from collections import defaultdict
from technique_labels import extract_techniques  # import the extractor
import io
import re

import subprocess, sys
import sys

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
SRC_PATH = BASE_DIR / "src"   # ✅ src is inside ICT3214-Sec-Analytics
sys.path.insert(0, str(SRC_PATH))
app = Flask(__name__)

# =======================================================
# Project-relative paths
# =======================================================
# BASE_DIR = Path(__file__).resolve().parent

def _run_mitigations_and_get_csv() -> Path:
    """
    Run mitigations.py synchronously and return the output CSV path:
      Data/mapped/mitigations.csv
    """
    script = BASE_DIR / "mitigations.py"
    out_csv = BASE_DIR / "Data" / "mapped" / "mitigations.csv"

    # Run mitigations.py in the same directory
    res = subprocess.run([sys.executable, str(script)], cwd=str(BASE_DIR))
    if res.returncode != 0:
        raise RuntimeError(f"mitigations.py failed with exit code {res.returncode}")
    if not out_csv.exists():
        raise FileNotFoundError(f"Expected mitigations CSV not found at: {out_csv}")
    return out_csv
#ADDITIONAL ADDED PATHS
#=================================================
ROOT = Path(__file__).resolve().parents[1]   
sys.path.insert(0, str(ROOT / "src"))        # make common/, data/, models/ importable
from paths.paths import (
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

# Scripts (relative to repo root)
EXTRACT_SCRIPT         = DATASCRIPT_ROOT  / "extract_pdfs.py"
ATTACK_SCRIPT          = DATASCRIPT_ROOT / "enterprise_attack.py"
BUILD_DATASET_SCRIPT = DATASCRIPT_ROOT / "build_dataset.py"
TRAIN_ROBERTA_SCRIPT = MODELS_ROOT  / "train_roberta.py"
PREDICT_SCRIPT         = MODELS_ROOT     / "predict_roberta.py"

#Trained model
BEST_MODEL_DIR = MODELS_ROOT / "best_roberta_for_predict"
BEST_REQUIRED  = [
    BEST_MODEL_DIR / "config.json",
    BEST_MODEL_DIR / "tokenizer.json",     
    BEST_MODEL_DIR / "id2label.json",
]
#=================================================

def output_dir_for_folds(n_folds: int, model_slug: str = "roberta_base"):
    return EXPERIMENTS_ROOT / f"{n_folds}foldruns" / model_slug


# Global cache for last results
LAST_RESULTS = {}
# -------------------------------------------------------
# Small helpers
# -------------------------------------------------------
def _ensure_score_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has numeric 'score' and 'rank' columns.

    Accepts common alternatives: 'prob', 'probability', 'confidence', 'logit', 'logprob'.
    If none are present, falls back to:
      - if 'rank' exists: score = 1 / (1 + rank)
      - else: score = numpy.linspace(1.0, 0.0, len(df), endpoint=False)
    """
    import numpy as np

    if df is None or df.empty:
        return df

    # 1) find a score-like column
    candidates = ["score", "prob", "probability", "confidence", "logit", "logprob"]
    src = next((c for c in candidates if c in df.columns), None)

    if src is None:
        # fallbacks
        if "rank" in df.columns:
            df["score"] = pd.to_numeric(df["rank"], errors="coerce")
            df["score"] = 1.0 / (1.0 + df["score"].fillna(df["score"].max() or 1))
        else:
            n = len(df)
            df["score"] = np.linspace(1.0, 0.0, n, endpoint=False)
    else:
        df["score"] = pd.to_numeric(df[src], errors="coerce").fillna(0.0)

    # 2) ensure rank exists (high score => low rank number)
    if "rank" not in df.columns or df["rank"].isna().all():
        # stable ranking: highest score gets rank 1
        df["rank"] = (-df["score"]).rank(method="first").astype(int)

    return df

def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(map(str, cmd))}")
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if res.returncode != 0:
        raise SystemExit(res.returncode)

def needs_run(outputs: list[Path], force: bool) -> bool:
    return force or not all(p.exists() for p in outputs)

def ensure_dirs():
    for p in [DATA_ROOT, RAW_DIR, EXTRACTED_PDFS_DIR, PROCESSED_DIR, MODELS_ROOT, EXPERIMENTS_ROOT]:
        p.mkdir(parents=True, exist_ok=True)

# ---- Result caches for export ----
LAST_RESULTS_RULE = {}
LAST_RESULTS_ROBERTA = {}

def _run_rule_match_flow(ttps: list[str]) -> dict:
    matched_df = match_ttps(ttps, DATA_DIR).copy()
    matched_df = _ensure_score_and_rank(matched_df)
    if "rank" in matched_df.columns and matched_df["rank"].notna().any():
        matched_df = matched_df.sort_values(by=["rank", "score"], ascending=[True, False])
    else:
        matched_df = matched_df.sort_values(by="score", ascending=False)
    top3_df = matched_df.head(3)

    matched_df.to_csv("matched_groups_rule.csv", index=False)
    top3_df.to_csv("matched_top3_rule.csv", index=False)
    pd.DataFrame({"TTP": ttps}).to_csv("inputted_ttps_rule.csv", index=False)

    # NEW: run mitigations and include in OpenAI context
    mit_csv_path = _run_mitigations_and_get_csv()

    gpt_response = analyze_TTP(ttps, matched_df, mitigations_csv=str(mit_csv_path))
    parsed = parse_ai_response(gpt_response)

    try:
        out_path = generate_word_report(gpt_response, ttps)
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


# --- PATCH inside _run_roberta_flow(ttps: list[str]) ---

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

    top3_df = df_ml.head(3)

    df_ml.to_csv("matched_groups_roberta.csv", index=False)
    top3_df.to_csv("matched_top3_roberta.csv", index=False)
    pd.DataFrame({"TTP": ttps}).to_csv("inputted_ttps_roberta.csv", index=False)

    # NEW: run mitigations and include in OpenAI context
    mit_csv_path = _run_mitigations_and_get_csv()

    gpt_response = analyze_TTP(ttps, df_ml, mitigations_csv=str(mit_csv_path))
    parsed = parse_ai_response(gpt_response)

    try:
        out_path = generate_word_report(gpt_response, ttps)
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


# -------------------------------------------------------
# Pipeline steps 
# -------------------------------------------------------
def step_extract_pdfs(force: bool) -> None:
    """
    Run extract_pdfs.py to produce:
      - data/extracted_pdfs/<doc-folder>/{pNNN.txt, full.txt, meta.json}
      - data/extracted_pdfs/extracted_iocs.csv
    """
    ensure_dirs()
    if not needs_run([EXTRACTED_IOCS_CSV], force):
        print(f"[SKIP] extract_pdfs.py — up to date: {EXTRACTED_IOCS_CSV}")
        return
    # extract_pdfs.py has CLI; we just call with defaults (input=RAW/pdfs, output=extracted_pdfs)
    run([sys.executable, str(EXTRACT_SCRIPT)])

def step_enterprise_attack(force: bool) -> None:
    """
    Run enterprise_attack.py to create:
      - data/attack_stix/processed/ti_groups_techniques.csv
    """
    ensure_dirs()
    if not needs_run([TI_GROUPS_TECHS_CSV], force):
        print(f"[SKIP] enterprise_attack.py — up to date: {TI_GROUPS_TECHS_CSV}")
        return
    run([sys.executable, str(ATTACK_SCRIPT)])

def step_build_dataset(force: bool) -> None:
    """
    Run build_dataset.py to create:
      - data/processed/dataset.csv
      - data/processed/labels.txt
    """
    ensure_dirs()
    if not needs_run([DATASET_CSV, LABELS_TXT], force):
        print(f"[SKIP] build_dataset.py — up to date: {DATASET_CSV}, {LABELS_TXT}")
        return

    run([sys.executable, str(BUILD_DATASET_SCRIPT)])

def step_train_roberta(force: bool) -> None:
    """
    Run train_roberta.py.
    Should copy the best run into src/models/best_roberta_for_predict/.
    """
    ensure_dirs()
    print(f"\n[DEBUG] BEST_MODEL_DIR: {BEST_MODEL_DIR}")
    if BEST_MODEL_DIR.exists():
        try:
            print("[DEBUG] best dir contents:", sorted(p.name for p in BEST_MODEL_DIR.iterdir()))
        except Exception as e:
            print("[DEBUG] failed to list best dir:", e)

    if not needs_run(BEST_REQUIRED, force):
        print(f"[SKIP] train_roberta.py — best model already present: {BEST_MODEL_DIR}")
        return
    run([sys.executable, str(TRAIN_ROBERTA_SCRIPT)])


# =======================================================
# INDEX ROUTE – Build dropdown of all MITRE TTPs
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
        # 1) Grab user selections (up to 5 TTPs from your UI)
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
        df_ml.to_csv("matched_groups.csv", index=False)
        pd.DataFrame({"TTP": ttps}).to_csv("inputted_ttps.csv", index=False)

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
    
@app.route('/submit_both', methods=['POST'])
def submit_both():
    try:
        # collect selected TTPs once
        ttps_input = [t.split()[0].upper() for t in request.form.getlist('ttps[]')]
        ttps = validate_ttps(ttps_input)

        # run both flows
        rule_res = _run_rule_match_flow(ttps)
        rob_res  = _run_roberta_flow(ttps)

        # cache both for export
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
            rule_matched=rule_res["matched_top3"],     # list of dicts
            rule_analysis=rule_res["analysis"],        # dict with keys summary/table/attacker/mitigation/suggestion
            rob_matched=rob_res["matched_top3"],       # list of dicts
            rob_analysis=rob_res["analysis"],          # same dict structure
            export_mode=False,
        )

    except Exception as e:
        return render_template('error.html', error=str(e)), 500

# -------------------------------------------------------
# PIPELINE ROUTE – Extract → Map → Build → Train
# -------------------------------------------------------
@app.route('/pipeline', methods=['POST'])
def pipeline():
    try:
        force = bool(request.form.get('force') or request.json.get('force') if request.is_json else request.form.get('force'))
        # Run in order
        step_extract_pdfs(force=force)
        step_enterprise_attack(force=force)
        step_build_dataset(force=force)
        step_train_roberta(force=force)

        ok = all(p.exists() for p in BEST_REQUIRED)
        return jsonify({
            "status": "ok" if ok else "warning",
            "message": "Pipeline completed" if ok else "Pipeline finished but best model looks incomplete.",
            "best_model_dir": str(BEST_MODEL_DIR),
            "have_best_files": {p.name: p.exists() for p in BEST_REQUIRED}
        })
    except SystemExit as e:
        # bubble up subprocess error codes nicely
        return jsonify({"status": "error", "message": f"Subprocess exited with code {int(e.code)}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

# -------------------------------------------------------
# PREDICT ROUTE – JSON in, ranked groups out
# -------------------------------------------------------
def _predict_with_module(payload: dict):
    """
    Try importing predict_roberta.py and using its functions directly.
    Fallback to subprocess if import fails (e.g., path mismatches).
    """
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
        # Subprocess fallback using CLI
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
        args += ["--threshold", str(payload.get("threshold", 0.5)), "--top-k", str(payload.get("top_k", 10))]

        res = subprocess.run(args, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(res.stderr or "predict_roberta failed")

        return {"raw": res.stdout}

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

# =======================================================
# MATCH ROUTE – Match selected TTPs to threat groups
# =======================================================
@app.route('/match', methods=['POST'])
def match():
    try:
        # Collect only clean Technique IDs
        ttps_input = [t.split()[0].upper() for t in request.form.getlist('ttps[]')]
        ttps = validate_ttps(ttps_input)

        matched_df = match_ttps(ttps, DATA_DIR)

        # Convert numeric fields
        matched_df['rank'] = pd.to_numeric(matched_df.get('rank', float('nan')), errors='coerce')
        matched_df['score'] = pd.to_numeric(matched_df.get('score', float('nan')), errors='coerce')

        # Sort results
        if matched_df['rank'].notna().any():
            matched_df = matched_df.sort_values(by='rank', ascending=True)
        else:
            matched_df = matched_df.sort_values(by='score', ascending=False)

        # Top 3
        top3_df = matched_df.head(3)

        # Save traceability outputs
        matched_df.to_csv("matched_groups.csv", index=False)
        top3_df.to_csv("matched_top3.csv", index=False)
        pd.DataFrame({"TTP": ttps}).to_csv("inputted_ttps.csv", index=False)

        # GPT Analysis
        mit_csv_path = _run_mitigations_and_get_csv()
        gpt_response = analyze_TTP(ttps, matched_df, mitigations_csv=str(mit_csv_path))
        parsed = parse_ai_response(gpt_response)
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

        # strip CSS link as you did
        rendered = re.sub(r'<link rel="stylesheet" href="[^"]*attribution\.css">', "", rendered, flags=re.IGNORECASE)

        response = make_response(rendered)
        response.headers["Content-Type"] = "application/msword"
        fname = "threat_report_roberta.doc" if which == "roberta" else "threat_report_rule.doc"
        response.headers["Content-Disposition"] = f"attachment; filename={fname}"
        return response

    except Exception as e:
        return f"Error exporting to .doc: {e}"

# =======================================================
# MAIN ENTRY POINT
# =======================================================
def main():
    import argparse
    ap = argparse.ArgumentParser(description="Build dataset + train RoBERTa (skip if already built).")
    ap.add_argument("--force", action="store_true", help="Re-run steps even if outputs exist.")
    ap.add_argument("--skip-build", action="store_true", help="Skip dataset build step.")
    ap.add_argument("--skip-train", action="store_true", help="Skip training step.")
    args = ap.parse_args()

    print(f"[ROOT] {BASE_DIR}")
    print(f"[DATA] {DATA_ROOT}")
    print(f"[PROC] {PROCESSED_DIR}")
    print(f"[MODELS] {MODELS_ROOT}")

    if not args.skip_build:
        step_build_dataset(force=args.force)
    else:
        print("[SKIP] Step: build dataset")

    if not args.skip_train:
        if not DATASET_CSV.exists():
            print(f"[WARN] {DATASET_CSV} not found; training may fail. Run without --skip-build or use --force.")
        step_train_roberta(force=args.force)
    else:
        print("[SKIP] Step: train roberta")

    ok = all(p.exists() for p in BEST_REQUIRED)
    print(f"\nBest model: {BEST_MODEL_DIR} {'(OK)' if ok else '(incomplete)'}")

if __name__ == '__main__':
    app.run(debug=True, threaded=True)
