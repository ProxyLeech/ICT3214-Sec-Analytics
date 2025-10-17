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
)
from datetime import datetime
from collections import defaultdict
from technique_labels import extract_techniques  # import the extractor
import io
import re
import subprocess
import sys
from pathlib import Path

# ============================================
# Paths / App
# ============================================
BASE_DIR = Path(__file__).resolve().parent
SRC_PATH = BASE_DIR / "src"  # ✅ src is inside ICT3214-Sec-Analytics
sys.path.insert(0, str(SRC_PATH))
app = Flask(__name__)

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
PDFS_IN_DIR         = RAW_DIR / "pdfs"
EXTRACTED_IOCS_CSV  = PROCESSED_DIR / "extracted_iocs.csv"
TI_GROUPS_TECHS_CSV = PROCESSED_DIR / "ti_groups_techniques.csv"
DATASET_CSV         = PROCESSED_DIR / "dataset.csv"
LABELS_TXT          = PROCESSED_DIR / "labels.txt"

# Scripts (relative to repo root)
EXTRACT_SCRIPT       = DATASCRIPT_ROOT / "extract_pdfs.py"
ATTACK_SCRIPT        = DATASCRIPT_ROOT / "enterprise_attack.py"
BUILD_DATASET_SCRIPT = DATASCRIPT_ROOT / "build_dataset.py"
TRAIN_ROBERTA_SCRIPT = MODELS_ROOT     / "train_roberta.py"
PREDICT_SCRIPT       = MODELS_ROOT     / "predict_roberta.py"

# Trained model
BEST_MODEL_DIR = MODELS_ROOT / "best_roberta_for_predict"
BEST_REQUIRED = [
    BEST_MODEL_DIR / "config.json",
    BEST_MODEL_DIR / "tokenizer.json",
    BEST_MODEL_DIR / "id2label.json",
]

def output_dir_for_folds(n_folds: int, model_slug: str = "roberta_base"):
    return EXPERIMENTS_ROOT / f"{n_folds}foldruns" / model_slug

# Global caches
LAST_RESULTS = {}
LAST_RESULTS_RULE = {}
LAST_RESULTS_ROBERTA = {}

# ============================================
# Small helpers
# ============================================
def _ensure_score_and_rank(df: pd.DataFrame) -> pd.DataFrame:
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

def _collect_group_ttps(df: pd.DataFrame, ttps_input: list[str]) -> list[str]:
    """
    Determine technique IDs relevant to matched groups for mitigation filtering.
    Priority:
      1) technique_id column in df (if present)
      2) group_ttps_detail.csv lookup by matched group_name
      3) fallback to the input ttps list
    """
    # 1) direct technique_id column
    if "technique_id" in df.columns:
        vals = (
            df["technique_id"]
            .dropna()
            .astype(str).str.upper()
            .unique()
            .tolist()
        )
        if vals:
            return vals

    # 2) lookup by group name
    map_path = BASE_DIR / "Data" / "mapped" / "group_ttps_detail.csv"
    if map_path.exists():
        try:
            g = pd.read_csv(map_path)
            groups = df["group_name"].dropna().unique().tolist() if "group_name" in df.columns else []
            vals = (
                g[g["group_name"].isin(groups)]["technique_id"]
                .dropna()
                .astype(str).str.upper()
                .unique()
                .tolist()
            )
            if vals:
                return vals
        except Exception as e:
            print(f"[WARN] Failed reading group_ttps_detail.csv: {e}")

    # 3) fallback to input ttps
    return list({t.strip().upper() for t in ttps_input if t})

# ============================================
# Index & Workflow
# ============================================
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

        return render_template('index.html', grouped_ttps=grouped_ttps)

    except Exception as e:
        return render_template('error.html', error=str(e))

@app.route('/workflow')
def workflow():
    return render_template('workflow.html')

# ============================================
# Rule-based flow helper
# ============================================
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

    # Mitigations (idempotent) + GPT analysis
    mit_csv_path = _run_mitigations_and_get_csv()
    gpt_response = analyze_TTP(ttps, matched_df, mitigations_csv=str(mit_csv_path))
    parsed = parse_ai_response(gpt_response)

    # Filter mitigations for the matched groups' techniques
    group_ttps = _collect_group_ttps(matched_df, ttps)
    mit_filtered = load_filtered_mitigations(str(mit_csv_path), group_ttps)
    parsed["mitigation"] = mit_filtered.to_dict(orient="records")

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

# ============================================
# RoBERTa flow helper
# ============================================
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
        preds = pr.predict(
            text=text,
            threshold=float(payload.get("threshold", 0.5)),
            top_k=int(payload.get("top_k", 10)),
        )
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
        args += ["--threshold", str(payload.get("threshold", 0.5)), "--top-k", str(payload.get("top_k", 10))]

        res = subprocess.run(args, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(res.stderr or "predict_roberta failed")

        return {"raw": res.stdout}

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

    # Keep file names distinct for clarity
    df_ml.to_csv("matched_groups_roberta.csv", index=False)
    top3_df.to_csv("matched_top3_roberta.csv", index=False)
    pd.DataFrame({"TTP": ttps}).to_csv("inputted_ttps_rule.csv", index=False)  # shared for report_generator

    # Mitigations (idempotent) + GPT analysis
    mit_csv_path = _run_mitigations_and_get_csv()
    gpt_response = analyze_TTP(ttps, df_ml, mitigations_csv=str(mit_csv_path))
    parsed = parse_ai_response(gpt_response)

    # Filter mitigations for the matched groups' techniques
    group_ttps = _collect_group_ttps(df_ml, ttps)
    mit_filtered = load_filtered_mitigations(str(mit_csv_path), group_ttps)
    parsed["mitigation"] = mit_filtered.to_dict(orient="records")

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

# ============================================
# Routes
# ============================================
@app.route('/roberta', methods=['POST'])
def roberta():
    try:
        ttps_input = [t.split()[0].upper() for t in request.form.getlist('ttps[]')]
        ttps = validate_ttps(ttps_input)

        res = _run_roberta_flow(ttps)

        global LAST_RESULTS
        LAST_RESULTS = {
            "ttps": res["ttps"],
            "matched": res["matched_top3"],
            "analysis": res["analysis"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return render_template(
            'results.html',
            ttps=res["ttps"],
            matched=res["matched_top3"],
            analysis=res["analysis"],
            timestamp=LAST_RESULTS["timestamp"]
        )

    except Exception as e:
        return render_template('error.html', error=str(e))

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
        return jsonify({"status": "error", "message": f"Subprocess exited with code {int(e.code)}"}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

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

        # Save traceability outputs
        matched_df.to_csv("matched_groups_rule.csv", index=False)
        top3_df.to_csv("matched_top3_rule.csv", index=False)
        pd.DataFrame({"TTP": ttps}).to_csv("inputted_ttps_rule.csv", index=False)

        # GPT Analysis
        mit_csv_path = _run_mitigations_and_get_csv()
        gpt_response = analyze_TTP(ttps, matched_df, mitigations_csv=str(mit_csv_path))
        parsed = parse_ai_response(gpt_response)

        # Filter mitigations to matched group techniques (NOT first lines)
        group_ttps = _collect_group_ttps(matched_df, ttps)
        mit_filtered = load_filtered_mitigations(str(mit_csv_path), group_ttps)
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

# ============================================
# Pipeline steps
# ============================================
def step_extract_pdfs(force: bool) -> None:
    ensure_dirs()
    if not needs_run([EXTRACTED_IOCS_CSV], force):
        print(f"[SKIP] extract_pdfs.py — up to date: {EXTRACTED_IOCS_CSV}")
        return
    run([sys.executable, str(EXTRACT_SCRIPT)])

def step_enterprise_attack(force: bool) -> None:
    ensure_dirs()
    if not needs_run([TI_GROUPS_TECHS_CSV], force):
        print(f"[SKIP] enterprise_attack.py — up to date: {TI_GROUPS_TECHS_CSV}")
        return
    run([sys.executable, str(ATTACK_SCRIPT)])

def step_build_dataset(force: bool) -> None:
    ensure_dirs()
    if not needs_run([DATASET_CSV, LABELS_TXT], force):
        print(f"[SKIP] build_dataset.py — up to date: {DATASET_CSV}, {LABELS_TXT}")
        return
    run([sys.executable, str(BUILD_DATASET_SCRIPT)])

def step_train_roberta(force: bool) -> None:
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

# ============================================
# Main
# ============================================
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
