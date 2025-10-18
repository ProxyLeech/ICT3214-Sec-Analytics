from flask import Flask, render_template, request, make_response, jsonify
from datetime import datetime
from collections import defaultdict
from typing import Iterable
from pathlib import Path
import subprocess, sys, os, tempfile, re, json

import pandas as pd

from matching import validate_ttps, match_ttps
from report_generator import (
    analyze_TTP,
    parse_ai_response,
    generate_word_report,
    load_filtered_mitigations,
    summarize_mitigations,
)

# =========================
# App & paths
# =========================
BASE_DIR = Path(__file__).resolve().parent
SRC_PATH = BASE_DIR / "src"
sys.path.insert(0, str(SRC_PATH))
app = Flask(__name__)

DATA_ROOT       = BASE_DIR / "data"
RAW_DIR         = DATA_ROOT / "raw"
PROCESSED_DIR   = DATA_ROOT / "processed"
SRC_ROOT        = BASE_DIR / "src"
MODELS_ROOT     = SRC_ROOT / "models"
DATASCRIPT_ROOT = SRC_ROOT / "data"
EXPERIMENTS_ROOT= BASE_DIR / "experiments"

DATA_DIR        = BASE_DIR / "Data" / "mapped"
EXCEL_PATH      = BASE_DIR / "Data" / "excel" / "enterprise-attack-v17.1-techniques.xlsx"
MAPPING_CSV     = BASE_DIR / "techniques_mapping.csv"

PDFS_IN_DIR          = RAW_DIR / "pdfs"
EXTRACTED_PDFS_DIR   = DATA_ROOT / "extracted_pdfs"
EXTRACTED_IOCS_CSV   = PROCESSED_DIR / "extracted_iocs.csv"
TI_GROUPS_TECHS_CSV  = PROCESSED_DIR / "ti_groups_techniques.csv"
DATASET_CSV          = PROCESSED_DIR / "dataset.csv"
LABELS_TXT           = PROCESSED_DIR / "labels.txt"

EXTRACT_SCRIPT       = DATASCRIPT_ROOT / "extract_pdfs.py"
ATTACK_SCRIPT        = DATASCRIPT_ROOT / "enterprise_attack.py"
BUILD_DATASET_SCRIPT = DATASCRIPT_ROOT / "build_dataset.py"
TRAIN_ROBERTA_SCRIPT = MODELS_ROOT     / "train_roberta.py"
PREDICT_SCRIPT       = MODELS_ROOT     / "predict_roberta.py"

BEST_MODEL_DIR = MODELS_ROOT / "best_roberta_for_predict"
BEST_REQUIRED  = [BEST_MODEL_DIR / p for p in ("config.json","tokenizer.json","id2label.json")]

# In-memory caches
LAST_RESULTS = {}
LAST_RESULTS_RULE = {}
LAST_RESULTS_ROBERTA = {}

# =========================
# Utilities
# =========================
def ensure_dirs():
    for p in [DATA_ROOT, RAW_DIR, EXTRACTED_PDFS_DIR, PROCESSED_DIR, MODELS_ROOT, EXPERIMENTS_ROOT]:
        p.mkdir(parents=True, exist_ok=True)

def run(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n$ {' '.join(map(str, cmd))}")
    res = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if res.returncode != 0:
        raise SystemExit(res.returncode)

def needs_run(outputs: Iterable[Path], inputs: Iterable[Path] = ()) -> bool:
    outs = list(outputs)
    if not outs or any(not p.exists() for p in outs):
        return True
    out_mtime = min(p.stat().st_mtime for p in outs)
    ins = [Path(p) for p in inputs if p is not None and Path(p).exists()]
    return bool(ins) and (max(p.stat().st_mtime for p in ins) > out_mtime)

def _atomic_to_csv(df: pd.DataFrame, path: str):
    d = Path(path).parent
    d.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=d, newline="", suffix=".tmp") as tmp:
        df.to_csv(tmp, index=False)
        tmp_name = tmp.name
    os.replace(tmp_name, path)

def _ensure_score_and_rank(df: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    if df is None or df.empty:
        return df
    # handle both classic and ML outputs
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

# =========================
# Mitigations CSV (idempotent)
# =========================
def _run_mitigations_and_get_csv() -> Path:
    script  = BASE_DIR / "mitigations.py"
    out_csv = BASE_DIR / "Data" / "mitigations" / "mitigations.csv"
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

# =========================
# Group → TTPs (strict) from mapped CSV
# =========================
def _collect_group_ttps(matched_df: pd.DataFrame) -> list[str]:
    map_path = BASE_DIR / "Data" / "mapped" / "group_ttps_detail.csv"
    if not map_path.exists():
        print(f"[ERROR] {map_path} not found.")
        return []
    try:
        g = pd.read_csv(map_path)
    except Exception as e:
        print(f"[ERROR] Failed reading {map_path}: {e}")
        return []
    expected_cols = {"group_name", "group_id", "matched_exact", "matched_root_only"}
    missing = expected_cols - set(c.lower() for c in g.columns)
    if missing:
        print(f"[WARN] group_ttps_detail.csv missing columns: {missing}; best-effort extraction.")
    id_re = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
    def _ids(text: str) -> list[str]:
        if not isinstance(text, str): return []
        return [s.upper() if s.upper().startswith("T") else f"T{s}" for s in id_re.findall(text)]
    all_ttps = []
    for col in ("matched_exact", "matched_root_only"):
        if col in g.columns:
            for entry in g[col].fillna("").tolist():
                all_ttps.extend(_ids(entry))
    def _sort_key(tid: str):
        m = re.match(r"T(\d{4})(?:\.(\d{3}))?", tid)
        return (int(m.group(1)), int(m.group(2) or 999)) if m else (9999, 999)
    return sorted(set(all_ttps), key=_sort_key)

# =========================
# Rule flow
# =========================
def _run_rule_match_flow(ttps: list[str]) -> dict:
    matched_df = match_ttps(ttps, DATA_DIR).copy()
    matched_df = _ensure_score_and_rank(matched_df)
    matched_df = (matched_df.sort_values(by=["rank","score"], ascending=[True,False])
                  if matched_df["rank"].notna().any()
                  else matched_df.sort_values(by="score", ascending=False))
    top3_df = matched_df.head(3)

    _atomic_to_csv(matched_df, "matched_groups_rule.csv")
    _atomic_to_csv(top3_df,   "matched_top3_rule.csv")
    _atomic_to_csv(pd.DataFrame({"TTP": ttps}), "inputted_ttps.csv")

    mit_csv_path = _run_mitigations_and_get_csv()
    gpt_response = analyze_TTP(ttps, matched_df, mitigations_csv=str(mit_csv_path))
    parsed = parse_ai_response(gpt_response)

    group_ttps = _collect_group_ttps(matched_df) or list({t.strip().upper() for t in ttps if t})
    mit_filtered = load_filtered_mitigations(str(mit_csv_path), group_ttps)
    parsed["mitigation"] = ("No mitigations found for these techniques."
                            if mit_filtered.empty
                            else summarize_mitigations(mit_filtered.to_dict(orient="records")))

    try:
        out_path = generate_word_report(gpt_response, ttps, mitigations_csv=str(mit_csv_path)) or "threat_report_rule.docx"
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

# =========================
# RoBERTa flow
# =========================
def _predict_with_module(payload: dict):
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("predict_roberta", str(PREDICT_SCRIPT))
        pr = importlib.util.module_from_spec(spec); assert spec and spec.loader
        spec.loader.exec_module(pr)  # type: ignore

        text   = pr.build_text(
            report_id=payload.get("id") or "adhoc",
            urls=payload.get("urls") or [], domains=payload.get("domains") or [],
            ips=payload.get("ips") or [], md5s=payload.get("md5s") or [],
            sha256s=payload.get("sha256s") or [], attack_ids=payload.get("attacks") or [],
            free_text=payload.get("text") or None,
        )
        preds  = pr.predict(text=text, threshold=float(payload.get("threshold", 0.5)),
                            top_k=int(payload.get("top_k", 10)))
        rows   = pr.load_attack_index()
        srcs   = pr.resolve_sources_from_inputs(
            payload.get("urls") or [], payload.get("domains") or [], payload.get("ips") or [],
            payload.get("md5s") or [], payload.get("sha256s") or [], payload.get("attacks") or []
        )
        origin_doc = ", ".join(srcs) if srcs else (payload.get("id") or "adhoc")
        attack_ids_in_input = {a.strip().upper() for a in (payload.get("attacks") or []) if a}
        flat   = pr.expand_to_attack_rows(preds, rows, attack_ids_in_input=attack_ids_in_input, origin_doc=origin_doc)
        groups = pr.aggregate_by_group(flat)
        return {"text": text, "groups": groups}

    except Exception as e:
        print(f"[WARN] Direct import predict_roberta failed: {e}; falling back to subprocess.")
        args = [sys.executable, str(PREDICT_SCRIPT), "--id", payload.get("id") or "adhoc"]
        for k, flag in [("urls","--url"),("domains","--domain"),("ips","--ip"),("md5s","--md5"),("sha256s","--sha256"),("attacks","--attack")]:
            for v in payload.get(k) or []:
                args += [flag, str(v)]
        if payload.get("text"): args += ["--text", payload["text"]]
        args += ["--threshold", str(payload.get("threshold", 0.5)), "--top-k", str(payload.get("top_k", 10))]
        res = subprocess.run(args, capture_output=True, text=True)
        if res.returncode != 0:
            print(res.stdout); print(res.stderr)
            raise RuntimeError(res.stderr or "predict_roberta failed")

        try:
            payload = json.loads(res.stdout)
            if isinstance(payload, dict) and "groups" in payload: return payload
            if isinstance(payload, list): return {"groups": payload}
            return {"groups": []}
        except Exception:
            print("[WARN] CLI output was not JSON; returning empty groups.")
            return {"groups": []}

def _run_roberta_flow(ttps: list[str]) -> dict:
    ml = _predict_with_module({"id":"from_ttps","attacks":ttps,"top_k":50,"threshold":0.0})
    group_rows = ml.get("groups", [])
    if not group_rows:
        raise RuntimeError("RoBERTa returned no groups. Check model artifacts and labels.")

    df_ml = pd.DataFrame(group_rows)
    if "group" in df_ml.columns and "group_name" not in df_ml.columns:
        df_ml.rename(columns={"group":"group_name"}, inplace=True)

    df_ml = _ensure_score_and_rank(df_ml).sort_values("score", ascending=False)
    _atomic_to_csv(df_ml, "matched_groups_roberta.csv")
    top3_df = df_ml.head(3)

    mit_csv_path = _run_mitigations_and_get_csv()
    gpt_response = analyze_TTP(ttps, df_ml, mitigations_csv=str(mit_csv_path))
    parsed = parse_ai_response(gpt_response)

    group_ttps = _collect_group_ttps(df_ml) or list({t.strip().upper() for t in ttps if t})
    mit_filtered = load_filtered_mitigations(str(mit_csv_path), group_ttps)
    parsed["mitigation"] = ("No mitigations found for these techniques."
                            if mit_filtered.empty
                            else summarize_mitigations(mit_filtered.to_dict(orient="records")))

    try:
        out_path = generate_word_report(gpt_response, ttps, mitigations_csv=str(mit_csv_path)) or "threat_report_roberta.docx"
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

# =========================
# Pipeline “steps” (optional on /)
# =========================
def step_extract_pdfs(in_dir: Path = PDFS_IN_DIR, out_dir: Path = EXTRACTED_PDFS_DIR):
    out_dir.mkdir(parents=True, exist_ok=True)
    inputs = [EXTRACT_SCRIPT] + list(in_dir.glob("*.pdf"))
    if not needs_run([EXTRACTED_IOCS_CSV], inputs=inputs):
        print(f"[SKIP] extract_pdfs.py — up to date: {EXTRACTED_IOCS_CSV}"); return
    run([sys.executable, str(EXTRACT_SCRIPT), "--in", str(in_dir), "--out", str(out_dir)])

def step_enterprise_attack():
    if not needs_run([TI_GROUPS_TECHS_CSV], inputs=[ATTACK_SCRIPT]):
        print(f"[SKIP] enterprise_attack.py — up to date: {TI_GROUPS_TECHS_CSV}"); return
    run([sys.executable, str(ATTACK_SCRIPT)])

def step_build_dataset():
    if not needs_run([DATASET_CSV, LABELS_TXT], inputs=[BUILD_DATASET_SCRIPT, EXTRACTED_IOCS_CSV, TI_GROUPS_TECHS_CSV]):
        print(f"[SKIP] build_dataset.py — up to date: {DATASET_CSV}, {LABELS_TXT}"); return
    run([sys.executable, str(BUILD_DATASET_SCRIPT)])

def _is_empty_dir(p: Path) -> bool:
    return (not p.exists()) or (next(p.iterdir(), None) is None)

def step_train_roberta():
    if not _is_empty_dir(BEST_MODEL_DIR):
        print(f"[SKIP] train_roberta.py — best model exists in {BEST_MODEL_DIR}"); return
    run([sys.executable, str(TRAIN_ROBERTA_SCRIPT)])

# =========================
# Routes
# =========================
from technique_labels import extract_techniques

@app.route("/")
def index():
    try:
        ensure_dirs()
        extract_techniques(EXCEL_PATH, MAPPING_CSV)
        df_map = pd.read_csv(MAPPING_CSV)

        ttp_dict = defaultdict(list)
        for tid, label in zip(df_map["id"], df_map["label"]):
            root = tid.split(".")[0]
            (ttp_dict[root].append((tid, label)) if "." in tid else ttp_dict.setdefault(tid, []))

        id_to_label = dict(zip(df_map["id"], df_map["label"]))
        grouped_ttps = []
        for root in sorted(ttp_dict.keys()):
            root_label = id_to_label.get(root, root)
            subs = [lbl for _, lbl in sorted(ttp_dict[root], key=lambda x: x[0])]
            grouped_ttps.append((root_label, subs))

        # (Optional) kick off pipeline steps if you want:
        # step_extract_pdfs(); step_enterprise_attack(); step_build_dataset(); step_train_roberta()

        return render_template("index.html", grouped_ttps=grouped_ttps)
    except Exception as e:
        return render_template("error.html", error=str(e))

@app.route("/workflow")
def workflow():
    return render_template("workflow.html")

@app.route("/roberta", methods=["POST"])
def roberta():
    try:
        ttps = validate_ttps([t.split()[0].upper() for t in request.form.getlist("ttps[]")])
        res = _run_roberta_flow(ttps)
        global LAST_RESULTS
        LAST_RESULTS = {
            "ttps": res["ttps"],
            "matched": res["matched_top3"],
            "analysis": res["analysis"],
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        return render_template("results.html",
                              ttps=res["ttps"], matched=res["matched_top3"],
                              analysis=res["analysis"], timestamp=LAST_RESULTS["timestamp"])
    except Exception as e:
        return render_template("error.html", error=str(e))

@app.route("/results", methods=["POST"])
def results():
    try:
        ttps = validate_ttps([t.split()[0].upper() for t in request.form.getlist("ttps[]")])
        rule_res = _run_rule_match_flow(ttps)
        rob_res  = _run_roberta_flow(ttps)

        global LAST_RESULTS_RULE, LAST_RESULTS_ROBERTA
        LAST_RESULTS_RULE = {
            "ttps": rule_res["ttps"], "matched": rule_res["matched_top3"],
            "analysis": rule_res["analysis"], "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        LAST_RESULTS_ROBERTA = {
            "ttps": rob_res["ttps"], "matched": rob_res["matched_top3"],
            "analysis": rob_res["analysis"], "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        return render_template("results.html",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            ttps=ttps,
            rule_matched=rule_res["matched_top3"], rule_analysis=rule_res["analysis"],
            rob_matched=rob_res["matched_top3"],  rob_analysis=rob_res["analysis"],
            export_mode=False)

    except Exception as e:
        return render_template("error.html", error=str(e)), 500

@app.route("/predict", methods=["POST"])
def predict_api():
    try:
        payload = request.get_json(force=True, silent=False) or {}
        result = _predict_with_module(payload)
        return jsonify({"status": "ok", **result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 400

@app.route("/match", methods=["POST"])
def match():
    try:
        ttps = validate_ttps([t.split()[0].upper() for t in request.form.getlist("ttps[]")])
        matched_df = match_ttps(ttps, DATA_DIR)
        matched_df["rank"]  = pd.to_numeric(matched_df.get("rank"), errors="coerce")
        matched_df["score"] = pd.to_numeric(matched_df.get("score"), errors="coerce")
        matched_df = (matched_df.sort_values(by="rank", ascending=True)
                      if matched_df["rank"].notna().any()
                      else matched_df.sort_values(by="score", ascending=False))
        top3_df = matched_df.head(3)

        _atomic_to_csv(matched_df, "matched_groups_rule.csv")
        _atomic_to_csv(top3_df,   "matched_top3_rule.csv")
        _atomic_to_csv(pd.DataFrame({"TTP": ttps}), "inputted_ttps.csv")

        mit_csv_path = _run_mitigations_and_get_csv()
        gpt_response = analyze_TTP(ttps, matched_df, mitigations_csv=str(mit_csv_path))
        parsed = parse_ai_response(gpt_response)

        group_ttps = list({t.strip().upper() for t in _collect_group_ttps(matched_df)})
        mit_filtered = load_filtered_mitigations(str(mit_csv_path), group_ttps)
        if not mit_filtered.empty:
            mit_filtered = mit_filtered.drop_duplicates(subset=["target id","target name","mapping description"], keep="first")
        parsed["mitigation"] = mit_filtered.to_dict(orient="records")

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        global LAST_RESULTS
        LAST_RESULTS = {"ttps": ttps, "matched": top3_df.to_dict(orient="records"),
                        "analysis": parsed, "timestamp": timestamp}

        return render_template("results.html",
            ttps=ttps, matched=top3_df.to_dict(orient="records"),
            analysis=parsed, timestamp=timestamp)
    except Exception as e:
        return render_template("error.html", error=str(e))

@app.route("/export")
def export():
    try:
        which = (request.args.get("which") or "rule").lower()
        ctx = LAST_RESULTS_ROBERTA if which == "roberta" else (LAST_RESULTS_RULE or LAST_RESULTS)
        if not ctx:
            return "No results available to export. Please generate a report first."
        rendered = render_template("results.html",
                                   ttps=ctx["ttps"], matched=ctx["matched"],
                                   analysis=ctx["analysis"], timestamp=ctx["timestamp"],
                                   export_mode=True)
        rendered = re.sub(r'<link rel="stylesheet" href="[^"]*attribution\.css">', "", rendered, flags=re.IGNORECASE)
        response = make_response(rendered)
        response.headers["Content-Type"] = "application/msword"
        response.headers["Content-Disposition"] = f'attachment; filename={"threat_report_roberta.doc" if which=="roberta" else "threat_report_rule.doc"}'
        return response
    except Exception as e:
        return f"Error exporting to .doc: {e}"

if __name__ == "__main__":
    app.run(debug=True, threaded=True)
