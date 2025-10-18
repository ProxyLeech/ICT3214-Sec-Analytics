# src/flows/roberta_flow.py
from __future__ import annotations
import json, subprocess, sys, pandas as pd
from pathlib import Path
from report_generator import analyze_TTP, parse_ai_response, load_filtered_mitigations, summarize_mitigations, generate_word_report
from common_flow import (
    BASE_DIR, MODELS_ROOT, PREDICT_SCRIPT,
    _run_mitigations_and_get_csv, _atomic_to_csv,
    _ensure_score_and_rank, _collect_top_group_ttps
)

def _save_roberta_traces(df_ml):
    _atomic_to_csv(df_ml, "matched_groups_roberta.csv")

def _predict_with_module(payload: dict):
    """
    Try importing predict_roberta.py directly, fallback to subprocess.
    (Signature unchanged)
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
        flat = pr.expand_to_attack_rows(preds, attack_rows, attack_ids_in_input=attack_ids_in_input, origin_doc=origin_doc)
        group_rows = pr.aggregate_by_group(flat)
        return {"text": text, "groups": group_rows}
    except Exception as e:
        print(f"[WARN] Direct import predict_roberta failed: {e}; falling back to subprocess.")
        args = [sys.executable, str(PREDICT_SCRIPT), "--id", payload.get("id") or "adhoc"]
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
            print(res.stdout); print(res.stderr)
            raise RuntimeError(res.stderr or "predict_roberta failed")
        try:
            pl = json.loads(res.stdout)
            if isinstance(pl, dict) and "groups" in pl:
                return pl
            if isinstance(pl, list):
                return {"groups": pl}
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

    _save_roberta_traces(df_ml)
    top3_df = df_ml.head(3).drop(columns=["origin"], errors="ignore")

    mit_csv_path = _run_mitigations_and_get_csv()
    gpt_response = analyze_TTP(ttps, df_ml, mitigations_csv=str(mit_csv_path))
    parsed = parse_ai_response(gpt_response)

    top_group_ttps = _collect_top_group_ttps(df_ml)
    if not top_group_ttps:
        print("[INFO] No top-group TTPs resolved for ROBERTA; skipping mitigations (no group mapping).")
        mit_for_docx = None
        parsed["mitigation"] = "No mitigations found for these techniques."
    else:
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

    try:
        out_path = generate_word_report(
            gpt_response,
            ttps,
            mitigations_csv=mit_for_docx
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
