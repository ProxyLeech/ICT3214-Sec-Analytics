# src/flows/rules_flow.py
from __future__ import annotations
import pandas as pd
from pathlib import Path
from matching import match_ttps
from report_generator import analyze_TTP, parse_ai_response, load_filtered_mitigations, summarize_mitigations, generate_word_report
from common_flow import (
    BASE_DIR, _run_mitigations_and_get_csv, _atomic_to_csv,
    _ensure_score_and_rank_rule, _collect_top_group_ttps
)

DATA_DIR = BASE_DIR / "Data" / "mapped"

def _run_rule_match_flow(ttps: list[str]) -> dict:
    matched_df = match_ttps(ttps, DATA_DIR).copy()
    matched_df = _ensure_score_and_rank_rule(matched_df)
    if "rank" in matched_df.columns and matched_df["rank"].notna().any():
        matched_df = matched_df.sort_values(by=["rank", "score"], ascending=[True, False])
    else:
        matched_df = matched_df.sort_values(by="score", ascending=False)
    top3_df = matched_df.head(3)

    matched_df.to_csv("matched_groups_rule.csv", index=False)
    pd.DataFrame({"TTP": ttps}).to_csv("inputted_ttps.csv", index=False)

    # Mitigations (TOP actor only)
    mit_csv_path = _run_mitigations_and_get_csv()
    gpt_response = analyze_TTP(ttps, matched_df, mitigations_csv=str(mit_csv_path))
    parsed = parse_ai_response(gpt_response)

    top_group_ttps = _collect_top_group_ttps(matched_df)
    # --- limit mitigations to the TOP matched actor (like RoBERTa) ---
    top_group_ttps = _collect_top_group_ttps(matched_df)
    if not top_group_ttps:
        # Fall back to the user’s input TTPS (keeps report useful if mapping fails)
        print("[INFO] No top-group TTPs resolved for RULES; falling back to inputted TTPs.")
        top_group_ttps = list({t.strip().upper() for t in ttps if t})

    # mit_filtered = load_filtered_mitigations(str(mit_csv_path), top_group_ttps)

    else:
        mit_filtered = load_filtered_mitigations(str(mit_csv_path), top_group_ttps)
        if not mit_filtered.empty:
            mit_filtered = mit_filtered.drop_duplicates(
                subset=["target id", "target name", "mapping description"], keep="first"
            )
            parsed["mitigation"] = summarize_mitigations(mit_filtered.to_dict(orient="records"))
            _atomic_to_csv(mit_filtered, "mitigations_rule_top.csv")
            mit_for_docx = "mitigations_rule_top.csv"
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
