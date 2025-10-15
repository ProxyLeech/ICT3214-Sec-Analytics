from flask import Flask, render_template, request, make_response
import pandas as pd
from matching import (
    validate_ttps,
    match_ttps,
    find_ttp_column,
    split_tokens,
    load_combined_dataset
)
from report_generator import analyze_TTP, parse_ai_response
from pathlib import Path
from collections import defaultdict
import re
from datetime import datetime
import io

app = Flask(__name__)

# =======================================================
# Project-relative data path
# =======================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data" / "mapped"

# Global cache for last results
LAST_RESULTS = {}


# =======================================================
# INDEX ROUTE – Build dropdown of all MITRE TTPs
# =======================================================
@app.route('/')
def index():
    try:
        # Load and merge both datasets
        df = load_combined_dataset(DATA_DIR)

        # =======================================================
        # Parse and normalize TTPs (scan all relevant columns)
        # =======================================================
        valid_ttp_pattern = re.compile(r"^T\d{4}(?:\.\d{3})?$")
        ttp_dict = defaultdict(set)
        all_ttps = set()

        # --- Identify all columns that may contain TTPs ---
        ttp_columns = [
            c for c in df.columns
            if any(k in c.lower() for k in ["matched", "ttp", "technique", "attack"])
        ]

        # --- Parse across all those columns ---
        for col in ttp_columns:
            for val in df[col].dropna():
                for ttp in split_tokens(val):
                    ttp = ttp.strip().upper()
                    if not valid_ttp_pattern.match(ttp):
                        continue
                    all_ttps.add(ttp)
                    if "." in ttp:
                        root = ttp.split(".")[0]
                        ttp_dict[root].add(ttp)
                        all_ttps.add(root)
                    else:
                        ttp_dict.setdefault(ttp, set())

        # --- Cleanup: ensure every root is initialized ---
        for ttp in list(all_ttps):
            if "." in ttp:
                root = ttp.split(".")[0]
                ttp_dict.setdefault(root, set())

        # --- Sort for dropdown ---
        grouped_ttps = [(root, sorted(subs)) for root, subs in sorted(ttp_dict.items())]

        # =======================================================
        # Debug / Sanity check counts
        # =======================================================
        all_subs = {s for subs in ttp_dict.values() for s in subs}
        main_ttps = [t for t in all_ttps if "." not in t]
        sub_ttps = [t for t in all_ttps if "." in t]

        print("\n================ TTP SUMMARY ================")
        print(f"✅ Total unique TTP IDs (main + sub): {len(all_ttps)}")
        print(f"   ├── Main techniques: {len(main_ttps)}")
        print(f"   └── Sub-techniques:  {len(sub_ttps)}")
        print(f"✅ Total main technique entries in dict: {len(grouped_ttps)}")
        print(f"🧪 Is T1110 in parsed TTPs? {'T1110' in all_ttps}")
        print(f"🧪 T1110 subtechniques: {ttp_dict.get('T1110', '❌ Not Present')}")
        print("============================================\n")

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
# ROBERTA ROUTE – ....
# =======================================================
@app.route('/roberta', methods=['POST'])
def roberta():
    try:

        return False 
    except Exception as e:
        return render_template('error.html', error=str(e))
    
# =======================================================
# MATCH ROUTE – Match selected TTPs to threat groups
# =======================================================

@app.route('/match', methods=['POST'])
def match():
    try:
        ttps_input = request.form.getlist('ttps[]')
        ttps = validate_ttps(ttps_input)
        matched_df = match_ttps(ttps, DATA_DIR)

        # Convert rank to numeric safely
        matched_df['rank'] = pd.to_numeric(matched_df.get('rank', float('nan')), errors='coerce')
        matched_df['score'] = pd.to_numeric(matched_df.get('score', float('nan')), errors='coerce')

        # Prefer rank for sorting if available; fallback to score
        if matched_df['rank'].notna().any():
            matched_df = matched_df.sort_values(by='rank', ascending=True)  # lower rank = higher priority
        else:
            matched_df = matched_df.sort_values(by='score', ascending=False)

        # Take only top 3 entries
        top3_df = matched_df.head(3)

        # Save outputs for traceability
        matched_df.to_csv("matched_groups.csv", index=False)
        top3_df.to_csv("matched_top3.csv", index=False)
        pd.DataFrame({"TTP": ttps}).to_csv("inputted_ttps.csv", index=False)

        # Analyze via GPT
        gpt_response = analyze_TTP(ttps, matched_df)
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


# =======================================================
# EXPORT ROUTE - Output the results to .doc file
# =======================================================
@app.route('/export')
def export():
    try:
        if not LAST_RESULTS:
            return "No results available to export. Please generate a report first."


        # Render HTML with same template
        rendered = render_template(
            "results.html",
            ttps=LAST_RESULTS["ttps"],
            matched=LAST_RESULTS["matched"],
            analysis=LAST_RESULTS["analysis"],
            timestamp=LAST_RESULTS["timestamp"],
            export_mode=True
        )

        # Convert HTML output to downloadable .doc file
        response = make_response(rendered)
        response.headers["Content-Type"] = "application/msword"
        response.headers["Content-Disposition"] = "attachment; filename=threat_report.doc"
        return response

    except Exception as e:
        return f"Error exporting to .doc: {e}"


# =======================================================
# MAIN ENTRY POINT
# =======================================================
if __name__ == '__main__':
    app.run(debug=True, threaded=True)
