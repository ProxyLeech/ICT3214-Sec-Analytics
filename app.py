from flask import Flask, render_template, request, make_response
import pandas as pd
from matching import (
    validate_ttps,
    match_ttps,
)
from report_generator import analyze_TTP, parse_ai_response
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from technique_labels import extract_techniques  # import the extractor
import io
import re

app = Flask(__name__)

# =======================================================
# Project-relative paths
# =======================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data" / "mapped"
EXCEL_PATH = BASE_DIR / "Data" / "excel" / "enterprise-attack-v17.1-techniques.xlsx"
MAPPING_CSV = BASE_DIR / "techniques_mapping.csv"

# Global cache for last results
LAST_RESULTS = {}


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
# ROBERTA ROUTE (placeholder)
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

        css_path = BASE_DIR / "static" / "css" / "attribution.css"
        # Read the file contents into a variable
        inline_css = ""
        if css_path.exists():
            with open(css_path, "r", encoding="utf-8") as f:
                inline_css = f.read()
        else:
            print(f"CSS file not found: {css_path}")

        # Replace the link tag with a <style> containing the CSS
        #pattern = r'<link rel="stylesheet" href="[^"]*attribution\.css">(\s*<style>)?'
        pattern = r'<link rel="stylesheet" href="[^"]*attribution\.css">'
        rendered = re.sub(pattern,"", rendered, flags=re.IGNORECASE)

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
