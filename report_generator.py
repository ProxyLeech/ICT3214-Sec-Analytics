import os
import re
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from docx import Document
from docx.shared import Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH
from datetime import datetime
from pathlib import Path

# ===========================
# Setup and Data Loading
# ===========================
load_dotenv()
key = os.getenv("OPENAI_API_KEY")
if not key:
    raise SystemExit("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=key)

def load_csv_data():
    if not os.path.exists("inputted_ttps.csv"):
        raise SystemExit("inputted_ttps.csv not found.")
    if not os.path.exists("matched_groups.csv"):
        raise SystemExit("matched_groups.csv not found.")

    ttps_df = pd.read_csv("inputted_ttps.csv")
    matched_df = pd.read_csv("matched_groups.csv")

    if "score" in matched_df.columns:
        matched_df = matched_df.sort_values(by="score", ascending=False)

    input_ttps = ttps_df["TTP"].dropna().tolist()
    return input_ttps, matched_df

# ===========================
# GPT Analysis
# ===========================
def analyze_TTP(input_ttps, matched_df):
    mapping_summary = "\n".join([
        f"{row['group_name']}: {row.to_dict()}"
        for _, row in matched_df.head(10).iterrows()
        if "group_name" in row
    ])

    prompt = f"""
You are a cyber threat intelligence analyst.
You are given:
1. A list of matched attacker groups (with details) from a TTP matching engine.
2. A list of detected MITRE ATT&CK TTPs from a security incident.

Your task:
- Compare the detected TTPs with known attacker groups.
- Identify the most likely actor(s) based on overlapping TTPs.
- Explain reasoning clearly and end with a confidence rating.

Matched Actor Groups:
{mapping_summary}

Detected Input TTPs:
{', '.join(input_ttps)}

Return your analysis formatted as:
---
**Analysis Summary:**
[Explanation]

**Overlap Table:**
| Actor Group | Overlap Count |
|--------------|---------------|
| Example APT  | 3 |
| Example B    | 2 |
| Example C    | 1 |

**Most Likely Attacker:**
[Actor Name] – Confidence: [High/Medium/Low]
---
"""

    prompt += """
Also include:
- A concise **summary** explaining overall findings and observed behavior patterns.
- A **justification** for why specific attacker groups are likely involved based on technique overlap.
- A section suggesting **defensive mitigations or detections** organizations can apply to counter these TTPs.
- Conclude with a short **confidence rating** (High/Medium/Low) and brief reasoning for this rating.

Keep the language concise, analytical, and professional.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.3,
        messages=[
            {"role": "system", "content": "You are an experienced MITRE ATT&CK threat analyst."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content.strip()

# ===========================
# Parse Response
# ===========================
def parse_ai_response(text):
    """Extract structured sections from GPT response."""
    sections = {
        "summary": "", "table": "", "attacker": "",
        "mitigation": "", "suggestion": ""
    }

    summary_match = re.search(r"\*\*Analysis Summary:\*\*(.*?)\*\*Overlap Table:", text, re.S)
    table_match = re.search(r"\*\*Overlap Table:\*\*(.*?)\*\*Most Likely Attacker:", text, re.S)
    attacker_match = re.search(r"\*\*Most Likely Attacker:\*\*(.*?)(\*\*Defensive Mitigations|\*\*Mitigations|\*\*Mitigation)", text, re.S)
    mitigation_match = re.search(r"\*\*Defensive Mitigations.*?\*\*(.*?)\*\*Confidence Rating", text, re.S)
    suggestion_match = re.search(r"\*\*Confidence Rating.*?\*\*(.*)", text, re.S)

    if summary_match:
        sections["summary"] = summary_match.group(1).strip()
    if table_match:
        sections["table"] = table_match.group(1).strip()
    if attacker_match:
        sections["attacker"] = attacker_match.group(1).strip()
    if mitigation_match:
        sections["mitigation"] = mitigation_match.group(1).strip()
    if suggestion_match:
        sections["suggestion"] = suggestion_match.group(1).strip()

    # Strip markdown bold markers (e.g. **text**)
    for key in sections:
        sections[key] = re.sub(r"\*\*(.*?)\*\*", r"\1", sections[key])

    return sections


# ===========================
# Generate Word Report
# ===========================
def generate_word_report(report_text, input_ttps):
    parsed = parse_ai_response(report_text)

    # Resolve base directory relative to this script
    base_dir = Path(__file__).resolve().parent
    template_path = base_dir / "templates" / "cleaned_report_template.docx"

    if not template_path.exists():
        raise FileNotFoundError(f"Template not found at: {template_path}")

    doc = Document(template_path)

    # Update metadata
    for para in doc.paragraphs:
        if "Generated on:" in para.text:
            para.text = f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        elif "Detected TTPs:" in para.text:
            para.text = f"Detected TTPs: {', '.join(input_ttps)}"

    # Populate main report sections
    for i, para in enumerate(doc.paragraphs):
        text = para.text.strip().lower()

        if "1. analysis summary" in text:
            doc.paragraphs[i + 1].text = parsed["summary"] or "N/A"

        elif "2. overlap table" in text:
            lines = [l.strip("| ").split("|") for l in parsed["table"].splitlines() if "|" in l]
            if len(lines) > 1 and len(doc.tables) > 0:
                table = doc.tables[0]
                for row_data in lines[1:]:
                    row = table.add_row().cells
                    row[0].text = row_data[0].strip()
                    row[1].text = row_data[1].strip()

        elif "3. most likely attacker" in text:
            doc.paragraphs[i + 1].text = parsed["attacker"] or "N/A"

        elif "4. defensive mitigations" in text:
            doc.paragraphs[i + 1].text = parsed["mitigation"] or "[Add blue-team detection strategies here.]"

        elif "5. analyst suggestions" in text:
            doc.paragraphs[i + 1].text = parsed["suggestion"] or "[Add reflection, lessons learned, or recommendations here.]"

    # Save generated file
    output_dir = base_dir / "Generated_Reports"
    output_dir.mkdir(exist_ok=True)
    filepath = output_dir / f"Threat_Report_{datetime.now():%Y%m%d_%H%M%S}.docx"
    doc.save(filepath)
    print(f"✅ Report saved: {filepath}")



# ===========================
# Run
# ===========================
if __name__ == "__main__":
    print("Running AI Threat Analysis...")
    input_ttps, matched_df = load_csv_data()
    report = analyze_TTP(input_ttps, matched_df)
    generate_word_report(report, input_ttps)
