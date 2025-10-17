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
# def parse_ai_response(text):
#     """Extract structured sections from GPT response."""
#     sections = {
#         "summary": "", "table": "", "attacker": "",
#         "mitigation": "", "suggestion": ""
#     }

#     summary_match = re.search(r"\*\*Analysis Summary:\*\*(.*?)\*\*Overlap Table:", text, re.S)
#     table_match = re.search(r"\*\*Overlap Table:\*\*(.*?)\*\*Most Likely Attacker:", text, re.S)
#     attacker_match = re.search(r"\*\*Most Likely Attacker:\*\*(.*?)(\*\*Defensive Mitigations|\*\*Mitigations|\*\*Mitigation)", text, re.S)
#     mitigation_match = re.search(r"\*\*Defensive Mitigations.*?\*\*(.*?)\*\*Confidence Rating", text, re.S)
#     suggestion_match = re.search(r"\*\*Confidence Rating.*?\*\*(.*)", text, re.S)

#     if summary_match:
#         sections["summary"] = summary_match.group(1).strip()
#     if table_match:
#         sections["table"] = table_match.group(1).strip()
#     if attacker_match:
#         sections["attacker"] = attacker_match.group(1).strip()
#     if mitigation_match:
#         sections["mitigation"] = mitigation_match.group(1).strip()
#     if suggestion_match:
#         sections["suggestion"] = suggestion_match.group(1).strip()

#     # Strip markdown bold markers (e.g. **text**)
#     for key in sections:
#         sections[key] = re.sub(r"\*\*(.*?)\*\*", r"\1", sections[key])

#     return sections
def parse_ai_response(text: str) -> dict:
    """
    Parse GPT output into sections expected by the HTML templates:
      - summary
      - table
      - attacker
      - mitigation
      - suggestion

    Tolerates multiple heading variants and formats (bold markdown, H3, plain).
    """
    import re

    # Normalize whitespace and keep a working copy
    t = text.replace("\r\n", "\n").strip()

    # Map many possible headings to canonical tokens
    heading_patterns = {
        "SUMMARY": [
            r"\*\*\s*Analysis\s+Summary\s*:\s*\*\*", r"^#{1,6}\s*Analysis\s+Summary\b", r"\bAnalysis\s+Summary\s*:"
        ],
        "TABLE": [
            r"\*\*\s*Overlap\s*Table\s*:\s*\*\*", r"\*\*\s*Technique\s+Overlap\s+Overview\s*:\s*\*\*",
            r"^#{1,6}\s*(Overlap\s*Table|Technique\s+Overlap\s+Overview)\b", r"\b(Overlap\s*Table|Technique\s+Overlap\s+Overview)\s*:"
        ],
        "ATTACKER": [
            r"\*\*\s*Most\s+Likely\s+Attacker\s*:\s*\*\*", r"\*\*\s*Probable\s+Threat\s+Actor\(s\)\s*:\s*\*\*",
            r"^#{1,6}\s*(Most\s+Likely\s+Attacker|Probable\s+Threat\s+Actor\(s\))\b",
            r"\b(Most\s+Likely\s+Attacker|Probable\s+Threat\s+Actor\(s\))\s*:"
        ],
        "MITIGATION": [
            r"\*\*\s*Defensive\s+Mitigations\s*.*?\*\*", r"\*\*\s*Mitigations?\s*.*?\*\*",
            r"^#{1,6}\s*(Defensive\s+Mitigations?|Mitigations?)\b",
            r"\b(Defensive\s+Mitigations?|Mitigations?)\s*:"
        ],
        "SUGGESTION": [
            r"\*\*\s*Analyst\s+Recommendations\s*.*?\*\*", r"\*\*\s*Recommendations\s*.*?\*\*",
            r"\*\*\s*Confidence\s+Rating\s*.*?\*\*",  # sometimes content ends up under this
            r"^#{1,6}\s*(Analyst\s+Recommendations|Recommendations|Confidence\s+Rating)\b",
            r"\b(Analyst\s+Recommendations|Recommendations|Confidence\s+Rating)\s*:"
        ],
    }

    # Build a single alternation that turns any matching heading into a token line
    token_map = {k: f"[[{k}]]" for k in heading_patterns}
    alternations = []
    for pats in heading_patterns.values():
        alternations.extend(pats)
    big_rx = re.compile("(" + "|".join(pats for pats in alternations) + ")", re.IGNORECASE | re.MULTILINE)

    # Substitute matches with canonical tokens
    def _sub(m):
        s = m.group(0)
        for key, pats in heading_patterns.items():
            for p in pats:
                if re.fullmatch(p, s, flags=re.IGNORECASE):
                    return token_map[key]
        # If we got here, it's a partial match – pick first that contains
        for key, pats in heading_patterns.items():
            if any(re.search(p, s, re.IGNORECASE) for p in pats):
                return token_map[key]
        return s

    normalized = big_rx.sub(_sub, t)

    # Split into sections by tokens
    sections = {"summary": "", "table": "", "attacker": "", "mitigation": "", "suggestion": ""}
    order = ["SUMMARY", "TABLE", "ATTACKER", "MITIGATION", "SUGGESTION"]
    # Ensure the first token exists to simplify splitting logic
    if "[[SUMMARY]]" not in normalized:
        normalized = "[[SUMMARY]]\n" + normalized

    # Extract text between tokens
    def grab(block, start_tok, end_tok=None):
        start_idx = block.find(start_tok)
        if start_idx == -1:
            return ""
        start_idx += len(start_tok)
        if end_tok:
            end_idx = block.find(end_tok, start_idx)
            return block[start_idx:end_idx].strip() if end_idx != -1 else block[start_idx:].strip()
        return block[start_idx:].strip()

    cur = normalized
    for i, key in enumerate(order):
        start_tok = f"[[{key}]]"
        end_tok = f"[[{order[i+1]}]]" if i < len(order) - 1 else None
        val = grab(cur, start_tok, end_tok)
        # move window forward
        if end_tok:
            cur = cur[cur.find(end_tok):]
        # save
        k = key.lower()
        if k == "suggestion" and not val:
            # sometimes people dump suggestions under "Confidence Rating"
            pass
        sections_map = {
            "summary": "summary",
            "table": "table",
            "attacker": "attacker",
            "mitigation": "mitigation",
            "suggestion": "suggestion",
        }
        sections[sections_map[k]] = val

    # Strip stray bold markers
    for k, v in sections.items():
        sections[k] = re.sub(r"\*\*(.*?)\*\*", r"\1", v or "").strip()

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
