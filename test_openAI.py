import os
import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from docx import Document
from datetime import datetime

load_dotenv()
key = os.getenv("OPENAI_API_KEY")

if not key:
    raise SystemExit("OPENAI_API_KEY not found in .env")

client = OpenAI(api_key=key)

def load_csv_data():
    if not os.path.exists("inputted_ttps.csv"):
        raise SystemExit("inputted_ttps.csv not found. Run matching_test.py first.")
    if not os.path.exists("matched_groups.csv"):
        raise SystemExit("matched_groups.csv not found. Run matching_test.py first.")

    ttps_df = pd.read_csv("inputted_ttps.csv")
    matched_df = pd.read_csv("matched_groups.csv")

    input_ttps = ttps_df["TTP"].dropna().tolist()
    return input_ttps, matched_df

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
[List top 3 matching groups with overlap count]

**Most Likely Attacker:**
[Actor Name] – Confidence: [High/Medium/Low]
---
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

def generate_word_report(report_text, input_ttps):
    doc = Document()
    doc.add_heading("Threat Attribution Report", level=1)
    doc.add_paragraph(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    doc.add_paragraph(f"Detected TTPs: {', '.join(input_ttps)}")
    doc.add_paragraph("")
    doc.add_heading("AI-Generated Analysis", level=2)
    doc.add_paragraph(report_text)
    filename = f"Threat_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.docx"
    doc.save(filename)
    print(f"Report saved as {filename}")

if __name__ == "__main__":
    print("Running AI Threat Analysis...")
    input_ttps, matched_df = load_csv_data()
    report = analyze_TTP(input_ttps, matched_df)
    generate_word_report(report, input_ttps)
