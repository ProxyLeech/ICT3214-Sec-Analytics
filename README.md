# LLM-Assisted Adversary Attribution

A modular pipeline for automating cyber threat attribution using MITRE ATT&CK, APTnotes, and GPT-based analysis.

---

## Table of Contents

- [System Overview](#system-overview)
- [Workflow Overview](#workflow-overview)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Step-by-Step Logic Flow and Preparation](#step-by-step-logic-flow-and-preparation)
  - [1. Extract IOCs from APTnotes PDFs](#1-extract-iocs-from-aptnotes-pdfs)
  - [2. Map IOCs to MITRE ATT&CK Groups](#2-map-iocs-to-mitre-attck-groups)
  - [3. Match Specific MITRE TTPs](#3-match-specific-mitre-ttps)
  - [4. Generate the Intelligence Report](#4-generate-the-intelligence-report)
- [How to Use (Web App)](#how-to-use-web-app)
- [Notes](#notes)
- [Troubleshooting](#troubleshooting)

---

## System Overview

This toolchain extracts Indicators of Compromise (IOCs) and MITRE ATT&CK Technique IDs (TTPs) from APT reports, maps them to known attacker groups using MITRE STIX data, and generates structured intelligence reports via GPT-based analysis.

---

## Workflow Overview

APTnotes PDFs  
│  
├──► `extract_pdfs.py` → Extracted IOCs & ATT&CK IDs  
│  
MITRE ATT&CK Enterprise JSON  
│  
└──► `map_iocs_to_attack.py` → Mapped Threat Groups (CSV)  
  │  
  ├──► `matching_test.py` → User-Selected TTPs  
  │  
  └──► `report_generator.py` → GPT-based DOCX Intelligence Report  

---

## Dependencies

All dependencies are listed in `requirements.txt`.

To install them, follow the setup steps in the Installation section.

---

## Installation

1. **Create and activate a virtual environment:**

- Windows:  
  `python -m venv .venv`  
  `.venv\Scripts\activate`

- macOS/Linux:  
  `python3 -m venv .venv`  
  `source .venv/bin/activate`

2. **Install the dependencies:**

  pip install -r requirements.txt



3. **Create a `.env` file in the root directory with your API key:**

  OPENAI_API_KEY=sk-your-api-key-here



---

## Step-by-Step Logic Flow and Preparation

### 1. Extract IOCs from APTnotes PDFs

python extract_pdfs.py


This script extracts text, metadata, and Indicators of Compromise (IOCs) such as URLs, domains, IPs, hashes, and emails from PDF files into a folder called `extracted_pdfs`. It saves the extracted data as text files and metadata JSON, and writes all found IOCs to a CSV file called `extracted_iocs.csv` for further analysis.

- Input: Folder containing PDF reports (`aptnotes_pdfs/*.pdf`)  
- Output: `Data/extracted_pdfs/extracted_iocs.csv`

---

### 2. Map IOCs to MITRE ATT&CK Groups

python map_iocs_to_attack.py



This script maps observed IOCs from PDF reports to MITRE ATT&CK techniques and threat groups by cross-referencing CSV and JSON data. It ranks threat groups based on technique matches, outputs detailed CSV reports, and generates mini STIX bundles for each analyzed file.

- Input: `Data/extracted_pdfs/extracted_iocs.csv`  
- Output:
  - `Data/mapped/ranked_groups.csv`
  - `Data/mapped/group_ttps_detail.csv`

---

### 3. Match Specific MITRE TTPs

python matching_test.py



- Prompts you to enter up to five MITRE ATT&CK Technique IDs (e.g., `T1059.003`, `T1110`)  
- Checks for overlap with known threat groups based on your input  

- Output:
  - `matched_groups.csv`
  - `inputted_ttps.csv`

Note: This script is part of the prototype logic behind the dropdown interface in the Flask web app. It is not the core logic of the attribution pipeline but acts as a lightweight test utility for debugging and manual use.

---

### 4. Generate the Intelligence Report

python report_generator.py



This script uses the OpenAI API to automatically generate a structured threat intelligence report, based on your selected or extracted TTPs and mapped threat groups.

It consolidates results from previous steps and produces a professional, human-readable DOCX report that includes:

- Threat group summary  
- TTP overlap table  
- Most likely adversaries  
- Recommended defensive mitigations  
- GPT-generated analyst insights

- Output: `/Generated_Reports/Threat_Report_<timestamp>.docx`

Note: This is currently a standalone CLI prototype. This logic will be triggered by the "Export Report" button in the Flask web app.

---

## How to Use (Web App)

This is the intended user-facing experience via the Flask web interface:

1. Start the app  
   Run: `python app.py`

2. Visit the interface  
   Open your browser and go to: `http://127.0.0.1:5000`

3. Choose TTPs  
   Use the dropdown to select up to five MITRE Technique IDs (e.g., T1059.003, T1110)

4. Click "Submit Button"  
   The system will match your TTPs to known threat groups and show you the result

5. Export the Report  
   Click the "Export" button to download the generated DOCX report

---

## Notes

- You must download the MITRE ATT&CK JSON (`enterprise-attack.json`) and place it in `Data/attack_stix/`
- Input TTPs should follow MITRE format (e.g., `T1059.001`)
- Limit input to five TTPs max for optimal GPT performance
- Output folders are automatically created if missing
- Reports are timestamped and saved in `/Generated_Reports/`

---

## Troubleshooting

| Issue                        | Cause                          | Solution                              |
|-----------------------------|---------------------------------|----------------------------------------|
| PyMuPDF not found           | Missing dependency              | Run `pip install pymupdf`              |
| OPENAI_API_KEY not found    | `.env` file missing             | Add your key to `.env`                 |
| No PDFs found               | Incorrect input folder          | Ensure path to `aptnotes_pdfs/` is correct |
| Empty report output         | Invalid TTP input format        | Use valid MITRE IDs (e.g., `T1059.003`) |

---
