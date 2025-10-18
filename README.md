# LLM-Assisted Adversary Attribution

A modular pipeline for automating cyber threat attribution using MITRE ATT&CK, APTnotes, and GPT-based analysis. 

---

## Table of Contents

- [System Overview](#system-overview)
- [Workflow Overview](#workflow-overview)
- [System Architecture](#system-architecture)
- [Dependencies](#dependencies)
- [Installation](#installation)
- [Prerequisites to run](#prerequisites-to-run)
  - [1. Extract IOCs from APTnotes PDFs](#1-extract-iocs-from-aptnotes-pdfs)
  - [2. Map IOCs to MITRE ATT&CK Groups](#2-map-iocs-to-mitre-attck-groups)
  - [3. Process and Normalize the MITRE Dataset](#3-process-and-normalize-the-mitre-dataset)
  - [4. Build the Labeled Dataset for Roberta Model Training](#4-build-the-labeled-dataset-for-roberta-model-training)
- [Usage](#usage)
- [Notes](#notes)
- [Troubleshooting](#troubleshooting)

---

## System Overview

This toolchain extracts Indicators of Compromise (IOCs) and MITRE ATT&CK Technique IDs (TTPs) from APT reports, maps them to known attacker groups using MITRE STIX data, and generates structured intelligence reports via GPT-based analysis. It supports two analysis modes. Rule-based TTP matching and Roberta model inference, both of which converge into a unified GPT-assisted report generation stage.

---

## System Architecture
```
ICT3214-SEC-ANALYTICS/
├──► app.py - Flask web app
├──► data/
│     ├──► raw/
│           ├──► attack_stix/ - contains enterprise-attack from MITRE
│           ├──► pdfs/ - contains APTnotes PDF files
│     ├──► excel/ - contains extracted enterprise-attack MITRE ATT&CK & mitigation techniques
│     ├──► extracted_pdfs/ - contains extracted information from data/raw/pdfs. This will be generated when app.py is run
│     ├──► mapped/ - HELP
│     ├──► mappedPDFs/ - HELP
│     ├──► processed/ 
│              ├──► rules/ - HELP
│              ├──► extracted_iocs.csv - Stores extracted Indicators of Compromise (IOCs) such as domains, IPs, URLs, hashes, and emails parsed from APTnotes PDFs for later threat group mapping.
│              ├──► ti_groups_techniques.csv - HELP
├──► logs/ - contains materials used to train RoBERTa model  
├──► src/
│     ├──► data/ - contains scripts to create .csv in data/ folders 
│     ├──► models/ - contains latest and best RoBERTa model as well as scripts to train and predict
│     ├──► paths/ - contains script that has static variables used by other scripts that is related to path locations 
├──► templates/
│       ├──► common/ - contains html that are used by all other pages
│       ├──► cleaned_report_template.docx - report template used by OpenAI to generate the adversary attribution report
│       ├──► index.html - Serves as the main landing page and user interface for the Flask-based MITRE ATT&CK Threat Attribution system. 
│       ├──► error.html - Error page rendered when invalid input, missing files, or API-related exceptions occur during app execution.
│       ├──► results_compare.html - Displays the dual-output comparison between the rule-based and RoBERTa-based threat attribution flows.  
│       ├──► results.html -  Displays **LLM-assisted adversary attribution results** for both **rule-based matching pipeline** and the **machine learning (RoBERTa)**
│      
├──► matching.py - Validates and normalizes user-entered MITRE ATT&CK TTPs and performs correlation to identify matching threat groups based on exact and root-technique overlaps; outputs ranked matches and input TTP lists for downstream analysis and report generation.
├──► mitigations.py - HELP
├──► report_generation.py - script to generate report using OpenAI and the results of RoBERTa, Matching and Mitigations
├──► technique_labels.py - HELP
├──► requirements.txt - list of dependencies that need to be installed via "pip install -r requirements"
```
---

## Workflow Overview

```
APTnotes PDFs
│
├──► extract_pdfs.py → Extracts IOCs & TTPs from APTnotes PDF reports
│
├──► build_dataset.py → Creates labeled dataset for the RoBERTa model
│
├──► enterprise_attack.py → Processes and normalizes the MITRE ATT&CK Enterprise dataset, extracting relationships between techniques, mitigations, and associated tactics
│
├──► map_iocs_to_attack.py → Correlates extracted IOCs and ATT&CK technique IDs with known MITRE ATT&CK threat groups
│
├──► matching.py → Matches input TTPs using both rule-based and RoBERTa inference modes
│
├──► mitigations.py → Retrieves defensive mitigations corresponding to the TTPs associated with matched groups
│
└──► report_generator.py → Generates GenAI-based structured intelligence reports summarizing group matches, mitigations, and analyst insights
```


---

## Dependencies

All dependencies are listed in `requirements.txt`.

To install them, follow the setup steps in the Installation section.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/ProxyLeech/ICT3214-Sec-Analytics
cd ICT3214-Sec-Analytics

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# On macOS / Linux:
source .venv/bin/activate
# On Windows:
.venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Create a .env file and add your OpenAI API key
# (Replace YOUR_KEY_HERE with your actual API key, look in user manual for more details)
echo "OPENAI_API_KEY=YOUR_KEY_HERE" > .env
```



---

## Prerequisites to run

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

### 3. Process and Normalize the MITRE Dataset

python enterprise_attack.py


This script parses the local MITRE ATT&CK Enterprise bundle (enterprise-attack.json) and builds the relational mapping between intrusion sets (groups) and techniques/sub-techniques.
The resulting CSV is essential for both the IOC mapping process and Roberta model training.

- Input: `Data/attack_stix/enterprise-attack/enterprise-attack-<version>.json`

- Output: `Data/attack_stix/processed/ti_groups_techniques.csv`

Contains columns: `group_sid, group_id, group_name, technique_id, technique_name, is_subtechnique`

---

### 4. Build the Labeled Dataset for Roberta Model Training

python build_dataset.py

This script consolidates the outputs from the IOC mapping (map_iocs_to_attack.py) and MITRE ATT&CK dataset (enterprise_attack.py) into a labeled dataset suitable for Roberta model fine-tuning and evaluation.
It constructs a multi-label classification dataset that maps threat groups to their observed techniques.

- Input:
`Data/mapped/group_ttps_detail.csv`
`Data/attack_stix/processed/ti_groups_techniques.csv`

- Output:
`Data/processed/dataset.csv` – Text dataset for model training
`Data/processed/labels.txt` – Class label reference file

---

## Usage

```
## 1. Create all the necessary files
Run the following scripts sequentially to extract IOCs, build datasets, and map ATT&CK relationships:


python src\data\extract_pdfs.py
python src\data\build_dataset.py
python src\data\enterprise_attack.py
python Data\map_iocs_to_attack.py

## 2. Run the web application

python app.py
navigate to `http://127.0.0.1:5000`
```

---

## Notes

- Limit input to five TTPs max for optimal GPT performance
- Output folders are automatically created if missing

---

## Troubleshooting

| Issue                        | Cause                          | Solution                              |
|-----------------------------|---------------------------------|----------------------------------------|
| PyMuPDF not found           | Missing dependency              | Run `pip install pymupdf`              |
| OPENAI_API_KEY not found    | `.env` file missing             | Add your key to `.env`                 |
| No PDFs found               | Incorrect input folder          | Ensure path to `aptnotes_pdfs/` is correct |
| Empty report output         | Invalid TTP input format        | Use valid MITRE IDs (e.g., `T1059.003`) |

---



