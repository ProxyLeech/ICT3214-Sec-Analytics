# LLM-Assisted Adversary Attribution

A modular pipeline for automating cyber threat attribution using MITRE ATT&CK, APTnotes, and GPT-based analysis. 

---

## Table of Contents

- [System Overview](#system-overview)
- [Workflow Overview](#workflow-overview)
- [System Architecture](#system-architecture)
- [Dependencies](#dependencies)
- [Installation](#installation)
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
│     ├──► mapped/ - contains the pdfs with successful correlation of IOCs and ATT&CK techniques
│     ├──► processed/ 
│              ├──► rules/ - Regex based auto generated ATT&CK matching rules
│              ├──► extracted_iocs.csv - Stores extracted Indicators of Compromise (IOCs) such as domains, IPs, URLs, hashes, and emails parsed from APTnotes PDFs for later threat group mapping.
│              ├──► ti_groups_techniques.csv - Mapping of MITRE ATT&CK Intrusion Sets (Groups) to the Techniques/sub-techniques they use
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
├──► report_generator.py - script to generate report using OpenAI and the results of RoBERTa, Matching and Mitigations
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

<img width="329" height="554" alt="image" src="https://github.com/user-attachments/assets/12cbaa1c-3185-4f0f-be1f-b5814e8d2991" />
<img width="421" height="554" alt="image" src="https://github.com/user-attachments/assets/e17b00be-1d62-4069-95ba-094594ced439" />
<img width="755" height="616" alt="image" src="https://github.com/user-attachments/assets/ccbe8831-b622-4825-a193-c9af4b944aad" />


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
# On Windows:
.venv\Scripts\activate

# On macOS / Linux:
source .venv/bin/activate


# 4. Install dependencies
pip install -r requirements.txt

# 5. Create a .env file and add your OpenAI API key
# (Replace YOUR_KEY_HERE with your actual API key, look in user manual for more details)
echo "OPENAI_API_KEY=YOUR_KEY_HERE" > .env
```


---

## Usage

```bash
# 1. Run the web application
python app.py

# 2. navigate to `http://127.0.0.1:5000`
http://127.0.0.1:500

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





