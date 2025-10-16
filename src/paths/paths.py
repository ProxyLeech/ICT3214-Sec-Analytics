from __future__ import annotations
from pathlib import Path
import os
def find_project_root(start: Path | None = None) -> Path:
    p = (start or Path(__file__)).resolve()
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists():
            return parent
        if (parent / "README.md").exists() or (parent / "requirements.txt").exists():
            return parent
    return p.parent  

PROJECT_ROOT = find_project_root()
# Project root = the repo folder that contains this file (…/common/paths.py)

# roots
DATA_ROOT      = PROJECT_ROOT / "data"
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT       = PROJECT_ROOT / "src"

#Data Roots
RAW_DIR        = DATA_ROOT / "raw"
PROCESSED_DIR  = DATA_ROOT / "processed"
EXTRACTED_PDFS_DIR   = DATA_ROOT / "extracted_pdfs"

#SRC Roots
LOGS_ROOT      = SRC_ROOT / "logs"
MODELS_ROOT    = SRC_ROOT / "models"
DATASCRIPT_ROOT= SRC_ROOT / "data"

#processed
RULES_DIR = PROCESSED_DIR / "rules"

#Raw data
ATTACK_STIX_DIR      = RAW_DIR / "attack_stix"
PDFS_DIR = RAW_DIR / "pdfs"


# Ensure directories exist 
for d in [DATA_ROOT,EXPERIMENTS_ROOT, RAW_DIR, PROCESSED_DIR,SRC_ROOT, LOGS_ROOT, MODELS_ROOT, DATASCRIPT_ROOT,RULES_DIR,
          EXTRACTED_PDFS_DIR, ATTACK_STIX_DIR, PDFS_DIR,
          ]:
    d.mkdir(parents=True, exist_ok=True)



def output_dir_for_folds(n_folds: int, model_slug: str = "roberta_base"):
    return EXPERIMENTS_ROOT / f"{n_folds}foldruns" / model_slug
