# from __future__ import annotations
# from pathlib import Path
# import os
# def find_project_root(start: Path | None = None) -> Path:
#     p = (start or Path(__file__)).resolve()
#     for parent in [p] + list(p.parents):
#         if (parent / ".git").exists():
#             return parent
#         if (parent / "README.md").exists() or (parent / "requirements.txt").exists():
#             return parent
#     return p.parent  

# PROJECT_ROOT = find_project_root()
# # Project root = the repo folder that contains this file (…/common/paths.py)

# # roots
# DATA_ROOT      = PROJECT_ROOT / "data"
# EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
# SRC_ROOT       = PROJECT_ROOT / "src"

# #Data Roots
# RAW_DIR        = DATA_ROOT / "raw"
# PROCESSED_DIR  = DATA_ROOT / "processed"
# EXTRACTED_PDFS_DIR   = DATA_ROOT / "extracted_pdfs"

# #SRC Roots
# MODELS_ROOT    = SRC_ROOT / "models"
# DATASCRIPT_ROOT= SRC_ROOT / "data"

# #processed
# RULES_DIR = PROCESSED_DIR / "rules"

# #Raw data
# ATTACK_STIX_DIR      = RAW_DIR / "attack_stix"
# PDFS_DIR = RAW_DIR / "pdfs"


# # Ensure directories exist 
# for d in [DATA_ROOT,EXPERIMENTS_ROOT, RAW_DIR, PROCESSED_DIR,SRC_ROOT, MODELS_ROOT, DATASCRIPT_ROOT,RULES_DIR,
#           EXTRACTED_PDFS_DIR, ATTACK_STIX_DIR, PDFS_DIR,
#           ]:
#     d.mkdir(parents=True, exist_ok=True)



# def output_dir_for_folds(n_folds: int, model_slug: str = "roberta_base"):
#     return EXPERIMENTS_ROOT / f"{n_folds}foldruns" / model_slug
# paths.py (place at repo root, or inside a small package folder like `ict3214/paths.py`)
from __future__ import annotations
from pathlib import Path
import os

# --- Discover the repository root robustly ---
def find_project_root(start: Path | None = None) -> Path:
    """
    Walk up from `start` to find a directory that looks like the repo root.
    We treat a dir with .git/ or pyproject.toml/ or requirements.txt as the root.
    """
    here = (start or Path(__file__)).resolve()
    for p in [here, *here.parents]:
        if (p / ".git").exists() or (p / "pyproject.toml").exists() or (p / "requirements.txt").exists():
            return p
    # fallback: folder containing this file
    return here.parent

# Optional: allow override via env var if someone wants data elsewhere
PROJECT_ROOT = Path(os.environ.get("ICT3214_PROJECT_ROOT", find_project_root()))
DATA_ROOT    = Path(os.environ.get("ICT3214_DATA_ROOT", PROJECT_ROOT / "data"))
SRC_ROOT     = PROJECT_ROOT / "src"
MODELS_ROOT  = SRC_ROOT / "models"
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"

RAW_DIR       = DATA_ROOT / "raw"
PROCESSED_DIR = DATA_ROOT / "processed"
EXTRACTED_PDFS_DIR = DATA_ROOT / "extracted_pdfs"

# If you have a *capitalized* "Data" folder, pick ONE canonical spelling:
# Prefer lowercase on cross-platform repos; Linux is case-sensitive.
DATA_TITLECASE = PROJECT_ROOT / "data"
MAPPED_DIR     = DATA_TITLECASE / "mapped"     # if you must keep "Data/..."
EXCEL_DIR      = DATA_TITLECASE / "excel"
MITIGATIONS_DIR= DATA_TITLECASE / "mitigations"

#processed
RULES_DIR = PROCESSED_DIR / "rules"

#Raw data
ATTACK_STIX_DIR      = RAW_DIR / "attack_stix"
PDFS_DIR = RAW_DIR / "pdfs"

def output_dir_for_folds(n_folds: int, model_slug: str = "roberta_base"):
    return EXPERIMENTS_ROOT / f"{n_folds}foldruns" / model_slug
# Convenience
def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
