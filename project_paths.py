from __future__ import annotations
from pathlib import Path
import os
import sys

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
# ---------- Canonical roots (env overrides supported) ----------
PROJECT_ROOT     = Path(os.environ.get("ICT3214_PROJECT_ROOT", find_project_root()))
DATA_ROOT        = Path(os.environ.get("ICT3214_DATA_ROOT", PROJECT_ROOT / "data"))
EXPERIMENTS_ROOT = PROJECT_ROOT / "experiments"
SRC_ROOT         = PROJECT_ROOT / "src"
SCRIPTS_DIR      = SRC_ROOT / "scripts"       # <-- all helpers live here now
MODELS_ROOT      = SRC_ROOT / "models"

# ---------- Data layout ----------
RAW_DIR            = DATA_ROOT / "raw"
PROCESSED_DIR      = DATA_ROOT / "processed"
EXTRACTED_PDFS_DIR = DATA_ROOT / "extracted_pdfs"

# If you previously had a capitalized "Data" folder, keep a single spelling.
# Using lowercase "data" everywhere is best for cross-platform use.
MAPPED_DIR      = DATA_ROOT / "mapped"   #DATA_DIR
EXCEL_DIR       = DATA_ROOT / "excel"
MITIGATIONS_DIR = DATA_ROOT / "mitigations"

# Raw data subdirs
ATTACK_STIX_DIR = RAW_DIR / "attack_stix"
PDFS_DIR        = RAW_DIR / "pdfs"

# Processed subdirs
RULES_DIR       = PROCESSED_DIR / "rules"

# ---------- Script entry points (import/CLI) ----------
# Move these files into src/scripts/ to match:
EXTRACT_SCRIPT        = SCRIPTS_DIR / "extract_pdfs.py"
ATTACK_SCRIPT         = SCRIPTS_DIR / "enterprise_attack.py"
MAP_IOCS_SCRIPT    =  SCRIPTS_DIR / "map_iocs_to_attack.py"   
BUILD_DATASET_SCRIPT  = SCRIPTS_DIR / "build_dataset.py"
MITIGATIONS_SCRIPT    =  SCRIPTS_DIR / "mitigations.py"   
MATCHING_SCRIPT     =  SCRIPTS_DIR / "matching.py"  
REPORT_GENERATION_SCRIPT    =  SCRIPTS_DIR / "report_generator.py"   
TECHNIQUE_LABELS_SCRIPT    =  SCRIPTS_DIR / "technique_labels.py"   
# Models
TRAIN_ROBERTA_SCRIPT  = MODELS_ROOT / "train_roberta.py"
PREDICT_SCRIPT        = MODELS_ROOT / "predict_roberta.py"
BEST_MODEL_DIR        = MODELS_ROOT / "best_roberta_for_predict"

# Frequently-used files
MAPPING_CSV           = SCRIPTS_DIR / "techniques_mapping.csv"
MITIGATIONS_CSV       = MITIGATIONS_DIR / "mitigations.csv"
EXCEL_ATTACK_TECHS    = EXCEL_DIR / "enterprise-attack-v17.1-techniques.xlsx"

# Outputs produced by scripts
EXTRACTED_IOCS_CSV    = PROCESSED_DIR / "extracted_iocs.csv"
TI_GROUPS_TECHS_CSV   = PROCESSED_DIR / "ti_groups_techniques.csv"
DATASET_CSV           = PROCESSED_DIR / "dataset.csv"
LABELS_TXT            = PROCESSED_DIR / "labels.txt"

GROUP_TTPS_DETAIL_CSV = PROCESSED_DIR / "group_ttps_detail.csv"
RANKED_GROUPS_CSV     = PROCESSED_DIR / "ranked_groups.csv"



def output_dir_for_folds(n_folds: int, model_slug: str = "roberta_base"):
    return EXPERIMENTS_ROOT / f"{n_folds}foldruns" / model_slug
# Convenience
def project_path(*parts: str) -> Path:
    return PROJECT_ROOT.joinpath(*parts)
def ensure_dir_tree() -> None:
    """Create the directory tree if missing."""
    for d in [
        DATA_ROOT, RAW_DIR, PROCESSED_DIR, EXTRACTED_PDFS_DIR,
        EXPERIMENTS_ROOT, SRC_ROOT, SCRIPTS_DIR, MODELS_ROOT,
        MAPPED_DIR, EXCEL_DIR, MITIGATIONS_DIR, ATTACK_STIX_DIR, PDFS_DIR, RULES_DIR
    ]:
        d.mkdir(parents=True, exist_ok=True)

def add_src_to_syspath() -> None:
    """Make src/ and src/scripts importable (so 'import matching' works after move)."""
    for p in (SRC_ROOT, SCRIPTS_DIR):
        sp = str(p)
        if sp not in sys.path:
            sys.path.insert(0, sp)
# ROOT = Path(__file__).resolve().parents[2]  # repo root
# sys.path.insert(0, str(ROOT))
# from project_paths import (
#     PROJECT_ROOT, DATA_ROOT, EXPERIMENTS_ROOT, SRC_ROOT, MODELS_ROOT, EXPERIMENTS_ROOT,SCRIPTS_DIR,
#     RAW_DIR, PROCESSED_DIR, EXTRACTED_PDFS_DIR,
#     MAPPED_DIR, EXCEL_DIR, MITIGATIONS_DIR,
#     ATTACK_STIX_DIR,PDFS_DIR,RULES_DIR,EXTRACT_SCRIPT,ATTACK_SCRIPT,MAP_IOCS_SCRIPT,
#     BUILD_DATASET_SCRIPT,MITIGATIONS_SCRIPT,
#     GROUP_TTPS_DETAIL_CSV,MATCHING_SCRIPT,REPORT_GENERATION_SCRIPT,TECHNIQUE_LABELS_SCRIPT,
#     TRAIN_ROBERTA_SCRIPT,PREDICT_SCRIPT,BEST_MODEL_DIR,
#     MAPPING_CSV,MITIGATIONS_CSV,EXCEL_ATTACK_TECHS,
#     EXTRACTED_IOCS_CSV,TI_GROUPS_TECHS_CSV,DATASET_CSV,LABELS_TXT,GROUP_TTPS_DETAIL_CSV,RANKED_GROUPS_CSV,
#     output_dir_for_folds, project_path,ensure_dir_tree,add_src_to_syspath
# )