import pandas as pd
import re
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]  # repo root
sys.path.insert(0, str(ROOT))
from project_paths import (
    PROJECT_ROOT, DATA_ROOT, EXPERIMENTS_ROOT, SRC_ROOT, MODELS_ROOT, EXPERIMENTS_ROOT,SCRIPTS_DIR,
    RAW_DIR, PROCESSED_DIR, EXTRACTED_PDFS_DIR,
    MAPPED_DIR, EXCEL_DIR, MITIGATIONS_DIR,
    ATTACK_STIX_DIR,PDFS_DIR,RULES_DIR,EXTRACT_SCRIPT,ATTACK_SCRIPT,MAP_IOCS_SCRIPT,
    BUILD_DATASET_SCRIPT,MITIGATIONS_SCRIPT,
    GROUP_TTPS_DETAIL_CSV,MATCHING_SCRIPT,REPORT_GENERATION_SCRIPT,TECHNIQUE_LABELS_SCRIPT,
    TRAIN_ROBERTA_SCRIPT,PREDICT_SCRIPT,BEST_MODEL_DIR,
    MAPPING_CSV,MITIGATIONS_CSV,EXCEL_ATTACK_TECHS,
    EXTRACTED_IOCS_CSV,TI_GROUPS_TECHS_CSV,DATASET_CSV,LABELS_TXT,GROUP_TTPS_DETAIL_CSV,RANKED_GROUPS_CSV,
    output_dir_for_folds, project_path,ensure_dir_tree,add_src_to_syspath
)

def extract_techniques(excel_path: Path, output_csv: Path):
    df = pd.read_excel(excel_path, sheet_name="techniques", engine="openpyxl")
    df = df.iloc[:, [0, 2]]  # Column A (ID), Column C (name)
    df.columns = ["id", "name"]

    pattern = re.compile(r"^T\d{4}(?:\.\d{3})?$", re.IGNORECASE)
    df = df[df["id"].astype(str).str.match(pattern)]
    df = df.drop_duplicates(subset=["id"]).dropna()

    df["label"] = df["id"] + " (" + df["name"] + ")"
    df.to_csv(output_csv, index=False, encoding="utf-8")

    print(f"Extracted {len(df)} techniques -> {output_csv}")
    return output_csv

if __name__ == "__main__":
    excel_path = EXCEL_ATTACK_TECHS
    output_csv = MAPPING_CSV
    extract_techniques(excel_path, output_csv)