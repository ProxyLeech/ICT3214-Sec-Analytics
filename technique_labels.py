import pandas as pd
import re
from pathlib import Path

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
    base = Path(__file__).resolve().parent.parent  # project root
    excel_path = base / "Data" / "excel" / "enterprise-attack-v17.1-techniques.xlsx"
    output_csv = base / "techniques_mapping.csv"
    extract_techniques(excel_path, output_csv)