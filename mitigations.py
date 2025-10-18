from __future__ import annotations

import os
import sys
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

REQUIRED_SHEET = "associated mitigations"
DEFAULT_REQUIRED_COLUMNS = ["target id", "target name", "mapping description"]

COLUMN_SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "target id": ("target id", "technique id", "targetid", "technique_id"),
    "target name": ("target name", "technique name", "target", "name"),
    "mapping description": (
        "mapping description",
        "relationship description",
        "description",
        "mapping",
    ),
}


def _normalize(s: str) -> str:
    return s.strip().lower()


def choose_sheet(excel_path: str, requested: str) -> str | None:
    """Return a best-effort match for the desired sheet name (case-insensitive, startswith)."""
    xls = pd.ExcelFile(excel_path)
    wanted = _normalize(requested)
    lowers = {name: _normalize(name) for name in xls.sheet_names}
    for name, low in lowers.items():
        if low == wanted:
            return name
    for name, low in lowers.items():
        if low.startswith(wanted):
            return name
    for name, low in lowers.items():
        if wanted in low:
            return name
    return None


def pick_columns(df: pd.DataFrame, required_columns: Sequence[str]) -> Tuple[pd.DataFrame, List[str]]:
    """Return dataframe with required columns renamed to canonical names."""
    norm_to_original = {_normalize(col): col for col in df.columns}
    selected = {}
    missing = []

    for canon in required_columns:
        candidates: Iterable[str] = COLUMN_SYNONYMS.get(_normalize(canon), (canon,))
        found_col = None
        for cand in candidates:
            key = _normalize(cand)
            if key in norm_to_original:
                found_col = norm_to_original[key]
                break
        if found_col is None:
            missing.append(canon)
        else:
            selected[canon] = found_col

    if missing:
        return df, missing

    out = df[[selected[c] for c in required_columns]].copy()
    out.columns = list(required_columns)
    return out, []


def clean_text_columns(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Trim whitespace and replace NaN with empty strings."""
    for col in columns:
        if col in df.columns:
            df[col] = df[col].map(lambda v: v.strip() if isinstance(v, str) else ("" if pd.isna(v) else str(v)))
    return df

_ID_RE = re.compile(r'^\s*[tT]\s*(\d{4})(?:\.(\d{3}))?\s*$')
_WS_RE = re.compile(r'\s+')
_PUNCT_TRIM_RE = re.compile(r'^[\s\-\u2022•·\:;,_\.\|\(\)\[\]\{\}]+|[\s\-\u2022•·\:;,_\.\|\(\)\[\]\{\}]+$')

def _norm_ws(s: str) -> str:
    """Collapse internal whitespace to a single space and strip ends."""
    return _WS_RE.sub(' ', s).strip()

def _norm_punct_edges(s: str) -> str:
    """Trim common bullet/punctuation clutter on the edges."""
    return _PUNCT_TRIM_RE.sub('', s).strip()

def _norm_text(s: str) -> str:
    """General text normalization used for names & descriptions."""
    s = str(s)
    s = _norm_ws(s)
    s = _norm_punct_edges(s)
    return s

def _norm_desc_for_dedupe(s: str) -> str:
    """A stricter normalization for de-duplication: lower, no trailing dots, collapse spaces."""
    s = _norm_text(s)
    s = s.rstrip('.').lower()
    return s

def _norm_tech_id(s: str) -> str:
    """
    Normalize MITRE technique/sub-technique IDs:
    - Accepts 't1059', 'T 1059.003', '  t1059.003  ' etc.
    - Returns 'T1059' or 'T1059.003'
    """
    if s is None:
        return ''
    m = _ID_RE.match(str(s))
    if not m:
        # If it already looks uppercase and reasonable, just tidy whitespace
        s = _norm_ws(str(s))
        # Final fallback: uppercase T- prefix if present
        if s and s[0].lower() == 't':
            s = 'T' + s[1:]
        return s
    major, minor = m.group(1), m.group(2)
    return f"T{major}" + (f".{minor}" if minor else "")

def _id_sort_key(v: str):
    """
    Numeric sort key for MITRE technique IDs like 'T1059' or 'T1059.003'.
    Techniques sort before sub-techniques.
    """
    m = _ID_RE.match(v or "")
    if not m:
        return (9999, 999)  # Non-standard IDs go to bottom
    major = int(m.group(1))
    minor = int(m.group(2) or 999)  # plain technique first
    return (major, minor)


def tidy_mitigations_dataframe(df: pd.DataFrame,
                               col_id: str = "target id",
                               col_name: str = "target name",
                               col_desc: str = "mapping description",
                               group_by_description_across_ttps: bool = True) -> pd.DataFrame:
    """
    - Normalize columns (ID, Name, Desc)
    - Remove exact & near-duplicate rows
    - If group_by_description_across_ttps=True (default):
        Collapse rows sharing the same mapping description across different TTPs
        into ONE row with comma-separated IDs and names.
      Else:
        Group by (ID, Name) and merge unique descriptions.
    """
    # Ensure columns exist
    for c in (col_id, col_name, col_desc):
        if c not in df.columns:
            raise ValueError(f"Expected column '{c}' not found in mitigations dataframe.")

    # Normalize fields
    df[col_id] = df[col_id].map(_norm_tech_id)
    df[col_name] = df[col_name].map(lambda x: _norm_text(x).strip())
    df[col_desc] = df[col_desc].map(_norm_text)

    # Drop rows missing both id and name (junk)
    df = df[~(df[col_id].eq('') & df[col_name].eq(''))].copy()

    # Remove full-row duplicates first
    df = df.drop_duplicates().reset_index(drop=True)

    # De-duplicate by (id,name,desc) with stricter normalization on desc
    df["_dedupe_desc_norm"] = df[col_desc].map(_norm_desc_for_dedupe)
    df = df.drop_duplicates(subset=[col_id, col_name, "_dedupe_desc_norm"]).reset_index(drop=True)

    if group_by_description_across_ttps:
        # NEW: collapse across TTPs by identical description
        out = collapse_across_ttps_by_description(df, col_id, col_name, col_desc)
        return out

    # (original path) Group by (id,name) and merge unique descriptions
    grouped = []
    for (tid, tname), g in df.groupby([col_id, col_name], dropna=False, sort=False):
        seen = set()
        uniq_descs = []
        for desc, normd in zip(g[col_desc].tolist(), g["_dedupe_desc_norm"].tolist()):
            if not normd or normd in seen:
                continue
            seen.add(normd)
            uniq_descs.append(desc)
        merged_desc = "\n".join(f"• {d}" for d in uniq_descs)
        grouped.append({col_id: tid, col_name: tname, col_desc: merged_desc})

    out = pd.DataFrame(grouped, columns=[col_id, col_name, col_desc])

    # Sort
    def _id_sort_key(v: str):
        """
        Return a numeric sort key for MITRE technique IDs (T####.###).
        Techniques sort before sub-techniques.
        """
        m = _ID_RE.match(v or "")
        if not m:
            return (9999, 999)  # Non-standard IDs go to the bottom
        major = int(m.group(1))
        minor = int(m.group(2) or 999)
        return (major, minor)

    out = out.sort_values(by=[col_id, col_name],
                          key=lambda s: s.map(_id_sort_key) if s.name == col_id else s.str.lower(),
                          kind="mergesort").reset_index(drop=True)
    return out

def collapse_across_ttps_by_description(df: pd.DataFrame,
                                        col_id: str = "target id",
                                        col_name: str = "target name",
                                        col_desc: str = "mapping description") -> pd.DataFrame:
    """
    Collapse rows that share the same (normalized) mapping description across different TTPs.
    Output: one row per unique description, with comma-separated IDs and names.
    """
    if not all(c in df.columns for c in [col_id, col_name, col_desc]):
        raise ValueError("collapse_across_ttps_by_description: missing required columns.")

    if "_dedupe_desc_norm" not in df.columns:
        df = df.copy()
        df["_dedupe_desc_norm"] = df[col_desc].map(_norm_desc_for_dedupe)

    rows = []
    for desc_norm, g in df.groupby("_dedupe_desc_norm", dropna=False):
        if not desc_norm:
            continue

        # pick the longest original description as representative
        rep_desc = max(g[col_desc].astype(str), key=lambda s: len(s))

        # Collect unique (id, name) pairs
        pairs = []
        seen = set()
        for tid, tname in zip(g[col_id].astype(str), g[col_name].astype(str)):
            tid_n = _norm_tech_id(tid)
            tname_n = _norm_text(tname)
            key = (tid_n.lower(), tname_n.lower())
            if tid_n and key not in seen:
                seen.add(key)
                pairs.append((tid_n, tname_n))

        # Sort pairs by technique id numeric order (techniques before sub-techniques)
        pairs.sort(key=lambda p: _id_sort_key(p[0]))

        # Join with commas (space after comma for readability)
        joined_ids = ", ".join(p[0] for p in pairs)
        joined_names = ", ".join(p[1] for p in pairs)

        # Only keep if there is at least one ID/name
        if joined_ids:
            rows.append({col_id: joined_ids, col_name: joined_names, col_desc: rep_desc})

    out = pd.DataFrame(rows, columns=[col_id, col_name, col_desc])

    # Sort by first ID token in the joined list
    def _first_id_sortkey(s: str):
        first = (s or "").split(",")[0].strip()
        return _id_sort_key(first)

    if not out.empty:
        out = out.sort_values(by=[col_id, col_name],
                              key=lambda s: s.map(_first_id_sortkey) if s.name == col_id else s.str.lower(),
                              kind="mergesort").reset_index(drop=True)
    return out


def main() -> None:
    # Input Excel path
    excel_path = r"Data/excel/enterprise-attack-v17.1-techniques.xlsx"

    # Ensure output folder Data/mitigations/ exists
    output_dir = os.path.join(os.path.dirname(excel_path), "..", "mitigations")
    os.makedirs(output_dir, exist_ok=True)

    # Output file in Data/mitigations/
    out_path = os.path.join(output_dir, "mitigations.csv")

    required_columns = list(DEFAULT_REQUIRED_COLUMNS)
    sheet_name = choose_sheet(excel_path, REQUIRED_SHEET)
    if sheet_name is None:
        available = pd.ExcelFile(excel_path).sheet_names
        print(f"Error: required sheet '{REQUIRED_SHEET}' not found. Available sheets: {available}")
        sys.exit(2)

    df = pd.read_excel(excel_path, sheet_name=sheet_name, engine="openpyxl")
    df.columns = [c.strip() for c in df.columns]

    out_df, missing = pick_columns(df, required_columns)
    if missing:
        print(f"Error: missing required columns {missing}. Found columns: {list(df.columns)}")
        sys.exit(3)

    out_df = clean_text_columns(out_df, required_columns)

    out_df = tidy_mitigations_dataframe(
        out_df,
        col_id=required_columns[0],         # "target id"
        col_name=required_columns[1],       # "target name"
        col_desc=required_columns[2],       # "mapping description"
    )

    out_df.to_csv(out_path, index=False, encoding="utf-8")
    print(f"Wrote {len(out_df):,} cleaned mitigation rows to {out_path}")


if __name__ == "__main__":
    main()