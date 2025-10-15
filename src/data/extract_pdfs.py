"""
Extract text/metadata/IOCs from PDFs.

Requires:
  pip install pymupdf
"""

import argparse
import hashlib
import json
import re
import sys
import urllib.parse
from pathlib import Path
from datetime import datetime

from common.paths import (
    PROJECT_ROOT, DATA_ROOT, RAW_DIR, PROCESSED_DIR, SRC_ROOT, LOGS_ROOT, MODELS_ROOT, DATASCRIPT_ROOT,
    EXTRACTED_PDFS_DIR, EXTRACTED_PDFS_DIR,  
)
EXTRACTED_IOCS_CSV =EXTRACTED_PDFS_DIR / "extracted_iocs.csv"

try:
    import fitz  # PyMuPDF
except ImportError:
    print("Missing dependency: PyMuPDF. Install with:\n  pip install pymupdf", file=sys.stderr)
    sys.exit(1)


DEFAULT_IN_DIR  = RAW_DIR / "pdfs"     
DEFAULT_OUT_DIR = EXTRACTED_PDFS_DIR           


URL_RX    = re.compile(r'\bhttps?://[^\s<>"\'\]\)}]+', re.I)  # http + https
IPV4_RX   = re.compile(r'\b(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}\b')
MD5_RX    = re.compile(r'\b[a-f0-9]{32}\b', re.I)
SHA1_RX   = re.compile(r'\b[a-f0-9]{40}\b', re.I)
SHA256_RX = re.compile(r'\b[a-f0-9]{64}\b', re.I)
EMAIL_RX  = re.compile(r'\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b', re.I)
ATTACK_RX = re.compile(r'\bT\s*?(\d{4})(?:[\s\.\-_/]?(\d{1,3}))?\b', re.I)

BARE_DOMAIN_RX = re.compile(
    r'\b(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+(?:[a-z]{2,})\b', re.I
)


_BAD_TLD_LIKE_FILES = {
    "exe","dll","sys","ocx","scr","dat","bin",
    "doc","docx","hwp","pdf","xls","xlsx","ppt","pptx","rtf","txt","csv","db","cfg","xml","json","log",
    "js","vbs","ps1","bat","cmd","php",
    "zip","rar","7z","gz","tar","bz2",
    "png","jpg","jpeg","gif","bmp","svg","ico"
}

_NOT_DOMAINS = {
    "wscript.shell", "msxml2.serverxmlhttp", "responsetext", "open", "send", "run"
}

_VALID_TLDS = {
    "com","net","org","io","co","gov","edu","mil","info","biz","me","app","pro","site","shop",
    "xyz","top","club","online","tech","ai","cloud","news","live","blog","dev",
    "us","uk","de","fr","au","ca","sg","hk","tw","kr","jp","cn","ru","ua","by","pl","ro","bg","tr",
    "nl","be","es","it","pt","gr","se","no","fi","dk","cz","sk","si","hu","at","ch",
    "in","pk","bd","vn","th","my","id","ph","la","kh",
    "ir","iq","sa","ae","qa","om","il","eg",
    "br","ar","cl","co","mx","pe","uy","ve","bo","ec",
    "za"
}

def sha1sum(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()

def slugify(name: str, maxlen: int = 64) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", name).strip("-")
    return s[:maxlen] or "doc"

def _strip_punct(s: str) -> str:
    return s.strip(').,;:>]}\'"')

def _is_valid_hostname(host: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-z0-9.-]+", host)) and "." in host

def _domain_from_url(u: str) -> str | None:
    try:
        parsed = urllib.parse.urlparse(u)
        host = (parsed.netloc or "").split("@")[-1]  # drop userinfo
        host = host.split(":")[0]                    # drop port
        host = host.lstrip("[").rstrip("]")         # ipv6 brackets
        host = host.lower()
        if not _is_valid_hostname(host):
            return None
        if host.startswith("www."):
            host = host[4:]
        return host or None
    except Exception:
        return None

def _looks_like_filename_token(token: str) -> bool:
    parts = token.rsplit(".", 1)
    return len(parts) == 2 and parts[1].lower() in _BAD_TLD_LIKE_FILES

def _tld(token: str) -> str:
    return token.rsplit(".", 1)[-1].lower() if "." in token else ""

def _is_probable_domain(token: str) -> bool:
    t = token.lower()
    if t in _NOT_DOMAINS or _looks_like_filename_token(t) or "_" in t:
        return False
    if not BARE_DOMAIN_RX.fullmatch(t):
        return False
    if t.count(".") == 1 and _tld(t) not in _VALID_TLDS:
        return False
    return True


# IOC Extraction
def extract_iocs_from_text(text: str):
    out = []

    # URLs
    for m in URL_RX.finditer(text):
        url = _strip_punct(m.group(0))
        d = _domain_from_url(url)
        if d is None:
            continue
        out.append(("url", url))
        if _is_probable_domain(d):
            out.append(("domain", d))

    # Bare domains
    for m in BARE_DOMAIN_RX.finditer(text):
        d = _strip_punct(m.group(0)).lower()
        if _is_probable_domain(d):
            out.append(("domain", d))

    # Other indicators
    out += [("ipv4", m.group(0)) for m in IPV4_RX.finditer(text)]
    out += [("md5", m.group(0)) for m in MD5_RX.finditer(text)]
    out += [("sha1", m.group(0)) for m in SHA1_RX.finditer(text)]
    out += [("sha256", m.group(0)) for m in SHA256_RX.finditer(text)]
    out += [("email", _strip_punct(m.group(0)).lower()) for m in EMAIL_RX.finditer(text)]
    out += [("attack_id", m.group(0)) for m in ATTACK_RX.finditer(text)]

    # de-dup
    seen = set()
    deduped = []
    for kind, val in out:
        key = (kind, val.lower())
        if key not in seen:
            seen.add(key)
            deduped.append((kind, val))
    return deduped


def extract_pdf(pdf_path: Path, out_dir: Path, per_page: bool = True, collect_iocs: bool = True):
    file_hash = sha1sum(pdf_path)
    folder = f"{slugify(pdf_path.stem)}-{file_hash[:8]}"  # short + unique
    sample_dir = out_dir / folder
    sample_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "source_file": str(pdf_path),
        "original_name": pdf_path.name,
        "sha1": file_hash,
        "size_bytes": pdf_path.stat().st_size,
        "extracted_at": datetime.utcnow().isoformat() + "Z",
    }

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        print(f"[SKIP] {pdf_path.name}: cannot open ({e})")
        return None

    meta["page_count"] = len(doc)
    md = doc.metadata or {}
    meta.update({
        "title": md.get("title"),
        "author": md.get("author"),
        "creationDate": md.get("creationDate"),
        "modDate": md.get("modDate"),
        "producer": md.get("producer"),
    })

    doc_iocs = []
    all_text_parts = []

    for i in range(len(doc)):
        page = doc.load_page(i)
        text = page.get_text("text") or ""
        if per_page:
            (sample_dir / f"p{i+1:03d}.txt").write_text(text, encoding="utf-8")
        all_text_parts.append(f"\n\n=== [PAGE {i+1}] ===\n{text}")

        if collect_iocs:
            for kind, val in extract_iocs_from_text(text):
                doc_iocs.append((kind, val, i + 1))

    (sample_dir / "full.txt").write_text("\n".join(all_text_parts), encoding="utf-8")
    (sample_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    return {"dir": str(sample_dir), "stem": pdf_path.stem, "iocs": doc_iocs}

# CSV output
def write_iocs(all_iocs, out_csv: Path):
    if not all_iocs:
        return
    import csv
    headers = ["file", "page", "kind", "value"]
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        seen = set()
        for r in all_iocs:
            key = (r["file"], r["page"], r["kind"], (r["value"] or "").lower())
            if key in seen:
                continue
            seen.add(key)
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser(description="Extract text/metadata/IOCs from PDFs.")
    ap.add_argument("--in",  dest="in_dir",  default=str(DEFAULT_IN_DIR),  help="Input folder (default: Data/raw/aptnotes_pdfs)")
    ap.add_argument("--out", dest="out_dir", default=str(DEFAULT_OUT_DIR), help="Output folder (default: Data/extracted_pdfs)")
    ap.add_argument("--no-per-page", action="store_true", help="Do not write pXXX.txt files")
    ap.add_argument("--no-iocs",     action="store_true", help="Do not extract IOCs")
    args = ap.parse_args()

    in_dir  = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pdfs = sorted(in_dir.glob("*.pdf"))
    if not pdfs:
        print(f"No PDFs found in {in_dir.resolve()}")
        return

    all_iocs = []
    for pdf in pdfs:
        res = extract_pdf(pdf, out_dir, per_page=not args.no_per_page, collect_iocs=not args.no_iocs)
        if not res:
            continue
        for (kind, val, page) in res["iocs"]:
            all_iocs.append({"file": Path(res["dir"]).name, "page": page, "kind": kind, "value": val})

    if all_iocs and not args.no_iocs:
        write_iocs(all_iocs, EXTRACTED_IOCS_CSV)

    print(f"Done → {out_dir.resolve()}")

if __name__ == "__main__":
    main()
