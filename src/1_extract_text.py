"""
Step 1: Extract plain text from PDFs.

- If a PDF is "digital", we will use pdfplumber (fast, accurate).
- If it's a "scan", we will convert each page to an image and OCR with Tesseract.

Output: one .txt per input PDF in data/text/.
"""

from pathlib import Path
import pdfplumber

# OCR (for scanned PDFs)
import pytesseract
from pdf2image import convert_from_path

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "data" / "input"
OUT_DIR = ROOT / "data" / "text"
OUT_DIR.mkdir(parents=True, exist_ok=True)

def is_digital_pdf(pdf_path: Path, min_chars: int = 20) -> bool:
    """Return True if at least one of the first two pages contains extractable text."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages[:2]:
                txt = (page.extract_text() or "").strip()
                if len(txt) >= min_chars:
                    return True
    except Exception:
        pass
    return False

def extract_text_digital(pdf_path: Path) -> str:
    """Extract text from each page with pdfplumber."""
    chunks = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            txt = page.extract_text() or ""
            chunks.append(f"\n--- [PAGE {i}] ---\n{txt}")
    return "\n".join(chunks)

def extract_text_scanned(pdf_path: Path, dpi: int = 300, lang: str = "deu") -> str:
    """
    OCR fallback:
    - Convert pages to images with pdf2image (needs Poppler installed).
    - Run Tesseract OCR (needs Tesseract installed).
    """
    pages = convert_from_path(str(pdf_path), dpi=dpi)
    chunks = []
    for i, img in enumerate(pages, start=1):
        txt = pytesseract.image_to_string(img, lang=lang)
        chunks.append(f"\n--- [PAGE {i}] ---\n{txt}")
    return "\n".join(chunks)

def process_pdf(pdf_path: Path):
    print(f"▶ Processing: {pdf_path.name}")
    digital = is_digital_pdf(pdf_path)
    if digital:
        print("   Type: DIGITAL (no OCR needed)")
        text = extract_text_digital(pdf_path)
    else:
        print("   Type: SCANNED (using OCR)")
        text = extract_text_scanned(pdf_path)

    out_path = OUT_DIR / f"{pdf_path.stem}.txt"
    out_path.write_text(text, encoding="utf-8")
    print(f"   Saved: {out_path}\n")

def main():
    pdfs = sorted(IN_DIR.glob("*.pdf"))
    if not pdfs:
        print("No PDFs found in data/input/. Put your files there and run again.")
        return
    for p in pdfs:
        process_pdf(p)

if __name__ == "__main__":
    main()
