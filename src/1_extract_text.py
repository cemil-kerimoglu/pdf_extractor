"""
Step 1 (optimized): Extract plain text from PDFs.

- Per-page strategy: use pdfplumber on pages with text; OCR only on pages that truly need it.
- "Sparse digital page" detection: If pdfplumber finds very little text on a page
  (common in vectorized CAD/plan sheets), we FORCE OCR for that page.
- This prevents cases where only a few labels (e.g., 'TEBEA', 'ARBOX Plus') are captured
  and the rest of the page is missed.
- Parallel OCR: speed up scans across CPU cores.
- Safe Tesseract config and grayscale conversion for speed without quality loss.
- Caching: skip work if output is newer than input.

Output: one .txt per input PDF in data/text/.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Optional, Tuple
import pdfplumber

# OCR & conversion
import pytesseract
from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Paths ---
ROOT = Path(__file__).resolve().parents[1]
IN_DIR = ROOT / "data" / "input"
OUT_DIR = ROOT / "data" / "text"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# --- Tunables (safe defaults) ---
MIN_CHARS_PER_PAGE = 20           # for quick "has text?" check
SPARSE_TEXT_CHARS = 800           # if pdfplumber text shorter than this, treat as "sparse"
SPARSE_WORDS = 80                 # and fewer than this many words
VECTOR_OBJECT_THRESHOLD = 30      # lines+curves+rects+images >= this -> likely vector-heavy

DEFAULT_DPI = 300                 # OCR dpi
TESSERACT_LANG = "deu"            # keep original default
TESSERACT_CONFIG = "--oem 1 --psm 6"

# Max parallel OCR workers (can override with env)
MAX_WORKERS = min(os.cpu_count() or 2, 8)
_env = os.getenv("OCR_MAX_WORKERS")
if _env and _env.isdigit():
    MAX_WORKERS = max(1, int(_env))

# Strict, conservative normalizer for observed brand OCR glitches.
# - Word boundaries (\b) ensure we don't alter parts of other words.
# - We preserve the ® if present.
BRAND_GLITCH_PATTERNS = [
    # 1EBEA / IEBEA / lEBEA / |EBEA  → TEBEA (keep optional ®)
    (re.compile(r"\b[1Il\|]EBEA(\s*®)?\b", re.IGNORECASE), r"TEBEA\1"),
    # More patterns can be added here as needed
]

def normalize_known_ocr_brand_glitches(text: str) -> str:
    for pat, repl in BRAND_GLITCH_PATTERNS:
        text = pat.sub(repl, text)
    return text


def page_has_some_text(page) -> bool:
    """Quick test used only in mixed-mode logic; low threshold."""
    try:
        txt = (page.extract_text() or "").strip()
        return len(txt) >= MIN_CHARS_PER_PAGE
    except Exception:
        return False


def is_sparse_digital_page(page, txt: str) -> bool:
    """
    Decide if a 'digital' page is actually vector-heavy with only a few text objects.
    Heuristics:
      - very short text (chars < SPARSE_TEXT_CHARS) AND
      - few words (words < SPARSE_WORDS) AND
      - many vector objects (rects + lines + curves + images >= VECTOR_OBJECT_THRESHOLD)
    """
    if len(txt) >= SPARSE_TEXT_CHARS:
        return False

    # Count words (robust to None)
    try:
        words = page.extract_words() or []
    except Exception:
        words = []

    if len(words) >= SPARSE_WORDS:
        return False

    # Vector object "richness"
    rects = getattr(page, "rects", []) or []
    lines = getattr(page, "lines", []) or []
    curves = getattr(page, "curves", []) or []
    images = getattr(page, "images", []) or []
    vector_count = len(rects) + len(lines) + len(curves) + len(images)

    return vector_count >= VECTOR_OBJECT_THRESHOLD


def ocr_page_from_pdf(pdf_path: Path, page_number: int, dpi: int, lang: str, config: str) -> Tuple[int, str]:
    """
    Convert a single page to an image and OCR it.
    page_number is 1-based. Returns (page_number, text).
    """
    imgs = convert_from_path(str(pdf_path), dpi=dpi, first_page=page_number, last_page=page_number)
    if not imgs:
        return page_number, ""
    img = imgs[0].convert("L")  # grayscale for speed
    txt = pytesseract.image_to_string(img, lang=lang, config=config)
    return page_number, txt


def extract_text_mixed(pdf_path: Path) -> str:
    """
    Per-page strategy (robust):
    - Try pdfplumber on every page.
    - If a page has no text -> OCR.
    - If it has text but looks 'sparse digital' (likely vector-heavy) -> FORCE OCR.
    - Otherwise keep pdfplumber text.
    """
    pieces: List[Optional[str]] = []
    pages_for_ocr: List[int] = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        pieces = [None] * total_pages

        for idx, page in enumerate(pdf.pages, start=1):
            # Attempt text
            txt = (page.extract_text() or "").strip()

            if not txt or len(txt) < MIN_CHARS_PER_PAGE:
                # No or almost no text -> OCR
                pages_for_ocr.append(idx)
                continue

            # Page has some text. Check if it is suspiciously sparse AND vector-heavy
            if is_sparse_digital_page(page, txt):
                # FORCE OCR to capture the rest (tables/drawings converted to outlines)
                pages_for_ocr.append(idx)
            else:
                # Keep digital text (fast, accurate)
                pieces[idx - 1] = f"\n--- [PAGE {idx}] ---\n{txt}"

    # OCR only where needed (parallel)
    if pages_for_ocr:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [
                ex.submit(ocr_page_from_pdf, pdf_path, p, DEFAULT_DPI, TESSERACT_LANG, TESSERACT_CONFIG)
                for p in pages_for_ocr
            ]
            for fut in as_completed(futs):
                pno, ocr_txt = fut.result()
                pieces[pno - 1] = f"\n--- [PAGE {pno}] ---\n{normalize_known_ocr_brand_glitches(ocr_txt)}"

    # All pages should now be filled
    return "".join(pieces)


def is_up_to_date(pdf_path: Path, txt_path: Path) -> bool:
    """Skip if the output is newer than the input (simple cache)."""
    if not txt_path.exists():
        return False
    try:
        return txt_path.stat().st_mtime >= pdf_path.stat().st_mtime
    except Exception:
        return False


def process_pdf(pdf_path: Path):
    out_path = OUT_DIR / f"{pdf_path.stem}.txt"
    if is_up_to_date(pdf_path, out_path):
        print(f"▶ Skipping (up-to-date): {pdf_path.name}")
        return

    print(f"▶ Processing: {pdf_path.name}")
    try:
        text = extract_text_mixed(pdf_path)
    except Exception as e:
        # As a defensive fallback: full OCR (rare)
        print(f"   Mixed-mode failed ({e}). Falling back to full OCR…")
        images = convert_from_path(str(pdf_path), dpi=DEFAULT_DPI)
        results = [None] * len(images)

        def _ocr(idx_img):
            idx, img = idx_img
            t = pytesseract.image_to_string(img.convert("L"), lang=TESSERACT_LANG, config=TESSERACT_CONFIG)
            return idx, t

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futs = [ex.submit(_ocr, (i, im)) for i, im in enumerate(images, start=1)]
            for fut in as_completed(futs):
                i, t = fut.result()
                results[i - 1] = f"\n--- [PAGE {i}] ---\n{normalize_known_ocr_brand_glitches(t)}"
        text = "".join(results)

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