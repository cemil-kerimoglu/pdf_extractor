"""
Step 2: Extract Schöck-only items from the plain text files produced in Step 1.

Output: outputs/schoeck_extracted.csv with columns:
- document, page, lv_position, family, product_code, quantity, unit, source_line
"""

from pathlib import Path
import re
import pandas as pd

def make_fuzzy_token(word: str) -> re.Pattern:
    """
    Build a regex that matches the word even if random single spaces are inserted
    between letters. Case-insensitive.
    Example: 'Stacon' matches 'S t a c o n', 'S tacon', etc.
    """
    parts = [re.escape(ch) + r"\s*" for ch in word]
    pattern = r"(?<!\w)" + "".join(parts).rstrip(r"\s*") + r"(?!\w)"
    return re.compile(pattern, re.IGNORECASE)

# Canonical tokens we want to "repair" before matching
CANONICAL_FIXES = {
    "Schöck": make_fuzzy_token("Schöck"),
    "Schoeck": make_fuzzy_token("Schoeck"),
    "Isokorb": make_fuzzy_token("Isokorb"),
    "Tronsole": make_fuzzy_token("Tronsole"),
    "Tronsolen": make_fuzzy_token("Tronsolen"),
    "Stacon": make_fuzzy_token("Stacon"),
    "Brandschutzmanschette": make_fuzzy_token("Brandschutzmanschette"),
}

def clean_line_for_matching(line: str) -> str:
    """
    Quick, safe 'deglitching' to make matching robust:
    - Fix known tokens even if letters have random spaces
    - Normalize 'S t' unit to 'St'
    - Tighten hyphens and collapse digit-digit spaces for codes (e.g., 'Q 4 00' -> 'Q400')
    """
    fixed = line

    # 1) Fix known tokens (brand & key words) with fuzzy replacements
    for canonical, patt in CANONICAL_FIXES.items():
        fixed = patt.sub(canonical, fixed)

    # 2) Normalize unit 'St' if it appears as 'S t'
    fixed = re.sub(r"S\s*t\b", "St", fixed, flags=re.IGNORECASE)

    # 3) Normalize hyphens inside codes: ' - ' -> '-'
    fixed = re.sub(r"\s*-\s*", "-", fixed)

    # 4) Remove spaces between consecutive digits: '4 00' -> '400'
    fixed = re.sub(r"(?<=\d)\s+(?=\d)", "", fixed)

    return fixed


ROOT = Path(__file__).resolve().parents[1]
TEXT_DIR = ROOT / "data" / "text"
OUT_DIR = ROOT / "outputs"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "schoeck_extracted.csv"

# Competitor words to IGNORE (you can extend this list)
COMPETITORS = {"halfen", "hit", "tebea", "arbox"}

# Family keywords (very simple, case-insensitive)
FAMILY_PATTERNS = {
    "Isokorb": make_fuzzy_token("Isokorb"),     # covers "I sokorb"
    "Tronsole": make_fuzzy_token("Tronsole"),   # covers "T ronsole"
    "Stacon": make_fuzzy_token("Stacon"),       # covers "S tacon"
    "Tronsolen": make_fuzzy_token("Tronsolen"), # covers "T ronsolen"
}

# LV position pattern like "6.1.2310." or "5.3.2110."
# allow spaces around dots (e.g., "6.1.2320 ." or "6 . 1 . 2320 .")
LV_PATTERN = re.compile(r"\b\d+(?:\s*\.\s*\d+)+\s*\.", re.IGNORECASE)

# Quantity + unit like "10,000 St" or "45,000 m"
# Accepts 'St' and 'S t'; 'm' stays as-is
QTY_UNIT_PATTERN = re.compile(
    r"(?P<qty>\d{1,3}(?:[.,]\d{3})*)(?:\s*)(?P<unit>S\s*t|St|m)\b", re.IGNORECASE
)

# Simple product-code helpers (starter-level):
# - Isokorb often shows blocks like K-M6-V1-REI120-CV35-X120-H220 or Q V1 REI120 H80 X180
ISOKORB_CODE = re.compile(
    r"(?:K-\w+|Q(?:\s*[-]?\s*V\d+)?)[-\s\w]*REI\d+[-\s\w]*?(?:CV\d+)?[-\s\w]*X\d+[-\s\w]*H\d+",
    re.IGNORECASE,
)
# - Tronsole: capture "Typ ..." after the word "Typ"
TRONSOLE_CODE = re.compile(r"Typ[:\s]*([A-Z0-9/\-\s]+)", re.IGNORECASE)
# - Stacon: capture tokens starting with SLD or LS-Q
STACON_CODE = re.compile(r"(SLD[^\s,;/]+|LS-Q[^\s,;/]+)", re.IGNORECASE)

def looks_like_competitor(line_lower: str) -> bool:
    return any(w in line_lower for w in COMPETITORS)

def parse_txt_file(txt_path: Path):
    rows = []
    current_page = None
    current_lv = None

    # We look ahead a few lines to find quantity/unit near the product mention
    LOOKAHEAD = 3

    lines = txt_path.read_text(encoding="utf-8", errors="ignore").splitlines()

    for idx, raw in enumerate(lines):
        line = raw.strip()
        line_lower = line.lower()
        # Clean a copy for robust matching, keep original 'line' for source_line
        cline = clean_line_for_matching(line)
        clower = cline.lower()

        # Track page (Step 1 marked pages as: --- [PAGE N] ---)
        if line.startswith("--- [PAGE ") and line.endswith("] ---"):
            try:
                current_page = int(re.findall(r"\d+", line)[0])
            except Exception:
                current_page = None
            continue

        # Track LV position if present on this line
        m_lv = LV_PATTERN.search(cline)
        if m_lv:
            current_lv = m_lv.group(0)

        # Must contain "Schöck"/"Schoeck" and not a competitor (check on cleaned lower)
        if "schöck" not in clower and "schoeck" not in clower:
            continue
        if looks_like_competitor(clower):
            continue

        # Detect family
        family = None
        for fam, patt in FAMILY_PATTERNS.items():
            if patt.search(cline):
                family = fam
                break
        if not family:
            # Maybe family is on neighboring lines; quick peek back one line
            if idx > 0:
                prev_clean = clean_line_for_matching(lines[idx-1].strip())
                for fam, patt in FAMILY_PATTERNS.items():
                    if patt.search(prev_clean):
                        family = fam
                        break
        if not family:
            # If we cannot detect a family, skip for now (keeps results clean)
            continue

        # Try to extract a product code/type depending on the family
        product_code = ""
        segment = " ".join([cline] + [clean_line_for_matching(x.strip()) for x in lines[idx+1: idx+2]])

        if family == "Isokorb":
            m = ISOKORB_CODE.search(segment)
            if m:
                product_code = m.group(0).strip()
        elif family == "Tronsole":
            m = TRONSOLE_CODE.search(segment)
            if m:
                product_code = "Typ " + m.group(1).strip()
        elif family == "Stacon":
            m = STACON_CODE.search(segment)
            if m:
                product_code = m.group(1).strip()

        # Find quantity + unit in the current line or the next few lines
        qty, unit = None, None
        for j in range(idx, min(idx + 1 + LOOKAHEAD, len(lines))):
            cj = clean_line_for_matching(lines[j])
            m2 = QTY_UNIT_PATTERN.search(cj)
            if m2:
                q_raw = m2.group("qty")
                q_norm = int(q_raw.replace(".", "").replace(",", ""))  # "10,000" -> 10000
                qty = q_norm
                unit = m2.group("unit").replace(" ", "")  # "S t" -> "St"
                break

        # Build row if we have at least family and (qty+unit or product_code)
        if family and (qty is not None or product_code):
            rows.append({
                "document": txt_path.stem + ".pdf",
                "page": current_page,
                "lv_position": current_lv,
                "family": family,
                "product_code": product_code,
                "quantity": qty,
                "unit": unit,
                "source_line": line[:500],  # keep it short for CSV
            })

    return rows

def main():
    all_rows = []
    txt_files = sorted(TEXT_DIR.glob("*.txt"))
    if not txt_files:
        print("No .txt files found in data/text/. Run Step 1 first.")
        return

    for t in txt_files:
        print(f"▶ parsing {t.name}")
        rows = parse_txt_file(t)
        all_rows.extend(rows)

    if not all_rows:
        print("No Schöck items found. Check your text files and patterns.")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"Saved: {OUT_CSV}")
    print(df.head(10))

if __name__ == "__main__":
    main()
