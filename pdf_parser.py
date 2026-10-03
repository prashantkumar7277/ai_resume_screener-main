"""
pdf_parser.py — Phase 1: PDF Text Extraction & Cleaning

Uses PyMuPDF (fitz) to extract text from PDF files page by page,
then applies regex-based cleaning to produce readable, normalized text.
"""

import re
import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file using PyMuPDF.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        A single string of raw text extracted from all pages.
        Returns empty string if PDF has no extractable text (e.g., scanned).
    """
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise ValueError(f"Could not open PDF: {pdf_path}\nError: {e}")

    pages_text = []
    for page in doc:
        text = page.get_text("text")
        if text:
            pages_text.append(text)

    doc.close()
    full_text = "\n".join(pages_text)
    return full_text


def clean_text(text: str) -> str:
    """
    Clean and normalize extracted PDF text.

    Steps:
    1. Remove null bytes and control characters (except newlines/tabs).
    2. Remove page numbers (standalone digits on a line).
    3. Collapse multiple blank lines into a single blank line.
    4. Collapse runs of spaces/tabs into a single space per line.
    5. Strip leading/trailing whitespace from each line.
    6. Strip overall leading/trailing whitespace.

    Args:
        text: Raw text string from extract_text_from_pdf().

    Returns:
        Clean, readable text string.
    """
    if not text:
        return ""

    # 1. Remove null bytes and non-printable control chars (keep \n, \t, \r)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)

    # 2. Remove standalone page numbers (e.g., "1", "- 2 -", "Page 3")
    text = re.sub(r"(?i)^\s*(page\s*)?\d+\s*$", "", text, flags=re.MULTILINE)

    # 3. Per-line: collapse multiple spaces/tabs to a single space, strip edges
    cleaned_lines = []
    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line)  # collapse horizontal whitespace
        line = line.strip()
        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    # 4. Collapse 3+ consecutive blank lines into exactly 2 (one blank separator)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # 5. Final trim
    text = text.strip()

    return text


def parse_resume(pdf_path: str) -> str:
    """
    Full pipeline: extract + clean text from a resume PDF.

    Args:
        pdf_path: Path to the resume PDF.

    Returns:
        Clean, readable resume text. Empty string if scanned/no text.
    """
    raw_text = extract_text_from_pdf(pdf_path)
    cleaned = clean_text(raw_text)
    return cleaned
