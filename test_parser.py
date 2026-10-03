# -*- coding: utf-8 -*-
"""
test_parser.py -- Phase 1: PDF Parser Test Suite

Tests pdf_parser.py against 3 resume types:
  1. Simple (single column)
  2. Columns (two-column layout)
  3. Tables (table-based layout)

Verifies:
  - Text is extractable (non-empty)
  - No null bytes or garbage characters (\x00 etc.)
  - Expected keywords/names appear in the output
  - clean_text() output is shorter than raw text (whitespace was removed)
  - Text is human-readable

Run:
    python test_parser.py
"""

import sys
import os
from pathlib import Path

# Force UTF-8 output on Windows to avoid cp1252 encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Make sure utils is importable when running from project root
sys.path.insert(0, str(Path(__file__).parent))

from utils.pdf_parser import extract_text_from_pdf, clean_text, parse_resume


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def ok(msg):   return f"[PASS]  {msg}"
def fail(msg): return f"[FAIL]  {msg}"
def info(msg): return f"[INFO]  {msg}"
def sep():     return "\n" + "-" * 60


# ─────────────────────────────────────────────────────────
# Individual assertion helpers
# ─────────────────────────────────────────────────────────

def assert_non_empty(text: str, label: str) -> bool:
    if text and len(text.strip()) > 50:
        print(ok(f"[{label}] Text extracted, length = {len(text)} chars"))
        return True
    else:
        print(fail(f"[{label}] Text is empty or too short (got: {repr(text[:80])})"))
        return False


def assert_no_garbage(text: str, label: str) -> bool:
    garbage_chars = ["\x00", "\x01", "\x02", "\x03", "\x04", "\x05",
                     "\x06", "\x07", "\x08", "\x0b", "\x0c", "\x0e"]
    found = [c for c in garbage_chars if c in text]
    if not found:
        print(ok(f"[{label}] No null bytes or control characters found"))
        return True
    else:
        print(fail(f"[{label}] Garbage characters found: {found}"))
        return False


def assert_keywords(text: str, keywords: list, label: str) -> bool:
    text_lower = text.lower()
    missing = [kw for kw in keywords if kw.lower() not in text_lower]
    if not missing:
        print(ok(f"[{label}] All expected keywords found: {keywords}"))
        return True
    else:
        print(fail(f"[{label}] Missing keywords: {missing}"))
        return False


def assert_cleaned_shorter(raw: str, cleaned: str, label: str) -> bool:
    if len(cleaned) <= len(raw):
        reduction = len(raw) - len(cleaned)
        print(ok(f"[{label}] Cleaning reduced text by {reduction} chars "
                 f"({len(raw)} -> {len(cleaned)})"))
        return True
    else:
        print(fail(f"[{label}] Cleaned text is LONGER than raw - something is wrong"))
        return False


def assert_no_triple_blanks(text: str, label: str) -> bool:
    if "\n\n\n" not in text:
        print(ok(f"[{label}] No excessive blank lines (3+ newlines in a row)"))
        return True
    else:
        print(fail(f"[{label}] Found 3+ consecutive blank lines - cleaning incomplete"))
        return False


# ─────────────────────────────────────────────────────────
# Test cases
# ─────────────────────────────────────────────────────────

SAMPLES_DIR = Path(__file__).parent / "data" / "samples"

TEST_CASES = [
    {
        "name": "Simple Resume (single column)",
        "file": SAMPLES_DIR / "resume_simple.pdf",
        "keywords": ["Priya Sharma", "FastAPI", "Python", "Razorpay", "PostgreSQL", "BITS Pilani"],
    },
    {
        "name": "Columns Resume (two-column layout)",
        "file": SAMPLES_DIR / "resume_columns.pdf",
        "keywords": ["Arjun Mehta", "Machine Learning", "TensorFlow", "Flipkart", "IIT Bombay"],
    },
    {
        "name": "Tables Resume (table-based layout)",
        "file": SAMPLES_DIR / "resume_tables.pdf",
        "keywords": ["Neha Kapoor", "React", "Node.js", "Swiggy", "PostgreSQL"],
    },
]


def run_tests():
    all_passed = True
    results_summary = []

    for tc in TEST_CASES:
        label = tc["name"]
        pdf_path = tc["file"]

        print(sep())
        print(f"\nTesting: {label}")
        print(f"   File: {pdf_path}\n")

        if not pdf_path.exists():
            print(fail(f"[{label}] PDF file not found at {pdf_path}"))
            print("   Run: python data/samples/generate_test_pdfs.py first!\n")
            all_passed = False
            results_summary.append((label, False, "File not found"))
            continue

        # Extract raw text
        try:
            raw_text = extract_text_from_pdf(str(pdf_path))
        except Exception as e:
            print(fail(f"[{label}] extract_text_from_pdf raised exception: {e}"))
            all_passed = False
            results_summary.append((label, False, f"Exception: {e}"))
            continue

        # Clean text
        cleaned_text = clean_text(raw_text)

        # Run assertions
        passed = True
        passed &= assert_non_empty(cleaned_text, label)
        passed &= assert_no_garbage(cleaned_text, label)
        passed &= assert_keywords(cleaned_text, tc["keywords"], label)
        passed &= assert_cleaned_shorter(raw_text, cleaned_text, label)
        passed &= assert_no_triple_blanks(cleaned_text, label)

        all_passed &= passed
        results_summary.append((label, passed, "OK" if passed else "Some assertions failed"))

        # Print a preview of extracted text
        preview = cleaned_text[:400].replace("\n", " ")
        print(info(f"Text preview (first 400 chars):\n   {preview}\n"))

    # Summary
    print(sep())
    print("\nTEST SUMMARY\n")
    for name, passed, note in results_summary:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {name}  ({note})")

    print()
    if all_passed:
        print("All Phase 1 tests passed! PDF parser is working correctly.")
    else:
        print("Some tests failed. Review output above.")
        sys.exit(1)


if __name__ == "__main__":
    print("AI Resume Screener -- Phase 1: PDF Parser Tests")
    print("=" * 60)
    run_tests()
