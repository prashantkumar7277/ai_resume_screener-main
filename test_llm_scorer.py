# -*- coding: utf-8 -*-
"""
test_llm_scorer.py -- Phase 4: Groq LLM Scoring Tests

Tests utils/llm_scorer.py:
  1. API key is present (skips gracefully if not set)
  2. Simple "hello" call to Groq works (connectivity test)
  3. build_scoring_prompt() produces correct template output
  4. score_resume() returns valid JSON structure
  5. Qualified candidate scores higher than unqualified (>= 20 pts difference)
  6. Score is always in [0, 100]
  7. Retry/fallback: _extract_json handles malformed and fenced JSON
  8. score_resumes_batch() processes 3 candidates correctly

Run:
    python test_llm_scorer.py
"""

import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def ok(msg):   print(f"[PASS]  {msg}")
def fail(msg): print(f"[FAIL]  {msg}")
def skip(msg): print(f"[SKIP]  {msg}")
def sep():     print("\n" + "-" * 60)


def assert_true(condition, msg_pass, msg_fail) -> bool:
    if condition:
        ok(msg_pass)
        return True
    else:
        fail(msg_fail)
        return False


# ─────────────────────────────────────────────────────────
# Check for API key before importing scorer
# ─────────────────────────────────────────────────────────

GROQ_KEY = os.getenv("GROQ_API_KEY", "").strip()
HAS_KEY  = bool(GROQ_KEY)

if not HAS_KEY:
    print("\n" + "=" * 60)
    print("GROQ_API_KEY is not set in .env")
    print("Get a free key at: https://console.groq.com")
    print("Then add to resume_screener/.env:")
    print("  GROQ_API_KEY=gsk_...")
    print("=" * 60)
    print("\nRunning offline tests only (prompt builder + JSON parser)...\n")

from utils.llm_scorer import (
    build_scoring_prompt,
    _extract_json,
    _validate_result,
    score_resume,
    score_resumes_batch,
)
from groq import Groq


# ─────────────────────────────────────────────────────────
# Test data
# ─────────────────────────────────────────────────────────

JD = """
Senior Python Backend Developer | Fintech Startup
Requirements: Python 3.10+, FastAPI, PostgreSQL, Redis, Docker, AWS.
5+ years experience. Responsible for designing and building payment APIs.
Must have: REST API design, async programming, CI/CD pipelines.
Nice to have: Kafka, Kubernetes, performance optimisation.
"""

RESUME_QUALIFIED = """
Priya Sharma - Senior Python Backend Developer, 5 years experience.
Skills: Python 3.11, FastAPI, PostgreSQL, Redis, Docker, AWS EC2/S3, Kafka.
At Razorpay: built payment processing APIs handling 10,000 TPS using async FastAPI.
Reduced latency by 40% via Redis caching. 92% test coverage. Kubernetes deployments.
B.Tech CS BITS Pilani 2019. Led team of 5 engineers.
"""

RESUME_UNQUALIFIED = """
Sneha Patel - Senior Graphic Designer, 6 years experience.
Skills: Adobe Photoshop, Illustrator, InDesign, Figma, After Effects, Canva.
Created brand identities for 30+ FMCG companies at Ogilvy India.
Won 3 national design awards. Expert in typography and color theory.
BFA Sir JJ School of Art, Mumbai 2018.
"""

RESUME_MODERATE = """
Aryan Gupta - Backend Developer, 3 years experience.
Skills: Python, Django, MySQL, basic Docker, some AWS.
Built internal CRUD APIs at Infosys. Some REST API experience.
Learning FastAPI. No Redis or Kafka experience.
B.Tech IT NIT Trichy 2021.
"""


# ─────────────────────────────────────────────────────────
# Offline tests (no API key needed)
# ─────────────────────────────────────────────────────────

def test_prompt_builder():
    sep()
    print("\nTest 1 (offline): build_scoring_prompt() formats correctly\n")
    prompt = build_scoring_prompt(JD, RESUME_QUALIFIED)
    all_ok = True
    all_ok &= assert_true(
        "JOB DESCRIPTION:" in prompt,
        "Prompt contains 'JOB DESCRIPTION:' header",
        "Prompt missing 'JOB DESCRIPTION:' header"
    )
    all_ok &= assert_true(
        "CANDIDATE RESUME:" in prompt,
        "Prompt contains 'CANDIDATE RESUME:' header",
        "Prompt missing 'CANDIDATE RESUME:' header"
    )
    all_ok &= assert_true(
        '"score"' in prompt,
        "Prompt contains JSON schema with 'score' field",
        "Prompt missing 'score' field in JSON schema"
    )
    all_ok &= assert_true(
        '"strengths"' in prompt and '"gaps"' in prompt and '"verdict"' in prompt,
        "Prompt contains all 4 JSON fields (score, strengths, gaps, verdict)",
        "Prompt is missing some JSON fields"
    )
    all_ok &= assert_true(
        len(prompt) < 8000,
        f"Prompt length is within token budget ({len(prompt)} chars)",
        f"Prompt is too long ({len(prompt)} chars)"
    )
    return all_ok


def test_json_extractor_clean():
    sep()
    print("\nTest 2 (offline): _extract_json() parses clean JSON\n")
    sample = '{"score": 85, "strengths": ["a", "b", "c"], "gaps": ["x"], "verdict": "Great fit."}'
    result = _extract_json(sample)
    return assert_true(
        result is not None and result.get("score") == 85,
        f"Clean JSON parsed correctly: score={result.get('score') if result else 'N/A'}",
        "Failed to parse clean JSON"
    )


def test_json_extractor_fenced():
    sep()
    print("\nTest 3 (offline): _extract_json() handles markdown code fences\n")
    fenced = '```json\n{"score": 72, "strengths": ["x"], "gaps": ["y"], "verdict": "Ok fit."}\n```'
    result = _extract_json(fenced)
    return assert_true(
        result is not None and result.get("score") == 72,
        f"Fenced JSON parsed correctly: score={result.get('score') if result else 'N/A'}",
        "Failed to parse fenced JSON"
    )


def test_json_extractor_embedded():
    sep()
    print("\nTest 4 (offline): _extract_json() finds JSON embedded in prose\n")
    prose = 'Here is my analysis: {"score": 55, "strengths": ["a"], "gaps": ["b"], "verdict": "Moderate."} Hope that helps!'
    result = _extract_json(prose)
    return assert_true(
        result is not None and result.get("score") == 55,
        f"Embedded JSON extracted correctly: score={result.get('score') if result else 'N/A'}",
        "Failed to extract JSON from prose"
    )


def test_validate_result_clamping():
    sep()
    print("\nTest 5 (offline): _validate_result() clamps score to [0, 100]\n")
    over  = _validate_result({"score": 150, "strengths": [], "gaps": [], "verdict": "test"})
    under = _validate_result({"score": -20, "strengths": [], "gaps": [], "verdict": "test"})
    all_ok = True
    all_ok &= assert_true(over["score"] == 100, f"Score 150 clamped to 100 (got {over['score']})", f"Expected 100, got {over['score']}")
    all_ok &= assert_true(under["score"] == 0,  f"Score -20 clamped to 0 (got {under['score']})", f"Expected 0, got {under['score']}")
    return all_ok


# ─────────────────────────────────────────────────────────
# Live tests (require GROQ_API_KEY)
# ─────────────────────────────────────────────────────────

def test_groq_connectivity():
    sep()
    print("\nTest 6 (live): Simple Groq API call works (connectivity check)\n")
    try:
        client = Groq(api_key=GROQ_KEY)
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": 'Reply with just the word "OK"'}],
            max_tokens=5,
            temperature=0,
        )
        reply = response.choices[0].message.content.strip()
        print(f"  [INFO]  Groq response: '{reply}'")
        return assert_true(
            len(reply) > 0,
            f"Groq API responded: '{reply}'",
            "Groq API returned empty response"
        )
    except Exception as e:
        fail(f"Groq API connectivity test failed: {e}")
        return False


def test_score_resume_structure():
    sep()
    print("\nTest 7 (live): score_resume() returns valid JSON structure\n")
    result = score_resume(JD, RESUME_QUALIFIED, candidate_name="Priya", add_delay=False)
    if result is None:
        fail("score_resume() returned None (check API key and network)")
        return False

    all_ok = True
    all_ok &= assert_true("score"     in result, "Result has 'score' key",     "Result missing 'score'")
    all_ok &= assert_true("strengths" in result, "Result has 'strengths' key", "Result missing 'strengths'")
    all_ok &= assert_true("gaps"      in result, "Result has 'gaps' key",      "Result missing 'gaps'")
    all_ok &= assert_true("verdict"   in result, "Result has 'verdict' key",   "Result missing 'verdict'")
    all_ok &= assert_true(
        isinstance(result["score"], int) and 0 <= result["score"] <= 100,
        f"Score is valid int in [0,100]: {result['score']}",
        f"Score invalid: {result.get('score')}"
    )
    all_ok &= assert_true(
        isinstance(result["strengths"], list) and len(result["strengths"]) > 0,
        f"Strengths is non-empty list: {result['strengths']}",
        "Strengths is not a non-empty list"
    )
    print(f"\n  [INFO]  Full result:")
    print(f"          Score:     {result['score']}")
    print(f"          Strengths: {result['strengths']}")
    print(f"          Gaps:      {result['gaps']}")
    print(f"          Verdict:   {result['verdict']}")
    return all_ok


def test_qualified_vs_unqualified():
    sep()
    print("\nTest 8 (live): Qualified candidate scores >= 20 pts higher than unqualified\n")
    print("  Scoring qualified candidate (Priya - Python/FastAPI)...")
    qualified_result = score_resume(JD, RESUME_QUALIFIED, candidate_name="Priya", add_delay=False)

    print("  Scoring unqualified candidate (Sneha - Graphic Designer)...")
    unqualified_result = score_resume(JD, RESUME_UNQUALIFIED, candidate_name="Sneha", add_delay=True)

    if qualified_result is None or unqualified_result is None:
        fail("One or both score_resume() calls returned None")
        return False

    q_score = qualified_result["score"]
    u_score = unqualified_result["score"]
    diff    = q_score - u_score

    print(f"\n  [INFO]  Qualified (Priya):   {q_score}/100")
    print(f"  [INFO]  Unqualified (Sneha):  {u_score}/100")
    print(f"  [INFO]  Difference:           {diff} points")

    return assert_true(
        diff >= 20,
        f"Score gap = {diff} pts (>= 20 pts threshold) -- LLM discriminates correctly",
        f"Score gap = {diff} pts -- expected >= 20, LLM may not be discriminating"
    )


def test_batch_scoring():
    sep()
    print("\nTest 9 (live): score_resumes_batch() processes 3 candidates\n")
    candidates = [
        {"name": "Priya_Sharma.pdf",  "text": RESUME_QUALIFIED},
        {"name": "Sneha_Patel.pdf",   "text": RESUME_UNQUALIFIED},
        {"name": "Aryan_Gupta.pdf",   "text": RESUME_MODERATE},
    ]
    batch   = score_resumes_batch(JD, candidates, max_candidates=3)
    results = batch["results"]   # list of candidate dicts
    all_ok  = True
    all_ok &= assert_true(
        len(results) == 3,
        f"Batch returned {len(results)} results (expected 3)",
        f"Batch returned {len(results)} results, expected 3"
    )
    all_ok &= assert_true(
        all("llm_result" in r for r in results),
        "All batch results have 'llm_result' key",
        "Some batch results missing 'llm_result' key"
    )
    all_ok &= assert_true(
        "rate_limit_hit" in batch,
        "Batch result has 'rate_limit_hit' flag",
        "Batch result missing 'rate_limit_hit' flag"
    )
    print(f"\n  [INFO]  Batch scores:")
    for r in results:
        score = r["llm_result"]["score"] if r["llm_result"] else "N/A"
        print(f"    {r['name']:<25}  score={score}")
    return all_ok


# ─────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────

def run_all():
    print("AI Resume Screener -- Phase 4: Groq LLM Scoring Tests")
    print("=" * 60)

    # Offline tests (always run)
    offline_tests = [
        ("Prompt builder output",          test_prompt_builder),
        ("JSON parser: clean",             test_json_extractor_clean),
        ("JSON parser: markdown fenced",   test_json_extractor_fenced),
        ("JSON parser: embedded in prose", test_json_extractor_embedded),
        ("Score clamping [0,100]",         test_validate_result_clamping),
    ]

    # Live tests (skip if no API key)
    live_tests = [
        ("Groq API connectivity",          test_groq_connectivity),
        ("score_resume() JSON structure",  test_score_resume_structure),
        ("Qualified vs unqualified gap",   test_qualified_vs_unqualified),
        ("score_resumes_batch() 3 items",  test_batch_scoring),
    ]

    results = []
    for name, fn in offline_tests:
        try:
            passed = fn()
        except Exception as e:
            fail(f"{name} raised exception: {e}")
            passed = False
        results.append((name, passed, False))

    for name, fn in live_tests:
        if not HAS_KEY:
            skip(f"[LIVE] {name} -- no GROQ_API_KEY")
            results.append((name, None, True))
            continue
        try:
            passed = fn()
        except Exception as e:
            fail(f"{name} raised exception: {e}")
            passed = False
        results.append((name, passed, False))

    sep()
    print("\nTEST SUMMARY\n")
    all_passed = True
    for name, passed, skipped in results:
        if skipped:
            print(f"  [SKIP]  {name}")
        elif passed:
            print(f"  [PASS]  {name}")
        else:
            print(f"  [FAIL]  {name}")
            all_passed = False

    print()
    if all_passed:
        if not HAS_KEY:
            print("All offline tests passed.")
            print("Add GROQ_API_KEY to .env to run live tests.")
        else:
            print("All Phase 4 tests passed! Groq LLM scoring is working correctly.")
    else:
        print("Some tests failed. Review output above.")
        sys.exit(1)


if __name__ == "__main__":
    run_all()
