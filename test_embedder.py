# -*- coding: utf-8 -*-
"""
test_embedder.py -- Phase 2: Embedding Engine Tests

Tests utils/embedder.py:
  1. Model loads and returns correct shape
  2. Matching JD/resume pair -> cosine similarity > 0.6
  3. Unrelated JD/resume pair -> cosine similarity < 0.4
  4. Hash cache works (second call skips model)
  5. embed_batch() returns correct shape and matches embed_text()
  6. Plan-specified test: "Python developer with FastAPI experience" vs
     "Machine Learning Engineer with TensorFlow" scored against
     a "Python backend developer" JD -- FastAPI candidate should score higher

Run:
    python test_embedder.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils.embedder import (
    embed_text, embed_batch, cosine_similarity, cache_size, clear_cache, EMBEDDING_DIM
)

# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def ok(msg):   print(f"[PASS]  {msg}")
def fail(msg): print(f"[FAIL]  {msg}"); return False
def sep():     print("\n" + "-" * 60)


def assert_true(condition: bool, msg_pass: str, msg_fail: str) -> bool:
    if condition:
        ok(msg_pass)
        return True
    else:
        fail(msg_fail)
        return False


# ─────────────────────────────────────────────────────────
# Test data
# ─────────────────────────────────────────────────────────

JD_PYTHON_BACKEND = """
We are looking for a Python Backend Developer with 2+ years of experience.
Required skills: Python, FastAPI, REST APIs, PostgreSQL, Docker.
Experience with AWS, Redis, and CI/CD pipelines is a plus.
You will build scalable microservices and APIs for our fintech platform.
"""

RESUME_PYTHON_MATCH = """
Priya Sharma - Python Backend Developer
Skills: Python, FastAPI, Django, PostgreSQL, Redis, Docker, AWS, REST APIs
Experience: 3 years at Razorpay building payment processing APIs using FastAPI.
Reduced API latency by 40% with Redis caching. 92% test coverage with Pytest.
Education: B.Tech Computer Science, BITS Pilani 2021
"""

RESUME_UNRELATED = """
Rahul Verma - Graphic Designer
Skills: Adobe Photoshop, Illustrator, InDesign, Figma, Canva, Video Editing
Experience: 4 years creating brand identities and marketing materials for FMCG clients.
Designed packaging for 20+ consumer products. Won 2 national design awards.
Education: BFA, Sir JJ School of Art, Mumbai 2020
"""

# Plan-specified test texts (Phase 2, micro-step 5)
FASTAPI_CANDIDATE = "Python developer with FastAPI experience building REST APIs"
ML_CANDIDATE = "Machine Learning Engineer with TensorFlow and PyTorch deep learning"
PYTHON_BACKEND_JD = "Python backend developer building scalable REST APIs and microservices"


# ─────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────

def test_shape_and_dtype():
    sep()
    print("\nTest 1: Model loads, output shape and dtype correct\n")
    t0 = time.time()
    vec = embed_text(JD_PYTHON_BACKEND)
    elapsed = time.time() - t0

    all_ok = True
    all_ok &= assert_true(
        vec.shape == (EMBEDDING_DIM,),
        f"Shape is correct: {vec.shape}",
        f"Expected shape ({EMBEDDING_DIM},) but got {vec.shape}"
    )
    all_ok &= assert_true(
        str(vec.dtype) == "float32",
        f"dtype is float32",
        f"Expected float32 but got {vec.dtype}"
    )
    all_ok &= assert_true(
        abs(float((vec ** 2).sum()) - 1.0) < 1e-5,
        f"Vector is L2-normalised (norm = {float((vec**2).sum()):.6f})",
        f"Vector is NOT normalised (norm = {float((vec**2).sum()):.6f})"
    )
    print(f"  [INFO]  Encoding time: {elapsed:.2f}s (includes model load on first call)")
    return all_ok


def test_matching_pair_similarity():
    sep()
    print("\nTest 2: Matching JD + resume pair -> similarity > 0.6\n")
    jd_vec = embed_text(JD_PYTHON_BACKEND)
    match_vec = embed_text(RESUME_PYTHON_MATCH)
    sim = cosine_similarity(jd_vec, match_vec)
    print(f"  [INFO]  Matching pair cosine similarity: {sim:.4f}")
    return assert_true(
        sim > 0.6,
        f"Matching pair similarity = {sim:.4f} (> 0.6 threshold)",
        f"Matching pair similarity = {sim:.4f} -- expected > 0.6"
    )


def test_unrelated_pair_similarity():
    sep()
    print("\nTest 3: Unrelated resume (graphic designer) -> similarity < 0.4\n")
    jd_vec = embed_text(JD_PYTHON_BACKEND)
    unrelated_vec = embed_text(RESUME_UNRELATED)
    sim = cosine_similarity(jd_vec, unrelated_vec)
    print(f"  [INFO]  Unrelated pair cosine similarity: {sim:.4f}")
    return assert_true(
        sim < 0.4,
        f"Unrelated pair similarity = {sim:.4f} (< 0.4 threshold)",
        f"Unrelated pair similarity = {sim:.4f} -- expected < 0.4"
    )


def test_separation_between_similar_and_unrelated():
    sep()
    print("\nTest 4: Gap between matching and unrelated similarity >= 0.2\n")
    jd_vec = embed_text(JD_PYTHON_BACKEND)
    match_sim = cosine_similarity(jd_vec, embed_text(RESUME_PYTHON_MATCH))
    unrel_sim = cosine_similarity(jd_vec, embed_text(RESUME_UNRELATED))
    gap = match_sim - unrel_sim
    print(f"  [INFO]  Match sim: {match_sim:.4f} | Unrelated sim: {unrel_sim:.4f} | Gap: {gap:.4f}")
    return assert_true(
        gap >= 0.2,
        f"Score gap = {gap:.4f} (>= 0.2) -- model discriminates well",
        f"Score gap = {gap:.4f} -- expected >= 0.2, model may not discriminate"
    )


def test_cache():
    sep()
    print("\nTest 5: Hash cache prevents re-encoding same text\n")
    clear_cache()

    t0 = time.time()
    vec1 = embed_text(JD_PYTHON_BACKEND)
    t1 = time.time() - t0

    t0 = time.time()
    vec2 = embed_text(JD_PYTHON_BACKEND)  # should hit cache
    t2 = time.time() - t0

    cache_hits = cache_size()
    all_ok = True
    all_ok &= assert_true(
        (vec1 == vec2).all(),
        "Cached vector identical to original",
        "Cached vector differs from original -- cache corrupted?"
    )
    all_ok &= assert_true(
        t2 < t1,
        f"Cache hit is faster: {t1*1000:.1f}ms -> {t2*1000:.2f}ms",
        f"Cache hit not faster than first encode ({t1*1000:.1f}ms vs {t2*1000:.1f}ms)"
    )
    print(f"  [INFO]  Cache size after 2 calls to same text: {cache_hits} (expected 1)")
    return all_ok


def test_embed_batch():
    sep()
    print("\nTest 6: embed_batch() returns correct shape and consistent results\n")
    texts = [JD_PYTHON_BACKEND, RESUME_PYTHON_MATCH, RESUME_UNRELATED]
    batch = embed_batch(texts)

    all_ok = True
    all_ok &= assert_true(
        batch.shape == (3, EMBEDDING_DIM),
        f"Batch shape correct: {batch.shape}",
        f"Expected (3, {EMBEDDING_DIM}) but got {batch.shape}"
    )

    # Each row should match the single-encode result
    for i, text in enumerate(texts):
        single = embed_text(text)
        match = (abs(batch[i] - single) < 1e-5).all()
        all_ok &= assert_true(
            match,
            f"batch[{i}] matches embed_text() result",
            f"batch[{i}] differs from embed_text() result"
        )

    return all_ok


def test_plan_specified():
    """
    Direct test from plan.md Phase 2, micro-step 5:
    'Run embed_text("Python developer with FastAPI experience") and
     embed_text("Machine Learning Engineer with TensorFlow") --
     compute cosine similarity. Should be noticeably different scores
     against a "Python backend developer" JD.'
    """
    sep()
    print("\nTest 7: Plan-specified test (Phase 2 micro-step 5)\n")
    print(f"  JD: '{PYTHON_BACKEND_JD.strip()}'")

    jd_vec = embed_text(PYTHON_BACKEND_JD)
    fastapi_vec = embed_text(FASTAPI_CANDIDATE)
    ml_vec = embed_text(ML_CANDIDATE)

    sim_fastapi = cosine_similarity(jd_vec, fastapi_vec)
    sim_ml = cosine_similarity(jd_vec, ml_vec)

    print(f"\n  Candidate A: '{FASTAPI_CANDIDATE}'")
    print(f"               similarity = {sim_fastapi:.4f}")
    print(f"\n  Candidate B: '{ML_CANDIDATE}'")
    print(f"               similarity = {sim_ml:.4f}")
    print(f"\n  Difference:  {abs(sim_fastapi - sim_ml):.4f}")

    all_ok = True
    all_ok &= assert_true(
        sim_fastapi > sim_ml,
        f"FastAPI candidate ({sim_fastapi:.4f}) scores higher than ML candidate ({sim_ml:.4f}) for Python backend JD",
        f"FastAPI candidate ({sim_fastapi:.4f}) did NOT score higher than ML candidate ({sim_ml:.4f}) -- unexpected"
    )
    all_ok &= assert_true(
        abs(sim_fastapi - sim_ml) >= 0.05,
        f"Score difference is noticeable: {abs(sim_fastapi - sim_ml):.4f} (>= 0.05)",
        f"Score difference too small: {abs(sim_fastapi - sim_ml):.4f} (expected >= 0.05)"
    )
    return all_ok


# ─────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────

def run_all():
    print("AI Resume Screener -- Phase 2: Embedding Engine Tests")
    print("=" * 60)

    tests = [
        ("Shape & dtype", test_shape_and_dtype),
        ("Matching pair similarity > 0.6", test_matching_pair_similarity),
        ("Unrelated pair similarity < 0.4", test_unrelated_pair_similarity),
        ("Score gap >= 0.2", test_separation_between_similar_and_unrelated),
        ("Hash cache correctness", test_cache),
        ("embed_batch() shape & consistency", test_embed_batch),
        ("Plan-specified FastAPI vs ML test", test_plan_specified),
    ]

    results = []
    for name, fn in tests:
        try:
            passed = fn()
        except Exception as e:
            print(f"[FAIL]  {name} raised exception: {e}")
            passed = False
        results.append((name, passed))

    sep()
    print("\nTEST SUMMARY\n")
    all_passed = True
    for name, passed in results:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status}  {name}")
        all_passed &= passed

    print()
    if all_passed:
        print("All Phase 2 tests passed! Embedding engine is working correctly.")
    else:
        print("Some tests failed. Review output above.")
        sys.exit(1)


if __name__ == "__main__":
    run_all()
