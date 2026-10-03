# -*- coding: utf-8 -*-
"""
test_ranker.py -- Phase 5: Ranking Engine Tests

Tests utils/ranker.py:
  1. combine_scores() with 30/70 default weights
  2. combine_scores() with 50/50 weights (verify different result)
  3. combine_scores() with llm_score=None falls back to FAISS only
  4. rank_candidates() sorts correctly and assigns sequential ranks
  5. rank_candidates() rank changes when weights change
  6. Full end-to-end pipeline: 5 resumes + 1 JD -> ranked table
  7. Rank #1 is the most qualified candidate (eye test)
  8. FAISS-only candidates still appear in results (below LLM threshold)

Run:
    python test_ranker.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
load_dotenv()

from utils.ranker import combine_scores, rank_candidates, run_pipeline, CandidateResult


# ─────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────

def ok(msg):   print(f"[PASS]  {msg}")
def fail(msg): print(f"[FAIL]  {msg}")
def sep():     print("\n" + "-" * 60)


def assert_true(condition, msg_pass, msg_fail) -> bool:
    if condition:
        ok(msg_pass)
        return True
    else:
        fail(msg_fail)
        return False


# ─────────────────────────────────────────────────────────
# Test data: 5 resumes with known quality order
# ─────────────────────────────────────────────────────────

JD = """
Senior Python Backend Developer | Fintech Startup
Requirements: Python 3.10+, FastAPI, PostgreSQL, Redis, Docker, AWS.
5+ years experience. Building scalable payment APIs and microservices.
Must have: REST API design, async Python, CI/CD pipelines.
"""

RESUMES = [
    {
        "name": "Priya_Sharma.pdf",
        "text": (
            "Priya Sharma - Senior Python Backend Developer, 5 years. "
            "Skills: Python 3.11, FastAPI, async, PostgreSQL, Redis, Docker, AWS, Kafka, Kubernetes. "
            "At Razorpay: led 5-engineer team building payment APIs handling 10,000 TPS. "
            "Reduced latency 40% with Redis caching. CI/CD via GitHub Actions. BITS Pilani."
        ),
    },
    {
        "name": "Aryan_Gupta.pdf",
        "text": (
            "Aryan Gupta - Python Backend Developer, 3 years. "
            "Skills: Python, Django, DRF, PostgreSQL, basic Docker, some AWS S3. "
            "Built REST APIs for e-commerce at Flipkart. No Redis or async experience. "
            "Learning FastAPI. B.Tech IT NIT Trichy."
        ),
    },
    {
        "name": "Kavya_Nair.pdf",
        "text": (
            "Kavya Nair - DevOps and Cloud Engineer, 5 years. "
            "Skills: AWS, Kubernetes, Terraform, Docker, CI/CD, Jenkins, Python scripting. "
            "Managed infrastructure for microservices at Infosys. Not a backend developer. "
            "B.E. ECE, NITK."
        ),
    },
    {
        "name": "Rohan_Mehta.pdf",
        "text": (
            "Rohan Mehta - Java Backend Developer, 5 years. "
            "Skills: Java, Spring Boot, Hibernate, MySQL, Kafka, RabbitMQ. "
            "Enterprise banking APIs at TCS. No Python or FastAPI experience. "
            "B.Tech CS, Pune University."
        ),
    },
    {
        "name": "Sneha_Patel.pdf",
        "text": (
            "Sneha Patel - Senior Graphic Designer, 6 years. "
            "Skills: Adobe Photoshop, Illustrator, Figma, After Effects. "
            "Brand identities for 30+ FMCG companies at Ogilvy. "
            "BFA Sir JJ School of Art."
        ),
    },
]


# ─────────────────────────────────────────────────────────
# Offline tests (no API calls)
# ─────────────────────────────────────────────────────────

def test_combine_scores_default_weights():
    sep()
    print("\nTest 1: combine_scores() with default 30/70 weights\n")
    # FAISS=0.80 (80%), LLM=90 -> 0.3*80 + 0.7*90 = 24 + 63 = 87.0
    score = combine_scores(0.80, 90, weights=(0.3, 0.7))
    all_ok = True
    all_ok &= assert_true(
        score == 87.0,
        f"combine_scores(0.80, 90, (0.3,0.7)) = {score} (expected 87.0)",
        f"Expected 87.0, got {score}"
    )
    # FAISS=0.50 (50%), LLM=50 -> 0.3*50 + 0.7*50 = 15 + 35 = 50.0
    score2 = combine_scores(0.50, 50, weights=(0.3, 0.7))
    all_ok &= assert_true(
        score2 == 50.0,
        f"combine_scores(0.50, 50, (0.3,0.7)) = {score2} (expected 50.0)",
        f"Expected 50.0, got {score2}"
    )
    return all_ok


def test_combine_scores_equal_weights():
    sep()
    print("\nTest 2: combine_scores() with 50/50 weights gives different result\n")
    # With 30/70: 0.3*80 + 0.7*20 = 24 + 14 = 38.0
    score_3070 = combine_scores(0.80, 20, weights=(0.3, 0.7))
    # With 50/50: 0.5*80 + 0.5*20 = 40 + 10 = 50.0
    score_5050 = combine_scores(0.80, 20, weights=(0.5, 0.5))
    all_ok = True
    all_ok &= assert_true(
        score_3070 != score_5050,
        f"Different weights give different scores: 30/70={score_3070}, 50/50={score_5050}",
        f"Weights had no effect: both returned {score_3070}"
    )
    all_ok &= assert_true(
        score_5050 > score_3070,
        f"50/50 weights favour high FAISS score more ({score_5050} > {score_3070})",
        f"Unexpected ordering: 50/50={score_5050}, 30/70={score_3070}"
    )
    return all_ok


def test_combine_scores_no_llm():
    sep()
    print("\nTest 3: combine_scores() with llm_score=None falls back to FAISS only\n")
    score = combine_scores(0.75, None, weights=(0.3, 0.7))
    all_ok = True
    all_ok &= assert_true(
        score == 75.0,
        f"combine_scores(0.75, None) = {score} (expected 75.0 = FAISS only)",
        f"Expected 75.0, got {score}"
    )
    return all_ok


def test_rank_candidates_sorting():
    sep()
    print("\nTest 4: rank_candidates() sorts by final_score and assigns correct ranks\n")

    # Create 4 unordered mock results
    mock = [
        CandidateResult(rank=0, name="C", text="", faiss_score=0.5, llm_score=60, final_score=60.0),
        CandidateResult(rank=0, name="A", text="", faiss_score=0.9, llm_score=95, final_score=95.0),
        CandidateResult(rank=0, name="D", text="", faiss_score=0.2, llm_score=10, final_score=10.0),
        CandidateResult(rank=0, name="B", text="", faiss_score=0.7, llm_score=75, final_score=75.0),
    ]
    ranked = rank_candidates(mock)

    all_ok = True
    all_ok &= assert_true(
        [r.name for r in ranked] == ["A", "B", "C", "D"],
        f"Sorted correctly: {[r.name for r in ranked]}",
        f"Wrong order: {[r.name for r in ranked]}, expected [A, B, C, D]"
    )
    all_ok &= assert_true(
        [r.rank for r in ranked] == [1, 2, 3, 4],
        f"Ranks are 1-based sequential: {[r.rank for r in ranked]}",
        f"Wrong ranks: {[r.rank for r in ranked]}"
    )
    all_ok &= assert_true(
        ranked[0].final_score >= ranked[-1].final_score,
        f"Rank 1 ({ranked[0].final_score}) >= Rank last ({ranked[-1].final_score})",
        "Sorting is incorrect"
    )
    return all_ok


def test_weight_change_affects_ranking():
    sep()
    print("\nTest 5: Changing weights from 30/70 to 50/50 changes rankings\n")

    # Candidate A: low FAISS (0.3), high LLM (90) -> 30/70 favours A
    # Candidate B: high FAISS (0.9), low LLM (20) -> 50/50 favours B more

    a_3070 = combine_scores(0.30, 90, (0.3, 0.7))  # 0.3*30 + 0.7*90 = 9+63=72
    b_3070 = combine_scores(0.90, 20, (0.3, 0.7))  # 0.3*90 + 0.7*20 = 27+14=41

    a_5050 = combine_scores(0.30, 90, (0.5, 0.5))  # 0.5*30 + 0.5*90 = 15+45=60
    b_5050 = combine_scores(0.90, 20, (0.5, 0.5))  # 0.5*90 + 0.5*20 = 45+10=55

    print(f"  Candidate A (low FAISS=0.30, high LLM=90): 30/70={a_3070} | 50/50={a_5050}")
    print(f"  Candidate B (high FAISS=0.90, low LLM=20): 30/70={b_3070} | 50/50={b_5050}")
    print(f"  30/70 leader: {'A' if a_3070 > b_3070 else 'B'}")
    print(f"  50/50 leader: {'A' if a_5050 > b_5050 else 'B'}")

    all_ok = True
    all_ok &= assert_true(
        a_3070 > b_3070,
        f"30/70 weights: high-LLM candidate A wins ({a_3070} > {b_3070})",
        f"30/70 weights: unexpected winner"
    )
    all_ok &= assert_true(
        abs(a_5050 - b_5050) < abs(a_3070 - b_3070),
        f"50/50 weights narrow the gap: {abs(a_5050-b_5050):.1f} < {abs(a_3070-b_3070):.1f}",
        "Changing weights did not affect score gap as expected"
    )
    return all_ok


# ─────────────────────────────────────────────────────────
# Live pipeline test (calls Groq)
# ─────────────────────────────────────────────────────────

def test_full_pipeline():
    sep()
    print("\nTest 6 (live): Full end-to-end pipeline with 5 resumes + 1 JD\n")
    print("  This will call FAISS + Groq LLM for top candidates...\n")

    results = run_pipeline(
        jd_text     = JD,
        resumes     = RESUMES,
        top_k_faiss = 5,
        top_k_llm   = 5,     # score all 5 via LLM for thorough test
        weights     = (0.3, 0.7),
        verbose     = True,
    )

    all_ok = True
    all_ok &= assert_true(
        len(results) == 5,
        f"Pipeline returned {len(results)} results (expected 5)",
        f"Expected 5 results, got {len(results)}"
    )
    all_ok &= assert_true(
        results == sorted(results, key=lambda r: r.final_score, reverse=True),
        "Results are sorted by final_score descending",
        "Results are NOT sorted correctly"
    )
    all_ok &= assert_true(
        results[0].name == "Priya_Sharma.pdf",
        f"Rank #1 is Priya_Sharma.pdf (best Python/FastAPI candidate)",
        f"Rank #1 is {results[0].name} -- expected Priya_Sharma.pdf"
    )
    all_ok &= assert_true(
        results[-1].name == "Sneha_Patel.pdf",
        f"Last place is Sneha_Patel.pdf (graphic designer)",
        f"Last place is {results[-1].name} -- expected Sneha_Patel.pdf"
    )
    all_ok &= assert_true(
        all(r.llm_used for r in results),
        "All 5 candidates were scored by LLM",
        "Some candidates were not scored by LLM"
    )

    # Print detailed results for each candidate
    print("\n  Detailed results:")
    for r in results:
        print(f"\n  Rank #{r.rank}: {r.name}")
        print(f"    FAISS:  {r.faiss_score*100:.1f}%")
        print(f"    LLM:    {r.llm_score}/100")
        print(f"    Final:  {r.final_score:.1f}/100")
        print(f"    Verdict: {r.verdict}")
        if r.strengths:
            print(f"    Top strength: {r.strengths[0]}")

    return all_ok


def test_top_k_llm_filter():
    sep()
    print("\nTest 7 (live): Only top-2 go to LLM, rest get FAISS-only scores\n")

    results = run_pipeline(
        jd_text     = JD,
        resumes     = RESUMES,
        top_k_faiss = 5,
        top_k_llm   = 2,     # only top 2 get LLM scoring
        weights     = (0.3, 0.7),
        verbose     = False,
    )

    llm_used_count = sum(1 for r in results if r.llm_used)
    no_llm_count   = sum(1 for r in results if not r.llm_used)

    all_ok = True
    all_ok &= assert_true(
        llm_used_count == 2,
        f"Exactly 2 candidates scored by LLM (got {llm_used_count})",
        f"Expected 2 LLM-scored candidates, got {llm_used_count}"
    )
    all_ok &= assert_true(
        no_llm_count == 3,
        f"3 candidates used FAISS-only fallback (got {no_llm_count})",
        f"Expected 3 FAISS-only candidates, got {no_llm_count}"
    )
    all_ok &= assert_true(
        len(results) == 5,
        f"All 5 candidates still appear in results",
        f"Expected 5 results, got {len(results)}"
    )

    print(f"\n  LLM used:    {[r.name for r in results if r.llm_used]}")
    print(f"  FAISS only:  {[r.name for r in results if not r.llm_used]}")
    return all_ok


# ─────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────

def run_all():
    print("AI Resume Screener -- Phase 5: Ranking Engine Tests")
    print("=" * 60)

    tests = [
        ("combine_scores() default 30/70 weights",    test_combine_scores_default_weights),
        ("combine_scores() 50/50 weights differ",     test_combine_scores_equal_weights),
        ("combine_scores() None LLM fallback",        test_combine_scores_no_llm),
        ("rank_candidates() sorting + ranks",         test_rank_candidates_sorting),
        ("Weight change affects ranking gap",         test_weight_change_affects_ranking),
        ("Full end-to-end pipeline (5 resumes)",      test_full_pipeline),
        ("top_k_llm filter (only top 2 get LLM)",    test_top_k_llm_filter),
    ]

    results = []
    for name, fn in tests:
        try:
            passed = fn()
        except Exception as e:
            fail(f"{name} raised exception: {e}")
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
        print("All Phase 5 tests passed! Ranking engine is working correctly.")
    else:
        print("Some tests failed. Review output above.")
        sys.exit(1)


if __name__ == "__main__":
    run_all()
