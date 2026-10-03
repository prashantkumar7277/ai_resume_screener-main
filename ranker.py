"""
ranker.py -- Phase 5: Ranking Engine

Combines FAISS cosine similarity scores (30%) with Groq LLM scores (70%)
into a single final score, then produces a ranked leaderboard.

Pipeline flow:
  1. All resumes go through FAISS -> get cosine similarity scores (0-1)
  2. Only top-10 FAISS candidates go to Groq LLM (saves API calls)
  3. Scores are combined with configurable weights
  4. Candidates are sorted by final score descending

Public API:
  run_pipeline(jd_text, resumes, top_k_faiss, top_k_llm, weights)
      -> list[CandidateResult]
  combine_scores(faiss_score, llm_score, weights)  -> float
  rank_candidates(results)                          -> list[CandidateResult]
  print_ranking_table(results)                      -> None
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from utils.embedder import embed_text
from utils.vector_store import build_index, SearchResult
from utils.llm_scorer import score_resumes_batch


# ─────────────────────────────────────────────────────────────────────────────
# Data types
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateResult:
    """Complete result record for one candidate."""
    rank:        int
    name:        str
    text:        str             # resume text (for display/export)
    faiss_score: float           # cosine similarity (0-1)
    llm_score:   Optional[int]   # Groq score (0-100), None if skipped/failed
    final_score: float           # weighted combination (0-100 scale)
    strengths:   list[str]       = field(default_factory=list)
    gaps:        list[str]       = field(default_factory=list)
    verdict:     str             = ""
    llm_used:    bool            = False   # True if LLM was called for this candidate

    def __repr__(self) -> str:
        return (f"CandidateResult(rank={self.rank}, name={self.name!r}, "
                f"final_score={self.final_score:.1f}, llm_score={self.llm_score})")


# ─────────────────────────────────────────────────────────────────────────────
# Core scoring functions
# ─────────────────────────────────────────────────────────────────────────────

def combine_scores(
    faiss_score: float,
    llm_score: Optional[int],
    weights: tuple[float, float] = (0.3, 0.7),
) -> float:
    """
    Combine FAISS cosine score and LLM score into a single 0-100 score.

    Args:
        faiss_score: Cosine similarity in [0, 1] from FAISS.
        llm_score:   Integer score in [0, 100] from Groq LLM, or None.
        weights:     (faiss_weight, llm_weight). Must sum to 1.0.
                     Default: 30% FAISS + 70% LLM (as per plan.md).

    Returns:
        Float in [0, 100]. If llm_score is None, uses FAISS score only
        (normalised to 0-100 scale) so the candidate still gets ranked.
    """
    faiss_w, llm_w = weights

    # Normalise FAISS cosine score (0-1) to 0-100 scale
    faiss_100 = faiss_score * 100.0

    if llm_score is None:
        # LLM was not called or failed — fall back to FAISS score only
        return round(faiss_100, 2)

    combined = faiss_w * faiss_100 + llm_w * float(llm_score)
    return round(combined, 2)


def rank_candidates(results: list[CandidateResult]) -> list[CandidateResult]:
    """
    Sort candidates by final_score descending and assign sequential rank numbers.

    Args:
        results: Unranked list of CandidateResult objects.

    Returns:
        New sorted list with rank fields updated (1 = best).
    """
    sorted_results = sorted(results, key=lambda r: r.final_score, reverse=True)
    for i, r in enumerate(sorted_results, start=1):
        r.rank = i
    return sorted_results


# ─────────────────────────────────────────────────────────────────────────────
# Display helper
# ─────────────────────────────────────────────────────────────────────────────

def print_ranking_table(results: list[CandidateResult], weights: tuple = (0.3, 0.7)) -> None:
    """Print a formatted ranking table to the console."""
    faiss_w, llm_w = weights
    print()
    print(f"{'RANK':<5} {'CANDIDATE':<28} {'FAISS':>6} {'LLM':>5} {'FINAL':>7}  VERDICT")
    print("-" * 90)
    for r in results:
        llm_str   = f"{r.llm_score:>5}" if r.llm_score is not None else "  N/A"
        verdict   = r.verdict[:45] + "..." if len(r.verdict) > 45 else r.verdict
        llm_flag  = " *" if not r.llm_used else ""
        print(f"  #{r.rank:<4} {r.name:<28} {r.faiss_score*100:>5.1f}% {llm_str} {r.final_score:>7.1f}  {verdict}{llm_flag}")
    print()
    print(f"  Weights: FAISS={faiss_w*100:.0f}%  LLM={llm_w*100:.0f}%")
    print(f"  * = LLM not called (below top-N FAISS threshold)")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline orchestrator
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(
    jd_text:      str,
    resumes:      list[dict],
    top_k_faiss:  int = 20,
    top_k_llm:    int = 10,
    weights:      tuple[float, float] = (0.3, 0.7),
    verbose:      bool = True,
) -> dict:
    """
    Run the full resume screening pipeline end-to-end.

    Steps:
      1. Build FAISS index from all resumes
      2. Search FAISS for top_k_faiss candidates
      3. Call Groq LLM on top_k_llm of those (saves API calls)
      4. Combine scores and produce final ranked leaderboard

    Args:
        jd_text:     Job description text.
        resumes:     List of {"name": str, "text": str} dicts.
        top_k_faiss: How many FAISS results to retrieve.
        top_k_llm:   Of those FAISS results, how many to send to Groq LLM.
        weights:     (faiss_weight, llm_weight) for combine_scores().
        verbose:     Print progress messages.

    Returns:
        dict with keys:
          "ranked"          -> list[CandidateResult] sorted by final_score descending
          "rate_limit_hit"  -> bool
          "rate_limit_msg"  -> str (empty if no rate limit issue)
    """
    if verbose:
        print(f"[INFO] Starting pipeline: {len(resumes)} resume(s), "
              f"top_k_faiss={top_k_faiss}, top_k_llm={top_k_llm}, "
              f"weights=FAISS {weights[0]*100:.0f}% + LLM {weights[1]*100:.0f}%")

    # ── Step 1: FAISS search ────────────────────────────────────────────────
    if verbose:
        print(f"\n[STEP 1] Building FAISS index and searching...")
    store = build_index(resumes)
    faiss_results: list[SearchResult] = store.search(jd_text, top_k=top_k_faiss)

    # Map name -> faiss_score for quick lookup
    faiss_map = {r.name: r for r in faiss_results}

    # ── Step 2: LLM scoring on top candidates ───────────────────────────────
    top_for_llm = faiss_results[:top_k_llm]
    llm_candidates = [{"name": r.name, "text": r.text} for r in top_for_llm]

    if verbose:
        print(f"\n[STEP 2] Calling Groq LLM on top {len(llm_candidates)} candidate(s)...")
    batch = score_resumes_batch(jd_text, llm_candidates, max_candidates=top_k_llm)
    llm_results    = batch["results"]
    rate_limit_hit = batch["rate_limit_hit"]
    rate_limit_msg = batch["rate_limit_msg"]

    if rate_limit_hit and verbose:
        print(f"[WARN] Rate limit was hit. Some candidates will use FAISS-only scores.")

    # Map name -> llm_result
    llm_map = {r["name"]: r["llm_result"] for r in llm_results}

    # ── Step 3: Combine scores ───────────────────────────────────────────────
    if verbose:
        print(f"\n[STEP 3] Combining scores (FAISS {weights[0]*100:.0f}% + LLM {weights[1]*100:.0f}%)...")

    candidate_results: list[CandidateResult] = []

    for faiss_r in faiss_results:
        llm_result = llm_map.get(faiss_r.name)      # None if not in LLM batch
        llm_used   = faiss_r.name in llm_map

        # Safety guard: llm_result must be a dict; treat anything else as None
        # so a malformed LLM response never crashes score assembly.
        if llm_result is not None and not isinstance(llm_result, dict):
            print(f"[WARN] [{faiss_r.name}] llm_result is not a dict "
                  f"(got {type(llm_result).__name__}). Falling back to FAISS-only scoring.")
            llm_result = None

        llm_score  = llm_result.get("score")      if llm_result else None
        strengths  = llm_result.get("strengths", []) if llm_result else []
        gaps       = llm_result.get("gaps", [])      if llm_result else []
        verdict    = llm_result.get("verdict", "Not analysed by LLM") if llm_result else "Not analysed by LLM"

        final_score = combine_scores(faiss_r.similarity, llm_score, weights)

        candidate_results.append(CandidateResult(
            rank        = 0,            # assigned by rank_candidates()
            name        = faiss_r.name,
            text        = faiss_r.text,
            faiss_score = faiss_r.similarity,
            llm_score   = llm_score,
            final_score = final_score,
            strengths   = strengths,
            gaps        = gaps,
            verdict     = verdict,
            llm_used    = llm_used,
        ))

    # ── Step 4: Rank ─────────────────────────────────────────────────────────
    ranked = rank_candidates(candidate_results)

    if verbose:
        print(f"\n[STEP 4] Final ranked leaderboard ({len(ranked)} candidates):")
        print_ranking_table(ranked, weights=weights)

    return {
        "ranked":         ranked,
        "rate_limit_hit": rate_limit_hit,
        "rate_limit_msg": rate_limit_msg,
    }
