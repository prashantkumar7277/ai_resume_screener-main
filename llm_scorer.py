"""
llm_scorer.py -- Phase 4: Groq LLM Scoring

Uses Groq's Llama 3.3 70B to deeply analyse a candidate resume against
a job description and return a structured JSON score with:
  - score (0-100)
  - strengths (list of 3 bullet points)
  - gaps (list of 2 weak points)
  - verdict (one sentence summary)

Design:
  - Prompt forces JSON-only output (no prose wrapping)
  - Retry logic for Groq's 30 RPM rate limit
  - Validates and clamps score to [0, 100]
  - Falls back to None on parse failure so the pipeline can continue
    using only FAISS scores for that candidate
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Optional

from dotenv import load_dotenv
from groq import Groq, RateLimitError, APIError

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

load_dotenv()

MODEL_NAME       = "openai/gpt-oss-120b"
MAX_RETRIES      = 3
RETRY_SLEEP_SEC  = 4        # wait between retries on rate limit (exponential)
CALL_DELAY_SEC   = 2        # polite delay between successive calls (30 RPM limit)
MAX_RESUME_CHARS = 3000     # truncate very long resumes to save tokens
MAX_JD_CHARS     = 2000     # truncate very long JDs


# ─────────────────────────────────────────────────────────────────────────────
# Custom exceptions
# ─────────────────────────────────────────────────────────────────────────────

class GroqRateLimitExhausted(Exception):
    """
    Raised when the Groq API rate limit is hit and all retries are exhausted.
    The app.py frontend catches this and shows a graceful Streamlit warning
    instead of an ugly traceback.
    """
    pass


# ─────────────────────────────────────────────────────────────────────────────
# Prompt template (from plan.md)
# ─────────────────────────────────────────────────────────────────────────────

SCORING_PROMPT = """\
You are an expert recruiter evaluating candidates for a job.

JOB DESCRIPTION:
{jd}

CANDIDATE RESUME:
{resume}

Analyse the candidate's fit for this role carefully.
Respond ONLY with valid JSON — no explanation, no markdown, no extra text:
{{
  "score": <integer 0-100>,
  "strengths": ["<strength1>", "<strength2>", "<strength3>"],
  "gaps": ["<gap1>", "<gap2>"],
  "verdict": "<one concise sentence summarising overall fit>"
}}
"""

# ─────────────────────────────────────────────────────────────────────────────
# Groq client (lazy init so import doesn't fail if key is missing)
# ─────────────────────────────────────────────────────────────────────────────

_client: Optional[Groq] = None


def _get_client() -> Groq:
    global _client
    if _client is None:
        api_key = os.getenv("GROQ_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file: GROQ_API_KEY=gsk_..."
            )
        _client = Groq(api_key=api_key)
    return _client


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builder
# ─────────────────────────────────────────────────────────────────────────────

def build_scoring_prompt(jd: str, resume: str) -> str:
    """
    Build the scoring prompt by injecting (and truncating) JD and resume text.

    Args:
        jd:     Job description text.
        resume: Candidate resume text.

    Returns:
        Formatted prompt string ready to send to Groq.
    """
    jd_text     = jd.strip()[:MAX_JD_CHARS]
    resume_text = resume.strip()[:MAX_RESUME_CHARS]
    return SCORING_PROMPT.format(jd=jd_text, resume=resume_text)


# ─────────────────────────────────────────────────────────────────────────────
# JSON extraction helper
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> Optional[dict]:
    """
    Extract and validate the JSON object from a raw LLM response string.

    Handles cases where the model wraps the JSON in markdown code fences
    or adds a small amount of prose before/after.

    If the LLM returns a JSON array (e.g. [{...}]) instead of a bare object,
    we unwrap the first element that is a dict so downstream code never
    receives a list where a dict is expected.

    Returns None if no valid JSON dict can be found.
    """
    def _unwrap(parsed):
        """If parsed is a list, try to extract the first dict inside it."""
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            for item in parsed:
                if isinstance(item, dict):
                    return item
        return None

    # 1. Try direct parse first
    try:
        result = json.loads(text.strip())
        unwrapped = _unwrap(result)
        if unwrapped is not None:
            return unwrapped
    except json.JSONDecodeError:
        pass

    # 2. Strip markdown code fences (```json ... ``` or ``` ... ```)
    fenced = re.search(r"```(?:json)?\s*(\{.*?\}|\[.*?\])\s*```", text, re.DOTALL)
    if fenced:
        try:
            result = json.loads(fenced.group(1))
            unwrapped = _unwrap(result)
            if unwrapped is not None:
                return unwrapped
        except json.JSONDecodeError:
            pass

    # 3. Find the first {...} block in the response
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        try:
            result = json.loads(brace_match.group(0))
            unwrapped = _unwrap(result)
            if unwrapped is not None:
                return unwrapped
        except json.JSONDecodeError:
            pass

    # 4. Find the first [...] block (array-only response)
    bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
    if bracket_match:
        try:
            result = json.loads(bracket_match.group(0))
            unwrapped = _unwrap(result)
            if unwrapped is not None:
                return unwrapped
        except json.JSONDecodeError:
            pass

    return None


def _validate_result(data) -> dict:
    """
    Validate and sanitise the parsed JSON scoring result.

    - Accepts a dict; returns a safe default if data is not a dict
    - Clamps score to [0, 100]
    - Ensures strengths/gaps are lists of strings
    - Ensures verdict is a string
    """
    # Guard: if data is a list (e.g. LLM wrapped the object in an array)
    # try to unwrap the first dict element before failing.
    if isinstance(data, list):
        data = next((item for item in data if isinstance(item, dict)), {})

    if not isinstance(data, dict):
        # Unexpected type — return safe defaults rather than crashing
        print(f"[WARN] _validate_result received unexpected type: {type(data).__name__}. "
              "Returning default result.")
        return {"score": 0, "strengths": [], "gaps": [], "verdict": "Parse error — unexpected response format."}

    try:
        score = data.get("score", 0)
        score = max(0, min(100, int(score)))
    except (TypeError, ValueError):
        score = 0

    strengths = data.get("strengths", [])
    if not isinstance(strengths, list):
        strengths = [str(strengths)]
    strengths = [str(s) for s in strengths[:5]]     # max 5 items

    gaps = data.get("gaps", [])
    if not isinstance(gaps, list):
        gaps = [str(gaps)]
    gaps = [str(g) for g in gaps[:5]]

    verdict = str(data.get("verdict", "No verdict provided."))

    return {
        "score":     score,
        "strengths": strengths,
        "gaps":      gaps,
        "verdict":   verdict,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main scoring function
# ─────────────────────────────────────────────────────────────────────────────

def score_resume(
    jd: str,
    resume: str,
    candidate_name: str = "Candidate",
    add_delay: bool = True,
) -> Optional[dict]:
    """
    Score a single resume against a job description using Groq LLM.

    Args:
        jd:             Job description text.
        resume:         Candidate resume text.
        candidate_name: Used only for logging.
        add_delay:      If True, sleeps CALL_DELAY_SEC before the API call
                        to respect Groq's 30 RPM rate limit.

    Returns:
        dict with keys: score, strengths, gaps, verdict
        Returns None if the API call failed after all retries or JSON was invalid.
    """
    if add_delay:
        time.sleep(CALL_DELAY_SEC)

    prompt = build_scoring_prompt(jd, resume)
    client = _get_client()

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,       # low temp for consistent JSON output
                max_tokens=1500,        # score JSON is small
            )
            raw_text = response.choices[0].message.content
            parsed   = _extract_json(raw_text)

            if parsed is None:
                print(f"[WARN] [{candidate_name}] Could not parse JSON from LLM response.")
                print(f"       Raw response: {raw_text[:200]}")
                return None

            result = _validate_result(parsed)
            print(f"[INFO] [{candidate_name}] LLM score: {result['score']}/100 — {result['verdict']}")
            return result

        except RateLimitError:
            if attempt < MAX_RETRIES:
                wait = RETRY_SLEEP_SEC * attempt   # exponential: 4s, 8s, 12s
                print(f"[WARN] [{candidate_name}] Rate limit hit (attempt {attempt}/{MAX_RETRIES}). "
                      f"Retrying in {wait}s...")
                time.sleep(wait)
            else:
                print(f"[ERROR] [{candidate_name}] Rate limit exhausted after {MAX_RETRIES} retries.")
                raise GroqRateLimitExhausted(
                    f"Groq API 30 RPM rate limit was hit for '{candidate_name}' and could not "
                    f"recover after {MAX_RETRIES} retries. "
                    "Tip: reduce the number of LLM candidates in Advanced Settings, "
                    "or wait 60 seconds and try again."
                )

        except APIError as e:
            print(f"[ERROR] [{candidate_name}] Groq API error: {e}")
            return None

        except Exception as e:
            print(f"[ERROR] [{candidate_name}] Unexpected error: {e}")
            return None

    return None


# ─────────────────────────────────────────────────────────────────────────────
# Batch scoring (used by Phase 5 Ranking Engine)
# ─────────────────────────────────────────────────────────────────────────────

def score_resumes_batch(
    jd: str,
    candidates: list[dict],
    max_candidates: int = 10,
) -> dict:
    """
    Score a batch of candidates against a JD (only top-N to save API calls).

    Args:
        jd:             Job description text.
        candidates:     List of dicts with "name" and "text" keys.
        max_candidates: Maximum number to score (plan: top-10 from FAISS only).

    Returns:
        dict with keys:
          "results"         -> list of candidate dicts + "llm_result" key
          "rate_limit_hit"  -> bool, True if Groq rate limit was exhausted
          "rate_limit_msg"  -> str, human-readable message for the UI
    """
    to_score = candidates[:max_candidates]
    print(f"[INFO] Scoring {len(to_score)} candidate(s) via Groq LLM...")

    results = []
    rate_limit_hit = False
    rate_limit_msg = ""

    for i, candidate in enumerate(to_score, start=1):
        name = candidate.get("name", f"Candidate {i}")
        text = candidate.get("text", "")
        print(f"[INFO] [{i}/{len(to_score)}] Scoring: {name}")
        try:
            llm_result = score_resume(
                jd=jd,
                resume=text,
                candidate_name=name,
                add_delay=(i > 1),    # skip delay on first call
            )
            results.append({**candidate, "llm_result": llm_result})
        except GroqRateLimitExhausted as e:
            # Rate limit exhausted — mark remaining candidates as unscored
            rate_limit_hit = True
            rate_limit_msg = str(e)
            print(f"[WARN] Rate limit exhausted at candidate {i}/{len(to_score)}. "
                  "Remaining candidates will use FAISS-only scores.")
            # Append current candidate as unscored
            results.append({**candidate, "llm_result": None})
            # All remaining candidates get unscored too
            for remaining in to_score[i:]:
                results.append({**remaining, "llm_result": None})
            break

    return {
        "results":        results,
        "rate_limit_hit": rate_limit_hit,
        "rate_limit_msg": rate_limit_msg,
    }
