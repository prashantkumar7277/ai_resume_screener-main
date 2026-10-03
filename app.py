"""
app.py -- Phase 6: Streamlit Frontend
AI Resume Screener — Full web application entry point
"""

import os
import sys
import time
import tempfile
from pathlib import Path

import fitz  # PyMuPDF
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# ── Make utils importable ────────────────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
load_dotenv()

from utils.pdf_parser import clean_text
from utils.ranker import run_pipeline, CandidateResult

# ─────────────────────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Resume Screener",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# Custom CSS — premium dark theme
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Main background ── */
.stApp {
    background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    min-height: 100vh;
}

/* ── Hero header ── */
.hero-header {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
    border-radius: 20px;
    padding: 2.5rem 2rem;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(102, 126, 234, 0.3);
}
.hero-header h1 {
    color: white;
    font-size: 2.4rem;
    font-weight: 700;
    margin: 0 0 0.5rem 0;
    letter-spacing: -0.5px;
}
.hero-header p {
    color: rgba(255,255,255,0.85);
    font-size: 1.05rem;
    margin: 0;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e1e3a 0%, #16213e 100%);
    border-right: 1px solid rgba(102, 126, 234, 0.2);
}
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] label {
    color: #c8d3f5 !important;
}
[data-testid="stSidebar"] h3 {
    color: #a78bfa !important;
    font-weight: 600;
}

/* ── Cards ── */
.metric-card {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(102, 126, 234, 0.25);
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    backdrop-filter: blur(10px);
}
.metric-card .value {
    font-size: 2rem;
    font-weight: 700;
    color: #a78bfa;
}
.metric-card .label {
    font-size: 0.8rem;
    color: #7c8db5;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin-top: 4px;
}

/* ── Candidate rank card ── */
.rank-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: 16px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: all 0.2s ease;
    position: relative;
    overflow: hidden;
}
.rank-card:hover {
    border-color: rgba(102, 126, 234, 0.5);
    background: rgba(255,255,255,0.07);
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(102, 126, 234, 0.15);
}
.rank-card.top1 {
    border-color: rgba(255, 215, 0, 0.4);
    background: rgba(255, 215, 0, 0.04);
}
.rank-card.top1:hover {
    border-color: rgba(255, 215, 0, 0.7);
}
.rank-badge {
    position: absolute;
    top: 1rem;
    right: 1rem;
    width: 36px;
    height: 36px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    font-size: 0.85rem;
}
.rank-badge.gold   { background: linear-gradient(135deg, #f6d365, #fda085); color: #5d3a00; }
.rank-badge.silver { background: linear-gradient(135deg, #c0c0c0, #e8e8e8); color: #333; }
.rank-badge.bronze { background: linear-gradient(135deg, #cd7f32, #e8a87c); color: #3d1f00; }
.rank-badge.other  { background: rgba(102, 126, 234, 0.2); color: #a78bfa; border: 1px solid #a78bfa; }

.candidate-name {
    font-size: 1.15rem;
    font-weight: 600;
    color: #e2e8f0;
    margin-bottom: 0.3rem;
}
.verdict-text {
    color: #94a3b8;
    font-size: 0.9rem;
    font-style: italic;
    margin-top: 0.4rem;
}

/* ── Score pill ── */
.score-pill {
    display: inline-block;
    padding: 0.2rem 0.8rem;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.95rem;
}
.score-high   { background: rgba(34, 197, 94, 0.15);  color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.3); }
.score-medium { background: rgba(251, 191, 36, 0.15); color: #fbbf24; border: 1px solid rgba(251, 191, 36, 0.3); }
.score-low    { background: rgba(239, 68, 68, 0.15);  color: #f87171; border: 1px solid rgba(239, 68, 68, 0.3); }

/* ── Tags ── */
.tag {
    display: inline-block;
    padding: 0.2rem 0.65rem;
    border-radius: 6px;
    font-size: 0.78rem;
    margin: 2px 3px 2px 0;
}
.tag-strength { background: rgba(34, 197, 94, 0.12);  color: #4ade80; border: 1px solid rgba(74, 222, 128, 0.2); }
.tag-gap      { background: rgba(239, 68, 68, 0.12);  color: #f87171; border: 1px solid rgba(239, 68, 68, 0.2); }

/* ── Progress bar override ── */
.stProgress > div > div > div {
    background: linear-gradient(90deg, #667eea, #764ba2);
    border-radius: 999px;
}

/* ── Section header ── */
.section-header {
    color: #a78bfa;
    font-size: 1rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin: 0.4rem 0 0.5rem 0;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea, #764ba2) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 2rem !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5) !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    background: rgba(34, 197, 94, 0.15) !important;
    color: #4ade80 !important;
    border: 1px solid rgba(74, 222, 128, 0.3) !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}

/* ── Text inputs ── */
.stTextArea textarea {
    background: rgba(255,255,255,0.05) !important;
    border: 1px solid rgba(102, 126, 234, 0.3) !important;
    border-radius: 10px !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', sans-serif !important;
}
.stTextArea textarea:focus {
    border-color: #a78bfa !important;
    box-shadow: 0 0 0 2px rgba(167, 139, 250, 0.2) !important;
}

/* ── Divider ── */
.divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, rgba(102,126,234,0.4), transparent);
    margin: 1.5rem 0;
}

/* ── Warning / info boxes ── */
.info-box {
    background: rgba(102, 126, 234, 0.1);
    border: 1px solid rgba(102, 126, 234, 0.3);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #c8d3f5;
    font-size: 0.9rem;
    margin: 0.5rem 0;
}
.warn-box {
    background: rgba(251, 191, 36, 0.1);
    border: 1px solid rgba(251, 191, 36, 0.3);
    border-radius: 10px;
    padding: 0.8rem 1rem;
    color: #fcd34d;
    font-size: 0.9rem;
    margin: 0.5rem 0;
}

/* ── Dataframe ── */
.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden;
}

/* ── Spinner ── */
.stSpinner > div {
    border-color: #a78bfa !important;
}

/* ── Expander ── */
[data-testid="stExpander"] {
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(102, 126, 234, 0.2);
    border-radius: 12px;
    overflow: hidden;
}
[data-testid="stExpander"]:hover {
    border-color: rgba(102, 126, 234, 0.4);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Cached model loading (runs once across Streamlit sessions)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading AI embedding model...")
def load_embedding_model():
    """Pre-load the sentence-transformer model into memory."""
    from utils.embedder import _get_model
    return _get_model()


# ─────────────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────────────

def score_color_class(score: float) -> str:
    if score >= 70:
        return "score-high"
    elif score >= 40:
        return "score-medium"
    return "score-low"


def rank_badge_class(rank: int) -> str:
    return {1: "gold", 2: "silver", 3: "bronze"}.get(rank, "other")


def parse_uploaded_pdf(uploaded_file) -> tuple[str, str, str]:
    """
    Save an uploaded Streamlit file to a temp path and parse it.

    Returns:
        (filename, text, error_type)
        error_type is one of: "" (ok), "scanned", "corrupted", "empty"
    """
    name   = uploaded_file.name
    suffix = Path(name).suffix.lower()

    # Validate file extension
    if suffix != ".pdf":
        return name, "", "corrupted"

    # Write to a temp file
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name
    except Exception:
        return name, "", "corrupted"

    try:
        # Try to open with PyMuPDF
        try:
            doc = fitz.open(tmp_path)
        except Exception:
            return name, "", "corrupted"

        # Check if it's an image-only / scanned PDF
        total_pages = len(doc)
        full_text   = []

        for page in doc:
            raw = page.get_text("text").strip()
            if raw:
                full_text.append(raw)

        doc.close()

        joined = "\n".join(full_text).strip()

        if not joined:
            # All pages extracted zero text — almost certainly scanned
            if total_pages > 0:
                return name, "", "scanned"
            return name, "", "empty"

        # Run the full cleaning pipeline
        cleaned = clean_text(joined)
        if len(cleaned) < 30:
            # Suspiciously short — treat as scanned
            return name, "", "scanned"

        return name, cleaned, ""

    except Exception:
        return name, "", "corrupted"

    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def results_to_dataframe(results: list[CandidateResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append({
            "Rank":         r.rank,
            "Candidate":    r.name.replace(".pdf", "").replace("_", " "),
            "FAISS Score":  f"{r.faiss_score*100:.1f}%",
            "LLM Score":    f"{r.llm_score}/100" if r.llm_score is not None else "N/A",
            "Final Score":  f"{r.final_score:.1f}",
            "Verdict":      r.verdict,
        })
    return pd.DataFrame(rows)


def results_to_csv(results: list[CandidateResult]) -> str:
    rows = []
    for r in results:
        rows.append({
            "Rank":         r.rank,
            "Candidate":    r.name,
            "FAISS Score":  round(r.faiss_score * 100, 1),
            "LLM Score":    r.llm_score if r.llm_score is not None else "",
            "Final Score":  r.final_score,
            "Strengths":    " | ".join(r.strengths),
            "Gaps":         " | ".join(r.gaps),
            "Verdict":      r.verdict,
        })
    return pd.DataFrame(rows).to_csv(index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Hero header
# ─────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="hero-header">
    <h1>🧠 AI Resume Screener</h1>
    <p>Screen up to 20 resumes against a job description in 30 seconds &nbsp;·&nbsp;
       Powered by Groq Llama 3.3 70B &nbsp;·&nbsp; Built for Indian job market</p>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — inputs
# ─────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # JD input mode
    jd_mode = st.radio(
        "Job Description Input",
        options=["Paste Text", "Upload PDF"],
        horizontal=True,
        label_visibility="visible",
    )

    jd_text = ""
    if jd_mode == "Paste Text":
        jd_text = st.text_area(
            "Job Description",
            placeholder="Paste the full job description here...\n\nExample:\nWe are looking for a Senior Python Developer with 5+ years experience in FastAPI, PostgreSQL, and AWS...",
            height=240,
            label_visibility="collapsed",
        )
        st.caption("💡 Press **Ctrl+Enter** after pasting to confirm the JD.")
    else:
        jd_pdf = st.file_uploader("Upload JD as PDF", type=["pdf"], key="jd_pdf")
        if jd_pdf:
            with st.spinner("Parsing JD..."):
                _, jd_text, jd_err = parse_uploaded_pdf(jd_pdf)
            if jd_text:
                st.success(f"JD parsed: {len(jd_text)} characters")
            else:
                err_detail = "scanned image" if jd_err == "scanned" else "corrupted or unreadable file"
                st.error(f"Could not extract text from JD PDF ({err_detail}).")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Advanced settings
    with st.expander("🔧 Advanced Settings", expanded=False):
        top_k_llm = st.slider(
            "Max LLM candidates (saves API calls)",
            min_value=1, max_value=20, value=10,
            help="Only the top-N FAISS results are sent to Groq LLM."
        )
        faiss_weight = st.slider(
            "FAISS weight (%)",
            min_value=0, max_value=100, value=30, step=5,
            help="Weight of semantic similarity score. LLM gets the remaining %."
        )
        llm_weight = 100 - faiss_weight
        st.markdown(f"<small style='color:#94a3b8'>LLM weight: **{llm_weight}%**</small>", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    st.markdown("""
    <div style='color:#475569; font-size:0.78rem; text-align:center'>
        Built with ❤️ using Groq + LangChain + FAISS<br>
        <a href='https://console.groq.com' style='color:#a78bfa'>Groq API</a> &nbsp;·&nbsp;
        <a href='https://sbert.net' style='color:#a78bfa'>Sentence Transformers</a>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main area — resume upload
# ─────────────────────────────────────────────────────────────────────────────

col_upload, col_help = st.columns([3, 1])

with col_upload:
    uploaded_resumes = st.file_uploader(
        "Upload Resumes (PDFs)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload up to 20 resume PDFs. Each file should be one candidate's resume.",
        label_visibility="visible",
    )

with col_help:
    if uploaded_resumes:
        st.markdown(f"""
        <div class="metric-card" style="margin-top:1.8rem">
            <div class="value">{len(uploaded_resumes)}</div>
            <div class="label">Resumes ready</div>
        </div>
        """, unsafe_allow_html=True)

# ── Warnings ────────────────────────────────────────────────────────────────

groq_key = os.getenv("GROQ_API_KEY", "").strip()
if not groq_key:
    st.markdown("""
    <div class="warn-box">
        ⚠️ <strong>GROQ_API_KEY not set</strong> — Add it to your <code>.env</code> file.
        Get a free key at <a href="https://console.groq.com" style="color:#fcd34d">console.groq.com</a>
    </div>
    """, unsafe_allow_html=True)

# ── Screen button ────────────────────────────────────────────────────────────

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

col_btn, col_info = st.columns([2, 5])
with col_btn:
    screen_clicked = st.button(
        "🚀 Screen Resumes",
        disabled=(not jd_text.strip() or not uploaded_resumes or not groq_key),
        use_container_width=True,
    )

with col_info:
    if not jd_text.strip():
        st.markdown('<div class="info-box">📝 Paste or upload a job description in the sidebar to get started.</div>', unsafe_allow_html=True)
    elif not uploaded_resumes:
        st.markdown('<div class="info-box">📂 Upload at least one resume PDF above.</div>', unsafe_allow_html=True)
    elif not groq_key:
        st.markdown('<div class="warn-box">🔑 Add GROQ_API_KEY to .env to enable AI scoring.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="info-box">✅ Ready! {len(uploaded_resumes)} resume(s) loaded. Click <strong>Screen Resumes</strong> to start.</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Run pipeline
# ─────────────────────────────────────────────────────────────────────────────

if screen_clicked:
    # Pre-load model (cached after first run)
    load_embedding_model()

    # ── Step 1: Parse PDFs ───────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    progress_bar = st.progress(0, text="📄 Parsing resume PDFs...")

    resumes  = []
    skipped_scanned   = []
    skipped_corrupted = []

    for i, uploaded_file in enumerate(uploaded_resumes):
        progress_bar.progress(
            int((i + 1) / len(uploaded_resumes) * 30),
            text=f"Parsing {uploaded_file.name}..."
        )
        name, text, err = parse_uploaded_pdf(uploaded_file)
        if err == "scanned":
            skipped_scanned.append(name)
        elif err in ("corrupted", "empty"):
            skipped_corrupted.append(name)
        else:
            resumes.append({"name": name, "text": text})

    if skipped_scanned:
        st.warning(
            f"Skipped {len(skipped_scanned)} scanned PDF(s) — "
            "these are image-only and cannot be read by AI. "
            "Please upload text-based PDFs.\n\n"
            f"Files: {', '.join(skipped_scanned)}"
        )
    if skipped_corrupted:
        st.warning(
            f"Skipped {len(skipped_corrupted)} corrupted or unreadable PDF(s).\n\n"
            f"Files: {', '.join(skipped_corrupted)}"
        )

    if not resumes:
        st.error("No readable resumes found. Please upload text-based (not scanned) PDFs.")
        st.stop()

    # ── Step 2: Run pipeline ─────────────────────────────────────────────────
    progress_bar.progress(35, text="🔍 Building FAISS vector index...")
    time.sleep(0.3)

    progress_bar.progress(50, text=f"🤖 Scoring top candidates with Groq LLM (this takes ~{len(resumes)*2}s)...")

    try:
        weights  = (faiss_weight / 100, llm_weight / 100)
        pipeline = run_pipeline(
            jd_text     = jd_text,
            resumes     = resumes,
            top_k_faiss = len(resumes),
            top_k_llm   = top_k_llm,
            weights     = weights,
            verbose     = False,
        )
        results: list[CandidateResult] = pipeline["ranked"]

        if pipeline["rate_limit_hit"]:
            st.warning(
                "**Groq rate limit was hit mid-batch.**\n\n"
                "Some candidates were ranked using FAISS semantic similarity only "
                "(no LLM deep analysis). They are marked below.\n\n"
                f"Tip: {pipeline['rate_limit_msg'].split('Tip:')[-1].strip()}"
            )

    except Exception as e:
        st.error(
            f"**Pipeline error:** {e}\n\n"
            "Please check that your GROQ_API_KEY is valid and try again."
        )
        st.stop()

    progress_bar.progress(100, text="✅ Done!")
    time.sleep(0.5)
    progress_bar.empty()

    # Store in session state for persistence
    st.session_state["results"] = results
    st.session_state["jd_text"] = jd_text


# ─────────────────────────────────────────────────────────────────────────────
# Display results (from session state, survives reruns)
# ─────────────────────────────────────────────────────────────────────────────

if "results" in st.session_state:
    results: list[CandidateResult] = st.session_state["results"]

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Summary metrics row ──────────────────────────────────────────────────
    top = results[0]
    llm_scored = sum(1 for r in results if r.llm_used)

    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{len(results)}</div>
            <div class="label">Candidates Ranked</div>
        </div>""", unsafe_allow_html=True)
    with m2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{llm_scored}</div>
            <div class="label">AI Scored</div>
        </div>""", unsafe_allow_html=True)
    with m3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{top.final_score:.0f}</div>
            <div class="label">Top Score</div>
        </div>""", unsafe_allow_html=True)
    with m4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="value">{top.name.replace('.pdf','').replace('_',' ')[:12]}</div>
            <div class="label">Best Candidate</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Tab view: Cards | Table ──────────────────────────────────────────────
    tab_cards, tab_table = st.tabs(["🏆 Ranked Cards", "📊 Data Table"])

    with tab_cards:
        for r in results:
            score_cls  = score_color_class(r.final_score)
            badge_cls  = rank_badge_class(r.rank)
            card_cls   = "rank-card top1" if r.rank == 1 else "rank-card"
            name_clean = r.name.replace(".pdf", "").replace("_", " ")
            llm_str    = f"{r.llm_score}/100" if r.llm_score is not None else "N/A"

            strengths_html = "".join(
                f'<span class="tag tag-strength">✓ {s}</span>'
                for s in r.strengths
            )
            gaps_html = "".join(
                f'<span class="tag tag-gap">✗ {g}</span>'
                for g in r.gaps
            )

            rank_num = {1: "🥇", 2: "🥈", 3: "🥉"}.get(r.rank, f"#{r.rank}")

            extra_html = ""
            if strengths_html:
                extra_html += f'<div style="margin-top:0.8rem">{strengths_html}</div>'
            if gaps_html:
                extra_html += f'<div style="margin-top:0.3rem">{gaps_html}</div>'

            card_html = (
                f'<div class="{card_cls}">'
                f'<div class="rank-badge {badge_cls}">{rank_num if r.rank > 3 else r.rank}</div>'
                f'<div class="candidate-name">{rank_num} &nbsp; {name_clean}</div>'
                f'<div style="margin: 0.6rem 0; display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">'
                f'<span class="score-pill {score_cls}">Final: {r.final_score:.1f}/100</span>'
                f'<span style="color:#64748b; font-size:0.85rem;">FAISS: {r.faiss_score*100:.1f}%&nbsp;&nbsp;|&nbsp;&nbsp;LLM: {llm_str}</span>'
                + ('<span style="background:rgba(251,191,36,0.1);color:#fbbf24;padding:0.1rem 0.5rem;border-radius:4px;font-size:0.75rem;border:1px solid rgba(251,191,36,0.3)">FAISS only</span>' if not r.llm_used else '')
                + '</div>'
                + f'<div class="verdict-text">"{r.verdict}"</div>'
                + extra_html
                + '</div>'
            )
            st.markdown(card_html, unsafe_allow_html=True)

            # Score bar
            st.progress(int(r.final_score), text="")
            st.markdown('<div style="margin-bottom:0.5rem"></div>', unsafe_allow_html=True)

    with tab_table:
        df = results_to_dataframe(results)
        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True,
        )

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Expandable detailed view per candidate ───────────────────────────────
    st.markdown("### 🔍 Detailed Candidate Analysis")
    st.markdown('<div style="height:0.5rem"></div>', unsafe_allow_html=True)

    for r in results:
        name_clean = r.name.replace(".pdf", "").replace("_", " ")
        rank_emoji = {1: "🥇", 2: "🥈", 3: "🥉"}.get(r.rank, f"#{r.rank}")
        with st.expander(f"{rank_emoji}  {name_clean}   —   {r.final_score:.1f}/100", expanded=(r.rank == 1)):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Final Score", f"{r.final_score:.1f}/100")
            with col_b:
                st.metric("Semantic (FAISS)", f"{r.faiss_score*100:.1f}%")
            with col_c:
                st.metric("AI Score (LLM)", f"{r.llm_score}/100" if r.llm_score is not None else "N/A")

            st.markdown(f"**Verdict:** *{r.verdict}*")

            if r.strengths:
                st.markdown("**Strengths:**")
                for s in r.strengths:
                    st.markdown(f"- ✅ {s}")

            if r.gaps:
                st.markdown("**Gaps:**")
                for g in r.gaps:
                    st.markdown(f"- ⚠️ {g}")

            if not r.llm_used:
                st.info("ℹ️ This candidate was ranked by semantic similarity only (below the LLM scoring threshold). Increase 'Max LLM candidates' in Advanced Settings to include them.")

    # ── CSV download ─────────────────────────────────────────────────────────
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    csv_data = results_to_csv(results)
    col_dl, col_clear = st.columns([2, 5])
    with col_dl:
        st.download_button(
            label="📥 Download Results as CSV",
            data=csv_data,
            file_name="screening_results.csv",
            mime="text/csv",
            use_container_width=True,
        )
    with col_clear:
        if st.button("🔄 Clear & Start Over", use_container_width=False):
            del st.session_state["results"]
            st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
# Empty state
# ─────────────────────────────────────────────────────────────────────────────

elif "results" not in st.session_state:
    st.markdown("""
    <div style="text-align:center; padding: 4rem 2rem; color:#475569;">
        <div style="font-size:4rem; margin-bottom:1rem">📋</div>
        <div style="font-size:1.2rem; font-weight:600; color:#64748b; margin-bottom:0.5rem">
            No results yet
        </div>
        <div style="font-size:0.95rem; color:#475569">
            Add a job description in the sidebar, upload resume PDFs above,<br>
            then click <strong style="color:#a78bfa">Screen Resumes</strong> to begin.
        </div>
    </div>
    """, unsafe_allow_html=True)
