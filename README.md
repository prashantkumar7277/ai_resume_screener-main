# 🧠 AI Resume Screener

A **production-grade, local-first AI application** that screens resumes against a job description using a two-stage hybrid ranking system: semantic search (FAISS) + deep LLM analysis (Groq Llama 3.3 70B). Built for the Indian job market.

---

## ✨ Features

| Feature | Details |
|---|---|
| **2-Stage Hybrid Ranking** | Stage 1: FAISS cosine similarity for fast semantic matching. Stage 2: Groq LLM for deep reasoning-based scoring |
| **Weighted Final Score** | Combine scores with configurable weights (default: 30% FAISS + 70% LLM) |
| **Batch Processing** | Upload up to 20 PDFs at once; only top-N go to LLM (saves API quota) |
| **Premium Dark UI** | Glassmorphism dark theme with animated rank cards, color-coded scores, strength/gap tags |
| **Detailed Analysis** | Per-candidate expandable section: score breakdown, strengths, gaps, one-line verdict |
| **CSV Export** | One-click download of the full ranked results table |
| **Scanned PDF Detection** | Graceful warning when a PDF is image-only or corrupted — no crash |
| **Rate Limit Safety** | Groq 30 RPM limit handled with exponential retry + graceful UI degradation |
| **Session Persistence** | Results survive Streamlit reruns (sidebar changes, slider moves) |
| **Model Caching** | `@st.cache_resource` keeps the 90MB embedding model in memory across sessions |

---

## 🏗️ Architecture

```
PDF Upload
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 1: Semantic Search (FAISS)                               │
│                                                                 │
│  PDF → PyMuPDF text extraction → SentenceTransformers embed    │
│  (all-MiniLM-L6-v2, 384-dim) → faiss.IndexFlatIP (cosine)     │
│  → top-K results ranked by cosine similarity                    │
└─────────────────────────────────────────────────────────────────┘
    │   top-N candidates (default: 10)
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Stage 2: Deep LLM Analysis (Groq)                              │
│                                                                 │
│  Prompt → Groq Llama 3.3 70B → JSON {score, strengths,        │
│  gaps, verdict} → validated + clamped to [0, 100]              │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────────────────┐
│  Ranking Engine                                                 │
│                                                                 │
│  final_score = 0.30 × (faiss_score × 100)                      │
│              + 0.70 × llm_score                                 │
│  → Sorted descending → Ranked leaderboard                       │
└─────────────────────────────────────────────────────────────────┘
    │
    ▼
Streamlit UI — ranked cards, table, CSV export
```

---

## 🛠️ Tech Stack

| Component | Library |
|---|---|
| PDF Parsing | `PyMuPDF (fitz)` |
| Embedding Model | `sentence-transformers` — `all-MiniLM-L6-v2` (90MB, 384-dim) |
| Vector Store | `faiss-cpu` — `IndexFlatIP` (inner product = cosine on L2-normalised vectors) |
| LLM Scoring | `groq` — `llama-3.3-70b-versatile` |
| Ranking Engine | Custom Python — weighted combination |
| Frontend | `Streamlit` — dark theme + custom CSS |
| Env Management | `python-dotenv` |
| Package Manager | `uv` |

---

## ⚙️ Local Setup

### Prerequisites

- Python 3.10+
- [`uv`](https://github.com/astral-sh/uv) package manager
- A free [Groq API key](https://console.groq.com) (no credit card needed)

### 1. Clone / open the project

```bash
cd resume_screener
```

### 2. Create virtual environment and install dependencies

```bash
uv venv .venv
.venv\Scripts\activate        # Windows
# or: source .venv/bin/activate   # Mac/Linux

uv pip install -r requirements.txt
```

### 3. Set up your API key

Open `.env` and paste your Groq key:

```env
GROQ_API_KEY=gsk_your_key_here
```

Get a free key at [console.groq.com](https://console.groq.com) → Create API Key.

### 4. (Optional) Generate demo data for testing

```bash
python data/samples/create_demo_data.py
```

This creates 4 sample resume PDFs and 1 job description in `data/samples/`.

### 5. Run the app

```bash
.venv\Scripts\streamlit run app.py
# or on Mac/Linux:
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 🚀 How to Use

1. **Paste the Job Description** in the sidebar text area (or upload as PDF)
2. **Upload Resume PDFs** — drag and drop up to 20 files
3. **(Optional)** Adjust settings in **Advanced Settings**:
   - **Max LLM candidates** — how many top FAISS results to send to Groq (saves API quota)
   - **FAISS weight** — how much semantic similarity matters vs LLM reasoning
4. Click **🚀 Screen Resumes**
5. View results:
   - **🏆 Ranked Cards** — visual ranked leaderboard with scores, strengths, gaps
   - **📊 Data Table** — sortable table view
   - Expand any candidate card for full AI analysis
6. Click **📥 Download Results as CSV** to export

---

## 📁 Project Structure

```
resume_screener/
├── app.py                          # Streamlit frontend (entry point)
├── requirements.txt                # All dependencies
├── .env                            # API keys (not committed to git)
├── .gitignore
│
├── utils/
│   ├── pdf_parser.py               # Phase 1: PDF text extraction + cleaning
│   ├── embedder.py                 # Phase 2: SentenceTransformers embedding + cache
│   ├── vector_store.py             # Phase 3: FAISS index + search
│   ├── llm_scorer.py               # Phase 4: Groq LLM scoring + retry logic
│   └── ranker.py                   # Phase 5: Score combination + ranking engine
│
├── data/
│   └── samples/
│       ├── create_demo_data.py     # Generates demo JD + 4 resume PDFs
│       └── generate_test_pdfs.py   # Test PDFs for unit tests
│
├── test_parser.py                  # Phase 1 tests (15 tests)
├── test_embedder.py                # Phase 2 tests (7 tests)
├── test_vector_store.py            # Phase 3 tests (8 tests)
├── test_llm_scorer.py              # Phase 4 tests (9 tests)
└── test_ranker.py                  # Phase 5 tests (7 tests)
```

---

## 🔒 Constraints & Limits

| Constraint | Detail | Mitigation |
|---|---|---|
| **Groq Free Tier: 30 RPM** | 30 API calls/minute | Only top-N FAISS results go to LLM (default 10). `time.sleep(2)` between calls. Exponential retry on 429. |
| **Scanned PDFs** | Image-only PDFs extract no text | Detected and shown as a clean warning. No crash. Future: add Tesseract OCR. |
| **PyMuPDF License** | AGPL — fine for personal/open-source | If commercialising, switch to `pdfplumber` (MIT) |
| **Model first load** | `all-MiniLM-L6-v2` downloads ~90MB | `@st.cache_resource` loads it once and keeps it in RAM |
| **FAISS not persistent** | Index resets each session | `st.session_state` stores results between reruns |

---

## 🧪 Running Tests

Run all 46 unit tests across the 5 pipeline phases:

```bash
# Individual phases
.venv\Scripts\python test_parser.py
.venv\Scripts\python test_embedder.py
.venv\Scripts\python test_vector_store.py
.venv\Scripts\python test_llm_scorer.py       # requires GROQ_API_KEY
.venv\Scripts\python test_ranker.py            # requires GROQ_API_KEY
```

> **Note:** `test_llm_scorer.py` and `test_ranker.py` make live Groq API calls.
> The 5 offline tests in `test_llm_scorer.py` work without a key.

---

## 🎯 Score Interpretation

| Score | Colour | Meaning |
|---|---|---|
| 70 – 100 | 🟢 Green | Strong fit — shortlist immediately |
| 40 – 69 | 🟡 Yellow | Moderate fit — worth a call |
| 0 – 39 | 🔴 Red | Poor fit — likely to reject |

---

## 📄 License

This project is for personal/portfolio use. PyMuPDF is AGPL licensed.
For commercial use, replace `fitz` with `pdfplumber` (MIT).

---

*Built with ❤️ using Groq + FAISS + SentenceTransformers + Streamlit*
