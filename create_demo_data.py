"""
create_demo_data.py -- Phase 7: Local Demo Dataset Generator

Creates a self-contained demo dataset in data/samples/ so anyone can
test the AI Resume Screener without sourcing their own PDFs.

What this generates:
  data/samples/demo_jd.txt              -- sample job description
  data/samples/demo_Priya_Sharma.pdf    -- strong Python/FastAPI match
  data/samples/demo_Aryan_Gupta.pdf    -- moderate Python/Django match
  data/samples/demo_Kavya_Nair.pdf     -- DevOps engineer (partial match)
  data/samples/demo_Sneha_Patel.pdf    -- graphic designer (no match)

Usage:
  python create_demo_data.py
  # Then open http://localhost:8501, paste demo_jd.txt, upload the PDFs
"""

import sys
from pathlib import Path

# ── Ensure PyMuPDF is available ───────────────────────────────────────────────
try:
    import fitz
except ImportError:
    print("ERROR: PyMuPDF is not installed. Run: uv pip install pymupdf")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

SAMPLES_DIR = Path(__file__).parent
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────────────────────────────────────
# Demo Job Description
# ─────────────────────────────────────────────────────────────────────────────

DEMO_JD = """\
Senior Python Backend Developer | FinZen Technologies (Fintech Startup)
Location: Bengaluru, India (Hybrid)
Experience: 4-6 years

About the Role:
We are building the next-generation payment infrastructure for India's
growing fintech ecosystem. We need a seasoned Python backend developer
to architect and own our core payment APIs.

Key Responsibilities:
- Design and build high-throughput REST APIs using FastAPI (async Python)
- Manage PostgreSQL databases with query optimisation and indexing
- Implement Redis caching layers for sub-10ms response times
- Containerise services with Docker and deploy on AWS (EC2, ECS, S3)
- Build CI/CD pipelines using GitHub Actions
- Collaborate with frontend and mobile teams via documented APIs

Required Skills:
- Python 3.10+ (must have 4+ years)
- FastAPI or Django REST Framework
- PostgreSQL (schema design, migrations)
- Redis, Docker, AWS
- REST API design principles
- Git, code review, async programming

Nice to Have:
- Kafka / message queues
- Kubernetes
- Prior fintech or high-scale startup experience

Compensation: 18-28 LPA depending on experience
"""


# ─────────────────────────────────────────────────────────────────────────────
# Sample resume content
# ─────────────────────────────────────────────────────────────────────────────

RESUMES = [
    {
        "filename": "demo_Priya_Sharma.pdf",
        "lines": [
            ("PRIYA SHARMA", 18, True),
            ("Senior Python Backend Developer", 13, False),
            ("priya.sharma@email.com | +91-9876543210 | LinkedIn: /in/priyasharma | GitHub: github.com/priya-dev", 9, False),
            ("", 6, False),
            ("SUMMARY", 13, True),
            ("Passionate Python backend engineer with 5 years of experience building scalable, high-availability "
             "payment APIs in the fintech space. Expert in FastAPI, async Python, PostgreSQL, Redis, Docker, and AWS. "
             "Led a team of 5 engineers at Razorpay to deliver payment APIs processing 10,000+ TPS.", 9, False),
            ("", 6, False),
            ("SKILLS", 13, True),
            ("Languages: Python 3.11, SQL, Bash", 9, False),
            ("Frameworks: FastAPI, Django REST Framework, Celery", 9, False),
            ("Databases: PostgreSQL, Redis, MongoDB", 9, False),
            ("Cloud & DevOps: AWS (EC2, ECS, S3, Lambda), Docker, Kubernetes, GitHub Actions (CI/CD)", 9, False),
            ("Messaging: Kafka, RabbitMQ", 9, False),
            ("", 6, False),
            ("EXPERIENCE", 13, True),
            ("Senior Backend Engineer | Razorpay | Bengaluru | Jan 2022 – Present", 11, True),
            ("- Led design and implementation of payment processing microservices using async FastAPI", 9, False),
            ("- Built payment APIs handling 10,000+ TPS with 99.99% uptime SLA", 9, False),
            ("- Reduced API latency by 40% via Redis caching and PostgreSQL query optimisation", 9, False),
            ("- Containerised all services with Docker; deployed on AWS ECS with Kubernetes orchestration", 9, False),
            ("- Implemented CI/CD pipelines via GitHub Actions; cut deployment time from 45 min to 8 min", 9, False),
            ("- Mentored team of 5 junior engineers in async Python and REST API best practices", 9, False),
            ("", 6, False),
            ("Backend Engineer | PhonePe | Bengaluru | Jul 2019 – Dec 2021", 11, True),
            ("- Developed Django REST APIs for merchant onboarding and KYC workflows", 9, False),
            ("- Designed PostgreSQL schemas and wrote complex analytical queries for transaction reporting", 9, False),
            ("- Integrated Kafka for event-driven order processing pipeline", 9, False),
            ("", 6, False),
            ("EDUCATION", 13, True),
            ("B.Tech Computer Science | BITS Pilani | 2019 | CGPA: 8.7/10", 9, False),
            ("", 6, False),
            ("ACHIEVEMENTS", 13, True),
            ("- Razorpay Hackathon Winner 2023 – built real-time fraud detection using ML + FastAPI", 9, False),
            ("- Speaker at PyCon India 2022 – 'Scaling FastAPI to 10,000 RPS'", 9, False),
        ],
    },
    {
        "filename": "demo_Aryan_Gupta.pdf",
        "lines": [
            ("ARYAN GUPTA", 18, True),
            ("Python Backend Developer", 13, False),
            ("aryan.gupta@email.com | +91-9823456789 | Pune, India", 9, False),
            ("", 6, False),
            ("SUMMARY", 13, True),
            ("Backend developer with 3 years of experience in Python and Django. "
             "Built REST APIs for e-commerce platforms. Learning FastAPI and looking to grow in fintech.", 9, False),
            ("", 6, False),
            ("SKILLS", 13, True),
            ("Languages: Python, SQL", 9, False),
            ("Frameworks: Django, Django REST Framework", 9, False),
            ("Databases: MySQL, PostgreSQL (basic)", 9, False),
            ("DevOps: Docker (basic), some AWS S3 experience", 9, False),
            ("Tools: Git, Postman, Jira", 9, False),
            ("", 6, False),
            ("EXPERIENCE", 13, True),
            ("Backend Developer | Flipkart | Pune | Aug 2021 – Present", 11, True),
            ("- Built and maintained Django REST APIs for product catalogue and search", 9, False),
            ("- Wrote MySQL queries for order analytics dashboards", 9, False),
            ("- Used basic Docker for local development; no production Docker experience", 9, False),
            ("- Some exposure to AWS S3 for static file storage", 9, False),
            ("", 6, False),
            ("EDUCATION", 13, True),
            ("B.Tech Information Technology | NIT Trichy | 2021 | CGPA: 7.8/10", 9, False),
            ("", 6, False),
            ("CURRENTLY LEARNING", 13, True),
            ("- FastAPI and async Python (in progress)", 9, False),
            ("- Redis fundamentals", 9, False),
        ],
    },
    {
        "filename": "demo_Kavya_Nair.pdf",
        "lines": [
            ("KAVYA NAIR", 18, True),
            ("Senior DevOps & Cloud Engineer", 13, False),
            ("kavya.nair@email.com | +91-9911234567 | Kochi, India", 9, False),
            ("", 6, False),
            ("SUMMARY", 13, True),
            ("5 years of DevOps and cloud engineering experience. Expert in AWS, Kubernetes, Terraform, "
             "and CI/CD pipelines. Managed infrastructure for 200+ microservices at Infosys. "
             "Strong Python scripting for automation but not a software developer by role.", 9, False),
            ("", 6, False),
            ("SKILLS", 13, True),
            ("Cloud: AWS (EC2, ECS, RDS, S3, CloudWatch), GCP", 9, False),
            ("Container Orchestration: Kubernetes, Docker, Helm", 9, False),
            ("IaC: Terraform, Ansible, CloudFormation", 9, False),
            ("CI/CD: Jenkins, GitHub Actions, ArgoCD", 9, False),
            ("Scripting: Python (automation scripts), Bash, PowerShell", 9, False),
            ("Monitoring: Prometheus, Grafana, ELK Stack", 9, False),
            ("", 6, False),
            ("EXPERIENCE", 13, True),
            ("Senior DevOps Engineer | Infosys | Bengaluru | Mar 2019 – Present", 11, True),
            ("- Managed AWS infrastructure for 200+ microservices across 12 production environments", 9, False),
            ("- Automated provisioning with Terraform; reduced setup time by 70%", 9, False),
            ("- Built and maintained CI/CD pipelines with Jenkins and GitHub Actions", 9, False),
            ("- Wrote Python automation scripts for log parsing and cost optimisation", 9, False),
            ("- Set up Prometheus + Grafana monitoring dashboards for SLO tracking", 9, False),
            ("", 6, False),
            ("EDUCATION", 13, True),
            ("B.E. Electronics & Communication | NITK Surathkal | 2019 | CGPA: 8.2/10", 9, False),
        ],
    },
    {
        "filename": "demo_Sneha_Patel.pdf",
        "lines": [
            ("SNEHA PATEL", 18, True),
            ("Senior Graphic Designer & Brand Identity Specialist", 13, False),
            ("sneha.patel@email.com | +91-9022345678 | Mumbai, India", 9, False),
            ("", 6, False),
            ("SUMMARY", 13, True),
            ("Award-winning graphic designer with 6 years of experience creating brand identities, "
             "packaging designs, and digital campaigns for leading FMCG and retail brands. "
             "Expert in Adobe Creative Suite and Figma.", 9, False),
            ("", 6, False),
            ("SKILLS", 13, True),
            ("Design Tools: Adobe Photoshop, Illustrator, InDesign, After Effects, Figma, Canva Pro", 9, False),
            ("Specialisations: Brand Identity, Packaging, Typography, UI/UX Wireframing", 9, False),
            ("Soft Skills: Client communication, art direction, campaign management", 9, False),
            ("", 6, False),
            ("EXPERIENCE", 13, True),
            ("Senior Graphic Designer | Ogilvy India | Mumbai | Jan 2018 – Present", 11, True),
            ("- Created brand identities and visual guidelines for 30+ FMCG companies", 9, False),
            ("- Led packaging redesign for Hindustan Unilever resulting in 15% sales uplift", 9, False),
            ("- Won 3 national design awards (Kyoorius, ABBY) for campaign work", 9, False),
            ("- Managed team of 4 junior designers and interns", 9, False),
            ("", 6, False),
            ("EDUCATION", 13, True),
            ("Bachelor of Fine Arts (BFA) | Sir J.J. School of Art, Mumbai | 2018 | First Class", 9, False),
            ("", 6, False),
            ("AWARDS", 13, True),
            ("- Kyoorius Design Award – Bronze 2022, Silver 2021", 9, False),
            ("- ABBY Award – Gold 2023 (Brand Identity Category)", 9, False),
        ],
    },
]


# ─────────────────────────────────────────────────────────────────────────────
# PDF generation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _add_text_block(page, y: float, text: str, font_size: int, bold: bool) -> float:
    """
    Insert a text block at position y on the page.
    Returns the new y position after inserting the text.
    """
    font_name = "helv" if not bold else "hebo"
    rect = fitz.Rect(50, y, 545, y + font_size + 6)
    page.insert_textbox(
        rect,
        text,
        fontname=font_name,
        fontsize=font_size,
        color=(0, 0, 0),
    )
    return y + font_size + 4


def create_resume_pdf(output_path: Path, lines: list[tuple]) -> None:
    """
    Create a clean, text-extractable PDF resume from a list of content lines.

    Args:
        output_path: Where to write the PDF.
        lines: List of (text, font_size, bold) tuples.
    """
    doc  = fitz.open()
    page = doc.new_page(width=595, height=842)   # A4
    y    = 50.0
    PAGE_BOTTOM = 800.0

    for (text, font_size, bold) in lines:
        if y > PAGE_BOTTOM:
            page = doc.new_page(width=595, height=842)
            y = 50.0

        if text == "":
            y += 4
            continue

        # Draw a light underline under section headers
        if bold and font_size == 13:
            page.draw_line(
                fitz.Point(50, y + font_size + 1),
                fitz.Point(545, y + font_size + 1),
                color=(0.7, 0.7, 0.7),
                width=0.5,
            )

        y = _add_text_block(page, y, text, font_size, bold)
        y += 2   # small gap between lines

    doc.save(str(output_path))
    doc.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("Creating demo dataset in:", SAMPLES_DIR)
    print()

    # 1. Write JD text file
    jd_path = SAMPLES_DIR / "demo_jd.txt"
    jd_path.write_text(DEMO_JD, encoding="utf-8")
    print(f"[OK] Created: {jd_path.name}")

    # 2. Generate resume PDFs
    for resume in RESUMES:
        pdf_path = SAMPLES_DIR / resume["filename"]
        create_resume_pdf(pdf_path, resume["lines"])
        print(f"[OK] Created: {pdf_path.name}")

    print()
    print("Demo dataset ready!")
    print()
    print("How to use:")
    print("  1. Run the app:   .venv\\Scripts\\streamlit run app.py")
    print("  2. Open:          http://localhost:8501")
    print("  3. In sidebar:    Paste the contents of demo_jd.txt")
    print("  4. Upload:        demo_Priya_Sharma.pdf, demo_Aryan_Gupta.pdf,")
    print("                    demo_Kavya_Nair.pdf, demo_Sneha_Patel.pdf")
    print("  5. Click:         Screen Resumes")
    print()
    print("Expected ranking: Priya > Aryan > Kavya > Sneha")


if __name__ == "__main__":
    main()
