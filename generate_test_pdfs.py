"""
generate_test_pdfs.py — Phase 1 Testing Helper

Generates 3 sample resume PDFs using reportlab to test our pdf_parser.py:
  1. resume_simple.pdf    — basic single-column flowing text
  2. resume_columns.pdf   — two-column layout
  3. resume_tables.pdf    — content organized inside a table

Run this script once to populate data/samples/ before running test_parser.py.
"""

import os
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Frame,
    PageTemplate, BaseDocTemplate, FrameBreak
)
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT, TA_CENTER

OUTPUT_DIR = Path(__file__).parent
styles = getSampleStyleSheet()


# ─────────────────────────────────────────────
# 1. SIMPLE RESUME (single column, plain text)
# ─────────────────────────────────────────────

def make_simple_resume():
    path = OUTPUT_DIR / "resume_simple.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    content = []

    title_style = ParagraphStyle("Title", parent=styles["Normal"],
                                 fontSize=18, spaceAfter=6, alignment=TA_CENTER)
    heading_style = ParagraphStyle("Heading", parent=styles["Normal"],
                                   fontSize=13, spaceBefore=10, spaceAfter=4,
                                   textColor=colors.HexColor("#1A56DB"))
    body_style = styles["Normal"]

    content.append(Paragraph("Priya Sharma", title_style))
    content.append(Paragraph("priya.sharma@email.com | +91-9876543210 | Bengaluru, India", body_style))
    content.append(Spacer(1, 0.4*cm))

    content.append(Paragraph("SUMMARY", heading_style))
    content.append(Paragraph(
        "Results-driven Python Backend Developer with 3 years of experience building "
        "scalable REST APIs using FastAPI and Django. Strong foundation in SQL, PostgreSQL, "
        "and cloud deployments on AWS. Passionate about clean code and test-driven development.",
        body_style))

    content.append(Paragraph("SKILLS", heading_style))
    content.append(Paragraph(
        "Python · FastAPI · Django · PostgreSQL · Redis · Docker · AWS · Git · REST APIs · "
        "Pytest · CI/CD · Linux · SQLAlchemy",
        body_style))

    content.append(Paragraph("EXPERIENCE", heading_style))
    content.append(Paragraph("<b>Software Engineer — Razorpay, Bengaluru (2022–Present)</b>", body_style))
    content.append(Paragraph(
        "• Built high-throughput payment processing APIs handling 10,000+ TPS using FastAPI and async Python.<br/>"
        "• Reduced API latency by 40% by introducing Redis caching for frequently accessed merchant data.<br/>"
        "• Wrote comprehensive unit and integration tests achieving 92% code coverage.",
        body_style))
    content.append(Spacer(1, 0.2*cm))
    content.append(Paragraph("<b>Junior Developer — TechStartup Pvt. Ltd., Hyderabad (2021–2022)</b>", body_style))
    content.append(Paragraph(
        "• Developed internal HR management portal using Django and PostgreSQL.<br/>"
        "• Integrated third-party APIs (Razorpay, Twilio) for payments and SMS notifications.",
        body_style))

    content.append(Paragraph("EDUCATION", heading_style))
    content.append(Paragraph(
        "<b>B.Tech in Computer Science</b> — BITS Pilani, 2021 | CGPA: 8.6/10",
        body_style))

    content.append(Paragraph("PROJECTS", heading_style))
    content.append(Paragraph(
        "<b>AI Resume Screener (2024):</b> Built an LLM-powered resume ranking tool using LangChain, "
        "FAISS, and Groq API. Deployed on Hugging Face Spaces.",
        body_style))

    doc.build(content)
    print(f"[OK] Created: {path}")


# ─────────────────────────────────────────────
# 2. COLUMNS RESUME (two-column layout)
# ─────────────────────────────────────────────

def make_columns_resume():
    path = OUTPUT_DIR / "resume_columns.pdf"
    width, height = A4

    # Two-frame layout per page
    left_frame = Frame(1.5*cm, 2*cm, 7*cm, height - 4*cm, id="left")
    right_frame = Frame(9.5*cm, 2*cm, 9.5*cm, height - 4*cm, id="right")

    class TwoColDoc(BaseDocTemplate):
        def __init__(self, filename, **kwargs):
            super().__init__(filename, **kwargs)
            template = PageTemplate(id="TwoCol", frames=[left_frame, right_frame])
            self.addPageTemplates([template])

    doc = TwoColDoc(str(path), pagesize=A4)

    body = styles["Normal"]
    h2 = ParagraphStyle("H2", parent=styles["Normal"], fontSize=11, spaceBefore=8,
                        spaceAfter=3, textColor=colors.HexColor("#1A56DB"))

    content = []

    # LEFT COLUMN
    content.append(Paragraph("<b>Arjun Mehta</b>", ParagraphStyle("name", fontSize=16)))
    content.append(Spacer(1, 0.2*cm))
    content.append(Paragraph("Machine Learning Engineer", body))
    content.append(Paragraph("arjun.mehta@gmail.com", body))
    content.append(Paragraph("+91-9123456789", body))
    content.append(Paragraph("Mumbai, India", body))
    content.append(Spacer(1, 0.4*cm))

    content.append(Paragraph("SKILLS", h2))
    skills = [
        "Python", "TensorFlow", "PyTorch", "Scikit-learn",
        "Pandas", "NumPy", "SQL", "Docker", "MLflow",
        "Hugging Face", "LangChain", "AWS SageMaker"
    ]
    for s in skills:
        content.append(Paragraph(f"• {s}", body))

    content.append(Spacer(1, 0.4*cm))
    content.append(Paragraph("EDUCATION", h2))
    content.append(Paragraph("<b>M.Tech, AI & ML</b>", body))
    content.append(Paragraph("IIT Bombay, 2023", body))
    content.append(Spacer(1, 0.2*cm))
    content.append(Paragraph("<b>B.E., Computer Engineering</b>", body))
    content.append(Paragraph("Mumbai University, 2021", body))

    # Switch to right column
    content.append(FrameBreak())

    # RIGHT COLUMN
    content.append(Paragraph("EXPERIENCE", h2))
    content.append(Paragraph("<b>ML Engineer — Flipkart, Mumbai (2023–Present)</b>", body))
    content.append(Paragraph(
        "• Developed recommendation engine using collaborative filtering and deep learning, "
        "increasing CTR by 22%.<br/>"
        "• Fine-tuned BERT-based models for product review sentiment analysis.<br/>"
        "• Built MLflow pipelines for experiment tracking and model versioning.",
        body))
    content.append(Spacer(1, 0.3*cm))
    content.append(Paragraph("<b>Data Science Intern — Analytics Vidhya (2022)</b>", body))
    content.append(Paragraph(
        "• Built customer churn prediction model using XGBoost achieving 87% accuracy.<br/>"
        "• Created interactive dashboards using Plotly and Streamlit.",
        body))

    content.append(Paragraph("PROJECTS", h2))
    content.append(Paragraph(
        "<b>LLM Document Q&A System:</b> RAG pipeline using LangChain + FAISS + OpenAI GPT-4. "
        "Handles 50-page PDFs with 95% answer accuracy.",
        body))
    content.append(Spacer(1, 0.2*cm))
    content.append(Paragraph(
        "<b>Real-time Object Detection:</b> YOLOv8 deployment on edge devices using ONNX Runtime. "
        "Achieves 30 FPS on Raspberry Pi 4.",
        body))

    content.append(Paragraph("CERTIFICATIONS", h2))
    content.append(Paragraph("• AWS Certified ML Specialty", body))
    content.append(Paragraph("• DeepLearning.AI TensorFlow Developer", body))
    content.append(Paragraph("• Google Professional Data Engineer", body))

    doc.build(content)
    print(f"[OK] Created: {path}")


# ─────────────────────────────────────────────
# 3. TABLES RESUME (structured table layout)
# ─────────────────────────────────────────────

def make_tables_resume():
    path = OUTPUT_DIR / "resume_tables.pdf"
    doc = SimpleDocTemplate(str(path), pagesize=A4,
                            leftMargin=2*cm, rightMargin=2*cm,
                            topMargin=2*cm, bottomMargin=2*cm)
    content = []

    body = styles["Normal"]
    h2 = ParagraphStyle("H2T", parent=styles["Normal"], fontSize=12, spaceBefore=10,
                        spaceAfter=4, textColor=colors.HexColor("#1A56DB"))
    bold = ParagraphStyle("Bold", parent=styles["Normal"], fontName="Helvetica-Bold")

    content.append(Paragraph("Neha Kapoor", ParagraphStyle("n", fontSize=18, alignment=TA_CENTER)))
    content.append(Paragraph(
        "Full Stack Developer | neha.kapoor@outlook.com | LinkedIn: linkedin.com/in/nehakapoor | Pune, India",
        ParagraphStyle("sub", fontSize=9, alignment=TA_CENTER)))
    content.append(Spacer(1, 0.4*cm))

    # Skills table
    content.append(Paragraph("TECHNICAL SKILLS", h2))
    skill_data = [
        [Paragraph("<b>Category</b>", body), Paragraph("<b>Technologies</b>", body)],
        ["Frontend", "React.js, Next.js, TypeScript, Tailwind CSS, HTML5, CSS3"],
        ["Backend", "Node.js, Express.js, Python, FastAPI, GraphQL"],
        ["Database", "PostgreSQL, MongoDB, Redis, Prisma ORM"],
        ["DevOps", "Docker, Kubernetes, GitHub Actions, AWS (EC2, S3, RDS)"],
        ["Tools", "Git, Jira, Figma, VS Code, Postman"],
    ]
    skill_table = Table(skill_data, colWidths=[4*cm, 12*cm])
    skill_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A56DB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FF")]),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    content.append(skill_table)
    content.append(Spacer(1, 0.3*cm))

    # Experience table
    content.append(Paragraph("WORK EXPERIENCE", h2))
    exp_data = [
        [Paragraph("<b>Role</b>", body), Paragraph("<b>Company</b>", body),
         Paragraph("<b>Period</b>", body), Paragraph("<b>Key Achievement</b>", body)],
        ["Senior Frontend Dev", "Swiggy, Pune", "2023–Present",
         "Rebuilt order tracking UI reducing load time by 60%"],
        ["Full Stack Dev", "Infosys, Pune", "2021–2023",
         "Delivered 3 enterprise portals for Fortune 500 clients"],
        ["Web Dev Intern", "Startup XYZ", "2020",
         "Built company website with 99.9% uptime using Next.js"],
    ]
    exp_table = Table(exp_data, colWidths=[4*cm, 4*cm, 3*cm, 5*cm])
    exp_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A56DB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FF")]),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 5),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    content.append(exp_table)
    content.append(Spacer(1, 0.3*cm))

    # Education
    content.append(Paragraph("EDUCATION", h2))
    content.append(Paragraph("<b>B.Tech, Information Technology</b> — Pune University | 2021 | 8.9 CGPA", body))
    content.append(Spacer(1, 0.2*cm))

    # Projects
    content.append(Paragraph("NOTABLE PROJECTS", h2))
    proj_data = [
        [Paragraph("<b>Project</b>", body), Paragraph("<b>Tech Stack</b>", body),
         Paragraph("<b>Impact</b>", body)],
        ["E-commerce Platform", "Next.js, Node.js, PostgreSQL, Stripe",
         "10K+ monthly active users, ₹2Cr GMV"],
        ["Real-time Chat App", "Socket.IO, React, Redis, MongoDB",
         "Supports 500 concurrent users"],
        ["AI Image Generator", "React, FastAPI, Stable Diffusion",
         "500+ images generated in beta"],
    ]
    proj_table = Table(proj_data, colWidths=[5*cm, 6*cm, 5*cm])
    proj_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1A56DB")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F5F7FF")]),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("PADDING", (0, 0), (-1, -1), 5),
    ]))
    content.append(proj_table)

    doc.build(content)
    print(f"[OK] Created: {path}")


if __name__ == "__main__":
    print("[*] Generating sample resume PDFs...")
    make_simple_resume()
    make_columns_resume()
    make_tables_resume()
    print("\n[DONE] All 3 sample PDFs created in:", OUTPUT_DIR)
