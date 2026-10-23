import io
import re
from xml.sax.saxutils import escape

import pymupdf
import streamlit as st
from dotenv import load_dotenv
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer

from connect_db import process_pdf
from chains import ask_question, generate_full_pdf_questions

load_dotenv()

st.set_page_config(
    page_title="PDF Study Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------- Styling ----------------
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1600px; }
      .hero {
        padding: 1.35rem 1.5rem; border-radius: 18px;
        background: linear-gradient(120deg, #102a43 0%, #175d72 100%);
        color: white; margin-bottom: 1rem;
      }
      .hero h1 { color: white; margin: 0 0 .35rem 0; font-size: 2rem; }
      .hero p { color: #e2f1f5; margin: 0; }
      .metric-card {
        border: 1px solid rgba(128,128,128,.25); border-radius: 14px;
        padding: .85rem 1rem; min-height: 88px;
      }
      .metric-label { font-size: .82rem; opacity: .75; margin-bottom: .25rem; }
      .metric-value { font-size: 1.45rem; font-weight: 700; }
      .section-card {
        border: 1px solid rgba(128,128,128,.22); border-radius: 16px;
        padding: 1rem 1.1rem; margin-bottom: .75rem;
      }
      div[data-testid="stButton"] button { border-radius: 10px; }
      div[data-testid="stDownloadButton"] button { border-radius: 10px; }
      .small-note { font-size: .85rem; opacity: .72; }
      [data-testid="stImage"] img { max-width: 100%; height: auto; object-fit: contain; }
      @media (max-width: 640px) {
        .block-container { padding-left: .65rem; padding-right: .65rem; padding-top: .8rem; }
        .hero { padding: 1rem; border-radius: 12px; }
        .hero h1 { font-size: 1.45rem; }
        .hero p { font-size: .9rem; }
        [data-testid="stMetric"] { padding: .5rem; }
        button, [data-testid="stDownloadButton"] button { width: 100%; }
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------- PDF export ----------------
def questions_to_pdf(title: str, content: str) -> bytes:
    buffer = io.BytesIO()
    pdf = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=title,
    )

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="StudyTitle",
            parent=styles["Title"],
            alignment=TA_CENTER,
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="StudyBody",
            parent=styles["BodyText"],
            leading=15,
            spaceAfter=8,
        )
    )

    story = [
        Paragraph(escape(title), styles["StudyTitle"]),
        Spacer(1, 6),
    ]

    # A small Markdown-to-ReportLab converter for headings, bullets and bold.
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if not line:
            story.append(Spacer(1, 5))
            continue

        if line.startswith("### "):
            story.append(Paragraph(escape(line[4:]), styles["Heading3"]))
        elif line.startswith("## "):
            story.append(Paragraph(escape(line[3:]), styles["Heading2"]))
        elif line.startswith("# "):
            story.append(Paragraph(escape(line[2:]), styles["Heading1"]))
        else:
            safe_line = escape(line)
            safe_line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", safe_line)
            if line.startswith(("- ", "* ")):
                safe_line = "&#8226; " + safe_line[2:]
            elif re.match(r"^\d+[\.\)]\s+", line):
                safe_line = re.sub(r"^(\d+[\.\)])\s+", r"<b>\1</b> ", safe_line)
            story.append(Paragraph(safe_line, styles["StudyBody"]))

    pdf.build(story)
    return buffer.getvalue()


# ---------------- Session state ----------------
DEFAULTS = {
    "vectorstore": None,
    "chunks": [],
    "pdf_name": None,
    "pdf_bytes": None,
    "file_hash": None,
    "selected_page": 1,
    "total_pdf_pages": 0,
    "answer": None,
    "answer_pages": [],
    "generated_cache": {},
    "interview_result": None,
}
for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


# ---------------- Header ----------------
st.markdown(
    """
    <div class="hero">
      <h1>📚 PDF Study Assistant</h1>
      <p>Read your PDF, ask questions, prepare exam answers, and revise from one workspace.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.expander("Upload or switch PDF", expanded=st.session_state.vectorstore is None):
    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    process_clicked = st.button(
        "📥 Process / Open PDF",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=True,
    )

if process_clicked and uploaded_file is not None:
    try:
        with st.spinner("Extracting text/OCR and checking stored embeddings..."):
            vectorstore, chunks, reused, file_hash = process_pdf(uploaded_file)
            pdf_bytes = uploaded_file.getvalue()

            with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf_doc:
                total_pdf_pages = len(pdf_doc)

        st.session_state.vectorstore = vectorstore
        st.session_state.chunks = chunks
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.pdf_bytes = pdf_bytes
        st.session_state.file_hash = file_hash
        st.session_state.total_pdf_pages = total_pdf_pages
        st.session_state.selected_page = 1
        st.session_state.answer = None
        st.session_state.answer_pages = []
        st.session_state.generated_cache = {}
        st.session_state.interview_result = None

        if reused:
            st.success(
                f"Opened existing indexed PDF · {total_pdf_pages} pages · "
                f"{len(chunks)} stored chunks reused."
            )
        else:
            st.success(
                f"PDF processed · {total_pdf_pages} pages · "
                f"{len(chunks)} text chunks indexed."
            )
    except Exception as exc:
        st.error(f"Could not process PDF: {exc}")


if st.session_state.vectorstore is None:
    st.info("Upload a PDF and click **Process / Open PDF** to start studying.")
    st.stop()


# Keep the information useful to the user, without exposing extraction internals.
chunks = st.session_state.chunks
pdf_bytes = st.session_state.pdf_bytes
total_pages = st.session_state.total_pdf_pages or 1

st.markdown(f"### 📄 {st.session_state.pdf_name}")
col1, col2 = st.columns(2)
with col1:
    st.metric("PDF pages", total_pages)
with col2:
    st.metric("Study chunks", len(chunks))


# ---------------- Main workspace ----------------
read_tab, ask_tab, study_tab = st.tabs(
    ["📖 Read PDF", "💬 Ask your PDF", "🎯 Study & Practice"]
)

with read_tab:
    st.caption("Read one page at a time. The page automatically fits the available screen width.")

    # Keep the PDF page image responsive instead of embedding a fixed-width PDF viewer.
    # This avoids horizontal clipping on phones.
    nav_left, nav_right = st.columns(2)
    with nav_left:
        if st.button(
            "◀ Previous page",
            disabled=st.session_state.selected_page <= 1,
            use_container_width=True,
            key="pdf_previous_page",
        ):
            st.session_state.selected_page = max(1, st.session_state.selected_page - 1)
    with nav_right:
        if st.button(
            "Next page ▶",
            disabled=st.session_state.selected_page >= total_pages,
            use_container_width=True,
            key="pdf_next_page",
        ):
            st.session_state.selected_page = min(total_pages, st.session_state.selected_page + 1)

    st.caption(f"Page {st.session_state.selected_page} of {total_pages}")

    try:
        with pymupdf.open(stream=pdf_bytes, filetype="pdf") as pdf_doc:
            page = pdf_doc.load_page(st.session_state.selected_page - 1)
            # Render at a higher resolution; Streamlit scales it down to the screen width.
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False)
            page_image = pix.tobytes("png")

        st.image(page_image, use_container_width=True)

    except Exception as exc:
        st.error("This page could not be displayed.")
        st.caption(f"Viewer error: {exc}")

    st.download_button(
        "Download original PDF",
        data=pdf_bytes,
        file_name=st.session_state.pdf_name,
        mime="application/pdf",
        use_container_width=True,
    )

    with st.expander("View extracted text (optional)"):
        page_docs = [
            doc for doc in chunks
            if doc.metadata.get("page") == int(st.session_state.selected_page)
        ]
        if page_docs:
            for doc in page_docs:
                st.text(doc.page_content)
        else:
            st.info("No searchable text was extracted from this page.")


with ask_tab:
    st.subheader("Ask questions about this PDF")
    st.caption("Answers are grounded in retrieved PDF passages and include page references.")

    with st.form("question_form", clear_on_submit=False):
        question = st.text_input(
            "Your question",
            placeholder="e.g. Explain renewable and non-renewable resources",
        )
        ask_clicked = st.form_submit_button("Get answer", type="primary")

    if ask_clicked:
        if not question.strip():
            st.warning("Please enter a question.")
        else:
            try:
                with st.spinner("Searching relevant passages..."):
                    answer, pages = ask_question(
                        st.session_state.vectorstore,
                        question.strip(),
                    )
                st.session_state.answer = answer
                st.session_state.answer_pages = pages
            except Exception as exc:
                st.error(f"Could not answer the question: {exc}")

    if st.session_state.answer:
        st.markdown("#### Answer")
        st.markdown(st.session_state.answer)

        pages = st.session_state.answer_pages
        if pages:
            st.caption("Open a cited source page:")
            page_cols = st.columns(min(len(pages), 5))
            for i, page in enumerate(pages):
                with page_cols[i % len(page_cols)]:
                    if st.button(f"📖 Page {page}", key=f"qa_page_{page}"):
                        st.session_state.selected_page = min(max(1, page), total_pages)
                        st.toast(f"Selected page {page}. Open the Read PDF tab to view it.")


with study_tab:
    st.subheader("Generate study material")
    st.write("Create interview/viva questions or longer exam-style answers from the indexed PDF.")

    controls1, controls2 = st.columns(2)
    with controls1:
        answer_style = st.selectbox(
            "Answer length",
            ["Short", "Mixed", "Detailed"],
            index=1,
            help=(
                "Short: quick revision. Mixed: short definitions and longer explanations "
                "where useful. Detailed: structured descriptive answers."
            ),
        )
    with controls2:
        number = st.slider(
            "Number of questions",
            min_value=5,
            max_value=30,
            value=10,
        )
        st.caption(
            "Detailed answers use more tokens and may take longer. "
            "Check generated page references against the source."
        )

    cache_key = (
        st.session_state.file_hash,
        answer_style,
        number,
    )

    if st.button("✨ Generate study material", type="primary", use_container_width=True):
        try:
            if cache_key in st.session_state.generated_cache:
                result = st.session_state.generated_cache[cache_key]
                st.info("Loaded previously generated material from this session.")
            else:
                progress = st.progress(0)
                status = st.empty()

                def update_progress(done, total):
                    progress.progress(done / total if total else 1.0)
                    status.write(f"Analysing PDF batches: {done} of {total}")

                # Requires generate_full_pdf_questions(..., answer_style=...)
                # in the updated chains.py.
                with st.spinner("Generating questions and answers..."):
                    result = generate_full_pdf_questions(
                        chunks=chunks,
                        num_questions=number,
                        batch_size=8,
                        progress_callback=update_progress,
                        max_workers=3,
                        answer_style=answer_style,
                    )

                st.session_state.generated_cache[cache_key] = result
                status.success("Generation complete.")

            st.session_state.interview_result = result
            st.session_state.interview_result_key = cache_key

        except Exception as exc:
            st.error(f"Study material generation failed: {exc}")

    result = st.session_state.get("interview_result")
    result_key = st.session_state.get("interview_result_key")

    if result and result_key == cache_key:
        st.markdown("---")
        st.markdown(result)

        safe_name = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            st.session_state.pdf_name.rsplit(".", 1)[0],
        ).strip("_") or "study_material"

        download_col1, download_col2 = st.columns(2)
        with download_col1:
            st.download_button(
                "📄 Download as PDF",
                data=questions_to_pdf(
                    f"Study Questions — {st.session_state.pdf_name}",
                    result,
                ),
                file_name=f"{safe_name}_study_material.pdf",
                mime="application/pdf",
                use_container_width=True,
            )
        with download_col2:
            st.download_button(
                "📝 Download as Markdown",
                data=result,
                file_name=f"{safe_name}_study_material.md",
                mime="text/markdown",
                use_container_width=True,
            )

        referenced_pages = sorted({
            int(page)
            for page in re.findall(r"\[Page\s+(\d+)\]", result)
            if 1 <= int(page) <= total_pages
        })

        if referenced_pages:
            st.caption("Jump to a page cited in the generated material:")
            ref_cols = st.columns(min(5, len(referenced_pages)))
            for i, page in enumerate(referenced_pages):
                with ref_cols[i % len(ref_cols)]:
                    if st.button(f"📖 Page {page}", key=f"study_page_{page}"):
                        st.session_state.selected_page = page
                        st.toast(f"Selected page {page}. Open the Read PDF tab to view it.")
