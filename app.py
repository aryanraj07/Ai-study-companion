import io
import re
import streamlit as st

# Pdf download from questions 
# ********************************
from xml.sax.saxutils import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
)
# ***************************************
from dotenv import load_dotenv
from streamlit_pdf_viewer import pdf_viewer

from connect_db import process_pdf
from chains import ask_question, generate_full_pdf_questions

load_dotenv()
def questions_to_pdf(title, content):
    buffer = io.BytesIO()

    doc = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=18 * mm,
        leftMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=title,
    )

    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(
        name="StudyTitle",
        parent=styles["Title"],
        alignment=TA_CENTER,
        spaceAfter=14,
    ))
    styles.add(ParagraphStyle(
        name="StudyBody",
        parent=styles["BodyText"],
        leading=15,
        spaceAfter=8,
    ))

    story = [
        Paragraph(escape(title), styles["StudyTitle"]),
        Spacer(1, 6),
    ]

    # Simple Markdown formatting ko readable PDF text mein convert karo.
    for line in content.splitlines():
        line = line.strip()

        if not line:
            story.append(Spacer(1, 5))
            continue

        if line.startswith("### "):
            story.append(Paragraph(
                escape(line[4:]),
                styles["Heading3"],
            ))
        elif line.startswith("## "):
            story.append(Paragraph(
                escape(line[3:]),
                styles["Heading2"],
            ))
        elif line.startswith("# "):
            story.append(Paragraph(
                escape(line[2:]),
                styles["Heading1"],
            ))
        else:
            safe_line = escape(line)
            safe_line = re.sub(
                r"\*\*(.+?)\*\*",
                r"<b>\1</b>",
                safe_line,
            )
            story.append(Paragraph(
                safe_line,
                styles["StudyBody"],
            ))

    doc.build(story)
    return buffer.getvalue()

st.set_page_config(
    page_title="PDF Study Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("📚 PDF Study Assistant")
st.caption("Upload once, reuse stored embeddings, and study with page references.")

# Session state
defaults = {
    "vectorstore": None,
    "chunks": [],
    "pdf_name": None,
    "pdf_bytes": None,
    "file_hash": None,
    "selected_page": 1,
    "answer": None,
    "answer_pages": [],
    "generated_cache": {},
}

for key, value in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value


uploaded_file = st.file_uploader(
    "Upload a PDF",
    type=["pdf"],
)

if st.button("📥 Process / Open PDF", disabled=uploaded_file is None):
    progress = st.progress(0)
    status = st.empty()
    try:
        with st.spinner("Checking for an existing PDF..."):
            vectorstore, chunks, reused, file_hash = process_pdf(
                uploaded_file
            )

        st.session_state.vectorstore = vectorstore
        st.session_state.chunks = chunks
        st.session_state.pdf_name = uploaded_file.name
        st.session_state.pdf_bytes = uploaded_file.getvalue()
        st.session_state.file_hash = file_hash
        st.session_state.selected_page = 1
        st.session_state.answer = None
        st.session_state.answer_pages = []

        if reused:
            st.success(
                "Existing PDF collection reused. "
                "No new chunk embeddings were generated."
            )
        else:
            st.success(
                f"PDF processed: {len(chunks)} chunks stored."
            )
        
    except Exception as exc:
        st.error(f"Could not process PDF: {exc}")


if st.session_state.vectorstore is not None:
    chunks = st.session_state.chunks

    page_numbers = [
        doc.metadata.get("page")
        for doc in chunks
        if isinstance(doc.metadata.get("page"), int)
    ]
    total_pages = max(page_numbers, default=0) + 1

    st.caption(
        f"📄 {st.session_state.pdf_name}  ·  "
        f"{total_pages} pages  ·  {len(chunks)} chunks"
    )

    left, right = st.columns([1, 1.1], gap="large")

    with left:
        st.subheader("💬 Ask your PDF")

        with st.form("question_form"):
            question = st.text_input(
                "What do you want to know?"
            )
            ask_clicked = st.form_submit_button("Get Answer")

        if ask_clicked and question.strip():
            with st.spinner("Finding relevant passages..."):
                answer, pages = ask_question(
                    st.session_state.vectorstore,
                    question.strip(),
                )

            st.session_state.answer = answer
            st.session_state.answer_pages = pages

        if st.session_state.answer:
            st.markdown("### Answer")
            st.markdown(st.session_state.answer)

            pages = st.session_state.answer_pages
            if pages:
                st.caption("Open a source page in the PDF viewer:")
                page_cols = st.columns(min(len(pages), 4))

                for i, page in enumerate(pages):
                    with page_cols[i % len(page_cols)]:
                        if st.button(
                            f"📖 Open Page {page}",
                            key=f"answer_page_{page}",
                        ):
                            st.session_state.selected_page = page

        st.divider()
        st.subheader("🎯 Interview Preparation")

        number = st.slider(
            "Number of final questions",
            min_value=5,
            max_value=30,
            value=10,
        )

        cache_key = (
            st.session_state.file_hash,
            number,
        )

        if st.button("Generate Interview Questions"):
            if cache_key in st.session_state.generated_cache:
                st.info("Using questions already generated in this session.")
                result = st.session_state.generated_cache[cache_key]
            else:
                progress = st.progress(0)
                status = st.empty()

                def update_progress(done, total):
                    progress.progress(done / total)
                    status.write(
                        f"Analysing PDF: batch {done} of {total}"
                    )

                try:
                    with st.spinner(
                        "Generating questions from PDF batches..."
                    ):
                        result = generate_full_pdf_questions(
                            chunks=chunks,
                            num_questions=number,
                            batch_size=8,
                            progress_callback=update_progress,
                            max_workers=4,
                        )

                    st.session_state.generated_cache[cache_key] = result
                    status.write("Generation complete.")

                except Exception as exc:
                    st.error(f"Question generation failed: {exc}")
                    result = None

            if result:
                st.markdown(result)
                safe_name = "biology_interview_questions"
                st.download_button(
                label="📄 Download Questions as PDF",
                data=questions_to_pdf(
                    "Biology Interview Questions",
                    result,
                ),
                file_name=f"{safe_name}.pdf",
                mime="application/pdf",
                use_container_width=True,
            )

            st.download_button(
                label="📝 Download as Markdown",
                data=result,
                file_name=f"{safe_name}.md",
                mime="text/markdown",
                use_container_width=True,
            )


            st.session_state.interview_result = result

        if st.session_state.get("interview_result"):
            st.markdown("### Your Questions & Answers")
            st.markdown(st.session_state.interview_result)

            # Page references in generated questions are clickable too.
            referenced_pages = sorted({
                int(page)
                for page in re.findall(
                    r"\[Page\s+(\d+)\]",
                    st.session_state.interview_result,
                )
            })

            if referenced_pages:
                st.caption("Jump to a referenced page:")
                ref_cols = st.columns(4)

                for i, page in enumerate(referenced_pages):
                    with ref_cols[i % 4]:
                        if st.button(
                            f"📖 Page {page}",
                            key=f"interview_page_{page}",
                        ):
                            st.session_state.selected_page = page

    with right:
        st.subheader("📄 PDF Viewer")

        st.session_state.selected_page = min(
            max(1, st.session_state.selected_page),
            total_pages,
        )

        selected_page = st.selectbox(
            "Jump to page",
            options=list(range(1, total_pages + 1)),
            key="selected_page",
        )

        try:
            pdf_viewer(
                input=st.session_state.pdf_bytes,
                pages_to_render=[selected_page],
                width=700,
                height=850,
            )
        except Exception as exc:
            st.warning(
                "PDF preview could not be rendered. "
                "You can still use the PDF's page references."
            )
            st.caption(str(exc))
else:
    st.info("Upload a PDF and click **Process / Open PDF** to begin.")
from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_retries=2,
)

# ---------------- PDF Q&A ----------------

qa_prompt = ChatPromptTemplate.from_template("""
You are a careful study assistant. Answer using only the supplied PDF context.
If the answer is not supported, say you could not find it in this PDF.
Keep the answer clear and concise. Cite source pages exactly as [Page N].

Context:
{context}

Question:
{question}
""")

qa_chain = qa_prompt | llm | StrOutputParser()


def ask_question(vectorstore, question):
    docs = vectorstore.similarity_search(question, k=4)
    if not docs:
        return "No relevant content found in the PDF.", []

    context_parts = []
    pages = set()

    for doc in docs:
        page = doc.metadata.get("page")
        page_number = page + 1 if isinstance(page, int) else None
        label = f"[Page {page_number}]" if page_number else "[Page unknown]"
        if page_number:
            pages.add(page_number)
        context_parts.append(f"{label}\n{doc.page_content}")

    answer = qa_chain.invoke({
        "context": "\n\n".join(context_parts),
        "question": question,
    })
    return answer, sorted(pages)


# ---------------- Interview questions ----------------

batch_prompt = ChatPromptTemplate.from_template("""
Create 3 distinct interview or viva questions with concise answers from this context.
Use only the supplied context. Cover important concepts, not minor details.
Every answer must cite its source page as [Page N].
Do not add an introduction or extra commentary.

Context:
{context}
""")

reduce_prompt = ChatPromptTemplate.from_template("""
Merge these candidate questions into a concise, non-repetitive set.
Keep the most useful distinct questions, accurate answers, and [Page N] references.
Use only the supplied candidates; do not invent facts.

Candidates:
{candidates}
""")

final_prompt = ChatPromptTemplate.from_template("""
Create up to {num_questions} high-quality interview/viva questions with concise answers.
Cover distinct topics, remove duplicates, preserve [Page N] references, and do not invent facts.
If the candidates do not support the requested number, return fewer rather than inventing.

Candidates:
{candidates}
""")

batch_chain = batch_prompt | llm | StrOutputParser()
reduce_chain = reduce_prompt | llm | StrOutputParser()
final_chain = final_prompt | llm | StrOutputParser()


def _make_context(docs):
    parts = []
    for doc in docs:
        page = doc.metadata.get("page")
        page_number = page + 1 if isinstance(page, int) else "unknown"
        parts.append(f"[Page {page_number}]\n{doc.page_content}")
    return "\n\n".join(parts)


def generate_full_pdf_questions(
    chunks,
    num_questions=10,
    batch_size=8,
    progress_callback=None,
    max_workers=4,
):
    if not chunks:
        return "No PDF chunks available."

    batches = [
        chunks[start:start + batch_size]
        for start in range(0, len(chunks), batch_size)
    ]
    candidates = [None] * len(batches)
    completed = 0

    # Parallel map stage. Concurrency is bounded to limit API bursts.
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(batch_chain.invoke, {"context": _make_context(batch)}): i
            for i, batch in enumerate(batches)
        }
        for future in as_completed(futures):
            candidates[futures[future]] = future.result()
            completed += 1
            if progress_callback:
                progress_callback(completed, len(batches))

    candidates = [item for item in candidates if item]

    # Reduce calls only when candidate volume warrants it.
    while len(candidates) > 4:
        groups = [
            candidates[i:i + 4]
            for i in range(0, len(candidates), 4)
        ]
        candidates = [
            reduce_chain.invoke({"candidates": "\n\n".join(group)})
            for group in groups
        ]

    return final_chain.invoke({
        "num_questions": num_questions,
        "candidates": "\n\n".join(candidates),
    })
