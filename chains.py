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
