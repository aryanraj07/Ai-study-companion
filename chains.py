from concurrent.futures import ThreadPoolExecutor, as_completed

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_chroma import Chroma


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
    print("\n" + "=" * 60)
    print("RAG DEBUG START")
    print("Question:", repr(question))
    print("=" * 60)

    if not question or not question.strip():
        return "Please enter a question.", []

    try:
        # 1. Check stored collection
        try:
            print("Collection count:", vectorstore._collection.count())
        except Exception as e:
            print("Could not get collection count:", repr(e))

        # 2. Retrieve relevant chunks
        docs = vectorstore.similarity_search(question, k=5)

        print("Retrieved documents:", len(docs))

        if not docs:
            print("FAIL: No documents retrieved.")
            return "No relevant content found in the PDF.", []

        context_parts = []
        pages = set()

        # 3. Inspect retrieved content
        for i, doc in enumerate(docs, start=1):
            page = doc.metadata.get("page")
            page_number = page + 1 if isinstance(page, int) else None

            label = (
                f"[Page {page_number}]"
                if page_number is not None
                else "[Page unknown]"
            )

            print(f"\n--- Retrieved chunk {i} ---")
            print("Metadata:", doc.metadata)
            print("Text preview:", repr(doc.page_content[:500]))

            if page_number is not None:
                pages.add(page_number)

            context_parts.append(f"{label}\n{doc.page_content}")

        context = "\n\n".join(context_parts)

        print("\nTotal context characters:", len(context))
        print("Context preview:\n", context[:2000])

        # 4. Generate answer
        print("\nCalling LLM...")
        answer = qa_chain.invoke({
            "context": context,
            "question": question.strip(),
        })
        print("\nLLM ANSWER:", repr(answer))
        print("RAG DEBUG END\n")

        return answer, sorted(pages)

    except Exception as e:
        import traceback
        print("RAG ERROR:", repr(e))
        traceback.print_exc()
        raise


# ---------------- Interview questions ----------------

batch_prompt = ChatPromptTemplate.from_template("""
You are preparing study questions from a PDF.

Create 4 distinct, useful questions from the supplied context.
Include a mix of:
- Definition or recall questions
- Conceptual understanding questions
- Comparison or difference questions, when supported
- Explanation or descriptive questions, when supported

For every question:
- Provide an accurate answer using only this context.
- Cite source pages as [Page N].
- Do not invent facts or unsupported examples.
- Avoid duplicate questions.
- Preserve important technical terms from the PDF.

Use this answer style: {answer_style}

For "Short", answers should be concise.
For "Detailed", answers should explain the concept with relevant
points and details from the context.
For "Mixed", use concise answers for simple questions and
longer, structured answers for descriptive questions.

Return readable Markdown with numbered questions.

PDF context:
{context}
""")

reduce_prompt = ChatPromptTemplate.from_template("""
Merge these candidate study questions into a non-repetitive set.

Rules:
- Preserve correct answers and their source page references.
- Keep different concepts instead of repeating the same idea.
- Do not introduce facts absent from the candidates.
- Keep answers appropriate to this style: {answer_style}.
- Retain descriptive answers where they add value.

Candidates:
{candidates}
""")

final_prompt = ChatPromptTemplate.from_template("""
Create up to {num_questions} high-quality study questions
and answers from the candidate material.

Answer style: {answer_style}

Requirements:
- Cover as many distinct important topics as the candidates support.
- Include both recall and conceptual questions when available.
- Include comparison questions only when supported by the source.
- Keep descriptive answers structured with headings or bullet points.
- Do not invent information or source page references.
- Preserve [Page N] citations in every answer.
- Return fewer questions if the candidates cannot support the
  requested number.
- Do not add an introduction or conclusion.

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
    max_workers=3,
    answer_style="Mixed",
):
    if not chunks:
        return "No PDF chunks available."

    batches = [
        chunks[start:start + batch_size]
        for start in range(0, len(chunks), batch_size)
    ]

    candidates = [None] * len(batches)
    completed = 0

    with ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
        futures = {
            executor.submit(
                batch_chain.invoke,
                {
                    "context": _make_context(batch),
                    "answer_style": answer_style,
                },
            ): index
            for index, batch in enumerate(batches)
        }

        for future in as_completed(futures):
            index = futures[future]
            candidates[index] = future.result()
            completed += 1

            if progress_callback:
                progress_callback(completed, len(batches))

    candidates = [
        item for item in candidates if item
    ]

    # Reduce candidate volume without losing page citations.
    while len(candidates) > 4:
        groups = [
            candidates[i:i + 4]
            for i in range(0, len(candidates), 4)
        ]

        reduced = []

        for group in groups:
            reduced.append(
                reduce_chain.invoke({
                    "candidates": "\n\n".join(group),
                    "answer_style": answer_style,
                })
            )

        candidates = reduced

    return final_chain.invoke({
        "num_questions": num_questions,
        "answer_style": answer_style,
        "candidates": "\n\n".join(candidates),
    })
