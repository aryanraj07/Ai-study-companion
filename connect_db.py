
import hashlib
import os
import tempfile
from io import BytesIO
from typing import List, Tuple

import chromadb
import pymupdf
import pytesseract
from dotenv import load_dotenv
from PIL import Image

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# --------------------------------------------------
# 1. Environment and configuration
# --------------------------------------------------

load_dotenv()

embedding_model = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

chroma_client = chromadb.CloudClient(
    tenant=os.environ["CHROMA_TENANT"],
    database=os.environ["CHROMA_DATABASE"],
    api_key=os.environ["CHROMA_API_KEY"],
)

REGISTRY_NAME = "pdf_registry"

# Update this path if Tesseract is installed elsewhere.
# On a hosted Linux server, configure the server's Tesseract
# executable instead of using this Windows-specific path.
TESSERACT_PATH = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

if os.path.exists(TESSERACT_PATH):
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_PATH


# --------------------------------------------------
# 2. Get the PDF registry collection
# --------------------------------------------------

def get_registry() -> Chroma:
    """Store mappings between PDF hashes and collection names."""
    return Chroma(
        client=chroma_client,
        collection_name=REGISTRY_NAME,
        embedding_function=embedding_model,
    )


# --------------------------------------------------
# 3. Extract text from a PDF using native extraction
#    and OCR fallback
# --------------------------------------------------

def extract_pdf_documents(
    pdf_path: str,
    source_name: str,
    file_hash: str,
) -> List[Document]:
    """
    Extract native PDF text first.
    Run OCR on pages where native extraction returns no text.

    Note:
    This first version does not OCR image text on pages
    that already contain native text.
    """

    documents: List[Document] = []

    with pymupdf.open(pdf_path) as pdf:
        print(f"Total PDF pages: {len(pdf)}")

        for page_index, page in enumerate(pdf):
            page_number = page_index + 1

            # Try native text extraction first.
            text = page.get_text("text").strip()
            method = "native"

            # OCR fallback for image-only/scanned pages.
            if not text:
                print(f"Page {page_number}: running OCR...")

                pix = page.get_pixmap(
                    matrix=pymupdf.Matrix(2, 2),
                    alpha=False,
                )

                image_bytes = pix.tobytes("png")

                with Image.open(BytesIO(image_bytes)) as image:
                    text = pytesseract.image_to_string(
                        image,
                        lang="eng",
                    ).strip()

                method = "ocr"

            if not text:
                print(
                    f"Page {page_number}: "
                    "no text extracted; skipping."
                )
                continue

            document = Document(
                page_content=text,
                metadata={
                    "source": source_name,
                    "file_hash": file_hash,
                    "page": page_number,
                    "extraction_method": method,
                },
            )

            documents.append(document)

            print(
                f"Page {page_number}: "
                f"{len(text)} characters; method={method}"
            )

    return documents


# --------------------------------------------------
# 4. Load existing chunks from Chroma
# --------------------------------------------------

def load_chunks(vectorstore: Chroma) -> List[Document]:
    """Load stored chunk text and metadata without re-embedding."""

    result = vectorstore.get(
        include=["documents", "metadatas"]
    )

    texts = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    rows = []

    for text, metadata in zip(texts, metadatas):
        if not text:
            continue

        metadata = metadata or {}
        chunk_index = int(metadata.get("chunk_index", 0))

        rows.append(
            (
                chunk_index,
                Document(
                    page_content=text,
                    metadata=metadata,
                ),
            )
        )

    rows.sort(key=lambda row: row[0])

    return [document for _, document in rows]


# --------------------------------------------------
# 5. Register the processed PDF
# --------------------------------------------------

def _register_hash(
    registry: Chroma,
    file_hash: str,
    collection_name: str,
    filename: str,
) -> None:
    """Save a PDF hash-to-collection mapping."""

    existing = registry.get(
        ids=[file_hash],
        include=["metadatas"],
    )

    if existing.get("ids"):
        return

    registry.add_texts(
        texts=[file_hash],
        ids=[file_hash],
        metadatas=[
            {
                "file_hash": file_hash,
                "collection_name": collection_name,
                "pdf_name": filename,
            }
        ],
    )


# --------------------------------------------------
# 6. Main PDF processing function
# --------------------------------------------------

def process_pdf(
    uploaded_file,
) -> Tuple[Chroma, list, bool, str]:
    """
    Returns:
        vectorstore,
        chunks,
        reused_existing_collection,
        file_hash

    Identical PDF bytes reuse the existing Chroma collection.
    """

    print("Called process_pdf()")

    # Read uploaded PDF bytes.
    pdf_bytes = uploaded_file.getvalue()

    if not pdf_bytes:
        raise ValueError("Uploaded PDF is empty.")

    if not pdf_bytes.startswith(b"%PDF-"):
        raise ValueError("The uploaded file does not appear to be a PDF.")

    # Generate a unique content hash.
    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    collection_name = f"pdf_{file_hash[:32]}"

    registry = get_registry()

    # ----------------------------------------------
    # A. Check whether this PDF is already registered
    # ----------------------------------------------

    registry_result = registry.get(
        ids=[file_hash],
        include=["metadatas"],
    )

    registry_metadata = registry_result.get("metadatas") or []

    if registry_metadata:
        saved_collection = registry_metadata[0].get(
            "collection_name"
        )

        if saved_collection:
            vectorstore = Chroma(
                client=chroma_client,
                collection_name=saved_collection,
                embedding_function=embedding_model,
            )

            chunks = load_chunks(vectorstore)

            if chunks:
                print("Reusing registered PDF collection.")

                return (
                    vectorstore,
                    chunks,
                    True,
                    file_hash,
                )

    # ----------------------------------------------
    # B. Recovery if the collection exists but the
    #    registry record was not saved successfully
    # ----------------------------------------------

    try:
        chroma_client.get_collection(
            name=collection_name
        )

        existing_vectorstore = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embedding_model,
        )

        existing_chunks = load_chunks(existing_vectorstore)

        if existing_chunks:
            _register_hash(
                registry,
                file_hash,
                collection_name,
                uploaded_file.name,
            )

            print("Recovered existing PDF collection.")

            return (
                existing_vectorstore,
                existing_chunks,
                True,
                file_hash,
            )

        # Empty collection: continue processing below.
        # The collection is reused for insertion.

    except Exception as error:
        print(
            "No reusable collection found; "
            f"processing PDF. Details: {error}"
        )

    # ----------------------------------------------
    # C. Save bytes to a temporary PDF file
    # ----------------------------------------------

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name

        # ------------------------------------------
        # D. Extract native text and OCR scanned pages
        # ------------------------------------------

        docs = extract_pdf_documents(
            pdf_path=temp_path,
            source_name=uploaded_file.name,
            file_hash=file_hash,
        )

        if not docs:
            raise ValueError(
                "No text could be extracted from this PDF, "
                "even after OCR."
            )

        # ------------------------------------------
        # E. Split extracted documents into chunks
        # ------------------------------------------

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )

        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError("No chunks were generated.")

        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index

        print(f"Total extracted pages: {len(docs)}")
        print(f"Total chunks generated: {len(chunks)}")

        # ------------------------------------------
        # F. Store chunks in Chroma Cloud
        # ------------------------------------------

        vectorstore = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embedding_model,
        )

        ids = [
            f"{file_hash}_{index}"
            for index in range(len(chunks))
        ]

        vectorstore.add_documents(
            documents=chunks,
            ids=ids,
        )

        # ------------------------------------------
        # G. Register PDF after successful insertion
        # ------------------------------------------

        _register_hash(
            registry,
            file_hash,
            collection_name,
            uploaded_file.name,
        )

        print("PDF processed and stored successfully.")

        return (
            vectorstore,
            chunks,
            False,
            file_hash,
        )

    finally:
        # Always remove the temporary PDF from disk.
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
