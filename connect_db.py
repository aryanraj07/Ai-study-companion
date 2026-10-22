import hashlib
import os
import tempfile
from typing import List, Tuple

import chromadb
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

chroma_client = chromadb.CloudClient(
    tenant=os.environ["CHROMA_TENANT"],
    database=os.environ["CHROMA_DATABASE"],
    api_key=os.environ["CHROMA_API_KEY"],
)

REGISTRY_NAME = "pdf_registry"


def get_registry() -> Chroma:
    """Persistent hash -> collection mapping in Chroma Cloud."""
    return Chroma(
        client=chroma_client,
        collection_name=REGISTRY_NAME,
        embedding_function=embedding_model,
    )


def load_chunks(vectorstore: Chroma) -> List[Document]:
    """Fetch stored chunk text/metadata; this does NOT create embeddings."""
    result = vectorstore.get(include=["documents", "metadatas"])
    texts = result.get("documents") or []
    metadatas = result.get("metadatas") or []

    rows = []
    for text, metadata in zip(texts, metadatas):
        if text:
            metadata = metadata or {}
            rows.append((metadata.get("chunk_index", 0), Document(
                page_content=text,
                metadata=metadata,
            )))

    rows.sort(key=lambda row: row[0])
    return [doc for _, doc in rows]


def _register_hash(registry: Chroma, file_hash: str, collection_name: str, filename: str):
    existing = registry.get(ids=[file_hash], include=["metadatas"])
    if existing.get("ids"):
        return
    # Registry is also a Chroma collection, so this tiny hash record has an embedding.
    registry.add_texts(
        texts=[file_hash],
        ids=[file_hash],
        metadatas=[{
            "file_hash": file_hash,
            "collection_name": collection_name,
            "pdf_name": filename,
        }],
    )


def process_pdf(uploaded_file) -> Tuple[Chroma, list, bool, str]:
    """
    Returns: vectorstore, chunks, reused_existing_collection, sha256_hash.
    Exact byte-identical PDFs reuse the existing collection regardless of filename.
    """
    pdf_bytes = uploaded_file.getvalue()
    if not pdf_bytes:
        raise ValueError("Uploaded PDF is empty.")

    file_hash = hashlib.sha256(pdf_bytes).hexdigest()
    collection_name = f"pdf_{file_hash[:32]}"
    registry = get_registry()

    # First, check the exact registry ID. No vector similarity is used here.
    registry_result = registry.get(ids=[file_hash], include=["metadatas"])
    registry_metadata = registry_result.get("metadatas") or []

    if registry_metadata:
        saved_collection = registry_metadata[0].get("collection_name")
        if saved_collection:
            vectorstore = Chroma(
                client=chroma_client,
                collection_name=saved_collection,
                embedding_function=embedding_model,
            )
            chunks = load_chunks(vectorstore)
            if chunks:
                return vectorstore, chunks, True, file_hash

    # Recovery path: collection may exist if the app crashed before registry write.
    try:
        chroma_client.get_collection(name=collection_name)
        existing_vectorstore = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embedding_model,
        )
        existing_chunks = load_chunks(existing_vectorstore)
        if existing_chunks:
            _register_hash(
                registry, file_hash, collection_name, uploaded_file.name
            )
            return existing_vectorstore, existing_chunks, True, file_hash
    except Exception:
        # Collection does not exist, so process the new file below.
        pass

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
            temp_file.write(pdf_bytes)
            temp_path = temp_file.name

        docs = PyPDFLoader(temp_path).load()
        for doc in docs:
            doc.metadata["source"] = uploaded_file.name
            doc.metadata["file_hash"] = file_hash

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=150,
        )
        chunks = splitter.split_documents(docs)

        if not chunks:
            raise ValueError(
                "No readable text found. This may be a scanned PDF; "
                "OCR support would be needed for image-only pages."
            )

        for index, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = index

        vectorstore = Chroma(
            client=chroma_client,
            collection_name=collection_name,
            embedding_function=embedding_model,
        )

        ids = [f"{file_hash}_{index}" for index in range(len(chunks))]
        vectorstore.add_documents(documents=chunks, ids=ids)

        _register_hash(registry, file_hash, collection_name, uploaded_file.name)
        return vectorstore, chunks, False, file_hash

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
