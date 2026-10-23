
from dotenv import load_dotenv
load_dotenv()

from connect_db import get_registry, load_chunks, chroma_client,embedding_model
from langchain_chroma import Chroma
registry = get_registry()
all_docs = load_chunks(registry)
file_hash = (
    "95d1d20d66138ceac1a89354af4bc755"
    "6f289041688fa0d859f3118972e62582"
)
registry = get_registry()
result = registry.get(
    ids=[file_hash],
    include=["metadatas"],
)
metadatas = result.get("metadatas") or []
if not metadatas:
    raise SystemExit("PDF hash not found in registry.")
collection_name = metadatas[0].get("collection_name")

print("Registry documents:", len(all_docs))

for doc in all_docs:
    text = doc.page_content.lower()
    print(text)
    if any(term in text for term in [
        "renewable",
        "non-renewable",
        "solar energy",
        "wind energy",
        "hydropower",
        "geothermal",
        "biomass",
    ]):
        print("\nPage:", doc.metadata.get("page", "unknown"))
        print("Chunk:", doc.metadata.get("chunk_index", "unknown"))
        print(doc.page_content[:1500])


