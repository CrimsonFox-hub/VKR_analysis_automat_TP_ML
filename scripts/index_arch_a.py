# scripts/index_arch_a.py
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from qdrant_client import QdrantClient
from shared.vector_store import VectorStore
from shared.config import CATEGORY_TO_COLLECTION, PROCESSED_DATA_DIR, QDRANT_DB_DIR, CHUNK_MAX_TOKENS, CHUNK_OVERLAP
from transformers import AutoTokenizer
from shared.logger import get_logger

log = get_logger(__name__)

tokenizer = AutoTokenizer.from_pretrained("intfloat/multilingual-e5-small")

def chunk_document(text: str, max_tokens=CHUNK_MAX_TOKENS, overlap=CHUNK_OVERLAP):
    if not text:
        return []
    tokens = tokenizer.encode(text, add_special_tokens=False)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(chunk_text)
        if end == len(tokens):
            break
        start = end - overlap
    return chunks

def main():
    records_path = PROCESSED_DATA_DIR / "records.json"
    if not records_path.exists():
        log.error("records.json not found")
        return

    with open(records_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    grouped = {}
    for rec in records:
        cat = rec.get("category", "routine")
        if cat not in grouped:
            grouped[cat] = []
        grouped[cat].append(rec)

    client = QdrantClient(path=str(QDRANT_DB_DIR))

    for cat, docs in grouped.items():
        collection_name = CATEGORY_TO_COLLECTION.get(cat)
        if not collection_name:
            log.warning("No collection mapping", category=cat)
            continue
        vs = VectorStore(collection_name, client=client)
        for doc in docs:
            text = doc.get("text_clean", "")
            if not text:
                continue
            chunks = chunk_document(text)
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc['id']}_chunk_{i}"
                metadata = {
                    "source_type": doc.get("source", "unknown"),
                    "category": cat,
                    "doc_id": doc["id"],
                    "title": doc.get("title", ""),
                    "answer": doc.get("answer", ""),
                }
                vs.add([chunk_id], [chunk], [metadata])
            log.debug("Indexed", doc_id=doc["id"], chunks=len(chunks))
    log.info("Indexing for architecture A completed")

if __name__ == "__main__":
    main()