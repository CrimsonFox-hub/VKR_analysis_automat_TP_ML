import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from shared.vector_store import VectorStore, get_qdrant_client
from shared.config import PROCESSED_DATA_DIR, CHUNK_MAX_TOKENS, CHUNK_OVERLAP
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

    client = get_qdrant_client()
    vs = VectorStore("global_knowledge", client=client)

    for doc in records:
        text = doc.get("text_clean", "")
        if not text:
            continue
        chunks = chunk_document(text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['id']}_chunk_{i}"
            metadata = {
                "source_type": doc.get("source", "unknown"),
                "category": doc.get("category", "routine"),
                "doc_id": doc["id"],
                "title": doc.get("title", ""),
            }
            vs.add([chunk_id], [chunk], [metadata])
    log.info("Indexing for architecture B completed")

if __name__ == "__main__":
    main()