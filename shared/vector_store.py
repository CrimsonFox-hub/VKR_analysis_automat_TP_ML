import os
os.environ["BITSANDBYTES_NOWELCOME"] = "1"
os.environ["DISABLE_BITSANDBYTES"] = "1"

import uuid
from typing import List, Optional, Dict, Any
from qdrant_client import QdrantClient
from qdrant_client import models as qdrant_models
from sentence_transformers import SentenceTransformer
from .config import QDRANT_DB_DIR, EMBEDDING_MODEL_NAME, EMBEDDING_DIM
from .logger import get_logger

log = get_logger(__name__)

_global_client: Optional[QdrantClient] = None

def get_qdrant_client() -> QdrantClient:
    global _global_client
    if _global_client is None:
        _global_client = QdrantClient(path=str(QDRANT_DB_DIR))
        log.info("Qdrant инициализирован", path=str(QDRANT_DB_DIR))
    return _global_client

def _str_to_uuid(s: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, s))

class VectorStore:
    def __init__(self, collection_name: str, client: Optional[QdrantClient] = None):
        self.client = client if client is not None else get_qdrant_client()
        self.embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME,
            device="cpu"
        )
        self.collection_name = collection_name
        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections().collections
        exists = any(col.name == self.collection_name for col in collections)
        if not exists:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=qdrant_models.VectorParams(
                    size=EMBEDDING_DIM,
                    distance=qdrant_models.Distance.COSINE
                )
            )
            log.info("collection_created", collection=self.collection_name)

    def add(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]):
        embeddings = self.embedding_model.encode(texts).tolist()
        points = []
        for point_id, text, meta, vector in zip(ids, texts, metadatas, embeddings):
            payload = {"text": text, **meta}
            points.append(
                qdrant_models.PointStruct(
                    id=_str_to_uuid(point_id),
                    vector=vector,
                    payload=payload
                )
            )
        self.client.upsert(collection_name=self.collection_name, points=points)
        log.info("vectors_added", collection=self.collection_name, count=len(ids))

    def search(self, query: str, top_k: int = 10) -> List[Dict[str, Any]]:
        query_embedding = self.embedding_model.encode([query]).tolist()[0]
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        docs = []
        for hit in search_result:
            docs.append({
                "id": hit.id,
                "text": hit.payload.get("text", ""),
                "metadata": {k: v for k, v in hit.payload.items() if k != "text"},
                "distance": hit.score,
            })
        return docs

    def count(self) -> int:
        collection_info = self.client.get_collection(collection_name=self.collection_name)
        return collection_info.points_count