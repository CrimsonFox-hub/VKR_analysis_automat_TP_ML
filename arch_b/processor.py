import time
from shared.vector_store import VectorStore
from .rag import RAG
from shared.config import ARCH_B_TOP_K
from shared.logger import get_logger
from evaluation.BERT_metrics import BERTMetrics

log = get_logger(__name__)

class ArchBProcessor:
    def __init__(self):
        self.vector_store = VectorStore("global_knowledge")
        self.rag = RAG()
        self.metrics = BERTMetrics()

    def process(self, query: str) -> dict:
        start = time.time()
        # Поиск контекста
        results = self.vector_store.search(query, top_k=ARCH_B_TOP_K)
        # Ранжирование
        top_docs = self.rag.rerank(results, top_k=3)
        context_chunks = [r["text"] for r in results]
        # Генерация ответа
        answer = self.rag.generate(query, top_docs)
        # Источники
        sources = list(set(
            doc["metadata"].get("doc_id", "")
            for doc in top_docs
            if doc["metadata"].get("doc_id")
        ))

        p_time = time.time() - start
        
        rel = self.metrics.answer_relevancy(query, answer)
        context_prec = self.metrics.context_precision(query, context_chunks)
        
        return {
            "answer": answer,
            "sources": sources,
            "architecture_used": "B",
            "p_time": p_time,
            "category": None,
            "confidence": None,
            "tags": None,
            "context_chunks": context_chunks,
            "answer_precision": rel["precision"],
            "answer_recall": rel["recall"],
            "answer_f1": rel["f1"],
            "context_precision": context_prec,
        }