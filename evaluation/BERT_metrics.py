# evaluation/BERT_metrics.py
import torch
from sentence_transformers import SentenceTransformer
from bert_score import BERTScorer
from typing import List

class BERTMetrics:
    def __init__(self):
        self.embedder = SentenceTransformer("intfloat/multilingual-e5-small", device="cpu")
        self.scorer = BERTScorer(lang="ru", device="cpu", batch_size=1)

    def answer_relevancy(self, query: str, answer: str) -> dict:
        if not answer:
            return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
        P, R, F1 = self.scorer.score([answer], [query])
        return {
            "precision": round(P.item(), 4),
            "recall": round(R.item(), 4),
            "f1": round(F1.item(), 4)
        }

    def context_precision(self, query: str, chunks: List[str]) -> float:
        if not chunks:
            return 0.0
        q_emb = self.embedder.encode([query])[0]
        chunk_embs = self.embedder.encode(chunks)
        similarities = torch.nn.functional.cosine_similarity(
            torch.tensor(q_emb).unsqueeze(0), torch.tensor(chunk_embs), dim=1
        )
        return round(similarities.mean().item(), 4)