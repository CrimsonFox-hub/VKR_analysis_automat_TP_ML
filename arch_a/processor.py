import time
from shared.vector_store import VectorStore
from shared.llm import LLM, get_llm
from shared.config import CATEGORY_TO_COLLECTION, ARCH_A_TOP_K
from .classifier import RBClassifier
from .prompts import SYSTEM_PROMPTS
from .tagger import TaskTagger
from shared.logger import get_logger
from evaluation.BERT_metrics import BERTMetrics

log = get_logger(__name__)

class ArchAProcessor:
    def __init__(self):
        self.classifier = RBClassifier()
        self.vector_stores = {}
        for cat, col_name in CATEGORY_TO_COLLECTION.items():
            self.vector_stores[cat] = VectorStore(col_name)
        self.global_store = VectorStore("global_knowledge")
        self.llm = get_llm()
        self.tagger = TaskTagger()
        self.metrics = BERTMetrics()

    def process(self, query: str) -> dict:
        start = time.time()
        # Классификация
        category, confidence, reason = self.classifier.classify(query, "")
        log.info("classification", category=category, confidence=confidence, reason=reason)
        tags = self.tagger.generate_tags(query, category) if self.tagger else ""
        
        # Получение векторного хранилища
        vs = self.vector_stores.get(category)
        if not vs:
            vs = self.vector_stores.get("routine")

        # Поиск контекста
        results = vs.search(query, top_k=ARCH_A_TOP_K)
        results = sorted(results, key=lambda r: r.get("distance", 1))
        results = results[:3]
        context_chunks = [r["text"] for r in results]
        metadatas = [r["metadata"] for r in results]

        # Формирование промпта
        system_prompt = SYSTEM_PROMPTS.get(category, SYSTEM_PROMPTS["routine"])
        context_text = "\n\n---\n\n".join(
            f"[Источник: {m.get('source_type', 'unknown')}, {m.get('title', '')}]\n{chunk}"
            for chunk, m in zip(context_chunks, metadatas)
        )
        user_content = f"Контекст:\n{context_text}\n\nЗапрос пользователя: {query}\n\nОтвет:"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ]
        prompt = self.llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Генерация
        answer = self.llm.generate(prompt)

        # Сбор источников
        sources = list(set(m.get("doc_id", "") for m in metadatas if m.get("doc_id")))

        p_time = time.time() - start
        
        rel = self.metrics.answer_relevancy(query, answer)
        context_prec = self.metrics.context_precision(query, context_chunks)
        
        return {
            "answer": answer,
            "sources": sources,
            "architecture_used": "A",
            "p_time": p_time,
            "category": category,
            "confidence": confidence,
            "context_chunks": context_chunks,
            "answer_precision": rel["precision"],
            "answer_recall": rel["recall"],
            "answer_f1": rel["f1"],
            "context_precision": context_prec,
        }