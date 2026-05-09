from shared.llm import get_llm
from shared.logger import get_logger
from .prompts import SYSTEM_PROMPTS

log = get_logger(__name__)

class RAG:
    def __init__(self):
        self.llm = get_llm()

    def rerank(self, documents: list[dict], top_k: int = 3) -> list[dict]:
        # Cортировка
        sorted_docs = sorted(documents, key=lambda x: x.get("similarity", 0), reverse=True)
        return sorted_docs[:top_k]

    def generate(self, query: str, documents: list[dict]) -> str:
        context = "\n\n".join(
            f"Источник: {doc['metadata'].get('title', 'Без названия')}\n{doc['text']}"
            for doc in documents
        )
        context = context[:4000]

        system_prompt = SYSTEM_PROMPTS

        user_message = f"Контекст:\n{context}\n\nВопрос: {query}"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        prompt = self.llm.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return self.llm.generate(prompt)