import json
import re
from typing import Dict, List
from shared.llm import get_llm
from shared.logger import get_logger

# Вычисление метрик LLM judge
log = get_logger(__name__)

class LLMJudge:
    def __init__(self):
        self.llm = get_llm()

    def _call_judge(self, prompt: str) -> float:
        try:
            raw = self.llm.generate(prompt)
            match = re.search(r'\b([1-5])\b', raw)
            if match:
                return float(match.group(1))
            return 0.0
        except Exception as e:
            log.exception("LLM judge упало с ошибкой")
            return 0.0

    def faithfulness(self, answer: str, context: str) -> float:
        prompt = f"""Оцени, насколько все фактические утверждения в ответе основаны на предоставленном контексте.
                    Выведи только число от 1 до 5, где 1 – множество галлюцинаций, 5 – полное подтверждение контекстом.
                    Контекст: {context[:3500]}
                    Ответ: {answer[:1000]}
                    Оценка:"""
        return self._call_judge(prompt)

    def answer_relevancy(self, query: str, answer: str) -> float:
        prompt = f"""Оцени, насколько ответ семантически соответствует запросу пользователя.
                    Выведи только число от 1 до 5, где 1 – совершенно не соответствует, 5 – полностью соответствует.
                    Запрос: {query[:1000]}
                    Ответ: {answer[:1000]}
                    Оценка:"""
        return self._call_judge(prompt)

    def context_precision(self, query: str, context_chunks: List[str]) -> float:
        scores = []
        for i, chunk in enumerate(context_chunks[:5]):
            prompt = f"""Оцени, насколько приведённый фрагмент контекста релевантен запросу пользователя.
                        Выведи только число от 1 до 5, где 1 – нерелевантен, 5 – полностью релевантен.
                        Запрос: {query[:500]}
                        Фрагмент контекста: {chunk[:500]}
                        Оценка:"""
            s = self._call_judge(prompt)
            scores.append(s)
        return sum(scores) / len(scores) if scores else 0.0