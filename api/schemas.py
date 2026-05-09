from pydantic import BaseModel
from typing import Optional, Literal, List

# Схемы запросов

# Архитектура
class QueryRequest(BaseModel):
    text: str
    architecture: Optional[Literal["A", "B"]] = None

# Поля
class QueryResponse(BaseModel):
    answer: str
    sources: list
    architecture_used: str
    p_time: float
    category: Optional[str] = None
    confidence: Optional[float] = None
    tags: Optional[list] = None
    answer_precision: Optional[float] = None
    answer_recall: Optional[float] = None
    answer_f1: Optional[float] = None
    context_precision: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_relevance: Optional[float] = None

# Объединение вопроса к обеим архитектурам  
class CompareResponse(BaseModel):
    query: str
    architecture_a: QueryResponse
    architecture_b: QueryResponse
    
# Вывод оценок
class EvaluateResponse(QueryResponse):
    faithfulness: Optional[float] = None
    answer_relevancy: Optional[float] = None
    context_relevance: Optional[float] = None