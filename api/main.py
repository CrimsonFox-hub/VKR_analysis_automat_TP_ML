# api/main.py
from contextlib import asynccontextmanager
import asyncio
import atexit
import datetime
from pathlib import Path

from fastapi import BackgroundTasks
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks

from .schemas import QueryRequest, QueryResponse, CompareResponse, EvaluateResponse
from .dependencies import get_arch_a_processor, get_arch_b_processor
from .middleware import log_requests
from arch_a.processor import ArchAProcessor
from arch_b.processor import ArchBProcessor
from shared.logger import get_logger
from shared.config import DATA_DIR
from shared.llm import get_llm
from evaluation.collector import ResultCollector
from evaluation.llm_judge import LLMJudge

# Логи
collector = ResultCollector(DATA_DIR / "query_logs", "live")
logger = get_logger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Запуск...")
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, get_llm)
    yield
    collector.save()
    atexit.register(collector.save)
    logger.info("Остановка...")
    
# Метрики
judge = LLMJudge()

app = FastAPI(title="ТП автоматизация", version="1.0", lifespan=lifespan)
app.middleware("http")(log_requests)

# Блок формирования/записи вопроса
@app.post("/query", response_model=QueryResponse)
async def query(
    req: QueryRequest,
    arch_a: ArchAProcessor = Depends(get_arch_a_processor),
    arch_b: ArchBProcessor = Depends(get_arch_b_processor),
    log_results: bool = True
):
    if not req.text:
        raise HTTPException(status_code=400, detail="Текст запроса не может быть пустым")

    if req.architecture == "A":
        result = arch_a.process(req.text)
    elif req.architecture == "B":
        result = arch_b.process(req.text)
    else:
        result = arch_a.process(req.text)

    if log_results:
        collector.record(result)
    return QueryResponse(**result)

#  Блок Сравнения подходов
@app.post("/compare", response_model=CompareResponse)
async def compare_architectures(
    req: QueryRequest,
    arch_a: ArchAProcessor = Depends(get_arch_a_processor),
    arch_b: ArchBProcessor = Depends(get_arch_b_processor),
):
    if not req.text:
        raise HTTPException(status_code=400, detail="Текст запроса не может быть пустым")

    result_a = arch_a.process(req.text)
    result_b = arch_b.process(req.text)
    return CompareResponse(query=req.text, architecture_a=result_a, architecture_b=result_b)

# Блок оценки результатов
@app.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(
    req: QueryRequest,
    background_tasks: BackgroundTasks,
    arch_a: ArchAProcessor = Depends(get_arch_a_processor),
    arch_b: ArchBProcessor = Depends(get_arch_b_processor),
):
    if not req.text:
        raise HTTPException(status_code=400, detail="Текст запроса не может быть пустым")

    if req.architecture == "B":
        result = arch_b.process(req.text)
    else:
        result = arch_a.process(req.text)

    collector.record(result)

    # Фоновое вычисление LLM-метрик
    async def compute_llm_metrics():
        from evaluation.llm_judge import LLMJudge
        judge = LLMJudge()
        context = result.get("context_chunks", [])
        if not context:
            try:
                if req.architecture != "B":
                    vs = arch_a.vector_stores.get(result.get("category"), arch_a.vector_stores.get("routine"))
                    if vs:
                        search_res = vs.search(req.text, top_k=3)
                        context = [r["text"] for r in search_res]
                        
                if req.architecture == "B":
                    vs = arch_b.vector_store
                else:
                    vs = arch_a.vector_stores.get(result.get("category"), arch_a.vector_stores.get("routine"))
                if vs:
                    search_res = vs.search(req.text, top_k=3)
                    context = [r["text"] for r in search_res]
            except Exception:
                context = []

        faithfulness = judge.faithfulness(result["answer"], "\n".join(context))
        llm_rel = judge.answer_relevancy(req.text, result["answer"])
        llm_prec = judge.context_precision(req.text, context)

        llm_record = result.copy()
        llm_record["faithfulness"] = faithfulness
        llm_record["answer_relevancy"] = llm_rel
        llm_record["context_precision"] = llm_prec
        llm_record["timestamp"] = datetime.now().isoformat()
        collector.record(llm_record)
        logger.info("Метрики записаны", query=req.text[:30])

    #Досчет в фоновом режиме, если не нужен вывод пользователям при запросе
    background_tasks.add_task(compute_llm_metrics)

    return EvaluateResponse(**result)

# Блок проверки работоспособности
@app.get("/health")
async def health():
    return {"status": "ОК"}