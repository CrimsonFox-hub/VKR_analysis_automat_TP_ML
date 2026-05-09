import json
import time
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from arch_a.processor import ArchAProcessor
from arch_b.processor import ArchBProcessor
from evaluation.llm_judge import LLMJudge
from shared.config import DATA_DIR
from shared.logger import get_logger


log = get_logger(__name__)

# Массовая проверка тестовых запросов
# По сути именно так и должно работать на проде

def main():
    
    test_path = DATA_DIR / "test_queries_100.json"
    if not test_path.exists():
        log.error("Файл не найден")
        return

    with open(test_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    proc_a = ArchAProcessor()
    proc_b = ArchBProcessor()
    judge = LLMJudge()

    results = []
    for item in queries:
        text = item["text"]
        arch = item.get("architecture", "A")
        start = time.time()
        if arch == "B":
            result = proc_b.process(text)
        else:
            result = proc_a.process(text)
        elapsed = time.time() - start

        context = result.get("context_chunks", [])
        if not context:
            pass

        faithfulness = judge.faithfulness(result["answer"], "\n".join(context)) if context else None
        answer_rel = judge.answer_relevancy(text, result["answer"])
        context_prec = judge.context_precision(text, context) if context else None

        results.append({
            "query": text,
            "architecture": arch,
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "category": result.get("category"),
            "confidence": result.get("confidence"),
            "p_time": elapsed,
            "faithfulness": faithfulness,
            "answer_relevancy": answer_rel,
            "context_precision": context_prec,
        })
        log.info(f"Processed: {text[:30]}...")

    output_path = DATA_DIR / "batch_evaluation.csv"
    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    log.info(f"Saved to {output_path}")

if __name__ == "__main__":
    main()