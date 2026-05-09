import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List
from shared.logger import get_logger

log = get_logger(__name__)

# Полная запись логов

class ResultCollector:
    def __init__(self, output_dir: Path, experiment_name: str):
        self.output_dir = output_dir / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: List[Dict[str, Any]] = []
        self.experiment_name = experiment_name
        self.fields = [
            "timestamp", "query", "architecture_used", "answer", "sources",
            "p_time", "category", "confidence", "tags",
            "faithfulness", "answer_relevancy", "context_precision"
        ]

    def record(self, data: Dict[str, Any]) -> None:
        if 'timestamp' not in data:
            data['timestamp'] = datetime.now().isoformat()
        self.results.append(data)

        csv_path = self.output_dir / f"{self.experiment_name}.csv"
        file_exists = csv_path.exists()
        with open(csv_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.fields, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            writer.writerow(data)
        log.info("result_recorded", query=data.get('query', '')[:50])

    def save(self) -> Path:
        # Сохраняем также JSON
        json_path = self.output_dir / f"{self.experiment_name}.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
        log.info("results_saved", json=str(json_path))
        return json_path