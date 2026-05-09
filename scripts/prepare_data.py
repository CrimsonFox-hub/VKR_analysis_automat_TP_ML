import json
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from shared.config import RAW_DATA_DIR, PROCESSED_DATA_DIR
from shared.logger import get_logger
from arch_a.text_processor import TextProcessor
from arch_a.classifier import RBClassifier

#Скрипт первоначального МЛ обучения 

log = get_logger(__name__)

def normalize_category(cat_raw: str) -> str:
    cat_lower = str(cat_raw).strip().lower() if cat_raw and not pd.isna(cat_raw) else ""
    if "инцидент" in cat_lower or "incident" in cat_lower:
        return "incident"
    return "routine"

def load_training_excel(file_path: Path, processor: TextProcessor) -> list:
    df = pd.read_excel(file_path, sheet_name="Sheet1")
    records = []
    for idx, row in df.iterrows():
        text_raw = str(row['text']) if pd.notna(row['text']) else ""
        category_raw = row.get('category', '')
        tags_raw = row.get('tags', '')

        if not text_raw.strip():
            continue

        text_clean = processor.clean_text_for_search(text_raw)
        text_lemmatized = processor.text_cleaner(text_raw)
        category = normalize_category(category_raw)
        tags = [t.strip() for t in str(tags_raw).split(',') if t.strip()] if pd.notna(tags_raw) else []

        record = {
            "id": f"train_{idx+1}",
            "source": "excel_training",
            "text_raw": text_raw,
            "text_clean": text_clean,
            "text_lemmatized": text_lemmatized,
            "category": category,
            "tags": tags,
            "title": f"Обращение {idx+1}",
            "answer": "",
        }
        records.append(record)
    log.info(f"Loaded {len(records)} records from {file_path.name}")
    return records

def load_content_excel(file_path: Path, processor: TextProcessor, classifier: RBClassifier) -> list:
    df = pd.read_excel(file_path)
    records = []
    for idx, row in df.iterrows():
        query = str(row['Описание']) if pd.notna(row['Описание']) else ""
        solution = str(row['Решение']) if pd.notna(row['Решение']) else ""
        if not query or not solution:
            continue

        text_clean = processor.text_cleaner(query)
        category, confidence, _ = classifier.classify(text_clean, "")

        # Доп. метаданные
        doc_number = str(row.get('Главный документ.Номер', '')) if pd.notna(row.get('Главный документ.Номер')) else ''
        doc_date = str(row.get('Главный документ.Дата', '')) if pd.notna(row.get('Главный документ.Дата')) else ''
        department = str(row.get('Наряд.Ответственное организационное подразделение', '')) if pd.notna(row.get('Наряд.Ответственное организационное подразделение')) else ''

        record = {
            "id": f"obf_{idx}",
            "source": "obfuscated_orders_2025",
            "text_raw": query,
            "text_clean": text_clean,
            "category": category,
            "confidence": confidence,
            "tags": [],
            "title": f"Наряд {doc_number}" if doc_number else f"Наряд {idx}",
            "answer": solution,
            "metadata": {
                "doc_number": doc_number,
                "doc_date": doc_date,
                "department": department
            }
        }
        records.append(record)
    log.info(f"Loaded {len(records)} records from {file_path.name}")
    return records

def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    processor = TextProcessor()
    classifier = RBClassifier()

    all_records = []

    # Обучающие примеры
    train_path = RAW_DATA_DIR / "для обучения.xlsx"
    if train_path.exists():
        all_records.extend(load_training_excel(train_path, processor))
    else:
        log.warning(f"Training file not found: {train_path}")

    # Реальные наряды
    orders_path = RAW_DATA_DIR / "obfuscated_наряды_2025.xlsx"
    if orders_path.exists():
        all_records.extend(load_content_excel(orders_path, processor, classifier))
    else:
        log.warning(f"Orders file not found: {orders_path}")

    # Сохранение
    output_path = PROCESSED_DATA_DIR / "records.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    log.info(f"Total records saved: {len(all_records)} to {output_path}")

if __name__ == "__main__":
    main()