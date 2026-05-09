from pathlib import Path
import os
from dotenv import load_dotenv
load_dotenv()

PROJECT_DIR = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)

# Пути к данным
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
QDRANT_DB_DIR = DATA_DIR / "qdrant_db"
MODELS_DIR = DATA_DIR / "models"
SQLITE_DB_PATH = DATA_DIR / "documents.db"

# Модели
EMBEDDING_MODEL_NAME = "intfloat/multilingual-e5-small"
EMBEDDING_DIM = 384
LLM_MODEL_NAME = os.getenv("QWEN_MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct")

# Параметры чанкинга
CHUNK_MAX_TOKENS = 256
CHUNK_OVERLAP = 25

# Количество извлекаемых чанков
ARCH_A_TOP_K = 10
ARCH_B_TOP_K = 7

EVALUATE_ENABLED = True

# Параметры генерации
GENERATION_CONFIG = {
    "max_new_tokens": 300,
    "temperature": 0.1,
    "top_p": 0.9,
    "repetition_penalty": 1.1,
    "do_sample": True,
}

# Категории для архитектуры А
CATEGORIES = ["incident", "routine"]
CATEGORY_TO_COLLECTION = {
    "incident": "incidents",
    "routine": "routine",
}