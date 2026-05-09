# Система автоматизации технической поддержки (ТП)

Проект реализует две архитектуры обработки обращений:

- **Архитектура А** – классификация запроса (инцидент / рутинная работа) → выбор специализированного векторного хранилища → генерация ответа.
- **Архитектура Б** – единый подход: глобальный поиск, переранжирование документов, генерация ответа.

Обе архитектуры используют общий LLM (Qwen2.5-1.5B-Instruct), векторное хранилище Qdrant и предоставляют REST API для запросов, сравнения и оценки качества.

## Структура проекта

- `api/` – FastAPI приложение (main.py, schemas.py, dependencies.py, middleware.py)
- `arch_a/` – Архитектура A: RB-классификатор, хранилища по категориям, промпты, теггер
  - `classifier.py` – классификатор на основе правил
  - `processor.py` – основной обработчик
  - `prompts.py` – системные промпты
  - `tagger.py` – генерация тегов
  - `text_processor.py` – очистка, извлечение сущностей
- `arch_b/` – Архитектура B: глобальный поиск, RAG
  - `processor.py` – единый обработчик
  - `prompts.py` – единый промпт
  - `rag.py` – генерация с переранжированием
- `evaluation/` – метрики и логирование
  - `BERT_metrics.py` – быстрые метрики (BERTScore)
  - `llm_judge.py` – LLM-судья
  - `collector.py` – сбор результатов
- `shared/` – общие компоненты: конфигурация, LLM, логгер, векторное хранилище
  - `config.py`, `llm.py`, `logger.py`, `vector_store.py`, `moderation.py`
- `scripts/` – скрипты подготовки данных и индексации
  - `prepare_data.py`, `index_arch_a.py`, `index_arch_b.py`, `evaluate_batch.py`
- `data/` – каталог данных (создаётся автоматически)
- `ui_app.py` – Gradio интерфейс для тестирования
- `requirements.txt` – зависимости

## Установка

```bash
python -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate         # Windows
pip install -r requirements.txt
```

## Подготовка данных

- Поместите файлы для обучения в data/raw/: и измените их название в prepare_data.py
- Запустите обработку и индексацию:

```bash
python scripts/prepare_data.py
python scripts/index_arch_a.py
python scripts/index_arch_b.py
```

## Запуск API
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000

uvicorn api.main:app --reload
# В другом окне
python ui_app.py
```

## Использование

- POST /query – отправить запрос в выбранную архитектуру.
- POST /compare – сравнить ответы архитектур A и B.
- POST /evaluate – выполнить обработку + фоновый расчёт LLM-метрик.

### Интерактивный интерфейс: python ui_app.py → http://localhost:7860

## Примечания
Векторное хранилище: Qdrant с эмбеддингами intfloat/multilingual-e5-small.

LLM: Qwen2.5-1.5B-Instruct (по умолчанию на CPU). Для GPU настройте параметры в shared/config.py.

Метрики answer_relevancy и context_precision вычисляются через BERTScore; дополнительные LLM-оценки (faithfulness и др.) сохраняются в фоновом режиме.