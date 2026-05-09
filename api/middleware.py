import time
import logging
from fastapi import Request
from shared.logger import get_logger

logger = get_logger(__name__)

# Запрс по логированию
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - {response.status_code} - {process_time:.4f}s"
    )
    return response