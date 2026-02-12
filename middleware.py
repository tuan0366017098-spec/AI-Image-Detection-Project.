import time
import logging
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)


class LoggingMiddleware(BaseHTTPMiddleware):
    def __init__(self, app: ASGIApp):
        super().__init__(app)

    async def dispatch(self, request: Request, call_next):
        start_time = time.time()

        # Log request
        client_host = request.client.host if request.client else "unknown"
        logger.info(f"📥 Incoming request: {request.method} {request.url} from {client_host}")

        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"❌ Request failed: {request.method} {request.url} - {str(e)}")
            raise

        # Log response
        process_time = time.time() - start_time
        logger.info(
            f"📤 Response: {request.method} {request.url} "
            f"Status: {response.status_code} "
            f"Time: {process_time:.3f}s"
        )

        # Add headers
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Server"] = "AI-Image-Detector"

        return response


# For backward compatibility
async def log_requests_middleware(request: Request, call_next):
    start_time = time.time()

    # Log request
    client_host = request.client.host if request.client else "unknown"
    logger.info(f"📥 Incoming request: {request.method} {request.url} from {client_host}")

    response = await call_next(request)

    # Log response
    process_time = time.time() - start_time
    logger.info(
        f"📤 Response: {request.method} {request.url} "
        f"Status: {response.status_code} "
        f"Time: {process_time:.3f}s"
    )

    response.headers["X-Process-Time"] = str(process_time)
    return response