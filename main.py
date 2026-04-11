import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import HTTPException as FastAPIHTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import psycopg2

from signup.signup import router as signup_router
from login.login import router as login_router
from data_ingestion.upload_files import router as upload_router
from data_ingestion.etl_router import router as etl_router
from overview import router as overview_router
from analytics.router import router as analytics_router
from history import router as history_router
from automl_router import router as automl_router
from export_model import router as export_router
from config import *

app = FastAPI()
logger = logging.getLogger("Backend")


def _safe_frontend_error_message(path: str, status_code: int) -> str | None:
    normalized_path = (path or "").lower()

    # Dataset-train specific popup text requested by user.
    if "/automl/model-building/run/" in normalized_path:
        return "sorry We are unable process this dataset"

    # Hide technical details for all AutoML flow errors in UI.
    if normalized_path.startswith("/automl/") and status_code >= 400:
        return "Sorry, we are unable to process this request right now."

    # Keep a generic fallback for unhandled 5xx errors elsewhere.
    if status_code >= 500:
        return "Sorry, something went wrong. Please try again."

    return None

# CORS must be added BEFORE routers
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(signup_router)
app.include_router(login_router)
app.include_router(upload_router)
app.include_router(etl_router)
app.include_router(overview_router)
app.include_router(analytics_router)
app.include_router(history_router)
app.include_router(automl_router)
app.include_router(export_router)


@app.exception_handler(FastAPIHTTPException)
async def sanitized_http_exception_handler(request: Request, exc: FastAPIHTTPException):
    safe_message = _safe_frontend_error_message(request.url.path, int(exc.status_code))
    if safe_message is not None:
        logger.warning(
            "Sanitized HTTPException | path=%s | status=%s | detail=%r",
            request.url.path,
            exc.status_code,
            exc.detail,
        )
        return JSONResponse(status_code=exc.status_code, content={"detail": safe_message})

    return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})


@app.exception_handler(Exception)
async def sanitized_unhandled_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception | path=%s", request.url.path)
    safe_message = _safe_frontend_error_message(request.url.path, 500)
    return JSONResponse(
        status_code=500,
        content={"detail": safe_message or "Sorry, something went wrong. Please try again."},
    )

@app.on_event("startup")
def create_tables():
    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users(
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            fullname TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT NOW()
        )
    """)
    conn.commit()
    cur.close()
    conn.close()

@app.get("/")
def root():
    return {"message": "AutoML backend running"}