from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import psycopg2

from signup.signup import router as signup_router
from login.login import router as login_router
from data_ingestion.upload_files import router as upload_router
from data_ingestion.etl_router import router as etl_router
from overview import router as overview_router
from analytics.router import router as analytics_router
from history import router as history_router
from config import *

app = FastAPI()

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