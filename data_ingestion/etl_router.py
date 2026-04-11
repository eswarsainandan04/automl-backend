"""
ETL from external database servers.

POST /etl/connect  — test connection and return list of tables
POST /etl/extract  — export selected tables as CSV, upload to storage,
                     and trigger the preprocessing pipeline
"""

import csv
import io
import os
import sys
import threading
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

_BACKEND_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _BACKEND_ROOT not in sys.path:
    sys.path.insert(0, _BACKEND_ROOT)

from data_ingestion.upload_files import _pipeline_status, get_current_user_id
from data_preprocessing.supabase_storage import upload_file
from pipeline import PIPELINE_STEPS, run_preprocessing

router = APIRouter(prefix="/etl", tags=["ETL"])

# ─── Default ports ──────────────────────────────────────────────────────────
_DEFAULT_PORT = {"mysql": 3306, "postgresql": 5432, "sqlserver": 1433}


# ─── Pydantic models ─────────────────────────────────────────────────────────
class ConnectRequest(BaseModel):
    server_type: str          # "mysql" | "postgresql" | "sqlserver"
    host: str
    port: Optional[int] = None
    username: str
    password: str
    database: str


class ExtractRequest(BaseModel):
    server_type: str
    host: str
    port: Optional[int] = None
    username: str
    password: str
    database: str
    tables: List[str]


# ─── Connection factory ───────────────────────────────────────────────────────
def _get_connection(
    server_type: str, host: str, port: Optional[int],
    username: str, password: str, database: str,
):
    resolved_port = port or _DEFAULT_PORT.get(server_type.lower())
    stype = server_type.lower()

    if stype == "postgresql":
        try:
            import psycopg2
        except ImportError:
            raise HTTPException(500, "psycopg2 is not installed")
        try:
            return psycopg2.connect(
                host=host, port=resolved_port,
                database=database, user=username, password=password,
                connect_timeout=10,
            )
        except Exception as exc:
            raise HTTPException(400, f"PostgreSQL connection failed: {exc}")

    if stype == "mysql":
        try:
            import mysql.connector
        except ImportError:
            raise HTTPException(
                500,
                "mysql-connector-python is not installed. "
                "Run: pip install mysql-connector-python",
            )
        try:
            return mysql.connector.connect(
                host=host, port=resolved_port,
                database=database, user=username, password=password,
                connection_timeout=10,
            )
        except Exception as exc:
            raise HTTPException(400, f"MySQL connection failed: {exc}")

    if stype == "sqlserver":
        try:
            import pymssql
        except ImportError:
            raise HTTPException(
                500,
                "pymssql is not installed. "
                "Run: pip install pymssql",
            )
        try:
            return pymssql.connect(
                server=host, port=str(resolved_port),
                database=database, user=username, password=password,
                timeout=10,
            )
        except Exception as exc:
            raise HTTPException(400, f"SQL Server connection failed: {exc}")

    raise HTTPException(400, f"Unsupported server type: '{server_type}'. "
                             "Use 'mysql', 'postgresql', or 'sqlserver'.")


# ─── Helpers ─────────────────────────────────────────────────────────────────
def _list_tables(conn, server_type: str, database: str) -> List[str]:
    """Return all base table names from the target database (via information_schema)."""
    cur = conn.cursor()
    stype = server_type.lower()
    if stype == "postgresql":
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE' "
            "ORDER BY table_name"
        )
    elif stype == "mysql":
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = %s AND table_type = 'BASE TABLE' "
            "ORDER BY table_name",
            (database,),
        )
    elif stype == "sqlserver":
        cur.execute(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_type = 'BASE TABLE' "
            "ORDER BY table_name"
        )
    tables = [row[0] for row in cur.fetchall()]
    cur.close()
    return tables


def _table_to_csv(conn, table_name: str, server_type: str) -> bytes:
    """
    Export a *validated* table to CSV bytes.
    Table name is quoted properly per SQL dialect to prevent injection.
    """
    cur = conn.cursor()
    stype = server_type.lower()

    if stype == "postgresql":
        from psycopg2 import sql as pgsql
        cur.execute(pgsql.SQL("SELECT * FROM {}").format(pgsql.Identifier(table_name)))
    elif stype == "mysql":
        # Strip backticks before re-quoting — table name already validated
        safe = f"`{table_name.replace('`', '')}`"
        cur.execute(f"SELECT * FROM {safe}")
    elif stype == "sqlserver":
        safe = f"[{table_name.replace(']', '')}]"
        cur.execute(f"SELECT * FROM {safe}")

    columns = [desc[0] for desc in cur.description]
    rows = cur.fetchall()
    cur.close()

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(columns)
    for row in rows:
        writer.writerow(["" if v is None else str(v) for v in row])
    return buf.getvalue().encode("utf-8")


# ─── Endpoints ───────────────────────────────────────────────────────────────
@router.post("/connect")
def etl_connect(
    req: ConnectRequest,
    user_id: str = Depends(get_current_user_id),
):
    """Connect to a database server and return the list of available tables."""
    conn = _get_connection(
        req.server_type, req.host, req.port,
        req.username, req.password, req.database,
    )
    try:
        tables = _list_tables(conn, req.server_type, req.database)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(400, f"Failed to list tables: {exc}")
    finally:
        conn.close()
    return {"tables": tables}


@router.post("/extract")
def etl_extract(
    req: ExtractRequest,
    user_id: str = Depends(get_current_user_id),
):
    """
    Validate, extract, and upload the selected tables as CSVs, then
    trigger the preprocessing pipeline exactly like a normal file upload.
    """
    if not req.tables:
        raise HTTPException(400, "No tables selected")

    conn = _get_connection(
        req.server_type, req.host, req.port,
        req.username, req.password, req.database,
    )

    try:
        # Validate requested table names against actual schema (SQL-injection guard)
        allowed = set(_list_tables(conn, req.server_type, req.database))
        unknown = [t for t in req.tables if t not in allowed]
        if unknown:
            raise HTTPException(400, f"Unknown table(s): {unknown}")

        session_id = datetime.now().strftime("%Y%m%d%H%M%S")
        uploaded: List[str] = []

        for table_name in req.tables:
            csv_bytes = _table_to_csv(conn, table_name, req.server_type)
            filename  = f"{table_name}.csv"
            path      = f"input/{user_id}/{session_id}/{filename}"
            upload_file(path, csv_bytes, "text/csv")
            uploaded.append(filename)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, f"Extraction failed: {exc}")
    finally:
        conn.close()

    # Initialise pipeline status and fire background thread
    _pipeline_status[session_id] = {
        "user_id":       user_id,
        "session_id":    session_id,
        "status":        "running",
        "files":         uploaded,
        "steps":         [{"name": s, "status": "pending", "error": None} for s in PIPELINE_STEPS],
        "error":         None,
        "started_at":    datetime.utcnow().isoformat(),
        "completed_at":  None,
    }
    t = threading.Thread(
        target=run_preprocessing,
        args=(user_id, session_id, _pipeline_status[session_id]),
        daemon=True,
    )
    t.start()

    return {"session_id": session_id, "user_id": user_id, "files": uploaded}
