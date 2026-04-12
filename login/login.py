from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
import ast
import bcrypt
import psycopg2
from jose import JWTError
from jose import jwt as jose_jwt
from config import *
from jwt_handler import ALGORITHM, SECRET_KEY, create_token

router = APIRouter(prefix="/login", tags=["Login"])

class LoginRequest(BaseModel):
    email: str
    password: str


def _stored_hash_to_bytes(stored_password: str | bytes) -> bytes:
    if isinstance(stored_password, bytes):
        return stored_password

    raw = str(stored_password).strip()
    if raw.startswith(("b'", 'b"')):
        try:
            parsed = ast.literal_eval(raw)
            if isinstance(parsed, bytes):
                return parsed
        except (ValueError, SyntaxError):
            pass
    return raw.encode("utf-8")

@router.post("/")
def login(data: LoginRequest):
    email = data.email.strip().lower()
    password = data.password

    if not email or not password:
        raise HTTPException(status_code=400, detail="Email and password are required")

    if len(password.encode("utf-8")) > 72:
        raise HTTPException(
            status_code=400,
            detail="Password exceeds bcrypt limit (72 bytes).",
        )

    with psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    ) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT id, fullname, password FROM users WHERE email=%s",
                (email,),
            )
            user = cur.fetchone()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    try:
        password_ok = bcrypt.checkpw(
            password.encode("utf-8"),
            _stored_hash_to_bytes(user[2]),
        )
    except (TypeError, ValueError):
        # Treat malformed/legacy hash formats as invalid credentials.
        raise HTTPException(status_code=401, detail="Invalid password")

    if not password_ok:
        raise HTTPException(status_code=401, detail="Invalid password")

    token = create_token(email, user_id=str(user[0]))

    return {"access_token": token, "user_id": str(user[0]), "fullname": user[1], "email": email}


_security = HTTPBearer()

@router.post("/refresh")
def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(_security)):
    """Issue a fresh token for a still-valid access token (call before it expires)."""
    try:
        payload = jose_jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        email   = payload.get("sub")
        user_id = payload.get("user_id")
        if not email:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        new_token = create_token(email, user_id=user_id)
        return {"access_token": new_token}
    except JWTError:
        raise HTTPException(status_code=401, detail="Token expired or invalid")
