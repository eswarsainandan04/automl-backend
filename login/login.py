from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
import psycopg2
from passlib.context import CryptContext
from jose import JWTError
from jose import jwt as jose_jwt
from config import *
from jwt_handler import ALGORITHM, SECRET_KEY, create_token

router = APIRouter(prefix="/login", tags=["Login"])

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

class LoginRequest(BaseModel):
    email: str
    password: str

@router.post("/")
def login(data: LoginRequest):
    email, password = data.email, data.password

    if len(password.encode("utf-8")) > 72:
        raise HTTPException(
            status_code=400,
            detail="Password exceeds bcrypt limit (72 bytes).",
        )

    conn = psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
    )

    cur = conn.cursor()

    cur.execute(
        "SELECT id, fullname, password FROM users WHERE email=%s",
        (email,)
    )

    user = cur.fetchone()

    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    if not pwd.verify(password, user[2]):
        raise HTTPException(status_code=401, detail="Invalid password")

    token = create_token(email, user_id=str(user[0]))

    cur.close()
    conn.close()

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