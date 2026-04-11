from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import psycopg2
import uuid
from passlib.context import CryptContext
from config import *
from jwt_handler import create_token

router = APIRouter(prefix="/signup", tags=["Signup"])

pwd = CryptContext(schemes=["bcrypt"], deprecated="auto")

class SignupRequest(BaseModel):
    fullname: str
    email: str
    password: str
    confirm_password: str

@router.post("/")
def signup(data: SignupRequest):
    fullname = data.fullname.strip()
    email = data.email.strip().lower()
    password = data.password
    confirm_password = data.confirm_password

    if not fullname:
        raise HTTPException(status_code=400, detail="Full name is required")

    if not email:
        raise HTTPException(status_code=400, detail="Email is required")

    if len(password) < 8:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 8 characters.",
        )

    if password != confirm_password:
        raise HTTPException(
            status_code=400,
            detail="Password and confirm password do not match.",
        )

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

    hashed = pwd.hash(password)

    new_id = str(uuid.uuid4())
    try:
        cur.execute(
            "INSERT INTO users(id, fullname, email, password) VALUES(%s,%s,%s,%s)",
            (new_id, fullname, email, hashed)
        )
        conn.commit()
    except Exception:
        raise HTTPException(status_code=400, detail="Email already exists")

    cur.close()
    conn.close()

    token = create_token(email)

    return {"access_token": token, "user_id": new_id, "fullname": fullname, "email": email}