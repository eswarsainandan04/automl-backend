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

@router.post("/")
def signup(data: SignupRequest):
    fullname, email, password = data.fullname, data.email, data.password

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