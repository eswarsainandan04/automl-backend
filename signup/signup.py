from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
import bcrypt
import psycopg2
from psycopg2 import errorcodes
import uuid
from config import *
from jwt_handler import create_token

router = APIRouter(prefix="/signup", tags=["Signup"])

class SignupRequest(BaseModel):
    fullname: str
    email: str
    password: str
    confirm_password: str


def _hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

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

    hashed = _hash_password(password)
    new_id = str(uuid.uuid4())

    try:
        with psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        database=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD
        ) as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users(id, fullname, email, password) VALUES(%s,%s,%s,%s)",
                    (new_id, fullname, email, hashed)
                )
            conn.commit()
    except psycopg2.IntegrityError as exc:
        if exc.pgcode == errorcodes.UNIQUE_VIOLATION:
            raise HTTPException(status_code=400, detail="Email already exists")
        raise HTTPException(status_code=400, detail="Unable to create account")

    token = create_token(email)

    return {"access_token": token, "user_id": new_id, "fullname": fullname, "email": email}
