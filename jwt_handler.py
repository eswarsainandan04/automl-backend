from jose import jwt
from datetime import datetime, timedelta

SECRET_KEY = "secretkey123"
ALGORITHM = "HS256"

def create_token(email, user_id=None):

    payload = {
        "sub": email,
        "exp": datetime.utcnow() + timedelta(hours=2)
    }
    if user_id is not None:
        payload["user_id"] = str(user_id)

    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)