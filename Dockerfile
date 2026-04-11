# AutoGluon currently does not support Python 3.14; keep image on 3.10.
FROM python:3.10.14-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
	PYTHONUNBUFFERED=1 \
	PIP_NO_CACHE_DIR=1 \
	PIP_PREFER_BINARY=1 \
	PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System libraries needed by ML/DB dependencies (psycopg2, pymssql, lightgbm, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	gcc \
	g++ \
	libpq-dev \
	unixodbc-dev \
	freetds-dev \
	libgomp1 \
	curl \
	&& rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip setuptools wheel && \
	pip install -r requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
