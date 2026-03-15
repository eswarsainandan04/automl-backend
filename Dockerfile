FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# System packages for postgres/mysql/sqlserver connectors and scientific libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    build-essential \
    gcc \
    g++ \
    libpq-dev \
    freetds-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Miniconda for isolated Python 3.7 AutoGluon runtime
RUN curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/miniconda.sh \
    && bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm -f /tmp/miniconda.sh

ENV PATH=/opt/conda/bin:${PATH}

# Accept Anaconda channel terms for non-interactive CI/CD builds
RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
    && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Use conda-forge for legacy Python 3.7 packages
RUN conda config --add channels conda-forge \
    && conda config --set channel_priority flexible

# Create Python 3.7 env used by pipeline missing-values subprocess
RUN conda create -y -n py37 python=3.7 pip \
    && conda run -n py37 pip install --no-cache-dir \
       autogluon.tabular==0.6.2 \
       boto3 \
       python-dotenv \
    && conda clean -afy

# Switch back to system Python for FastAPI runtime packages.
ENV PATH=/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin

# Main backend runtime (Python 3.12)
RUN python -m pip install --no-cache-dir \
    fastapi \
    uvicorn \
    boto3 \
    python-dotenv \
    mysql-connector-python \
    pymssql \
    psycopg2-binary \
    pandas \
    numpy \
    scipy \
    python-jose[cryptography] \
    passlib==1.7.4 \
    bcrypt==4.0.1 \
    python-multipart \
    google-genai \
    flask

COPY . /app/

# Python path used by pipeline.py for the AutoGluon subprocess
ENV AUTOGLUON_PYTHON=/opt/conda/envs/py37/bin/python
ENV PYTHONPATH=/app

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
