# Image légère, multi-arch (linux/amd64 et arm64)
FROM python:3.11-slim AS base

# Variables d'env Python
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    OLLAMA_HOST=http://host.docker.internal:11434

# Dépendances système minimales
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copier requirements en premier pour profiter du cache Docker
COPY requirements.txt ./
RUN pip install -r requirements.txt && \
    pip install pymupdf pdfplumber

# Copier le reste du code
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY data/index/ ./data/index/
COPY .streamlit/ ./.streamlit/

# Port exposé par Streamlit
EXPOSE 8501

# Healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# Lancer l'app
CMD ["streamlit", "run", "src/app.py", \
     "--server.port=8501", \
     "--server.address=0.0.0.0", \
     "--server.enableCORS=false", \
     "--server.enableXsrfProtection=false"]
