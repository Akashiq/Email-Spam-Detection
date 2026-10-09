# ---------- Serving image (also reused for training / monitoring jobs) ----------
FROM python:3.11-slim

# REQUIREMENTS lets docker-compose build a "tools" variant that also contains MLflow
ARG REQUIREMENTS=requirements-api.txt

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# install deps first so Docker caches this layer until requirements change
COPY requirements.txt requirements-api.txt ./
RUN pip install -r ${REQUIREMENTS}

COPY app ./app
COPY src ./src
COPY scripts ./scripts
COPY models ./models

RUN mkdir -p /app/logs /app/reports /app/data

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8000/health').status==200 else 1)"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
