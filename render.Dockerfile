FROM python:3.12-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends \
    libexpat1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app/backend

COPY backend/requirements.txt /tmp/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --only-binary=:all: -r /tmp/requirements.txt

COPY backend/main.py /app/backend/main.py
COPY backend/engine /app/backend/engine
COPY ["backend/data/raw/M20_JezeroCrater_CTXDEM_20m (1).tif", "/app/backend/data/raw/"]
COPY backend/data/raw/jezero_10_scientific_sites.geojson /app/backend/data/raw/

EXPOSE 10000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:10000/api/health', timeout=3)" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000", "--workers", "1", "--limit-concurrency", "2", "--timeout-keep-alive", "15"]
