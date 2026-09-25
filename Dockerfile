# syntax=docker/dockerfile:1

# --- API (slim) -------------------------------------------------------------
FROM python:3.11-slim AS api

WORKDIR /opt/poster-eval

RUN useradd --create-home --uid 1000 app \
    && mkdir -p /opt/poster-eval/data/processed \
    && chown -R app:app /opt/poster-eval

COPY requirements-api.txt /opt/poster-eval/requirements-api.txt
RUN pip install --no-cache-dir -r /opt/poster-eval/requirements-api.txt

COPY src/__init__.py /opt/poster-eval/src/__init__.py
COPY src/api /opt/poster-eval/src/api

ENV PYTHONPATH=/opt/poster-eval \
    POSTEREVAL_HOME=/opt/poster-eval \
    POSTEREVAL_DATA_DIR=/opt/poster-eval/data/processed \
    POSTEREVAL_DB_PATH=/opt/poster-eval/data/processed/postereval.db

USER app
EXPOSE 8000
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]

# --- Airflow (SequentialExecutor) -------------------------------------------
FROM apache/airflow:2.9.3-python3.11 AS airflow

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /opt/poster-eval/data/processed \
    && chown -R airflow:root /opt/poster-eval

USER airflow
COPY --chown=airflow:root requirements.txt /opt/poster-eval/requirements.txt
RUN pip install --no-cache-dir -r /opt/poster-eval/requirements.txt

ENV PYTHONPATH=/opt/poster-eval \
    POSTEREVAL_HOME=/opt/poster-eval
WORKDIR /opt/airflow
