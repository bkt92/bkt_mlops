FROM python:3.11.1-slim

WORKDIR /mlflow/

RUN pip install --no-cache-dir mlflow==2.3.2
