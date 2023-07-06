FROM python:3.10-slim

RUN apt-get update

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /model_predictor

COPY ./mlopsmlflow/requirements.txt .
RUN pip install --upgrade wheel setuptools pip
RUN pip install -r requirements.txt

COPY ./mlopsmlflow/src ./src
COPY ./mlopsmlflow/data/model_config ./data/model_config
COPY ./mlopsmlflow/data/raw_data ./data/raw_data
COPY ./mlopsmlflow/data/train_data ./data/train_data
