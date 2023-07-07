FROM python:3.11.1-slim

RUN apt-get update
RUN apt-get install libgomp1

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /model_predictor

COPY requirements.txt .
RUN pip install --upgrade wheel setuptools pip
RUN pip install -r requirements.txt

COPY ./src ./src
COPY ./data ./data
COPY ./config ./config
COPY ./deploy/entrypoint.sh /
RUN chmod +x /entrypoint.sh
ENTRYPOINT ["/entrypoint.sh"]
