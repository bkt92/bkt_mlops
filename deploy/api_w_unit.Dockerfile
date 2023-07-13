FROM unit:1.30.0-python3.11

RUN apt-get update
RUN apt-get install libgomp1 mosquitto apt-utils -y

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
WORKDIR /model_predictor

COPY requirements.txt .
RUN pip install --upgrade wheel setuptools pip
RUN pip install -r requirements.txt

COPY ./deploy/mosquitto.conf /etc/mosquitto/mosquitto.conf
COPY ./src ./src
COPY ./data ./data
COPY ./config ./config
COPY ./deploy/mosquitto.sh .
RUN chmod +x mosquitto.sh
COPY ./deploy/config.json /docker-entrypoint.d/config.json
COPY ./deploy/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh
RUN chmod +x /usr/local/bin/docker-entrypoint.sh
