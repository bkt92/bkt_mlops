#!/bin/sh

echo "Start mqtt service"
/model_predictor/mosquitto.sh &

echo "Start load and compile models"
cd /model_predictor
python src/init_startup.py