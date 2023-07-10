call ../.venv/Scripts/activate.bat
set MLFLOW_TRACKING_URI=http://localhost:5000
set REDIS_ENDPOINT=localhost
set MQTT_ENDPOINT=localhost
set MQTT_PORT=1883
python src/dashboard.py