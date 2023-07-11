import logging
import os
from pathlib import Path

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

class AppPath:
    #ROOT_DIR = Path(".").absolute().parents[0]
    ROOT_DIR = Path(".")
    DATA_DIR = ROOT_DIR / "data"
    REQUEST_DATA_DIR = ROOT_DIR / "data" / "captured"
    CONFIG_DIR = ROOT_DIR / "config"
    # store configs for deployments
    MODEL_CONFIG_DIR = CONFIG_DIR
    # store problem config 
    PROB_CONFIG_DIR = CONFIG_DIR
    # store drifts reports
    REPORTS_DIR = ROOT_DIR / "reports"

AppPath.MODEL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
AppPath.PROB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
AppPath.REQUEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

class AppConfig:
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    REDIS_ENDPOINT = os.environ.get("REDIS_ENDPOINT", "127.0.0.1")
    MQTT_ENDPOINT = os.environ.get("MQTT_ENDPOINT", "127.0.0.1")
    MQTT_PORT = os.environ.get("MQTT_PORT", "1883")
    MLFLOW_MODEL_PREFIX = "model"