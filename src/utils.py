import logging
import os
from pathlib import Path

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

class AppPath:
    ROOT_DIR = Path(".")
    DATA_DIR = ROOT_DIR / "data"
    CONFIG_DIR = ROOT_DIR / "config"
    # store configs for deployments
    MODEL_CONFIG_DIR = CONFIG_DIR / "model_config"
    # store problem config 
    PROB_CONFIG_DIR = CONFIG_DIR / "problem_config"

AppPath.MODEL_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
AppPath.PROB_CONFIG_DIR.mkdir(parents=True, exist_ok=True)

class AppConfig:
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    REDIS_ENDPOINT = os.environ.get("REDIS_ENDPOINT", "127.0.0.1")
    MEMCACHED_ENDPOINT = os.environ.get("MEMCACHED_ENDPOINT", "127.0.0.1")
    MQTT_ENDPOINT = os.environ.get("MQTT_ENDPOINT", "127.0.0.1")
    CACHE_BACKEND = os.environ.get("CACHE_BACKEND")
    CACHE_REQUEST = os.environ.get("SET_CACHE_REQUEST")
    REMOTE_LOGGING = os.environ.get("REMOTE_LOGGING")
    MLFLOW_MODEL_PREFIX = "model"
    COMPILE_MODEL = os.environ.get("COMPILE_MODEL", True)
