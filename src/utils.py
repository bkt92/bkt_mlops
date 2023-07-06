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
    MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI")
    REDIS_ENDPOINT = os.environ.get("REDIS_ENDPOINT")
    MLFLOW_MODEL_PREFIX = "model"
