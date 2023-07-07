import sys
import os
import logging
sys.path.append('./src')
import pathlib
import yaml
import mlflow
import lleaves
from utils import AppPath
from utils import AppConfig
from problem_config import create_prob_config
import numpy as np
# Preload numba function and cached
from drift_detector import ks_drift_detect_async

config_path = {}
config_path[1] = (AppPath.MODEL_CONFIG_DIR / "phase-2_prob-1.yaml").as_posix()
config_path[2] = (AppPath.MODEL_CONFIG_DIR / "phase-2_prob-2.yaml").as_posix()

def init_startup(config_file_path):

    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    logging.info(f"model-config: {config}")

    mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)

    prob_config = create_prob_config(
        config["phase_id"], config["prob_id"]
    )

    # Compile model to llvm for faster speed and save to disk
    model_path = prob_config.data_path / f'{config["phase_id"]}_{config["prob_id"]}_lgbm.txt'
    model_classes_path = prob_config.data_path / f'{config["phase_id"]}_{config["prob_id"]}_classes.npy'
    llvm_model_path = prob_config.data_path / f'{config["phase_id"]}_{config["prob_id"]}_llvm_compiled.model'

    logging.info("Delete old files")
    # Delete old models
    if os.path.exists(model_path):
        os.remove(model_path)
    if os.path.exists(model_classes_path):
        os.remove(model_classes_path)
    if os.path.exists(llvm_model_path):
        os.remove(llvm_model_path)

    # load model from mlflow tracker
    logging.info("Loading and compiling models")
    model_uri = str(pathlib.Path(
        "models:/", config["model_name"], str(config["model_version"])
    ).as_posix())
    model = mlflow.pyfunc.load_model(model_uri)
    model._model_impl.booster_.save_model(filename=model_path)
    np.save(model_classes_path, model._model_impl.classes_)
    llvm_model = lleaves.Model(model_file=model_path)
    llvm_model.compile(cache=llvm_model_path)
    logging.info("Sucess Loading and Compiling Models")

    return True

if __name__ == "__main__":
    init_startup(config_path[1])
    init_startup(config_path[2])