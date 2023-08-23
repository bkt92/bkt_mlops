import sys
import os
#import glob
import logging
sys.path.append('./src')
import pathlib
import yaml
import mlflow
import lleaves
from utils import AppPath
from utils import AppConfig
from problem_config import create_prob_config
#import numpy as np
# Preload numba function and cached
from drift_detector import ks_drift_detect_async
#import daal4py as d4p
import treelite
import tl2cgen

config_path = {}
#config_path = glob.glob(str(AppPath.MODEL_CONFIG_DIR / '*.yaml')) # For auto load the config files
config_path[1] = (AppPath.MODEL_CONFIG_DIR / "phase-3_prob-1.yaml").as_posix()
config_path[2] = (AppPath.MODEL_CONFIG_DIR / "phase-3_prob-2.yaml").as_posix()

def init_startup(config_file_path):

    with open(config_file_path, "r") as f:
        config = yaml.safe_load(f)
    logging.info(f"model-config: {config}")

    mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)

    prob_config = create_prob_config(
        config["phase_id"], config["prob_id"]
    )

    # load model from mlflow tracker
    logging.info("Loading models")
    model_uri = str(pathlib.Path(
        "models:/", config["model_name"], str(config["model_version"])
    ).as_posix())
    model = mlflow.pyfunc.load_model(model_uri)

    # Init model paths
    if config["model_type"]=='lgbm':
        model_path = prob_config.data_path / f'{config["phase_id"]}_{config["prob_id"]}_lgbm.txt'
        #model_classes_path = prob_config.data_path / f'{config["phase_id"]}_{config["prob_id"]}_classes.npy'
        compiled_model_path = prob_config.data_path / f'{config["phase_id"]}_{config["prob_id"]}_llvm_compiled.model'
    if config["model_type"]=='xgb':
        model_path = prob_config.data_path / f'{config["phase_id"]}_{config["prob_id"]}_xgb.model'
        #model_classes_path = prob_config.data_path / f'{config["phase_id"]}_{config["prob_id"]}_classes.npy'
        compiled_model_path = prob_config.data_path / f'{config["phase_id"]}_{config["prob_id"]}_xgb_compiled.so'

    logging.info("Delete old files")
    # Delete old files
    if os.path.exists(model_path):
        os.remove(model_path)
    #if os.path.exists(model_classes_path):
    #    os.remove(model_classes_path)
    if os.path.exists(compiled_model_path):
        os.remove(compiled_model_path)

    if config["model_type"]=='lgbm':
        model._model_impl.booster_.save_model(filename=model_path)
        #np.save(model_classes_path, model._model_impl.classes_)
        if config["compile"] == 'true':
            logging.info("Compiling models")
            llvm_model = lleaves.Model(model_file=model_path)
            llvm_model.compile(cache=compiled_model_path)
    
    if config["model_type"]=='xgb':
        model._model_impl.save_model(model_path)
        if config["compile"] == 'true':
            treelite_model = treelite.Model.from_xgboost(model._model_impl._Booster)
            tl2cgen.export_lib(treelite_model, toolchain="gcc", libpath=compiled_model_path, params={})

    logging.info("Sucess Loading and Compiling Models")

    return True

if __name__ == "__main__":
    init_startup(config_path[1])
    init_startup(config_path[2])