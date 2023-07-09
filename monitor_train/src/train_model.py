import sys
sys.path.append('./src')
from data_processor import RawDataProcessor
from problem_config import create_prob_config
from utils import AppPath, AppConfig
from sklearn.metrics import roc_auc_score
import datetime
import pandas as pd
import numpy as np
import pickle

from mlflow.models.signature import infer_signature
import mlflow
import numpy as np

def log_model_to_tracker(model, test_x, metrics, desc, experiment_id="phase-2_prob-1_lgbm"):
    mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(experiment_id)
    mlflow.start_run(description=desc)
    mlflow.log_metrics(metrics)
    mlflow.log_params(model.get_params())
    predictions = model.predict(test_x)
    signature = infer_signature(test_x.astype(np.float64), predictions)
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path=AppConfig.MLFLOW_MODEL_PREFIX,
        signature=signature,
        pip_requirements ='src/requirements.txt'
        #registered_model_name="phase-1_prob-1_model-1"
    )
    experimentid = mlflow.active_run().info.run_id
    mlflow.end_run()
    return experimentid

from lightgbm import LGBMClassifier

def train(train_x, train_y, test_x, test_y, **args):
    model = LGBMClassifier(objective="binary", random_state=123)
    model.fit(train_x, train_y, verbose=False)
    predictions = model.predict_proba(test_x.astype(np.float64))[:,1]
    metrics = {"test_auc": roc_auc_score(test_y, predictions)}
    return model, metrics

def process_data(prob_config):
    training_data = pd.read_parquet(prob_config.raw_data_path)
    training_data, category_index = RawDataProcessor.build_category_features(
                training_data, prob_config.categorical_cols
            )
    target_col = prob_config.target_col
    train_x = training_data.drop([target_col], axis=1)
    train_y = training_data[[target_col]]
    # Store the category_index
    with open(prob_config.category_index_path, "wb") as f:
        pickle.dump(category_index, f)
    sample = training_data.sample(1000)
    test_x = sample.drop([target_col], axis=1)
    test_y = sample[[target_col]]
    return train_x, train_y, test_x, test_y

if __name__=="__main__":

    prob_config = create_prob_config("phase-2", "prob-1")
    current_dt = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
    run_description = f"""
    ### Header {current_dt}
    LGBM model, Model for PROB1
    Model: LGBM
        """
    log_model_to_tracker(model, metrics, run_description)