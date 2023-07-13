import sys
sys.path.append('./src')
from data_processor import DataProcessor
from sklearn.model_selection import train_test_split
from problem_config import create_prob_config
from utils import AppConfig
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
import numpy as np
import pickle

from mlflow.models.signature import infer_signature
import mlflow
import numpy as np
from lightgbm import LGBMClassifier

class ModelTrainer:
    @staticmethod
    def log_model_to_tracker(model, signature, metrics, desc, experiment_id="phase-2_prob-1_lgbm"):
        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_id)
        mlflow.start_run(description=desc)
        mlflow.log_metrics(metrics)
        mlflow.log_params(model.get_params())
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

    @staticmethod
    def train_lgb(phase_id, prob_id, split=False, newdata=None, **args):
        prob_config = create_prob_config(phase_id, prob_id)
        training_data = pd.read_parquet(prob_config.raw_data_path)
        training_data, category_index = DataProcessor.build_category_features(
                    training_data, prob_config.categorical_cols
                )
        target_col = prob_config.target_col
        train, dev = train_test_split(training_data, test_size=prob_config.test_size, random_state=prob_config.random_state)
        # Store the category_index
        with open(prob_config.category_index_path, "wb") as f:
            pickle.dump(category_index, f)
        if split:
            train_x = train.drop([target_col], axis=1)
            train_y = train[[target_col]]
        else:
            train_x = training_data.drop([target_col], axis=1)
            train_y = training_data[[target_col]]
        test_x = dev.drop(["label"], axis=1)
        test_y = dev[[target_col]]
        if (newdata is not None):
            train_x = pd.DataFrame(np.concatenate((train_x, newdata.drop([target_col], axis=1))), columns=train_x.columns)
            train_y = pd.DataFrame(np.concatenate((train_y, newdata[[target_col]])), columns=train_y.columns)
        model = LGBMClassifier(objective="binary", random_state=prob_config.random_state, **args)
        model.fit(train_x, train_y, verbose=False)
        if len(np.unique(train_y)) == 2:
            predictions = model.predict_proba(test_x.astype(np.float64))[:,1]
            metrics = {"test_auc": roc_auc_score(test_y, predictions)}
        else:
            predictions = model.predict(test_x)
            metrics = {"test_auc": accuracy_score(test_y, predictions)}
        signature = infer_signature(test_x.astype(np.float64), predictions)
        return model, metrics, signature