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
from flaml import AutoML

from mlflow.models.signature import infer_signature
import mlflow
import numpy as np
from lightgbm import LGBMClassifier

class ModelTrainer:
    @staticmethod
    def log_model_to_tracker(phase_id, prob_id, model, signature, metrics, desc, experiment_id):
        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)
        mlflow.set_experiment(experiment_id)
        mlflow.start_run(description=desc)
        mlflow.log_metrics(metrics)
        mlflow.log_params(model.get_params())
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path=AppConfig.MLFLOW_MODEL_PREFIX,
            signature=signature,
            pip_requirements ='src/requirements.txt',
            registered_model_name=f'{phase_id}_{prob_id}_model_lgbm'
        )
        experimentid = mlflow.active_run().info.run_id
        mlflow.end_run()
        return experimentid

    @staticmethod
    def train_lgb(phase_id, prob_id, split=False, newdata=None, hypertune=False):
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

        if len(np.unique(train_y)) == 2:
            model = LGBMClassifier(objective="binary", random_state=prob_config.random_state)
            model.fit(train_x, train_y, verbose=False)
            predictions = model.predict_proba(test_x.astype(np.float64))[:,1]
            metrics = {"test_auc": roc_auc_score(test_y, predictions)}
        else:
            if hypertune:
                automl = AutoML()
                settings = {
                    "time_budget": 600,  # total running time in seconds
                    "metric": 'accuracy', 
                    "estimator_list": ['lgbm'],#lgbm, xgboost
                    "task": 'classification',  # task type
                    "log_file_name": f'{phase_id}_{prob_id}_experiment.log',  # flaml log file
                    "seed": prob_config.random_state,    # random seed
                }

                automl.fit(X_train=train_x, y_train=train_y['label'], **settings)
                model = LGBMClassifier(objective="multiclass", random_state=prob_config.random_state, **automl.best_config)
            else:
                model = LGBMClassifier(objective="multiclass", random_state=prob_config.random_state)

            model.fit(train_x, train_y, verbose=False)
            predictions = model.predict(test_x)
            metrics = {"test_auc": accuracy_score(test_y, predictions)}

        signature = infer_signature(test_x.astype(np.float64), predictions)

        experiment_id = f'{phase_id}_{prob_id}_lgbm'

        desc = """
                ### Header
                LGBM model
                Model: LGBM (Original Model)
               """

        id = ModelTrainer.log_model_to_tracker(phase_id, prob_id, model, signature, metrics, desc, experiment_id)
        return id
    
if __name__ == "__main__":
    model1_id = ModelTrainer.train_lgb("phase-3", "prob-1")
    model2_id = ModelTrainer.train_lgb("phase-3", "prob-2")