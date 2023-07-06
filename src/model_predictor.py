import logging
import pathlib
import time

import mlflow
import pandas as pd
import yaml
from pandas.util import hash_pandas_object

from problem_config import create_prob_config
from data_processor import RawDataProcessor
from utils import AppConfig
import numpy as np

try:
    from ks_drift import ks_drift as ks_drift_detect
except:
    from drift_detector import ks_drift_detect, ks_drift_detect_async

from aiocache import Cache
from aiocache.serializers import PickleSerializer
import asyncio

class ModelPredictor(object):
    @classmethod
    async def create(cls, config_file_path):
        self = ModelPredictor()
        with open(config_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        logging.info(f"model-config: {self.config}")

        mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)

        self.prob_config = create_prob_config(
            self.config["phase_id"], self.config["prob_id"]
        )

        # load category_index
        self.category_index = RawDataProcessor.load_category_index(self.prob_config)

        # load feature cols
        self.feature_cols = self.prob_config.feature_cols

        # load model from mlflow tracker
        model_uri = str(pathlib.Path(
            "models:/", self.config["model_name"], str(self.config["model_version"])
        ).as_posix())
        self.model = mlflow.pyfunc.load_model(model_uri)

        # Init drift detect
        self.X_baseline = pd.read_parquet(self.prob_config.driff_ref_path)[self.prob_config.drift_cols].to_numpy()

        # Init cache
        self.cache = Cache(Cache.REDIS, endpoint=AppConfig.REDIS_ENDPOINT, port=6379, db=0, \
                           namespace=self.config["phase_id"]+self.config["prob_id"])
        self.cacherequest = Cache(Cache.REDIS, endpoint=AppConfig.REDIS_ENDPOINT, port=6379, \
                                  db=self.config["fet_db"], serializer=PickleSerializer())

        return self

    async def detect_drift(self, feature_df) -> int:
        # watch drift between coming requests and training data
        return ks_drift_detect(self.X_baseline, feature_df[self.prob_config.drift_cols].to_numpy())[0]
        #return await ks_drift_detect_async(self.X_baseline, feature_df[self.prob_config.drift_cols].to_numpy())

    async def predict(self, data):
        start_time = time.time()

        raw_df = pd.DataFrame(data.rows, columns=data.columns)

        # load cached data
        key = str(hash_pandas_object(raw_df).sum())

        if await self.cache.exists(key):
            run_time = round((time.time() - start_time) * 1000, 0)
            logging.info(f"cached {key} {run_time} ms")
            return await self.cache.get(key)
        
        # preprocess
        feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=self.prob_config.categorical_cols,
            category_index=self.category_index,
        )

        feature_df = feature_df.astype(np.float64)

        # Run prediction if no cache stored.
        prediction = self.model._model_impl.predict(feature_df[self.feature_cols])
        is_drifted = await self.detect_drift(feature_df.dropna()) #.sample(100, replace=True))

        # Save data
        asyncio.create_task(self.cacherequest.set(key, raw_df))

        result = {
            "id": data.id,
            "predictions": prediction.tolist(),
            "drift": is_drifted,
        }

        asyncio.create_task(self.cache.set(key, result))

        run_time = round((time.time() - start_time) * 1000, 0)
        logging.info(f"prediction response takes {run_time} ms")
        return result

    async def predict_proba(self, data):
        start_time = time.time()

        raw_df = pd.DataFrame(data.rows, columns=data.columns)

        # load cached data
        key = str(hash_pandas_object(raw_df).sum())
        if await self.cache.exists(key):
            run_time = round((time.time() - start_time) * 1000, 0)
            logging.info(f"cached {key} {run_time} ms")
            return await self.cache.get(key)
        
        # preprocess
        feature_df = RawDataProcessor.apply_category_features(
            raw_df=raw_df,
            categorical_cols=self.prob_config.categorical_cols,
            category_index=self.category_index,
        )

        feature_df = feature_df.astype(np.float64)

        # Run prediction if no cache stored.
        prediction = self.model._model_impl.predict_proba(feature_df[self.feature_cols])[:,1]
        is_drifted = await self.detect_drift(feature_df.dropna()) #.sample(100, replace=True)

        asyncio.create_task(self.cacherequest.set(key, raw_df))

        result = {
            "id": data.id,
            "predictions": prediction.tolist(),
            "drift": is_drifted,
        }

        asyncio.create_task(self.cache.set(key, result))

        run_time = round((time.time() - start_time) * 1000, 0)
        logging.info(f"prediction response takes {run_time} ms")
        return result

    async def clear_cache(self):
        return await self.cache.clear()