import logging
#import pathlib
import time
import datetime

#import mlflow
import lleaves
import pandas as pd
import yaml
from pandas.util import hash_pandas_object

from problem_config import create_prob_config
from data_processor import RawDataProcessor
from utils import AppConfig
import numpy as np

from drift_detector import ks_drift_detect_async
import paho.mqtt.client as mqtt
import orjson

from aiocache import Cache
from aiocache.serializers import PickleSerializer
import asyncio

class ModelPredictor(object):
    @classmethod
    async def create(cls, config_file_path):
        self = ModelPredictor()
        with open(config_file_path, "r") as f:
            self.config = yaml.safe_load(f)
        #logging.info(f"model-config: {self.config}")

        #mlflow.set_tracking_uri(AppConfig.MLFLOW_TRACKING_URI)

        self.prob_config = create_prob_config(
            self.config["phase_id"], self.config["prob_id"]
        )

        # load category_index
        self.category_index = RawDataProcessor.load_category_index(self.prob_config)

        # load feature cols
        self.feature_cols = self.prob_config.feature_cols

        # load model from mlflow tracker
        #model_uri = str(pathlib.Path(
        #    "models:/", self.config["model_name"], str(self.config["model_version"])
        #).as_posix())
        #self.model = mlflow.pyfunc.load_model(model_uri)

        # Load sample for drift detect
        self.X_baseline = pd.read_parquet(self.prob_config.driff_ref_path)[self.prob_config.drift_cols].to_numpy()

        # Init cache
        if AppConfig.CACHE_BACKEND == 'MEMCACHED':
            logging.info(f"Using Memcached as caching server for response at {AppConfig.MEMCACHED_ENDPOINT}")
            self.cache = Cache(Cache.MEMCACHED, endpoint=AppConfig.MEMCACHED_ENDPOINT, port=11211, \
                            namespace=self.config["phase_id"]+self.config["prob_id"])
            self.is_caching_response = True
        elif AppConfig.CACHE_BACKEND == 'REDIS':
            logging.info(f"Using Redis as caching server for response at {AppConfig.REDIS_ENDPOINT}")
            self.cache = Cache(Cache.REDIS, endpoint=AppConfig.REDIS_ENDPOINT, port=6379, db=0, \
                            namespace=self.config["phase_id"]+self.config["prob_id"])
            self.is_caching_response = True
        else:
            self.is_caching_response = False

        if   AppConfig.CACHE_REQUEST == 'False' or AppConfig.CACHE_REQUEST == '0':
            self.is_caching_request = False
        else:  
            logging.info(f'Caching request at {AppConfig.REDIS_ENDPOINT}, {self.config["phase_id"]}_{self.config["prob_id"]}, \
                            db: {self.config["fet_db"]}')
            self.cacherequest = Cache(Cache.REDIS, endpoint=AppConfig.REDIS_ENDPOINT, port=6379, \
                                    db=self.config["fet_db"], serializer=PickleSerializer())
            self.is_caching_request = True                         
        
        # Load compiled model for faster speed (run init_startup.py first)
        try:
            model_path = self.prob_config.data_path / f'{self.config["phase_id"]}_{self.config["prob_id"]}_lgbm.txt'
            llvm_model_path = self.prob_config.data_path / f'{self.config["phase_id"]}_{self.config["prob_id"]}_llvm_compiled.model'
            model_classes_path = self.prob_config.data_path / f'{self.config["phase_id"]}_{self.config["prob_id"]}_classes.npy'
            #self.model._model_impl.booster_.save_model(filename=model_path)
            self.llvm_model = lleaves.Model(model_file=model_path)
            self.llvm_model.compile(cache=llvm_model_path)
            self.model_classes = np.load(model_classes_path, allow_pickle=True)
        except:
            raise Exception("Sorry, model is not saved, run the init_startup.py script first")
        # Logging to mqtt
        if AppConfig.REMOTE_LOGGING == 'False' or AppConfig.REMOTE_LOGGING == '0':
            self.is_logging_to_mqtt = False
        else:
            self.mqtt_topic = f'logging/{self.config["phase_id"]}/{self.config["prob_id"]}'
            self.mqtt_cient = mqtt.Client()
            logging.info(f"Connecting to logging server: {AppConfig.MQTT_ENDPOINT}")
            try:
                self.mqtt_cient.connect(AppConfig.MQTT_ENDPOINT, port=1883)
                logging.info("Successfully connected to mqtt server")
                self.mqtt_cient.loop_start()
                self.is_logging_to_mqtt = True
            except:
                logging.info("Fail to connect mqtt server")
                self.is_logging_to_mqtt = False

        return self

    async def detect_drift(self, feature_df) -> int:
        # watch drift between coming requests and training data
        #return ks_drift_detect(self.X_baseline, feature_df[self.prob_config.drift_cols].to_numpy())[0]
        return await ks_drift_detect_async(self.X_baseline, feature_df[self.prob_config.drift_cols].to_numpy())
    
    async def caching_req_res(self, raw_df, result, runtime, key = None):
        if (not key and self.is_caching_request) or (not key and self.is_caching_response):
            key = str(hash_pandas_object(raw_df).sum())
        if self.is_caching_request:
            await self.cacherequest.set(key, raw_df)
        if self.is_caching_response:
            await self.cache.set(key, result)
        if self.is_logging_to_mqtt:
            current_dt = datetime.datetime.now().strftime("%d/%m/%Y, %H:%M:%S")
            result = {
                "key": key,
                "request_time": current_dt,
                "runtime": runtime
            }
            (rc, mid) = self.mqtt_cient.publish(self.mqtt_topic, orjson.dumps(result))
        return None

    async def predict(self, data):
        start_time = time.time()
        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        # load cached data
        key = None
        if self.is_caching_response:
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
        ).astype(np.float64)
        # Run prediction if no cache stored.
        #prediction = self.model._model_impl.predict(feature_df[self.feature_cols])
        prediction_prob = self.llvm_model.predict(feature_df[self.feature_cols])
        labels = np.argmax(prediction_prob, axis=1)
        prediction = [self.model_classes[i] for i in labels]
        is_drifted = await self.detect_drift(feature_df.dropna()) #.sample(100, replace=True))
        result = {
            "id": data.id,
            "predictions": prediction,
            "drift": is_drifted,
        }
        run_time = round((time.time() - start_time) * 1000, 0)
        logging.info(f"prediction response takes {run_time} ms")
        # Caching
        asyncio.create_task(self.caching_req_res(raw_df, result, run_time, key))
        return result

    async def predict_proba(self, data):
        start_time = time.time()
        raw_df = pd.DataFrame(data.rows, columns=data.columns)
        # load cached data
        key = None
        if self.is_caching_response:
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
        ).astype(np.float64)
        # Run prediction if no cache stored.
        #prediction = self.model._model_impl.predict_proba(feature_df[self.feature_cols])[:,1]
        prediction = self.llvm_model.predict(feature_df[self.feature_cols])
        is_drifted = await self.detect_drift(feature_df.dropna()) #.sample(100, replace=True)
        result = {
            "id": data.id,
            "predictions": prediction.tolist(),
            "drift": is_drifted,
        }
        run_time = round((time.time() - start_time) * 1000, 0)
        logging.info(f"prediction response takes {run_time} ms")
        # Caching
        asyncio.create_task(self.caching_req_res(raw_df, result, run_time, key))
        return result

    async def clear_cache(self):
        if self.is_caching_response:
            await self.cache.clear()
        return True
    
    async def stop(self):
        if self.is_caching_response:
            await self.cache.clear()
        if self.is_logging_to_mqtt:
            self.mqtt_cient.loop_stop()
        return True