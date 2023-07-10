import pickle
import pandas as pd
from problem_config import ProblemConfig
import pandas as pd
import redis
import pickle
from utils import AppConfig, AppPath
import datetime
import yaml

class DataProcessor:
    @staticmethod
    def build_category_features(data, categorical_cols=None):
        if categorical_cols is None:
            categorical_cols = []
        category_index = {}
        if len(categorical_cols) == 0:
            return data, category_index

        df = data.copy()
        # process category features
        for col in categorical_cols:
            df[col] = df[col].astype("category")
            category_index[col] = df[col].cat.categories
            df[col] = df[col].cat.codes
        return df, category_index

    @staticmethod
    def apply_category_features(
        raw_df, categorical_cols=None, category_index: dict = None
    ):
        if categorical_cols is None:
            categorical_cols = []
        if len(categorical_cols) == 0:
            return raw_df

        apply_df = raw_df.copy()
        for col in categorical_cols:
        #    apply_df[col] = apply_df[col].astype("category")
            apply_df[col] = pd.Categorical(
                apply_df[col],
                categories=category_index[col],
            ).codes
        return apply_df

    @staticmethod
    def load_category_index(prob_config: ProblemConfig):
        with open(prob_config.category_index_path, "rb") as f:
            return pickle.load(f)
    
    @staticmethod
    def load_and_save_data_redis(model_config, host = AppConfig.REDIS_ENDPOINT, clear_db=True):
        model_config_path = (AppPath.MODEL_CONFIG_DIR / model_config).as_posix()
        with open(model_config_path, "r") as f:
            config = yaml.safe_load(f)
        rc = redis.Redis(host=host, db=config["fet_db"], port=6379,  socket_keepalive=True)
        captured_x = {}
        print(f'Load request from {host}, db: {config["fet_db"]}')
        for key in rc.keys():
            captured_data = pickle.loads(rc.get(key))
            captured_x[key] = captured_data
            current_dt = datetime.datetime.now().strftime("%d_%m_%Y__%H_%M_%S")
            file_name = f'{host}_prob{config["fet_db"]}_{current_dt}.pkl'
        with open(AppPath.REQUEST_DATA_DIR / file_name , 'wb') as f:
            pickle.dump(captured_x, f)
        if clear_db:
            rc.flushdb()
            print("Cache is cleared")
        return file_name

    @staticmethod
    def load_saved_request(request_file):
        request_file_path = AppPath.REQUEST_DATA_DIR / request_file
        with open(request_file_path, "rb") as f:
            return pickle.load(f)