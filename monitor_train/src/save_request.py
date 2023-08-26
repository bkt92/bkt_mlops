import pandas as pd
import redis
import pickle
from data_processor import DataProcessor
from problem_config import create_prob_config
import os

def save_request(phase_id, prob_id, db, host, flush=True):
    prob_config = create_prob_config(phase_id=phase_id, prob_id=prob_id)

    capture_path = prob_config.data_path / f'{phase_id}_{prob_id}_captured.parquet'

    category_index = DataProcessor.load_category_index(prob_config)
    rc = redis.Redis(host=host, db=db, port=6379)
    captured_x = pd.DataFrame()

    if len(rc.keys()) > 0:
        for key in rc.keys():
            captured_data = pickle.loads(rc.get(key))
            captured_x = pd.concat([captured_x, captured_data])

        captured_x.drop_duplicates(inplace=True, ignore_index=True)

        captured_x = DataProcessor.apply_category_features(
            raw_df=captured_x[prob_config.feature_cols],
            categorical_cols=prob_config.categorical_cols,
            category_index=category_index,
        )
        if os.path.exists(capture_path):
            request_data = pd.read_parquet(capture_path)
            request_data = pd.concat([captured_x, request_data])
        else:
            request_data = captured_x
        request_data.drop_duplicates(inplace=True, ignore_index=True)
        request_data.to_parquet(capture_path)

    if flush:
        rc.flushdb()

if __name__ == "__main__":
    save_request("phase-3", "prob-1", 1, '127.0.0.1', flush=True)
    save_request("phase-3", "prob-2", 2, '127.0.0.1', flush=True)