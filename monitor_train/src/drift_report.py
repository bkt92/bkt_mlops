from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from utils import AppPath
import pandas as pd
from problem_config import create_prob_config
from data_processor import DataProcessor
import yaml

def generate_drift_report(model_config, request_file, key):
    model_config_path = (AppPath.MODEL_CONFIG_DIR / model_config).as_posix()
    with open(model_config_path, "r") as f:
        config = yaml.safe_load(f)
    prob_config = create_prob_config(
    config["phase_id"], config["prob_id"])
    reference = pd.read_parquet(prob_config.driff_ref_path)[prob_config.drift_cols]
    category_index = DataProcessor.load_category_index(prob_config)

    saved_request = DataProcessor.load_saved_request(request_file)
    current_raw = saved_request[key][prob_config.feature_cols]

    current = DataProcessor.apply_category_features(
    raw_df=current_raw,
    categorical_cols=prob_config.categorical_cols,
    category_index=category_index,
    )

    report = Report(metrics=[
        DataDriftPreset(num_stattest='ks'), 
    ])

    report.run(reference_data=reference, current_data=current)
    report_path = AppPath.REPORTS_DIR / f'{config["phase_id"]}_{config["prob_id"]}_{key}.html'
    report.save_html(report_path)
    return report_path

