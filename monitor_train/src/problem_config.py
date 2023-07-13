import json
from utils import AppPath

class ProblemConfig:
    # required inputs
    phase_id: str
    prob_id: str
    random_state: int

    # ml-problem properties
    target_col: str
    feature_cols: list
    numerical_cols: list
    categorical_cols: list
    drift_cols: list
    ml_type: str

    # For referent data for drift test
    driff_ref_path: str

def load_feature_configs_dict(config_path: str) -> dict:
    with open(config_path) as f:
        features_config = json.load(f)
    return features_config

def create_prob_config(phase_id: str, prob_id: str) -> ProblemConfig:
    prob_config = ProblemConfig()
    prob_config.prob_id = prob_id
    prob_config.phase_id = phase_id
    prob_config.random_state = 123
    prob_config.test_size = 0.1

    # construct data paths for original data
    prob_config.feature_config_path = (
        AppPath.PROB_CONFIG_DIR / f"features_{phase_id}_{prob_id}.json"
    )
    prob_config.data_path = AppPath.DATA_DIR / f"{phase_id}" / f"{prob_id}"
    prob_config.data_path.mkdir(parents=True, exist_ok=True)

    prob_config.category_index_path = (
        prob_config.data_path / "category_index.pickle"
    )
    prob_config.driff_ref_path = prob_config.data_path / "x_ref.parquet"
    prob_config.raw_data_path = prob_config.data_path / "raw_train.parquet"
    prob_config.kmeans_path = prob_config.data_path / "kmeans_path.cpk"

    # get properties of ml-problem
    feature_configs = load_feature_configs_dict(prob_config.feature_config_path)
    prob_config.target_col = feature_configs.get("target_column")
    prob_config.categorical_cols = feature_configs.get("category_columns")
    prob_config.numerical_cols = feature_configs.get("numeric_columns")
    prob_config.drift_cols = feature_configs.get("drift_cols")
    prob_config.feature_cols = feature_configs.get("feature_cols")
    prob_config.ml_type = feature_configs.get("ml_type")

    return prob_config

def get_prob_config(phase_id: str, prob_id: str):
    prob_config = create_prob_config(phase_id, prob_id)
    return prob_config