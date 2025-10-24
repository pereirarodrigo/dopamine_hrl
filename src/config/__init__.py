from yaml import safe_load

# Simulation parameters
# Model parameters
with open("src/config/default_model_params.yaml", "r") as f:
    MODEL_CONFIG = safe_load(f)

# Baseline parameters
with open("src/config/default_baseline_params.yaml", "r") as f:
    BASELINE_CONFIG = safe_load(f)

HRL_DATA_PATH = MODEL_CONFIG["experiment_params"].get("hrl_exp_save_path", "logs/hrl")
RND_BASELINE_DATA_PATH = BASELINE_CONFIG["experiment_params"].get("rnd_exp_save_path", "logs/baseline/random")
FLAT_TD_DATA_PATH = BASELINE_CONFIG["experiment_params"].get("flat_td_exp_save_path", "logs/baseline/flat_td")
SUMMARY_OUTPUT_PATH = "analysis/summary"
BEHAVIOURAL_OUTPUT_PATH = "analysis/behaviour"
IGT_HEALTHY_DATASET_PATH = "datasets/steingroever"

CONDITIONS = ["healthy", "depleted", "overactive"]
