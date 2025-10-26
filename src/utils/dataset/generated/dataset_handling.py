import os
import pandas as pd
from src.config import HRL_DATA_PATH, RND_BASELINE_DATA_PATH, FLAT_TD_DATA_PATH


def load_condition_data(
    cond: str, 
    is_baseline: bool = False, 
    is_random: bool = False, 
    n_trials: int = 100
) -> tuple[pd.DataFrame, str]:
    """
    Load data for a specific condition.
    """
    if not is_baseline and is_random:
        raise ValueError("To load the random baseline results, `is_baseline` must be set to True.")

    if not is_baseline:
        base_path = os.path.join(HRL_DATA_PATH, f"choice_{n_trials}")
        file_path = os.path.join(base_path, f"igt_dopamine_hrl_results_{cond}.csv")
        exp_type = "hrl"

    else:
        if is_random:
            base_path = os.path.join(RND_BASELINE_DATA_PATH, f"choice_{n_trials}")
            file_path = os.path.join(base_path, f"igt_random_results_{cond}.csv")
            exp_type = "baseline/random"

        else:
            base_path = os.path.join(FLAT_TD_DATA_PATH, f"choice_{n_trials}")
            file_path = os.path.join(base_path, f"igt_flat_td_results_{cond}.csv")
            exp_type = "baseline/flat_td"

    return pd.read_csv(file_path), exp_type