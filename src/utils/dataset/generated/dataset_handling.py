import pandas as pd
from src.config import HRL_DATA_PATH, RND_BASELINE_DATA_PATH, FLAT_TD_DATA_PATH


def load_condition_data(cond: str, is_baseline: bool = False, is_random: bool = False) -> tuple[pd.DataFrame, str]:
    """
    Load data for a specific condition.
    """
    if not is_baseline and is_random:
        raise ValueError("To load the random baseline results, `is_baseline` must be set to True.")

    if not is_baseline:
        path = f"{HRL_DATA_PATH}/igt_dopamine_hrl_results_{cond}.csv"
        exp_type = "hrl"

    else:
        if is_random:
            path = f"{RND_BASELINE_DATA_PATH}/igt_random_results_{cond}.csv"
            exp_type = "baseline/random"

        else:
            path = f"{FLAT_TD_DATA_PATH}/igt_flat_td_results_{cond}.csv"
            exp_type = "baseline/flat_td"

    return pd.read_csv(path), exp_type