import os
import numpy as np
import pandas as pd
from src.config import CONDITIONS, SUMMARY_OUTPUT_PATH
from src.utils.dataset.generated import load_condition_data
from src.utils.deck import (
    compute_blockwise_winlose,
    compute_blockwise_net_score,
    compute_total_cumulative_reward,
    compute_blockwise_cumulative_reward
)
from src.utils.plotting import (
    plot_winlose_trend,
    plot_net_score_trend,
    plot_cumulative_reward,
    plot_final_total_reward
)


def compute_summary(data: pd.DataFrame, n_blocks: int = 5):
    """
    Compute the four main behavioural metrics:

      1. Net score per block
      2. Win-stay/lose-shift per block
      3. Cumulative reward per block
      4. Total cumulative reward per agent
    """
    df_net = compute_blockwise_net_score(data, n_blocks)
    df_winlose = compute_blockwise_winlose(data, n_blocks)
    df_cum = compute_blockwise_cumulative_reward(data, n_blocks)
    df_total = compute_total_cumulative_reward(data)

    return df_net, df_winlose, df_cum, df_total


def plot_all_results(
    df_net: pd.DataFrame, 
    df_winlose: pd.DataFrame, 
    df_cum: pd.DataFrame, 
    df_total: pd.DataFrame, 
    output_path: str
) -> None:
    """
    Generate and save the four key behavioural plots.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_path, exist_ok = True)

    # 1. Net score trend
    plot_net_score_trend(df_net, output_path)

    # 2. Win-stay/lose-shift trend
    plot_winlose_trend(df_winlose, output_path)

    # 3. Cumulative reward trend
    plot_cumulative_reward(df_cum, output_path)

    # 4. Final total reward
    plot_final_total_reward(df_total, output_path)

    print(f"All plots saved in {output_path}")


def run_analysis(is_baseline: bool = False, is_random: bool = False, n_trials: int = 100) -> None:
    """
    Run full analysis pipeline and generate plots.
    """
    df_net_all, df_winlose_all, df_cum_all, df_total_all = [], [], [], []
    exp_type = ''

    for cond in CONDITIONS:
        data, exp_type = load_condition_data(cond, is_baseline, is_random, n_trials = n_trials)
        df_net, df_winlose, df_cum, df_total = compute_summary(data)

        df_net_all.append(df_net)
        df_winlose_all.append(df_winlose)
        df_cum_all.append(df_cum)
        df_total_all.append(df_total)

    # Concatenate across conditions
    df_net_all = pd.concat(df_net_all, ignore_index = True)
    df_winlose_all = pd.concat(df_winlose_all, ignore_index = True)
    df_cum_all = pd.concat(df_cum_all, ignore_index = True)
    df_total_all = pd.concat(df_total_all, ignore_index = True)

    # Output folder
    output_path = os.path.join(SUMMARY_OUTPUT_PATH, exp_type, f"choice_{n_trials}")

    os.makedirs(output_path, exist_ok=True)

    # Save data
    df_net_all.to_csv(f"{output_path}/blockwise_net_score.csv", index = False)
    df_winlose_all.to_csv(f"{output_path}/blockwise_winlose.csv", index = False)
    df_cum_all.to_csv(f"{output_path}/blockwise_cumulative_reward.csv", index = False)
    df_total_all.to_csv(f"{output_path}/final_total_reward.csv", index = False)

    # Plot
    plot_all_results(df_net_all, df_winlose_all, df_cum_all, df_total_all, output_path)


if __name__ == "__main__":
    # Run analysis for HRL model
    print("Starting analysis for the HRL model...")
    run_analysis(is_baseline = False, n_trials = 100)

    # Run analysis for flat TD baseline
    print("\nStarting analysis for the flat TD baseline...")
    run_analysis(is_baseline = True, is_random = False, n_trials = 100)

    # Run analysis for random baseline
    print("\nStarting analysis for the random baseline...")
    run_analysis(is_baseline = True, is_random = True, n_trials = 100)