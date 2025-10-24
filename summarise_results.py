import os
import numpy as np
import pandas as pd
from src.config import CONDITIONS, SUMMARY_OUTPUT_PATH
from src.utils.dataset.generated import load_condition_data
from src.utils.deck import (
    compute_blockwise_winlose,
    compute_block_reward_per_ep,
    compute_blockwise_reward_gain
)
from src.utils.plotting import (
    plot_metric,
    plot_advantage_trend,
    plot_reward_gain_trend,
    plot_blockwise_reward_trend,
    plot_blockwise_winlose_trend,
    plot_advantage_trend_with_agents
)


def compute_summary(dataframe: pd.DataFrame, n_blocks: int = 4) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute blockwise deck preferences, win-stay/lose-shift, and final performance across conditions, agents,
    and episodes.
    """
    # Determine bin edges based on max trials
    max_trials = dataframe["trial"].max()
    block_size = max_trials / n_blocks
    bins = [i * block_size for i in range(n_blocks + 1)]

    # Ensure last bin covers full range
    bins[-1] = np.ceil(max_trials)

    # Compute per-episode blocks
    dataframe["block"] = dataframe.groupby(["condition", "agent", "episode"])["trial"].transform(
        lambda x: pd.cut(
            x,
            bins = bins,
            labels = range(1, n_blocks + 1),
            include_lowest = True
        ).astype(int)
    )

    # Deck preferences
    deck_pref = (
        dataframe.groupby(["condition", "agent", "episode", "block"])["deck"]
        .value_counts(normalize = True)
        .unstack(fill_value = 0)
        .rename(columns = {0: "A", 1: "B", 2: "C", 3: "D"})
        .reset_index()
    )
    deck_pref["advantage_index"] = (deck_pref["C"] + deck_pref["D"]) - (deck_pref["A"] + deck_pref["B"])

    # Win-stay / lose-shift
    df_sorted = dataframe.sort_values(["condition", "agent", "episode", "trial"]).copy()
    df_sorted["prev_reward"] = df_sorted.groupby(["condition", "agent", "episode"])["reward"].shift(1)
    df_sorted["prev_deck"] = df_sorted.groupby(["condition", "agent", "episode"])["deck"].shift(1)

    df_sorted["reward_outcome"] = df_sorted["prev_reward"].apply(lambda r: "win" if pd.notnull(r) and r > 0 else "lose")

    df_sorted["win_stay"] = (
        (df_sorted["reward_outcome"] == "win") &
        (df_sorted["deck"] == df_sorted["prev_deck"])
    ).astype(int)

    df_sorted["lose_shift"] = (
        (df_sorted["reward_outcome"] == "lose") &
        (df_sorted["deck"] != df_sorted["prev_deck"])
    ).astype(int)

    winlose = (
        df_sorted.groupby(["condition", "agent"])
        [["win_stay", "lose_shift"]]
        .mean()
        .reset_index()
        .rename(columns = {"win_stay": "win_stay_rate", "lose_shift": "lose_shift_rate"})
    )

    # Final performance (total cumulative reward)
    final_perf = (
        df_sorted.groupby(["condition", "agent"])["reward"]
        .sum()
        .reset_index(name = "total_reward")
    )

    return deck_pref, winlose, final_perf


def plot_all_results(
    deck_all: pd.DataFrame, 
    winlose_all: pd.DataFrame, 
    perf_all: pd.DataFrame, 
    output_path: str,
    is_baseline: bool = False,
    is_random: bool = False,
) -> None:
    """
    Generate and save all plots based on the analysis results.
    """
    blockwise_all, block_reward_all, reward_gain_all = [], [], []

    # Calculate all important metrics beforehand
    for cond in CONDITIONS:
        data, _ = load_condition_data(cond, is_baseline, is_random)
        blockwise = compute_blockwise_winlose(data)

        # Compute blockwise reward gain
        reward_gain = compute_blockwise_reward_gain(data)

        # Compute block reward per episode (for plotting purposes)
        block_reward = compute_block_reward_per_ep(data)

        blockwise_all.append(blockwise)
        reward_gain_all.append(reward_gain)
        block_reward_all.append(block_reward)

    # Save block reward, blockwise winlose, and reward gain data
    block_reward_all = pd.concat(block_reward_all, ignore_index = True)
    block_reward_all.to_csv(f"{output_path}/blockwise_reward_trend.csv", index = False)

    blockwise_all = pd.concat(blockwise_all, ignore_index = True)
    blockwise_all.to_csv(f"{output_path}/blockwise_winlose.csv", index = False)

    reward_gain_all = pd.concat(reward_gain_all, ignore_index = True)
    reward_gain_all.to_csv(f"{output_path}/blockwise_reward_gain.csv", index = False)

    # Plot results
    # 1. Block-wise advantage index trend
    plot_advantage_trend(
        deck_all,
        output_path = output_path,
        filename = "advantage_index_trend"
    )

    # 2. Block-wise advantage index trend by agent
    plot_advantage_trend_with_agents(
        deck_all,
        output_path = output_path,
        filename = "advantage_index_trend_agent"
    )

    # 3. Win-stay / Lose-shift
    melted_winlose = winlose_all.melt(
        id_vars = ["condition", "agent"],
        value_vars = ["win_stay_rate", "lose_shift_rate"],
        var_name = "metric",
        value_name = "rate"
    )
    plot_metric(
        melted_winlose,
        x = "metric",
        y = "rate",
        hue = "condition",
        title = "Win-Stay/Lose-Shift Rates by Condition",
        ylabel = "Mean Probability",
        filename = "winlose_rates",
        output_path = output_path
    )

    # 4. Blockwise win-stay/lose-shift trends
    plot_blockwise_winlose_trend(
        blockwise_all,
        output_path = output_path,
        filename = "blockwise_winlose_trend"
    )

    # 5. Final performance
    plot_metric(
        perf_all,
        x = "condition",
        y = "total_reward",
        hue = None,
        title = "Final Cumulative Reward per Condition",
        ylabel = "Total Reward",
        filename = "final_performance",
        output_path = output_path
    )

    # 6. Blockwise cumulative reward trend (learning progression)
    plot_blockwise_reward_trend(
        block_reward_all,
        output_path = output_path,
        filename = "blockwise_reward_trend"
    )

    # 7. Learning gain (reward delta) over time
    plot_reward_gain_trend(
        reward_gain_all,
        output_path = output_path,
        filename = "blockwise_reward_gain_trend"
    )

    print(f"Analysis complete. Results and plots saved in: {output_path}")


def run_analysis(is_baseline: bool = False, is_random: bool = False) -> None:
    """
    Run full analysis pipeline and generate plots.
    """
    data = pd.DataFrame({})
    deck_all, winlose_all, perf_all = [], [], []
    exp_type = ''

    for cond in CONDITIONS:
        data, exp_type = load_condition_data(cond, is_baseline, is_random)
        deck_pref, winlose, final_perf = compute_summary(data)
        
        deck_all.append(deck_pref)
        winlose_all.append(winlose)
        perf_all.append(final_perf)

    deck_all = pd.concat(deck_all, ignore_index = True)
    winlose_all = pd.concat(winlose_all, ignore_index = True)
    perf_all = pd.concat(perf_all, ignore_index = True)

    # Append experiment type to output path, and create directory if it doesn't exist
    output_path = os.path.join(SUMMARY_OUTPUT_PATH, exp_type)

    os.makedirs(output_path, exist_ok = True)

    # Save all dataframes
    deck_all.to_csv(f"{output_path}/deck_preferences.csv", index = False)
    winlose_all.to_csv(f"{output_path}/winlose_rates.csv", index = False)
    perf_all.to_csv(f"{output_path}/final_performance.csv", index = False)

    # Generate and save plots
    plot_all_results(
        deck_all, 
        winlose_all, 
        perf_all,
        output_path,
        is_baseline,
        is_random
    )


if __name__ == "__main__":
    # Run analysis for HRL model
    print("Starting analysis for the HRL model...")
    run_analysis(is_baseline = False)

    # Run analysis for flat TD baseline
    print("\nStarting analysis for the flat TD baseline...")
    run_analysis(is_baseline = True, is_random = False)

    # Run analysis for random baseline
    print("\nStarting analysis for the random baseline...")
    run_analysis(is_baseline = True, is_random = True)