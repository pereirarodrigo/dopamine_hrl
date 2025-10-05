import os
import numpy as np
import pandas as pd
from src.utils.plotting import (
    plot_metric,
    plot_advantage_trend,
    plot_reward_gain_trend,
    plot_blockwise_reward_trend,
    plot_blockwise_winlose_trend,
    compute_blockwise_reward_gain,
    plot_advantage_trend_with_agents
)

# Paths and settings
DATA_PATH = "logs"
OUTPUT_PATH = "analysis"
os.makedirs(OUTPUT_PATH, exist_ok = True)

conditions = ["healthy", "depleted", "overactive"]


def load_condition_data(cond: str) -> pd.DataFrame:
    """
    Load data for a specific condition.
    """
    path = f"{DATA_PATH}/igt_dopamine_hrl_results_{cond}.csv"

    return pd.read_csv(path)


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


def compute_blockwise_winlose(dataframe: pd.DataFrame, n_blocks: int = 4) -> pd.DataFrame:
    """
    Compute blockwise win-stay and lose-shift probabilities for each agent and condition.
    """
    # Determine bin edges dynamically
    max_trials = dataframe["trial"].max()
    block_size = max_trials / n_blocks
    bins = [i * block_size for i in range(n_blocks + 1)]
    bins[-1] = np.ceil(max_trials)

    df_sorted = dataframe.sort_values(["condition", "agent", "episode", "trial"]).copy()
    df_sorted["block"] = df_sorted.groupby(["condition", "agent", "episode"])["trial"].transform(
        lambda x: pd.cut(x, bins = bins, labels = range(1, n_blocks + 1), include_lowest = True).astype(int)
    )

    # Previous reward and choice
    df_sorted["prev_reward"] = df_sorted.groupby(["condition", "agent", "episode"])["reward"].shift(1)
    df_sorted["prev_deck"] = df_sorted.groupby(["condition", "agent", "episode"])["deck"].shift(1)

    # Outcome classification
    df_sorted["reward_outcome"] = df_sorted["prev_reward"].apply(lambda r: "win" if pd.notnull(r) and r > 0 else "lose")

    df_sorted["win_stay"] = (
        (df_sorted["reward_outcome"] == "win") &
        (df_sorted["deck"] == df_sorted["prev_deck"])
    ).astype(int)

    df_sorted["lose_shift"] = (
        (df_sorted["reward_outcome"] == "lose") &
        (df_sorted["deck"] != df_sorted["prev_deck"])
    ).astype(int)

    # Aggregate per block
    blockwise = (
        df_sorted.groupby(["condition", "agent", "block"])[["win_stay", "lose_shift"]]
        .mean()
        .reset_index()
    )

    return blockwise


# Main analysis section
deck_all, winlose_all, perf_all = [], [], []

for cond in conditions:
    data = load_condition_data(cond)
    deck_pref, winlose, final_perf = compute_summary(data)
    
    deck_all.append(deck_pref)
    winlose_all.append(winlose)
    perf_all.append(final_perf)

deck_all = pd.concat(deck_all, ignore_index = True)
winlose_all = pd.concat(winlose_all, ignore_index = True)
perf_all = pd.concat(perf_all, ignore_index = True)

# Save all dataframes
deck_all.to_csv(f"{OUTPUT_PATH}/deck_preferences.csv", index = False)
winlose_all.to_csv(f"{OUTPUT_PATH}/winlose_rates.csv", index = False)
perf_all.to_csv(f"{OUTPUT_PATH}/final_performance.csv", index = False)

# Plot results
# 1. Block-wise advantage index trend
plot_advantage_trend(
    deck_all,
    output_path = OUTPUT_PATH,
    filename = "advantage_index_trend"
)

# 2. Block-wise advantage index trend by agent
plot_advantage_trend_with_agents(
    deck_all,
    output_path = OUTPUT_PATH,
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
    filename = "winlose_rates"
)

# 4. Blockwise win-stay/lose-shift trends
blockwise_all = []

for cond in conditions:
    data = load_condition_data(cond)
    blockwise = compute_blockwise_winlose(data)
    blockwise_all.append(blockwise)

blockwise_all = pd.concat(blockwise_all, ignore_index = True)
blockwise_all.to_csv(f"{OUTPUT_PATH}/blockwise_winlose.csv", index = False)

plot_blockwise_winlose_trend(
    blockwise_all,
    output_path = OUTPUT_PATH,
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
    filename = "final_performance"
)

# 6. Block-wise cumulative reward trend (learning progression)
block_reward_all = []

for cond in conditions:
    data = load_condition_data(cond)

    # Determine block membership per agent and episode
    max_trials = data["trial"].max()
    n_blocks = 4
    block_size = max_trials / n_blocks
    bins = [i * block_size for i in range(n_blocks + 1)]
    bins[-1] = np.ceil(max_trials)

    data["block"] = data.groupby(["condition", "agent", "episode"])["trial"].transform(
        lambda x: pd.cut(x, bins = bins, labels = range(1, n_blocks + 1), include_lowest = True).astype(int)
    )

    # Average cumulative reward per block
    block_reward = (
        data.groupby(["condition", "agent", "block"])["cumulative_reward"]
        .mean()
        .reset_index()
    )
    block_reward_all.append(block_reward)

block_reward_all = pd.concat(block_reward_all, ignore_index = True)
block_reward_all.to_csv(f"{OUTPUT_PATH}/blockwise_reward_trend.csv", index = False)

plot_blockwise_reward_trend(
    block_reward_all,
    output_path = OUTPUT_PATH,
    filename = "blockwise_reward_trend"
)

# 7. Learning gain (reward delta) over time
reward_gain_all = []

for cond in conditions:
    data = load_condition_data(cond)
    reward_gain = compute_blockwise_reward_gain(data)
    reward_gain_all.append(reward_gain)

reward_gain_all = pd.concat(reward_gain_all, ignore_index = True)
reward_gain_all.to_csv(f"{OUTPUT_PATH}/blockwise_reward_gain.csv", index = False)

plot_reward_gain_trend(
    reward_gain_all,
    output_path = OUTPUT_PATH,
    filename = "blockwise_reward_gain_trend"
)

print(f"Analysis complete. Results and plots saved in: {OUTPUT_PATH}")
