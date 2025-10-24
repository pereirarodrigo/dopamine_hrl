import numpy as np
import pandas as pd


def compute_deck_preferences(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Compute deck preferences and advantage index for each agent in the dataframe.
    """
    prefs = (
        dataframe.groupby(["agent", "condition"])["deck"]
        .value_counts(normalize = True)
        .unstack(fill_value = 0)
        .rename(columns = {0: "A", 1: "B", 2: "C", 3: "D"})
        .reset_index()
    )
    prefs["advantage_index"] = (prefs["C"] + prefs["D"]) - (prefs["A"] + prefs["B"])

    return prefs


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


def compute_blockwise_reward_gain(dataframe: pd.DataFrame, n_blocks: int = 4) -> pd.DataFrame:
    """
    Compute mean cumulative reward per block and derive block-to-block reward gains (delta of reward) to visualise 
    learning improvement across conditions.
    """
    # Determine bin edges dynamically
    max_trials = dataframe["trial"].max()
    block_size = max_trials / n_blocks
    bins = [i * block_size for i in range(n_blocks + 1)]
    bins[-1] = np.ceil(max_trials)

    # Assign block number per trial
    dataframe["block"] = dataframe.groupby(["condition", "agent", "episode"])["trial"].transform(
        lambda x: pd.cut(x, bins = bins, labels = range(1, n_blocks + 1), include_lowest = True).astype(int)
    )

    # 1️Sum reward per condition x agent x episode x block
    block_reward = (
        dataframe.groupby(["condition", "agent", "episode", "block"])["reward"]
        .sum()
        .reset_index(name="block_reward")
    )

    # Average over episodes = mean reward per agent x block
    agent_mean = (
        block_reward.groupby(["condition", "agent", "block"])["block_reward"]
        .mean()
        .reset_index()
    )

    # Compute per-agent reward delta between consecutive blocks
    agent_mean["delta_reward"] = agent_mean.groupby(["condition", "agent"])["block_reward"].diff().fillna(0)

    # Compute final mean reward delta across agents for plotting
    summary = (
        agent_mean.groupby(["condition", "block"])[["block_reward", "delta_reward"]]
        .mean()
        .reset_index()
    )

    return summary


def compute_block_reward_per_ep(data: pd.DataFrame, n_blocks: int = 4) -> pd.DataFrame:
    """
    Assign blockwise reward per episode for plotting purposes.
    """
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

    return block_reward