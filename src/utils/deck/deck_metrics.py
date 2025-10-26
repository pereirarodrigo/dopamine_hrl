import numpy as np
import pandas as pd


def assign_blocks(df: pd.DataFrame, n_blocks: int = 5) -> pd.DataFrame:
    """
    Assigns block indices (1..n_blocks) per episode based on trial number.
    """
    max_trials = df["trial"].max()
    block_edges = np.linspace(0, max_trials, n_blocks + 1)
    df = df.copy()
    df["block"] = df.groupby(["condition", "agent", "episode"])["trial"].transform(
        lambda x: pd.cut(x, bins = block_edges, labels = range(1, n_blocks + 1), include_lowest = True).astype(int)
    )

    return df


def compute_blockwise_net_score(df: pd.DataFrame, n_blocks: int = 5) -> pd.DataFrame:
    """
    Net score (advantageous - disadvantageous choices) per block: (C + D) - (A + B)
    """
    df = assign_blocks(df, n_blocks)
    block_choices = (
        df.groupby(["condition", "agent", "episode", "block"])["deck"]
        .value_counts(normalize = True)
        .unstack(fill_value = 0)
        .rename(columns = {0: "A", 1: "B", 2: "C", 3: "D"})
        .reset_index()
    )
    block_choices["net_score"] = (block_choices["C"] + block_choices["D"]) - (
        block_choices["A"] + block_choices["B"]
    )

    # Mean per agent across episodes
    return (
        block_choices.groupby(["condition", "agent", "block"])["net_score"]
        .mean()
        .reset_index()
    )


def compute_blockwise_winlose(df: pd.DataFrame, n_blocks: int = 5) -> pd.DataFrame:
    """
    Win-stay/lose-shift per block.
    """
    # Choose perceived reward if available
    reward_col = "perceived_reward" if "perceived_reward" in df.columns else "reward"

    df = assign_blocks(df, n_blocks).sort_values(
        ["condition", "agent", "episode", "trial"]
    )
    df["prev_reward"] = df.groupby(["condition", "agent", "episode"])[reward_col].shift(1)
    df["prev_deck"] = df.groupby(["condition", "agent", "episode"])["deck"].shift(1)
    df["outcome"] = np.where(df["prev_reward"] > 0, "win", "lose")

    df["win_stay"] = ((df["outcome"] == "win") & (df["deck"] == df["prev_deck"])).astype(int)
    df["lose_shift"] = ((df["outcome"] == "lose") & (df["deck"] != df["prev_deck"])).astype(int)

    return (
        df.groupby(["condition", "agent", "block"])[["win_stay", "lose_shift"]]
        .mean()
        .reset_index()
    )


def compute_blockwise_cumulative_reward(df: pd.DataFrame, n_blocks: int = 5) -> pd.DataFrame:
    """
    Mean cumulative reward per block.
    """
    df = assign_blocks(df, n_blocks)

    # Choose perceived reward if available
    reward_col = (
        "cumulative_perceived_reward" if "cumulative_perceived_reward" in df.columns 
        else "cumulative_reward"
    )

    # Last cumulative reward of each block per episode
    block_end = (
        df.groupby(["condition", "agent", "episode", "block"], as_index = False)
        .apply(lambda g: g.loc[g["trial"].idxmax()])
        .reset_index(drop = True)
    )

    # Average across episodes → agent-level mean
    return (
        block_end.groupby(["condition", "agent", "block"])[reward_col]
        .mean()
        .reset_index()
    )


def compute_total_cumulative_reward(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total reward per agent across all trials and episodes.
    """
    # Choose perceived reward if available
    reward_col = (
        "cumulative_perceived_reward" if "cumulative_perceived_reward" in df.columns 
        else "cumulative_reward"
    )

    return (
        df.groupby(["condition", "agent"])[reward_col]
        .sum()
        .reset_index(name = "total_reward")
    )
