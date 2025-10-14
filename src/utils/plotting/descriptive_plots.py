import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.deck import compute_blockwise_reward_gain


def plot_advantage_trend(dataframe: pd.DataFrame, output_path: str, filename: str) -> None:
    """
    Plot smooth learning trends (advantage index over blocks) with mean +/- standard error of the mean (SEM) 
    per condition.
    """
    plt.figure(figsize = (7, 5))
    
    # Compute mean +/- SEM per condition x block
    summary = (
        dataframe.groupby(["condition", "block"])["advantage_index"]
        .agg(["mean", "sem"])
        .reset_index()
    )
    
    # Plot each condition as a smooth line with shaded SEM
    palette = {
        "healthy": "#4C72B0",
        "depleted": "#DD8452",
        "overactive": "#55A868"
    }
    
    for cond, group in summary.groupby("condition"):
        plt.plot(group["block"], group["mean"], label = cond, color = palette.get(cond, None), lw = 2)
        plt.fill_between(
            group["block"],
            group["mean"] - group["sem"],
            group["mean"] + group["sem"],
            alpha = 0.2,
            color = palette.get(cond, None)
        )

    # Create path if it doesn't exist
    os.makedirs(output_path, exist_ok = True)

    plt.title("Advantageous Deck Preference over Time", fontsize = 12)
    plt.xlabel("Block")
    plt.ylabel("Positive = more advantageous")
    plt.legend(title = "Condition")
    plt.grid(alpha = 0.3)
    plt.tight_layout()
    plt.savefig(f"{output_path}/{filename}.png", dpi = 300)
    plt.close()


def plot_advantage_trend_with_agents(dataframe: pd.DataFrame, output_path: str, filename: str) -> None:
    """
    Plot smooth learning trends (advantage index over blocks) with mean +/- standard error of the mean (SEM) 
    per condition and agent.
    """
    plt.figure(figsize = (7, 5))
    palette = {"healthy": "#4C72B0", "depleted": "#DD8452", "overactive": "#55A868"}

    # Plot individual agent trends (faint lines)
    for cond, group in dataframe.groupby("condition"):
        for agent, agent_data in group.groupby("agent"):
            plt.plot(agent_data["block"], agent_data["advantage_index"], color=palette[cond], alpha=0.1, linewidth=0.7)

    # Overlay condition means +/- SEM
    summary = (
        dataframe.groupby(["condition", "block"])["advantage_index"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    for cond, group in summary.groupby("condition"):
        plt.plot(group["block"], group["mean"], label = cond, color = palette[cond], lw = 2.5)
        plt.fill_between(
            group["block"], 
            group["mean"] - group["sem"], 
            group["mean"] + group["sem"],
            color = palette[cond], 
            alpha = 0.25
        )

    # Create path if it doesn't exist
    os.makedirs(output_path, exist_ok = True)

    plt.title("Advantageous Deck Preference over Time (By Agent)", fontsize = 12)
    plt.xlabel("Block")
    plt.ylabel("Positive = more advantageous")
    plt.legend(title = "Condition")
    plt.grid(alpha = 0.3)
    plt.tight_layout()
    plt.savefig(f"{output_path}/{filename}.png", dpi = 300)
    plt.close()



def plot_blockwise_winlose_trend(dataframe: pd.DataFrame, output_path: str, filename: str) -> None:
    """
    Plot win-stay and lose-shift behaviour over time (by block) for each dopamine condition.
    """
    plt.figure(figsize = (7, 5))
    palette = {"healthy": "#4C72B0", "depleted": "#DD8452", "overactive": "#55A868"}

    melted = dataframe.melt(
        id_vars = ["condition", "agent", "block"],
        value_vars = ["win_stay", "lose_shift"],
        var_name = "Metric",
        value_name = "Rate"
    )

    sns.lineplot(
        data = melted,
        x = "block",
        y = "Rate",
        hue = "condition",
        style = "Metric",
        markers = True,
        dashes = False,
        lw = 2,
        palette = palette,
        errorbar = "se"
    )

    # Create path if it doesn't exist
    os.makedirs(output_path, exist_ok = True)

    plt.title("Behavioural Adaptation over Time (Win-Stay/Lose-Shift)", fontsize = 12)
    plt.xlabel("Block")
    plt.ylabel("Mean Probability")
    plt.legend(title = "Condition/Metric")
    plt.grid(alpha = 0.3)
    plt.tight_layout()
    plt.savefig(f"{output_path}/{filename}.png", dpi = 300)
    plt.close()


def plot_blockwise_reward_trend(dataframe: pd.DataFrame, output_path: str, filename: str) -> None:
    """
    Plot mean cumulative reward per block across conditions to show learning progression.
    Includes ±SEM shading for variability visualization.
    """
    plt.figure(figsize = (7, 5))
    palette = {"healthy": "#4C72B0", "depleted": "#DD8452", "overactive": "#55A868"}

    # Compute mean and SEM across agents per condition × block
    summary = (
        dataframe.groupby(["condition", "block"])["cumulative_reward"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    for cond, group in summary.groupby("condition"):
        plt.plot(group["block"], group["mean"], label = cond, color = palette.get(cond), lw = 2.5)
        plt.fill_between(
            group["block"],
            group["mean"] - group["sem"],
            group["mean"] + group["sem"],
            alpha = 0.25,
            color = palette.get(cond)
        )

    # Create path if it doesn't exist
    os.makedirs(output_path, exist_ok = True)

    plt.title("Blockwise Cumulative Reward Trend", fontsize = 12)
    plt.xlabel("Block")
    plt.ylabel("Mean Cumulative Reward")
    plt.legend(title = "Condition")
    plt.grid(alpha = 0.3)
    plt.tight_layout()
    plt.savefig(f"{output_path}/{filename}.png", dpi = 300)
    plt.close()


def plot_reward_gain_trend(dataframe: pd.DataFrame, output_path: str, filename: str) -> None:
    """
    Plot blockwise reward gain (reward delta) over time per condition.
    """
    plt.figure(figsize = (7, 5))
    palette = {"healthy": "#4C72B0", "depleted": "#DD8452", "overactive": "#55A868"}

    summary = (
        dataframe.groupby(["condition", "block"])["delta_reward"]
        .agg(["mean", "sem"])
        .reset_index()
    )

    for cond, group in summary.groupby("condition"):
        plt.plot(group["block"], group["mean"], label = cond, color = palette[cond], lw = 2)
        plt.fill_between(
            group["block"],
            group["mean"] - group["sem"],
            group["mean"] + group["sem"],
            color = palette[cond],
            alpha = 0.25
        )

    # Create path if it doesn't exist
    os.makedirs(output_path, exist_ok = True)

    plt.title("Blockwise Learning Gain (Δ Reward)", fontsize=12)
    plt.xlabel("Block")
    plt.ylabel("Δ Mean Reward from Previous Block")
    plt.legend(title = "Condition")
    plt.grid(alpha = 0.3)
    plt.tight_layout()
    plt.savefig(f"{output_path}/{filename}.png", dpi = 300)
    plt.close()


def plot_metric(
    dataframe: pd.DataFrame, 
    x: pd.Series, 
    y: pd.Series,
    hue: str, 
    title: str, 
    ylabel: str, 
    filename: str,
    output_path: str
) -> None:
    """
    Plot a given metric with error bars.
    """
    # Create path if it doesn't exist
    os.makedirs(output_path, exist_ok = True)

    plt.figure(figsize = (7, 5))
    sns.barplot(data = dataframe, x = x, y = y, hue = hue, errorbar = "sd", alpha = 0.8)
    plt.title(title, fontsize = 12)
    plt.ylabel(ylabel)
    plt.tight_layout()
    plt.savefig(f"{output_path}/{filename}.png", dpi = 300)
    plt.close()

