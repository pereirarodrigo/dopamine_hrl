import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Global aesthetic settings
sns.set_theme(style = "whitegrid", context = "talk")
plt.rcParams.update({
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlepad": 12,
    "axes.labelpad": 8,
    "legend.frameon": True
})

# Define a consistent color palette for conditions
palette = {
    "healthy": "#3B82F6",      # blue
    "overactive": "#10B981",   # green
    "depleted": "#F97316"      # orange
}


def plot_net_score_trend(df_net: pd.DataFrame, output_path: str) -> None:
    """
    Plot the trend of net scores across blocks.
    """
    plt.figure(figsize = (8,6))
    sns.lineplot(
        data = df_net, x = "block", y = "net_score", hue = "condition",
        palette = palette, marker = "o", err_style = "band", errorbar = "se"
    )
    plt.title("Mean Net Score Across Blocks")
    plt.xlabel("Block")
    plt.ylabel("Positive = More Advantageous")
    plt.legend(title = "Condition")
    plt.tight_layout()
    plt.savefig(f"{output_path}/net_score_trend.png", dpi = 300)
    plt.close()


def plot_winlose_trend(df_winlose: pd.DataFrame, output_path: str) -> None:
    """
    Plot separate trends of win-stay and lose-shift rates across blocks for each condition.
    """
    fig, axes = plt.subplots(1, 2, figsize = (14, 6), sharey = True)

    # Melt dataframe to long form
    melted = df_winlose.melt(
        id_vars = ["condition", "agent", "block"],
        value_vars = ["win_stay", "lose_shift"],
        var_name = "Metric",
        value_name = "Rate"
    )

    # Define metrics for looping
    metrics = ["win_stay", "lose_shift"]
    titles = ["Win-Stay Across Blocks", "Lose-Shift Across Blocks"]

    for ax, metric, title in zip(axes, metrics, titles):
        subset = melted[melted["Metric"] == metric]

        sns.lineplot(
            data = subset,
            x = "block",
            y = "Rate",
            hue = "condition",
            markers = True,
            dashes = False,
            lw = 2,
            errorbar = "se",
            palette = palette,
            ax = ax
        )

        ax.set_title(title)
        ax.set_xlabel("Block")
        ax.set_ylabel("Mean Probability" if metric == "win_stay" else "")
        ax.legend(title = "Condition")

    plt.tight_layout()
    plt.savefig(f"{output_path}/winlose_trend.png", dpi = 300)
    plt.close()


def plot_cumulative_reward(df_cum: pd.DataFrame, output_path: str) -> None:
    """
    Plot the trend of cumulative rewards across blocks.
    """
    reward_col = (
        "cumulative_perceived_reward" if "cumulative_perceived_reward" in df_cum.columns 
        else "cumulative_reward"
    )

    plt.figure(figsize = (8,6))
    sns.lineplot(
        data = df_cum, x = "block", y = reward_col, hue = "condition",
        palette = palette, marker = "o", err_style = "band", errorbar = "se"
    )
    plt.title("Cumulative Reward Across Blocks")
    plt.xlabel("Block")
    plt.ylabel("Mean Cumulative Reward")
    plt.legend(title = "Condition")
    plt.tight_layout()
    plt.savefig(f"{output_path}/cumulative_reward_trend.png", dpi = 300)
    plt.close()


def plot_final_total_reward(df_total: pd.DataFrame, output_path: str) -> None:
    """
    Plot the final total reward by condition.
    """
    plt.figure(figsize = (7,6))
    sns.barplot(
        data = df_total, x = "condition", y = "total_reward",
        palette = palette, hue = "condition", legend = False, errorbar = "se"
    )
    plt.title("Total Cumulative Reward by Condition")
    plt.xlabel("")
    plt.ylabel("Total Reward")
    plt.tight_layout()
    plt.savefig(f"{output_path}/final_total_reward.png", dpi = 300)
    plt.close()


def plot_deck_analysis(agent_df: pd.DataFrame, igt_df: pd.DataFrame, output_path: str, block_size: int) -> None:
    """
    Plot block-wise deck choice proportions and similarity metrics.
    """
    # Assign block indices if not present
    if "block" not in agent_df.columns:
        agent_df["block"] = ((agent_df["trial"] - 1) // block_size) + 1

    if "block" not in igt_df.columns:
        igt_df["block"] = ((igt_df["trial"] - 1) // block_size) + 1

    # Aggregate deck-choice proportions per block
    agent_block = (
        agent_df.groupby(["block", "deck"])
        .size().unstack(fill_value = 0)
        .apply(lambda x: x / x.sum(), axis = 1)
    )
    igt_block = (
        igt_df.groupby(["block", "deck"])
        .size().unstack(fill_value = 0)
        .apply(lambda x: x / x.sum(), axis = 1)
    )

    # Melt to long format
    agent_long = (
        agent_block.reset_index()
        .melt(id_vars = "block", var_name = "deck", value_name = "proportion")
    )
    igt_long = (
        igt_block.reset_index()
        .melt(id_vars = "block", var_name = "deck", value_name = "proportion")
    )

    # Distinct colors per deck
    deck_palette = {
        "A": "#ef4444",  # red
        "B": "#f59e0b",  # amber
        "C": "#10b981",  # green
        "D": "#3b82f6",  # blue
    }

    # Iterate over decks (ensure deck labels are strings for safety)
    for deck in sorted(agent_long["deck"].astype(str).unique()):
        deck_agent = agent_long[agent_long["deck"].astype(str) == deck]
        deck_igt = igt_long[igt_long["deck"].astype(str) == deck]

        plt.figure(figsize = (8, 6))

        # Agent curve (solid)
        sns.lineplot(
            data = deck_agent,
            x = "block",
            y = "proportion",
            color = deck_palette.get(deck, "gray"),
            label = "Agent",
            lw = 2.5,
            marker = "o"
        )

        # Human (IGT) curve (dashed)
        sns.lineplot(
            data = deck_igt,
            x = "block",
            y = "proportion",
            color = deck_palette.get(deck, "gray"),
            label = "Human",
            lw = 2,
            linestyle = "--",
            marker = "s"
        )

        plt.title(f"Deck {deck}: Agent vs. Human Choice Proportion per Block")
        plt.xlabel("Block (20 trials each)")
        plt.ylabel("Deck Selection Proportion")
        plt.legend(title="")
        plt.ylim(0, 1)
        plt.tight_layout()

        fname = f"{output_path}/deck_{deck}_similarity.png"

        plt.savefig(fname, dpi=300)
        plt.close()