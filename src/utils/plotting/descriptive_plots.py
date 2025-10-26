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
    Plot the trend of win-stay and lose-shift rates across blocks.
    """
    plt.figure(figsize = (8, 6))

    melted = df_winlose.melt(
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

    plt.title("Win-Stay/Lose-Shift Across Blocks")
    plt.xlabel("Block")
    plt.ylabel("Mean Probability")
    plt.legend(title = "Condition")
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
