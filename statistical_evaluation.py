import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, f_oneway
from src.utils.dataset.generated import load_condition_data
from src.utils.dataset.steingroever import build_igt_dataset
from src.config import BEHAVIOURAL_OUTPUT_PATH, IGT_HEALTHY_DATASET_PATH


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute Cohen's d for independent samples.
    """
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * x.std(ddof = 1) ** 2 + (ny - 1) * y.std(ddof = 1) ** 2) / dof)

    return (x.mean() - y.mean()) / pooled_std


def eta_squared_from_anova(f_stat: float, k: int, N: int) -> float:
    """
    Compute eta squared effect size from ANOVA.
    """
    return (f_stat * (k - 1)) / (f_stat * (k - 1) + (N - k))


def verify_behavioural_diff() -> None:
    """
    Verify behavioural differences across conditions by performing statistical tests on final performance.
    """
    # Load data for each condition
    healthy_df, _ = load_condition_data("healthy", is_baseline = False, is_random = False)
    depleted_df, _ = load_condition_data("depleted", is_baseline = False, is_random = False)
    overactive_df, _ = load_condition_data("overactive", is_baseline = False, is_random = False)

    # Compute final performance per agent per episode
    final_performance = (
        pd.concat([healthy_df, depleted_df, overactive_df])
        .groupby(["condition", "agent", "episode"])["reward"]
        .sum()
        .reset_index()
    )

    # Define results storage
    t_test_results = []
    anova_results = []

    # Create condition pairs
    condition_pairs = [
        ("healthy", "depleted"),
        ("healthy", "overactive"),
        ("depleted", "overactive")
    ]

    # Perform t-tests for each pair
    for cond1, cond2 in condition_pairs:
        group1 = final_performance[final_performance["condition"] == cond1]["reward"]
        group2 = final_performance[final_performance["condition"] == cond2]["reward"]

        t_stat, p_value = ttest_ind(group1, group2, equal_var=False)
        d = cohen_d(group1, group2)

        t_test_results.append({
            "condition_1": cond1,
            "condition_2": cond2,
            "t_stat": t_stat,
            "p_value": p_value,
            "cohen_d": d
        })

    # Apply Bonferroni correction
    num_tests = len(t_test_results)

    for r in t_test_results:
        r["p_value_bonferroni"] = min(r["p_value"] * num_tests, 1.0)

    # Perform ANOVA
    groups = [
        final_performance.loc[final_performance["condition"] == "healthy", "reward"],
        final_performance.loc[final_performance["condition"] == "depleted", "reward"],
        final_performance.loc[final_performance["condition"] == "overactive", "reward"],
    ]

    anova_stat, anova_p_value = f_oneway(*groups)

    # Compute eta squared effect size
    N = len(final_performance)
    k = 3
    eta_sq = eta_squared_from_anova(anova_stat, k, N)

    # Create folder if it doesn't exist
    os.makedirs(BEHAVIOURAL_OUTPUT_PATH, exist_ok = True)

    # Create and save DataFrames for t-test and ANOVA results
    t_test_results = pd.DataFrame(t_test_results)
    anova_results = pd.DataFrame([{
        "anova_stat": anova_stat,
        "anova_p_value": anova_p_value,
        "eta_squared": eta_sq
    }])

    t_test_results.to_csv(f"{BEHAVIOURAL_OUTPUT_PATH}/conditional_t_test_results.csv", index = False)
    anova_results.to_csv(f"{BEHAVIOURAL_OUTPUT_PATH}/conditional_anova_results.csv", index = False)

    print(f"Behavioural statistics complete. Results saved to: {BEHAVIOURAL_OUTPUT_PATH}")


def igt_validation() -> None:
    """
    Evaluate agent behaviours vs. empirical findings from IGT studies. The comparison is primarily reward and deck choice-
    based.
    """
    # Load IGT and agent data
    igt_df = build_igt_dataset(path = IGT_HEALTHY_DATASET_PATH, n_trials = 100)
    healthy_df, _ = load_condition_data("healthy", is_baseline = False, is_random = False)

    # Select comparable segment from agent data
    last_eps = healthy_df["episode"].unique()[-3:]
    agent_df = healthy_df[healthy_df["episode"].isin(last_eps)].copy()
    agent_df = agent_df.head(100).reset_index(drop = True)

    # Align lengths
    min_len = min(len(agent_df), len(igt_df))
    agent_df = agent_df.head(min_len)
    igt_df = igt_df.head(min_len)

    # Compute MSE metrics
    reward_mse = np.mean((np.array(agent_df["reward"], dtype=float) - np.array(igt_df["reward"], dtype=float)) ** 2)
    deck_mse = np.mean((np.array(agent_df["deck"], dtype=float) - np.array(igt_df["deck"], dtype=float)) ** 2)

    # Independent-samples t-test between simulated and empirical rewards
    t_reward, p_reward = ttest_ind(agent_df["reward"], igt_df["reward"], equal_var = False)
    t_deck, p_deck = ttest_ind(agent_df["deck"], igt_df["deck"], equal_var = False)

    # One-way ANOVA
    anova_reward, anova_p_reward = f_oneway(agent_df["reward"], igt_df["reward"])
    anova_deck, anova_p_deck = f_oneway(agent_df["deck"], igt_df["deck"])

    d_reward = cohen_d(agent_df["reward"], igt_df["reward"])
    d_deck = cohen_d(agent_df["deck"], igt_df["deck"])

    # Bonferroni correction for 2 comparisons (reward + deck)
    p_reward_bonf = min(float(p_reward) * 2, 1.0)
    p_deck_bonf = min(float(p_deck) * 2, 1.0)

    # Build results dataframe
    results_df = pd.DataFrame([{
        "reward_mse": reward_mse,
        "deck_mse": deck_mse,
        "t_stat_reward": t_reward,
        "p_reward": p_reward,
        "p_reward_bonferroni": p_reward_bonf,
        "cohen_d_reward": d_reward,
        "t_stat_deck": t_deck,
        "p_deck": p_deck,
        "p_deck_bonferroni": p_deck_bonf,
        "cohen_d_deck": d_deck,
        "anova_reward_stat": anova_reward,
        "anova_reward_p": anova_p_reward,
        "anova_deck_stat": anova_deck,
        "anova_deck_p": anova_p_deck
    }])

    # Create folder if it doesn't exist
    os.makedirs(BEHAVIOURAL_OUTPUT_PATH, exist_ok = True)

    # Save results
    results_df.to_csv(f"{BEHAVIOURAL_OUTPUT_PATH}/igt_behaviour_validation.csv", index = False)

    print(f"IGT validation complete. Results saved to: {BEHAVIOURAL_OUTPUT_PATH}")


if __name__ == "__main__":
    verify_behavioural_diff()
    igt_validation()