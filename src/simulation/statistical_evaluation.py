import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, f_oneway
from src.utils.dataset.generated import load_condition_data
from src.utils.dataset.steingroever import build_igt_dataset
from src.utils.plotting.descriptive_plots import plot_deck_analysis
from src.utils.deck.deck_metrics import compute_blockwise_deck_analysis
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


def verify_behavioural_diff(n_trials: int) -> None:
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
        .groupby(["condition", "agent", "episode"])["perceived_reward"]
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
        group1 = final_performance[final_performance["condition"] == cond1]["perceived_reward"]
        group2 = final_performance[final_performance["condition"] == cond2]["perceived_reward"]

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
        final_performance.loc[final_performance["condition"] == "healthy", "perceived_reward"],
        final_performance.loc[final_performance["condition"] == "depleted", "perceived_reward"],
        final_performance.loc[final_performance["condition"] == "overactive", "perceived_reward"],
    ]

    anova_stat, anova_p_value = f_oneway(*groups)

    # Compute eta squared effect size
    N = len(final_performance)
    k = 3
    eta_sq = eta_squared_from_anova(anova_stat, k, N)

    # Create folder if it doesn't exist
    output_path = os.path.join(BEHAVIOURAL_OUTPUT_PATH, f"choice_{n_trials}")

    os.makedirs(output_path, exist_ok = True)

    # Create and save DataFrames for t-test and ANOVA results
    t_test_results = pd.DataFrame(t_test_results)
    anova_results = pd.DataFrame([{
        "anova_stat": anova_stat,
        "anova_p_value": anova_p_value,
        "eta_squared": eta_sq
    }])

    t_test_results.to_csv(f"{output_path}/conditional_t_test_results.csv", index = False)
    anova_results.to_csv(f"{output_path}/conditional_anova_results.csv", index = False)

    print(f"Behavioural statistics complete. Results saved to: {output_path}")


def igt_validation(n_trials: int) -> None: 
    """
    Evaluate agent behaviours vs. empirical findings from IGT studies, using an agent-trial granularity. Each (agent, trial) combination is matched to its IGT counterpart, and metrics are averaged across agents and trials.
    """ 
    # Load IGT and simulated data 
    igt_df = build_igt_dataset(path = IGT_HEALTHY_DATASET_PATH, n_trials = n_trials) 
    healthy_df, _ = load_condition_data("healthy", is_baseline = False, is_random = False) 
    
    # Select the last episode of the healthy condition for comparison
    last_episode = healthy_df["episode"].max() 
    agent_df = healthy_df[healthy_df["episode"] == last_episode] 
    
    # Match number of agents between datasets 
    n_agents = min(agent_df["agent"].nunique(), igt_df["agent"].nunique()) 
    igt_df = igt_df[igt_df["agent"] <= n_agents] 
    agent_df = agent_df[agent_df["agent"] <= n_agents] 
    
    # Merge trial-by-trial across both datasets 
    merged = pd.merge( 
        agent_df, 
        igt_df, 
        on = ["agent", "trial"], 
        suffixes = ("_sim", "_igt"), 
        how = "inner" 
    ) 
    
    if merged.empty: 
        raise ValueError("Merged dataframe is empty — check alignment of agent/trial indices.") 
    
    # Compute trial-level error signals
    merged["reward_sq_error"] = (merged["true_reward"] - merged["reward"]) ** 2 
    merged["deck_sq_error"] = (merged["deck_sim"] - merged["deck_igt"]) ** 2 
    
    # Average across agents and trials 
    reward_mse = merged["reward_sq_error"].mean() 
    deck_mse = merged["deck_sq_error"].mean() 
    
    # Compute between-distribution statistics using the full population of paired trials 
    t_reward, p_reward = ttest_ind(merged["true_reward"], merged["reward"], equal_var = False) 
    t_deck, p_deck = ttest_ind(merged["deck_sim"], merged["deck_igt"], equal_var = False) 
    d_reward = cohen_d(merged["true_reward"], merged["reward"]) 
    d_deck = cohen_d(merged["deck_sim"], merged["deck_igt"]) 
    
    # Bonferroni correction 
    p_reward_bonf = min(float(p_reward) * 2, 1.0) 
    p_deck_bonf = min(float(p_deck) * 2, 1.0) 
    
    # Build summary 
    results_df = pd.DataFrame([{ 
        "reward_mse_mean": reward_mse, 
        "deck_mse_mean": deck_mse, 
        "t_stat_reward": t_reward, 
        "p_reward": p_reward, 
        "p_reward_bonferroni": p_reward_bonf, 
        "cohen_d_reward": d_reward, 
        "t_stat_deck": t_deck, 
        "p_deck": p_deck, 
        "p_deck_bonferroni": p_deck_bonf, 
        "cohen_d_deck": d_deck, 
        "n_agents": n_agents, 
        "n_trials_total": len(merged) 
    }]) 
    
    # Save main output 
    output_path = os.path.join(BEHAVIOURAL_OUTPUT_PATH, f"choice_{n_trials}") 
    
    os.makedirs(output_path, exist_ok = True)
    results_df.to_csv(f"{output_path}/igt_behaviour_validation.csv", index = False) 
    
    print(f"IGT behavioural validation complete. Results saved to: {output_path}") 
    
    block_df = compute_blockwise_deck_analysis(agent_df, igt_df) 
    
    # Plot and save results 
    plot_deck_analysis(agent_df, igt_df, output_path, block_size = 20) 
    
    block_df.to_csv(f"{output_path}/blockwise_deck_similarity.csv", index = False) 
    
    print(f"Block-wise deck analysis saved to: {output_path}")


if __name__ == "__main__":
    verify_behavioural_diff(n_trials = 100)
    igt_validation(n_trials = 100)