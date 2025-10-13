import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from yaml import safe_load
from src.modelling.environment import IGTEnv
from src.utils import compute_deck_preferences
from src.modelling.agent import TDPolicy, RandomPolicy
from src.modelling.modulation import DopamineModulation


# Simulation parameters
with open("default_params.yaml", "r") as f:
    config = safe_load(f)


def run_episode(env: IGTEnv, policy: TDPolicy | RandomPolicy, max_steps: int = 200) -> list[dict]:
    """
    Run a single IGT episode and collect detailed trial-level data.
    """
    obs, _ = env.reset()
    total_reward = 0.0
    episode_log = []

    for t in range(max_steps):
        state = tuple(obs.round(2))

        # Policy sampling
        action = policy.sample(state)

        # Environment step
        next_obs, reward, done, _, _ = env.step(action)
        total_reward += reward

        # Update the policy only if it's not a random policy
        if isinstance(policy, TDPolicy):
            policy.update(state, action, reward, tuple(next_obs.round(2)), done)

        # Log results
        # Compute win-stay/lose-shift if not first trial
        if t > 0:
            prev_reward = episode_log[-1]["reward"]
            prev_action = episode_log[-1]["deck"]
            win_stay = int(prev_reward > 0 and action == prev_action)
            lose_shift = int(prev_reward <= 0 and action != prev_action)

        else:
            win_stay = np.nan
            lose_shift = np.nan

        # Log results
        episode_log.append({
            "trial": t + 1,
            "deck": action,
            "reward": reward,
            "cumulative_reward": total_reward,
            "win_stay": win_stay,
            "lose_shift": lose_shift
        })

        obs = next_obs

        if done:
            break

    return episode_log


def run_condition(
    seed, 
    agent_type: str = "random",
    dysfunction: str = None, 
    agent_id: int = 1, 
    n_episodes: int = 20, 
    n_trials_per_ep: int = 10,
) -> pd.DataFrame:
    """
    Run a full condition (healthy, depleted, overactive) with reproducible seeding.
    """
    rng = np.random.default_rng(seed)
    env = IGTEnv(max_steps = n_episodes * n_trials_per_ep)

    # Load parameters from config
    params = config["healthy"] if dysfunction is None else config[dysfunction]

    # Initialise base policy parameters for the TD agent only
    if agent_type == "td":
        lr = params["policy_params"].get("learning_rate", 0.1)
        gamma = params["policy_params"].get("gamma", 0.9)
        epsilon = params["policy_params"].get("epsilon", 0.1)

        # Initialise the agent
        policy = TDPolicy(
            rng, 
            lr = lr, 
            gamma = gamma, 
            epsilon = epsilon, 
            num_states = 1, 
            num_actions = 4
        )

    # If random agent, set policy to RandomPolicy
    else:
        policy = RandomPolicy(rng, num_states = 1, num_actions = 4)
   
    # List to hold all episode data
    all_episodes = []

    for ep in range(n_episodes):
        ep_data = run_episode(env, policy, max_steps = n_trials_per_ep)
        df = pd.DataFrame(ep_data)
        df["episode"] = ep + 1

        all_episodes.append(df)

    df = pd.concat(all_episodes, ignore_index = True)

    df.insert(0, "agent", agent_id + 1)
    
    df["condition"] = dysfunction if dysfunction else "healthy"
    df["seed"] = seed
    
    return df


def main(baseline_agent_type: str = "random") -> None:
    """
    Execute experiments across multiple seeds and conditions, then save results for analysis.
    """
    # Assert valid agent type
    assert baseline_agent_type in ["random", "flat_td"], "Invalid baseline agent type. Choose 'random' or 'flat_td'."

    conditions = [None, "overactive", "depleted"]
    seeds = list(range(config["experiment_params"].get("num_seeds", 10)))
    agents_per_condition = config["experiment_params"].get("agents_per_condition", 50)
    num_episodes = config["experiment_params"].get("num_episodes", 10)
    trials_per_episode = config["experiment_params"].get("trials_per_episode", 20)

    if baseline_agent_type == "random":
        save_path = config["experiment_params"].get("rnd_exp_save_path", "logs/baseline/random")

    else:
        save_path = config["experiment_params"].get("flat_td_exp_save_path", "logs/baseline/flat_td")

    master_log = []

    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok = True)

    for cond in conditions:
        condition_name = cond if cond is not None else "healthy"

        print(f"\nRunning condition: {condition_name}")

        for s in tqdm(seeds, desc = f"Seeds ({condition_name})"):
            for agent in tqdm(range(agents_per_condition), desc = f"Agents ({condition_name}, seed = {s})", leave = False):
                df = run_condition(
                    seed = s, 
                    agent_type = baseline_agent_type,
                    dysfunction = cond, 
                    agent_id = agent, 
                    n_episodes = num_episodes, 
                    n_trials_per_ep = trials_per_episode
                )

                master_log.append(df)

        # Save each condition separately for clarity
        condition_name = cond if cond is not None else "healthy"
        results = pd.concat(master_log, ignore_index = True)

        results.to_csv(f"{save_path}/igt_{baseline_agent_type}_results_{condition_name}.csv", index = False)

        # Save deck preferences summary
        prefs = compute_deck_preferences(results)

        prefs.to_csv(f"{save_path}/igt_{baseline_agent_type}_deck_prefs_{condition_name}.csv", index = False)
        
        print(f"Completed: {condition_name} ({len(seeds)} seeds x {agents_per_condition} agents x {num_episodes * trials_per_episode} trials)")
        
        # Reset log for next condition
        master_log.clear()


if __name__ == "__main__":
    # Run an experiment with the random agent
    print(f"Starting baseline experiments with the random policy...")
    main(baseline_agent_type = "random")

    # Run an experiment with the flat TD agent
    print(f"\nStarting baseline experiments with the flat TD policy...")
    main(baseline_agent_type = "flat_td")
