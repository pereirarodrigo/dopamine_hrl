import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from yaml import safe_load
from src.modelling.environment import IGTEnv
from src.utils.deck import compute_deck_preferences
from src.modelling.modulation import DopamineModulation
from src.modelling.agent import (
    SoftmaxPolicy,
    DopamineTDPolicy, 
    SigmoidTermination
)


# Simulation parameters
with open("default_model_params.yaml", "r") as f:
    config = safe_load(f)


def run_episode(
    env: IGTEnv, 
    high_policy: DopamineTDPolicy, 
    low_policy: SoftmaxPolicy, 
    term_func: SigmoidTermination, 
    dopamine_mod: DopamineModulation, 
    max_steps: int = 200,
    base_alpha: float = 0.1,
    base_temperature: float = 1.0,
    reward_scaling: float = 0.5,
    punishment_sensitivity: float = 0.5
) -> list[dict]:
    """
    Run a single IGT episode and collect detailed trial-level data.
    """
    obs, _ = env.reset()
    total_reward = 0.0
    episode_log = []

    for t in range(max_steps):
        state = tuple(obs.round(2))

        # Dopamine-modulated parameters
        dop_level = getattr(dopamine_mod, "dopamine_level", 1.0)

        # Adjust learning rate and temperature based on dopamine level
        # Low dopamine, for example, could lead to lower learning rates and more rigid choices
        # The opposite occurs for high dopamine levels, leading to more exploratory behaviour
        effective_alpha = max(0.001, base_alpha * dop_level)
        effective_temp = max(0.005, base_temperature * (1.5 - dop_level))

        high_policy.alpha = effective_alpha
        low_policy.temperature = effective_temp

        # Policy sampling
        option = high_policy.sample(state)
        terminate = term_func.sample(state)
        action = low_policy.sample(state)

        # Environment step
        next_obs, reward, done, _, _ = env.step(action)
        total_reward += reward

        # Dopamine and RPE calculations
        expected = np.mean(high_policy.q_table[state])
        rpe = dopamine_mod.prediction_error(expected, reward, scaling = reward_scaling)
        dop_level = dopamine_mod.update_dopamine(reward, rpe, sensitivity = punishment_sensitivity)

        # Policy updates
        high_policy.update(
            state, 
            option, 
            reward, 
            tuple(next_obs.round(2)), 
            punishment_sensitivity, 
            reward_scaling, 
            done
        )
        low_policy.update(state, action, reward)
        term_func.update(state, advantage = reward)

        # Log results
        # Compute win-stay/lose-shift if not first trial
        if t > 0:
            prev_reward = episode_log[-1]["reward"]
            prev_action = episode_log[-1]["deck"]
            win_stay = bool(prev_reward > 0 and action == prev_action)
            lose_shift = bool(prev_reward <= 0 and action != prev_action)

        else:
            win_stay = np.nan
            lose_shift = np.nan

        # Log results
        episode_log.append({
            "trial": t + 1,
            "deck": action,
            "reward": reward,
            "learning_rate": effective_alpha,
            "temperature": effective_temp,
            "expl_rate": high_policy.epsilon,
            "rpe": rpe,
            "dopamine": dop_level,
            "cumulative_reward": total_reward,
            "win_stay": win_stay,
            "lose_shift": lose_shift
        })

        obs = next_obs

        if done:
            break

    return episode_log


def run_condition(seed, dysfunction: str = None, agent_id: int = 1, n_episodes: int = 20, n_trials_per_ep: int = 10) -> pd.DataFrame:
    """
    Run a full condition (healthy, depleted, overactive) with reproducible seeding.
    """
    rng = np.random.default_rng(seed)
    env = IGTEnv(max_steps = n_episodes * n_trials_per_ep)

    # Load parameters from config
    params = config["healthy"] if dysfunction is None else config[dysfunction]

    # Initialise base policy parameters
    base_alpha = params["policy_params"].get("learning_rate", 0.1)
    gamma = params["policy_params"].get("gamma", 0.9)
    base_epsilon = params["policy_params"].get("epsilon", 0.1)
    base_temperature = params["policy_params"].get("temperature", 1.0)

    # Initialise dopamine parameters
    reward_scaling = params["dopamine_modulation"].get("scaling", 0.5)
    punishment_sensitivity = params["dopamine_modulation"].get("sensitivity", 0.5)

    # Initialise an agent and dopamine modulator
    dopamine_mod = DopamineModulation(learning_rate = base_alpha, has_dysfunction = dysfunction)
    high_policy = DopamineTDPolicy(
        rng, 
        num_states = 1, 
        num_options = 4, 
        alpha = base_alpha, 
        gamma = gamma, 
        epsilon = base_epsilon,
        dopamine_modulation = dopamine_mod)
    low_policy = SoftmaxPolicy(rng, lr = base_alpha, num_states = 1, num_actions = 4, temperature = base_temperature)
    term_func = SigmoidTermination(rng, lr = 0.05, num_states = 1)

    # List to hold all episode data
    all_episodes = []

    for ep in range(n_episodes):
        ep_data = run_episode(
            env, 
            high_policy, 
            low_policy, 
            term_func,
            dopamine_mod, 
            max_steps = n_trials_per_ep,
            base_alpha = base_alpha,
            base_temperature = base_temperature,
            reward_scaling = reward_scaling,
            punishment_sensitivity = punishment_sensitivity
        )
        df = pd.DataFrame(ep_data)

        df.insert(0, "episode", ep + 1)
        all_episodes.append(df)

    df = pd.concat(all_episodes, ignore_index = True)

    df.insert(0, "agent", agent_id + 1)
    
    df["condition"] = dysfunction if dysfunction else "healthy"
    df["seed"] = seed
    
    return df


def main() -> None:
    """
    Execute experiments across multiple seeds and conditions, then save results for analysis.
    """
    conditions = [None, "overactive", "depleted"]
    seeds = list(range(config["experiment_params"].get("num_seeds", 10)))
    agents_per_condition = config["experiment_params"].get("agents_per_condition", 50)
    num_episodes = config["experiment_params"].get("num_episodes", 10)
    trials_per_episode = config["experiment_params"].get("trials_per_episode", 20)
    save_path = config["experiment_params"].get("hrl_exp_save_path", "logs/hrl")
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
                    dysfunction = cond, 
                    agent_id = agent, 
                    n_episodes = num_episodes, 
                    n_trials_per_ep = trials_per_episode
                )

                master_log.append(df)

        # Save each condition separately for clarity
        condition_name = cond if cond is not None else "healthy"
        results = pd.concat(master_log, ignore_index = True)

        results.to_csv(f"{save_path}/igt_dopamine_hrl_results_{condition_name}.csv", index = False)

        # Save deck preferences summary
        prefs = compute_deck_preferences(results)

        prefs.to_csv(f"{save_path}/igt_dopamine_hrl_deck_prefs_{condition_name}.csv", index = False)
        
        print(f"Completed: {condition_name} ({len(seeds)} seeds x {agents_per_condition} agents x {num_episodes * trials_per_episode} trials)")
        
        # Reset log for next condition
        master_log.clear()


if __name__ == "__main__":
    main()
