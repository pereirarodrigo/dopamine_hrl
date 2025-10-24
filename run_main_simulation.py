import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from yaml import safe_load
from src.config import MODEL_CONFIG
from src.modelling.environment import IGTEnv
from src.utils.deck import compute_deck_preferences
from src.modelling.modulation import DopamineModulation
from src.modelling.agent import (
    SoftmaxPolicy,
    DopamineTDPolicy, 
    SigmoidTermination
)


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
    punishment_sensitivity: float = 0.5,
    epsilon_decay: float = 0.99,
    epsilon_min: float = 0.05,
    epsilon_max: float = 1.0,
    reset_env: bool = False
) -> list[dict]:
    """
    Run a single IGT episode and collect detailed trial-level data.
    """
    # Reset environment at the beginning of each episode
    if reset_env:
        obs, _ = env.reset()
        env.cumulative_reward = 0.0
        env.current_step = 0

    else:
        obs = env._get_obs()

    total_reward = 0.0
    episode_log = []
    
    # Persistent total reward across episodes
    total_reward = env.cumulative_reward
    episode_log = []

    for t in range(max_steps):
        state = tuple(obs.round(2))
        dop_level = getattr(dopamine_mod, "dopamine_level", 0.5)

        # Adapt learning rate softly with dopamine to never vanish
        effective_alpha = base_alpha * (0.5 + 0.5 * dop_level)
        effective_alpha = np.clip(effective_alpha, 0.001, 0.3)

        # Decrease temperature slightly with dopamine (stability from confidence)
        effective_temp = base_temperature * (1.2 - 0.4 * dop_level)
        effective_temp = np.clip(effective_temp, 0.05, 2.5)

        high_policy.alpha = effective_alpha
        low_policy.temperature = effective_temp

        # Epsilon-greedy decay with boundaries ---
        high_policy.epsilon = max(epsilon_min, min(high_policy.epsilon * epsilon_decay, epsilon_max))

        # Select and execute actions
        option = high_policy.sample(state)
        terminate = term_func.sample(state)
        action = low_policy.sample(state)

        # Step in the environment (no reset between episodes)
        next_obs, reward, done, _, _ = env.step(action)

        # Clip reward to avoid extreme outliers
        reward = np.clip(reward, -400, 200)

        # Update total reward
        total_reward += reward

        # Dopamine and RPE
        expected = np.mean(high_policy.q_table[state])
        rpe = dopamine_mod.prediction_error(expected, reward, scaling=reward_scaling)
        dop_level = dopamine_mod.update_dopamine(reward, rpe, sensitivity=punishment_sensitivity)

        # Policy updates
        high_policy.update(state, option, reward, tuple(next_obs.round(2)), punishment_sensitivity, reward_scaling, done)
        low_policy.update(state, action, reward)
        term_func.update(state, advantage = reward)

        # Compute win-stay/lose-shift (if not first trial)
        if t > 0:
            prev_reward = episode_log[-1]["reward"]
            prev_action = episode_log[-1]["deck"]
            win_stay = bool(prev_reward > 0 and action == prev_action)
            lose_shift = bool(prev_reward <= 0 and action != prev_action)

        else:
            win_stay = np.nan
            lose_shift = np.nan

        # Log trial
        episode_log.append({
            "trial": env.current_step,  # cumulative trial number
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

    # Tonic dopamine drift between episodes
    dopamine_mod.dopamine_level += 0.3 * (dopamine_mod.baseline_dopamine - dopamine_mod.dopamine_level)
    dopamine_mod.dopamine_level = np.clip(dopamine_mod.dopamine_level, 0.05, 0.95)

    return episode_log


def run_condition(seed, dysfunction: str = None, agent_id: int = 1, n_episodes: int = 20, n_trials_per_ep: int = 10) -> pd.DataFrame:
    """
    Run a full condition (healthy, depleted, overactive) with reproducible seeding.
    """
    rng = np.random.default_rng(seed)
    env = IGTEnv(max_steps = n_trials_per_ep)

    # Load parameters from model config
    params = MODEL_CONFIG["healthy"] if dysfunction is None else MODEL_CONFIG[dysfunction]

    # Initialise base policy parameters
    base_alpha = params["policy_params"].get("learning_rate", 0.1)
    gamma = params["policy_params"].get("gamma", 0.9)
    base_epsilon = params["policy_params"].get("epsilon", 0.1)
    epsilon_decay = params["policy_params"].get("epsilon_decay", 0.99)
    epsilon_min = params["policy_params"].get("epsilon_min", 0.05)
    epsilon_max = params["policy_params"].get("epsilon_max", 1.0)
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
            punishment_sensitivity = punishment_sensitivity,
            epsilon_decay = epsilon_decay,
            epsilon_min = epsilon_min,
            epsilon_max = epsilon_max,
            reset_env = True
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
    seeds = list(range(MODEL_CONFIG["experiment_params"].get("num_seeds", 10)))
    agents_per_condition = MODEL_CONFIG["experiment_params"].get("agents_per_condition", 50)
    num_episodes = MODEL_CONFIG["experiment_params"].get("num_episodes", 10)
    trials_per_episode = MODEL_CONFIG["experiment_params"].get("trials_per_episode", 20)
    save_path = MODEL_CONFIG["experiment_params"].get("hrl_exp_save_path", "logs/hrl")
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
