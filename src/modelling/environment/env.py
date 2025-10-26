import numpy as np
import gymnasium as gym
from gymnasium import spaces


class IGTEnv(gym.Env):
    """
    Iowa Gambling Task (IGT) environment (Bechara et al., 1994 version).
    
    Observation:
        A simple vector with cumulative reward and deck selection counts.

    Actions:
        Type: Discrete(4)
        0 = Choose deck A
        1 = Choose deck B
        2 = Choose deck C
        3 = Choose deck D

    Reward:
        Stochastic reward based on payoff schedules.
    
    Episode termination:
        After max_steps draws (default: 100).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, max_steps: int = 100, render_mode: bool = None, seed: int = None) -> None:
        super().__init__()

        # Set the seed
        self.seed = np.random.default_rng(seed)

        # Define action (4 decks)
        self.action_space = spaces.Discrete(4)

        # Observation space: [cumulative_reward] + [deck counts for A, B, C, D]
        self.observation_space = spaces.Box(
            low = -np.inf, high = np.inf, shape = (5,), dtype = np.float32
        )

        # Deck payoff structures (classic Bechara version)
        self.decks = {
            0: [100, 100, 100, 100, 100, -1250, 100, 100, 100, -1250],  # A (bad)
            1: [100, 100, 100, -1250, 100, 100, -1250, 100, 100, 100],  # B (bad)
            2: [50, 50, -250, 50, 50, 50, 50, -250, 50, 50],            # C (good)
            3: [50, 50, 50, -250, 50, 50, 50, 50, 50, -250],            # D (good)
        }

        # Shuffle the decks to reflect uncertainty
        for deck in self.decks.values():
            self.seed.shuffle(deck)

        # Environment steps
        self.max_steps = max_steps
        self.current_step = 0

        # Set an initial balance for cumulative reward
        self.cumulative_reward = 2000

        # Tracking deck selections
        self.deck_counts = np.zeros(4, dtype = np.int32)
        self.deck_indices = np.zeros(4, dtype = np.int32)

        self.render_mode = render_mode


    def reset(self, seed: any = None, options: any = None) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to an initial state and return an initial observation.
        """
        super().reset(seed = seed)

        self.current_step = 0
        self.cumulative_reward = 2000.0
        self.deck_counts[:] = 0
        self.deck_indices[:] = 0

        return self._get_obs(), {}
    

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Perform one step in the environment given an action.
        """
        assert self.action_space.contains(action), f"Invalid action {action}"

        # Deterministic cycle reward
        idx = self.deck_indices[action]
        reward = self.decks[action][idx]

        # Increment deck index (loop back after 10)
        self.deck_indices[action] = (idx + 1) % len(self.decks[action])

        self.cumulative_reward += reward
        self.deck_counts[action] += 1
        self.current_step += 1

        obs = self._get_obs()
        terminated = self.current_step >= self.max_steps
        truncated = False

        return obs, float(reward), terminated, truncated, {}


    def _get_obs(self) -> np.ndarray:
        """
        Get a current observation of the environment.
        """
        return np.array(
            [self.cumulative_reward, *self.deck_counts], dtype = np.float32
        )


    def render(self) -> None:
        """
        Render the environment.
        """
        if self.render_mode == "human":
            print(
                f"Step: {self.current_step}, "
                f"Cumulative Reward: {self.cumulative_reward}, "
                f"Deck Counts: {self.deck_counts}"
            )


    def close(self):
        pass
