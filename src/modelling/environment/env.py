import numpy as np
import gymnasium as gym
from gymnasium import spaces


class IGTEnv(gym.Env):
    """
    Iowa Gambling Task (IGT) environment.
    
    Observation:
        A simple vector with cumulative reward and deck selection counts.
        (can be expanded with more features if needed)

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

    def __init__(self, max_steps: int = 100, render_mode: bool = None) -> None:
        super().__init__()

        # Define action (4 decks)
        self.action_space = spaces.Discrete(4)

        # Observation space: [cumulative_reward] + [deck counts for A, B, C, D]
        self.observation_space = spaces.Box(
            low = -np.inf, high = np.inf, shape = (5,), dtype = np.float32
        )

        # Deck payoff structures (classic Bechara version)
        self.decks = {
            0: {"gain": 100, "loss_prob": 0.1, "loss_amount": -1250},  # deck A
            1: {"gain": 100, "loss_prob": 0.1, "loss_amount": -1250},  # deck B
            2: {"gain": 50,  "loss_prob": 0.1, "loss_amount": -250},   # deck C
            3: {"gain": 50,  "loss_prob": 0.1, "loss_amount": -250},   # deck D
        }

        self.max_steps = max_steps
        self.current_step = 0
        self.cumulative_reward = 0
        self.deck_counts = np.zeros(4, dtype = np.int32)

        self.render_mode = render_mode


    def reset(self, seed: any = None, options: any = None) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to an initial state and return an initial observation.
        """
        super().reset(seed = seed)

        self.current_step = 0
        self.cumulative_reward = 0
        self.deck_counts = np.zeros(4, dtype = np.int32)
        obs = self._get_obs()
        info = {}

        return obs, info
    

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Perform one step in the environment given an action.
        """
        assert self.action_space.contains(action), "Invalid action"

        deck = self.decks[action]
        reward = deck["gain"]

        # Pick a random chance for loss
        if self.np_random.random() < deck["loss_prob"]:
            reward += deck["loss_amount"]

        self.cumulative_reward += reward
        self.deck_counts[action] += 1
        self.current_step += 1

        obs = self._get_obs()
        terminated = self.current_step >= self.max_steps
        truncated = False
        info = {}

        return obs, reward, terminated, truncated, info


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
