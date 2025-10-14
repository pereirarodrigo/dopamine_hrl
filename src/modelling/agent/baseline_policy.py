import numpy as np
from collections import defaultdict


class RandomPolicy:
    """
    Implementation of a random policy for action selection.

    Attributes:
        rng (np.random): Random number generator.
        num_states (int): Number of states in the environment.
        num_options (int): Number of options available.
        q_table (defaultdict): Q-value table for state-option pairs.

    Methods:
        sample(state): Sample an option uniformly at random.
    """
    def __init__(self, rng: np.random.Generator, num_states: int, num_actions: int) -> None:
        self.rng = rng
        self.num_states = num_states
        self.num_actions = num_actions
        self.q_table = defaultdict()


    def sample(self, state: tuple) -> int:
        """
        Sample an option uniformly at random.
        """
        # If the current state is not in the Q-table, initialize it with zeros
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)

        return int(self.rng.integers(0, self.num_actions))
    

class TDPolicy:
    """
    Implementation of temporal difference (TD) learning for policy estimation.

    Attributes:
        rng (np.random): Random number generator.
        num_states (int): Number of states in the environment.
        num_options (int): Number of options available.
        alpha (float): Learning rate for updating the Q-values.
        gamma (float): Discount factor for future rewards.
        epsilon (float): Exploration rate for epsilon-greedy action selection.
        q_table (defaultdict): Q-value table for state-option pairs.

    Methods:
        update(state, option, reward, next_state, done): Update the Q-value table using the TD learning rule.
        sample(state): Sample an option based on maximum Q-value with epsilon-greedy exploration.
    """
    def __init__(
        self, 
        rng: np.random.Generator,
        num_states: int, 
        num_actions: int, 
        lr: float = 0.1, 
        gamma: float = 0.9, 
        epsilon: float = 0.1
    ) -> None:
        self.rng = rng
        self.num_states = num_states
        self.num_actions = num_actions
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = defaultdict()


    def update(
        self, 
        state: tuple, 
        option: int, 
        reward: float, 
        next_state: tuple, 
        done: bool
    ) -> None:
        """
        Update the Q-value table using the TD learning rule.
        """
        # If the current state or next state is not in the Q-table, initialise it with zeros
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)

        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_actions)

        # TD update rule
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + (0 if done else self.gamma * self.q_table[next_state][best_next_action])
        td_delta = td_target - self.q_table[state][option]
        self.q_table[state][option] += self.lr * td_delta


    def sample(self, state: tuple) -> int:
        """
        Sample an option based on maximum Q-value with epsilon-greedy exploration.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_actions)

        # Randomly explore
        if self.rng.uniform() < self.epsilon:
            return int(self.rng.integers(0, self.num_actions))
        
        # Greedily exploit
        else:
            return int(np.argmax(self.q_table[state]))

    