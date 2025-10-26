import numpy as np
from scipy.special import expit
from collections import defaultdict


class DopamineTDPolicy:
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
        num_options: int, 
        alpha: float = 0.1, 
        gamma: float = 0.9, 
        epsilon: float = 0.1,
    ) -> None:
        self.rng = rng
        self.num_states = num_states
        self.num_options = num_options
        self.alpha = alpha
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
        Update the Q-value table using the TD learning rule and dopamine modulation
        """
        # If the current state or next state is not in the Q-table, initialize it with zeros
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_options)

        if next_state not in self.q_table:
            self.q_table[next_state] = np.zeros(self.num_options)

        # TD update rule
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + (0 if done else self.gamma * self.q_table[next_state][best_next_action])
        td_delta = td_target - self.q_table[state][option]
        self.q_table[state][option] += self.alpha * td_delta


    def sample(self, state: tuple) -> int:
        """
        Sample an option based on maximum Q-value with epsilon-greedy exploration.
        """
        if state not in self.q_table:
            self.q_table[state] = np.zeros(self.num_options)

        if self.rng.uniform() < self.epsilon:
            return int(self.rng.integers(0, self.num_options))  # explore: random option
        
        else:
            return int(np.argmax(self.q_table[state]))  # exploit: best option


class SigmoidTermination:
    """
    Sigmoid-based termination function for options, which will transition to the lower-level policy.

    Attributes:
        rng (np.random): Random number generator.
        lr (float): Learning rate for updating the weights.
        num_states (int): Number of states in the environment.
        weights (dict): Weights for states.

    Methods:
        pmf(state): Probability mass function (PMF) that computes the logistic sigmoid of the state given the temperature.
        sample(state): Sample the environment based on whether a uniform random distribution is smaller than the PMF of the state.
        gradient(state): Compute the gradient of the state and its PMF.
        update(state, advantage): Update the weights based on the state and advantage.
    """
    def __init__(self, rng: np.random, lr: float, num_states: int) -> None:
        self.rng = rng
        self.lr = lr
        self.num_states = num_states
        self.weights = {}


    def pmf(self, state: tuple) -> np.ndarray:
        """
        Probability mass function (PMF) that computes the logistic sigmoid of the state given the temperature.
        """
        if state not in self.weights:
            self.weights[state] = 0.0

        # Return the logistic sigmoid function of the state's weights
        return expit(self.weights[state])
    

    def sample(self, state: tuple) -> int:
        """
        Sample the environment based on whether a uniform random distribution is smaller than the PMF of the state.
        """
        return int(self.rng.uniform() < self.pmf(state))
    

    def gradient(self, state: tuple) -> tuple[np.ndarray, tuple]:
        """
        Compute the gradient of the state and its PMF.
        """
        return self.pmf(state) * (1.0 - self.pmf(state)), state
    

    def update(self, state: tuple, advantage: float) -> None:
        """
        Update the weights based on the state and advantage.
        """
        if state not in self.weights:
            self.weights[state] = 0.0

        magnitude, action = self.gradient(state)
        self.weights[action] -= self.lr * magnitude * advantage