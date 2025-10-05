import numpy as np
from scipy.special import logsumexp
from collections import defaultdict


class SoftmaxPolicy:
    """
    Policy that will be used for options, where the update is given by a Bellman equation.

    Attributes:
        rng (np.random): Random number generator.
        lr (float): Learning rate for updating the weights.
        num_states (int): Number of states in the environment.
        num_actions (int): Number of possible actions.
        temperature (float): Temperature parameter for the softmax function.
        weights (defaultdict): Weights for state-action pairs.

    Methods:
        Q_u(state, action = None): State-action function that returns the action taken by the lower policy.
        pmf(state): Probability mass function (PMF) that computes the probabilities of the state given the temperature.
        sample(state): Sample the environment based random choice and a PMF with probabilities associated with each state.
        update(state, action, new_Q_u): Update the weights based on new Q_u, state, and action values.
    """
    def __init__(self, rng: np.random, lr: float, num_states: int, num_actions: int, temperature: float = 1.0) -> None:
        self.rng = rng
        self.lr = lr
        self.num_states = num_states
        self.num_actions = num_actions
        self.temperature = temperature
        self.weights = defaultdict()


    def Q_u(self, state: tuple, action: int = None) -> any:
        """
        State-action function `Q_u` that returns the action taken by the lower policy.
        """
        if state not in self.weights:
            self.weights[state] = np.zeros(self.num_actions)

        if action is None:
            return self.weights[state]
        
        else:
            return self.weights[state][action]
        

    def pmf(self, state: tuple) -> np.ndarray:
        """
        Probability mass function (PMF) that computes the probabilities of the state given the temperature.
        """
        exponent = self.Q_u(state) / self.temperature

        return np.exp(exponent - logsumexp(exponent))
    

    def sample(self, state: tuple) -> int:
        """
        Sample the environment based random choice and a PMF with probabilities associated with each state.
        """
        return int(self.rng.choice(self.num_actions, p = self.pmf(state)))
    

    def update(self, state: tuple, action: int, new_Q_u: np.ndarray) -> None:
        """
        Update the weights based on new `Q_u`, state, and action values.
        """
        actions_pmf = self.pmf(state)
        self.weights[state] -= self.lr * actions_pmf * new_Q_u
        self.weights[state][action] += self.lr * new_Q_u
    