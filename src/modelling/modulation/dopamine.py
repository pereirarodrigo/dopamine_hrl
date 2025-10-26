import numpy as np


class DopamineModulation:
    """
    Class to simulate dopamine modulation in response to rewards and prediction errors.

    Attributes:
        baseline_dopamine (float): Baseline level of dopamine.
        learning_rate (float): Learning rate for updating dopamine levels.
        dopamine_level (float): Current level of dopamine.
        has_dysfunction (str or None): If dopamine signalling suffers from any dysfunction (None, 'overactive', or 'depleted').

    Methods:
        update_dopamine(reward, prediction_error): Update dopamine level based on reward and prediction error.
        prediction_error(expected_reward, actual_reward): Calculate the reward prediction error (RPE).
        reset(): Reset dopamine level to baseline.
    """
    def __init__(self, has_dysfunction: str | None, baseline_dopamine: float = 0.8, learning_rate: float = 0.1) -> None:
        self.has_dysfunction = has_dysfunction
        self.baseline_dopamine = baseline_dopamine
        self.learning_rate = learning_rate
        self.dopamine_level = baseline_dopamine

        # Tonic drift and decay
        self.recovery_rate = 0.010
        self.decay_rate = 0.008

        # Dysfunction-specific tuning
        if self.has_dysfunction == "overactive":
            self.recovery_rate = 0.030   # quick to rebuild
            self.decay_rate = 0.1    # dopamine bleeds very quickly

        elif self.has_dysfunction == "depleted":
            self.recovery_rate = 0.009   # very slow to rebuild
            self.decay_rate = 0.005       # dopamine bleeds slowly

        # Scaling limits
        self.max_rpe_effect = 0.3
        self.max_reward_influence = 0.7

        # Expanded physiological range
        self.clip_range = (0.05, 1.2)


    def prediction_error(self, expected_reward: float, actual_reward: float, scaling: float) -> float:
        """
        Calculate the reward prediction error (RPE) while scaling it based on current dopamine level.
        """
        rpe = actual_reward - expected_reward

        # Dysfunction-specific scaling to decrease RPE contribution
        if self.has_dysfunction == "overactive":
            scaling *= (0.2 if self.dopamine_level > 0.8 else 1.4)

        # Make RPEs more expressive for healthy agents
        elif self.has_dysfunction is None:
            scaling *= 1.1

        return scaling * rpe


    def update_dopamine(self, reward: float, rpe: float, sensitivity: float) -> tuple[float, float]:
        """
        Update dopamine level based on reward, prediction error, and punishment sensititivy. The update method will also depend on the 
        mode of dopamine signalling.
        """
        # Base reward + RPE influence
        # Reward effect is squashed to avoid extreme spikes
        # The RPE effect is also bounded and has some noise to reflect biological variability
        reward_effect = np.tanh(reward) * self.max_reward_influence
        rpe_effect = (np.clip(rpe, -1, 1) + np.random.uniform(-0.1, 0.1)) * self.max_rpe_effect
        delta = reward_effect + rpe_effect

        # Dysfunction-specific tuning
        if self.has_dysfunction == "overactive":
            # Amplify positive deltas, blunt negatives
            delta *= 2.5 if reward > 0 else 0.5
            sensitivity *= 0.5

        elif self.has_dysfunction == "depleted":
            # Flatten positive response, exaggerate negative
            delta *= 0.2 if reward > 0 else 1.5
            sensitivity *= 1.5

        # Apply learning
        self.dopamine_level += (
            self.recovery_rate * (self.baseline_dopamine - self.dopamine_level) + 0.05 * self.learning_rate * delta
        )
        self.dopamine_level = np.clip(self.dopamine_level, *self.clip_range)

        # Dopamine-weighted reward perception
        # Overactive leads to inflated positives and underweight negatives
        # Depleted leads to deflated positives and overweight negatives
        if self.has_dysfunction == "overactive":
            perceived_reward = reward * (1 + 0.5 * (self.dopamine_level - 0.8))
            
        elif self.has_dysfunction == "depleted":
            perceived_reward = reward * (0.2 - 1.0 * (self.dopamine_level - 0.5))

        else:
            # To avoid overpenalising healthy agents, add a small boost to dopamine level (an "optimism term")
            perceived_reward = reward * (self.dopamine_level + 0.4)

        # Punishment sensitivity amplification
        # Some noise is added to reflect variability in negative perception
        if reward < 0:
            if self.has_dysfunction == "depleted":
                perceived_reward *= (1 + 1.2 * sensitivity + np.random.uniform(0, 0.3))

            elif self.has_dysfunction == "overactive":
                perceived_reward *= (1 + 0.05 * sensitivity + np.random.uniform(0, 0.5))

            else:
                perceived_reward *= (1 + 0.2 * sensitivity + np.random.uniform(0, 0.2))

        return self.dopamine_level, perceived_reward
        

    def reset(self) -> None:
        """
        Reset dopamine level to baseline.
        """
        self.dopamine_level = self.baseline_dopamine