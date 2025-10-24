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
    def __init__(self, has_dysfunction: str | None, baseline_dopamine: float = 0.5, learning_rate: float = 0.1) -> None:
        self.has_dysfunction = has_dysfunction
        self.baseline_dopamine = baseline_dopamine
        self.learning_rate = learning_rate
        self.dopamine_level = baseline_dopamine

        # Tonic drift toward baseline parameters
        self.recovery_rate = 0.02

        # Maximum magnitude of RPE impact
        self.max_rpe_effect = 0.3

        # Cap for reward influence
        self.max_reward_influence = 0.4

        # Clipping range for dopamine level
        self.clip_range = (0.05, 0.95)


    def prediction_error(self, expected_reward: float, actual_reward: float, scaling: float) -> float:
        """
        Calculate the reward prediction error (RPE) while scaling it based on current dopamine level.
        """
        raw_rpe = actual_reward - expected_reward
        
        # Compress into [-1, 1] range
        rpe_bounded = np.tanh(raw_rpe / 500.0)

        return scaling * rpe_bounded


    def update_dopamine(self, reward: float, rpe: float, sensitivity: float) -> float:
        """
        Update dopamine level based on reward, prediction error, and punishment sensititivy. The update method will also depend on the 
        mode of dopamine signalling.
        """
        # Normalise reward contribution (smaller impact)
        reward_effect = np.tanh(reward / 500.0) * self.max_reward_influence

        # Core RPE influence
        rpe_effect = np.clip(rpe, -1, 1) * self.max_rpe_effect

        # Combine both influences
        delta = reward_effect + rpe_effect

        # Dopamine is strengthened, punishment sensitivity is weakened
        if self.has_dysfunction == "overactive":
            delta *= 1.5
            sensitivity *= 0.75

        # Punishments matter more, dopamine is blunted and sensitivity increased
        elif self.has_dysfunction == "depleted":
            delta *= 0.6
            sensitivity *= 1.5

        # Adjust with sensitivity (punishment bias)
        if reward < 0:
            delta *= (1 - 0.5 * sensitivity)
        
        # Update dopamine (phasic + tonic)
        self.dopamine_level += self.learning_rate * delta

        # Recovery toward baseline (homeostasis)
        self.dopamine_level += self.recovery_rate * (self.baseline_dopamine - self.dopamine_level)

        # Clamp to physiological range
        self.dopamine_level = np.clip(self.dopamine_level, *self.clip_range)

        return self.dopamine_level
        

    def reset(self) -> None:
        """
        Reset dopamine level to baseline.
        """
        self.dopamine_level = self.baseline_dopamine