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


    def prediction_error(self, expected_reward: float, actual_reward: float, scaling: float) -> float:
        """
        Calculate the reward prediction error (RPE) while scaling it based on current dopamine level.
        """
        rpe = actual_reward - expected_reward

        return rpe * (1 + scaling * (self.dopamine_level - self.baseline_dopamine))


    def update_dopamine(self, reward: float, rpe: float, sensitivity: float) -> float:
        """
        Update dopamine level based on reward, prediction error, and punishment sensititivy. The update method will also depend on the 
        mode of dopamine signalling.
        """
        # Dopamine dysfunction effects
        if self.has_dysfunction == "overactive":
            # Punishments matter less, rewards and RPE are amplified
            sensitivity = max(0.0, sensitivity - 0.5)

            # Use asymmetric exaggeration for biological realism
            rpe *= (1.0 + abs(rpe) * 0.001)
            
            # If reward is negative, dampen it
            if reward < 0:
                reward *= (1 - 0.5 * sensitivity)

            # Otherwise, amplify it to mimic addiction-like behaviour
            else:
                reward *= (1.5 * sensitivity)

        # Punishments matter more, rewards and RPE are blunted
        elif self.has_dysfunction == "depleted":
            sensitivity = min(1.0, sensitivity + 0.5)

            # Asymmetric dampening for biological realism
            rpe *= 0.5
            
            # If reward is negative, amplify it to mimic oversensitivity to punishment (Parkinson's-like)
            if reward < 0:
                reward *= (1 + sensitivity)

            # Otherwise, dampen it
            else:
                reward *= (0.5 * sensitivity)

        # Dopamine state update
        delta = reward + rpe - self.dopamine_level
        self.dopamine_level += self.learning_rate * delta

        # Clamp to [0, 1] range for stability
        self.dopamine_level = np.clip(self.dopamine_level, 0.0, 1.0)

        return self.dopamine_level
        

    def reset(self) -> None:
        """
        Reset dopamine level to baseline.
        """
        self.dopamine_level = self.baseline_dopamine