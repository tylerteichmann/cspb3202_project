###############################################################################
#
# Author: Tyler Teichmann
# Purpose: Learning agent for lander.py using Q-learning.
# Date created: 2025-04-18
#
#


import numpy as np
import gymnasium as gym
from collections import defaultdict


class QLanderAgent:
    def __init__(
            self,
            env: gym.Env,
            alpha: float,
            initial_epsilon: float,
            epsilon_decay: float,
            final_epsilon: float,
            gamma: float = 0.95,
    ):
        """Initialize a Q-Learning RL agent with an empty dictionary
        of state-action values (q_values), a learning rate and discount factor
        and an epsilon.

        Args:
            env: The training environment
            alpha: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            gamma: The discount factor for computing the Q-value
        """
        # Environment and intial Q values {[state]:[action]}
        self.env = env
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))

        # Alpha and Gamma values
        self.alpha = alpha
        self.gamma = gamma

        # Epsilon values
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # ??? not sure what for yet
        self.training_error = []


    def get_action(
            self,
            state: np.ndarray[
                np.float64, np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64, np.float64
                ],
        ) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure
        exploration.
        """

        if np.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(np.argmax(self.q_values[state]))


    def update(
            self,
            state: np.ndarray[
                np.float64, np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64, np.float64
                ],
            action: int,
            reward: float,
            terminal: bool,
            state_prime: np.ndarray[
                np.float64, np.float64, np.float64, np.float64,
                np.float64, np.float64, np.float64, np.float64
                ],
    ):
        """
        Update agent Q values
        """

        # Max Q(s', a')
        q_value_prime = (not terminal) * np.max(self.q_values[state_prime])

        # Q(s, a) <-- (1 - alpha)*(s, a) + alpha(r + gamma(Max Q(s', a')))
        self.q_values[state_prime][action] = (
            (1 - self.alpha) * self.q_values[state_prime][action] + 
            self.alpha * (reward + self.gamma * q_value_prime)
        )

        # Still not sure
        self.training_error.append(reward + self.gamma * q_value_prime)


    def decay_epsilon(self):
        """
        Decays the agent's epsilon value until it reaches the final value.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)