###############################################################################
#
# Author: Tyler Teichmann
# Purpose: Learning agent for lander.py using Q-learning or 
# Approximate Q-Learning.
# Date created: 2025-04-18
#
#


import numpy as np
import gymnasium as gym
from collections import defaultdict


class LanderAgent:
    def __init__(
            self,
            env,
            alpha,
            initial_epsilon,
            epsilon_decay,
            final_epsilon,
            gamma=0.95
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
        # Environment
        self.env = env

        # Alpha and Gamma values
        self.alpha = alpha
        self.gamma = gamma

        # Epsilon values
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        self.training_error = []


    def get_action(self, state) -> int:
        """
        Returns the best action with probability (1 - epsilon)
        otherwise a random action with probability epsilon to ensure
        exploration.
        """

        rng = np.random.default_rng()

        if rng.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            return int(
                np.argmax(
                    [self.get_q_values(state, action_prime) for action_prime in range(4)]
                )
            )


    def get_q_values(self, state, action):
        """
        Returns the q value for a given state and action. Q(s,a) = w*f(s, a)
        """
        raise NotImplementedError()


    def update(self, state, action, reward, terminal, state_prime):
        """
        Update agent Q values
        """
        raise NotImplementedError()


    def decay_epsilon(self):
        """
        Decays the agent's epsilon value until it reaches the final value.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


class QLander(LanderAgent):
    def __init__(self, **args):
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
        super().__init__(**args)
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))


    def get_q_values(self, state, action):
        """
        Returns the q value for a given state and action.
        """
        return self.q_values[tuple(state)][action]


    def update(self, state, action, reward, terminal, state_prime):
        """
        Update agent Q values
        """

        # Max Q(s', a')
        q_value_prime = (
            (not terminal) * np.max(self.q_values[tuple(state_prime)])
        )

        # r + gamma(Max Q(s', a') - Q(s, a)
        temporal_difference = (
            reward + self.gamma * q_value_prime - 
            self.q_values[tuple(state)][action]
        )

        # Q(s, a) <-- Q(s, a) + alpha(r + gamma(Max Q(s', a') - Q(s, a))
        self.q_values[tuple(state)][action] = (
            self.q_values[tuple(state)][action] + self.alpha * temporal_difference
        )

        self.training_error.append(temporal_difference)


class ApproximateQLander(LanderAgent):
    def __init__(self, **args):
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
        super().__init__(**args)
        self.weights = np.zeros(8)


    def get_q_values(self, state, action):
        """
        Returns the q value for a given state and action. Q(s,a) = w*f(s, a)
        """
        features = self.get_features(state, action)
        q_value = np.dot(self.weights, features)

        return q_value


    def update(
            self,
            state: np.ndarray,
            action: int,
            reward: float,
            terminal: bool,
            state_prime: np.ndarray,
    ):
        """
        Update agent Q values
        """

        # Q(s, a)
        q_value = self.get_q_values(state, action)

        # Max Q(s', a')
        q_value_prime = (
            (not terminal) * 
            np.max(
                [self.get_q_values(state_prime, action_prime) for action_prime in range(4)]
            )
        )

        # r + gamma(Max Q(s', a')) - Q(s, a)
        temporal_difference = (
            reward + (self.gamma * q_value_prime) - q_value
        )

        #f(s, a)
        features = self.get_features(state, action)

        # w <-- w + alpha(r + gamma(Max Q(s', a')) - Q(s, a)) * f(s,a)
        self.weights = (
            self.weights + 
            (self.alpha * temporal_difference * features)
        )


        self.training_error.append(temporal_difference)


    def get_features(self, state, action):
        """
        Returns the feature vector for a given state and action.

        Feature vector is 7 elements:
        - Manhattan Distance to Pad
        - Lander Speed
        - Lander Tilt
        - Right Leg in Contact with Ground
        - Left Leg in Contact with Ground
        - Side Engine Firing
        - Main Engine Firing
        """

        bias = 1

        # State Values
        x, y = state[0], state[1]
        x_speed, y_speed = state[2], state[3]
        tilt = state[4]
        tilt_speed = state[5]
        rleg = state[6]
        lleg = state[7]
        side_engine = 0
        main_engine = 0

        # Adjustments to the state based on the action.
        if action == 1:
            tilt_speed += 0.03
            side_engine = -0.03
        elif action == 2:
            x_speed = x_speed + (0.3 * np.cos(tilt))
            y_speed = y_speed + (0.3 * np.sin(tilt))
            main_engine = -0.3
        elif action == 3:
            tilt_speed += 0.03
            side_engine = -0.03

        dist_to_pad = abs(x + (x_speed / 100)) + abs(y + (y_speed / 100))
        speed = abs(x_speed) + abs(y_speed)
        tilt += (tilt_speed / 100)

        features = np.array([
            bias,
            (dist_to_pad),
            (speed),
            (tilt),
            rleg,
            lleg,
            side_engine,
            main_engine
        ])

        return features