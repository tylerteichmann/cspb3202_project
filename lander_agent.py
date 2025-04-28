###############################################################################
#
# Author: Tyler Teichmann
# Purpose: Learning agents for game.py using the "LunarLander-v3" environment
# Agetn options are Q-learning or Approximate Q-Learning.
# Date created: 2025-04-18
#
#


import numpy as np
from agent import ApproximateQLearner
import util


class ApproximateQLander(ApproximateQLearner):
    def __init__(
        self, env, alpha, epsilon, epsilon_decay, final_epsilon,
        gamma=0.95, weights=None
    ):
        """Initialize an approximate Q-Learning RL agent with an zero 
        vector of weights for the "LunarLander-v3" environment.

        Args:
            env: The training environment
            alpha: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            gamma: The discount factor for computing the Q-value
        """
        super().__init__(
            env,
            alpha,
            epsilon,
            epsilon_decay,
            final_epsilon,
            gamma=0.95
        )

        # Weight vector based on number of features
        self.weights = np.zeros(7)


    def get_features(self, state, action):
        """
        Returns the feature vector for a given state and action.

        Feature vector is  elements:
        - Bias
        - Manhattan Distance to Pad
        - Lander Speed
        - Lander Tilt
        - Lander Tilt Speed
        - Legs in contact with ground
        - Engines firing
        """

        # First, get the features for the next state after taking an action
        state_prime = self.get_next_state(state, action)

        # If the array is empty, the agent does not know the outcome.
        # In this case, use the current state
        # If multiple states have been observed, randomly select one
        # Otherwise, flatten the array
        if state_prime.size == 0:
            state_prime = state
        elif state_prime.shape[0] > 1:
            print(state_prime)
            idx = np.random.randint(0, state_prime.shape[0])
            state_prime = state_prime[idx,]
        else:
            state_prime = state_prime[0,]

        dist_to_pad = abs(state_prime[0]) + abs(state_prime[1])
        speed = abs(state_prime[2]) + abs(state_prime[3])
        tilt = abs(state_prime[4])
        tilt_speed = abs(state_prime[5])
        legs = (state_prime[6] + state_prime[7]) / 2

        if action == 0:
            engines = 1
        elif action == 2:
            engines = 0
        else:
            engines = 0.5

        features = np.array([
            dist_to_pad,
            speed,
            tilt,
            tilt_speed,
            legs,
            engines,
        ])

        features = features / np.linalg.norm(features)
        features = np.insert(features, 0, 1.0)

        return features



#######################################
#    Implemented for testing only     #
#######################################


###############################################################################
#
# PacQlander is a second implementation following the design in HW4A to test
# performance.
# 
#


class PacQLander(ApproximateQLearner):
    def __init__(
        self, env, alpha, epsilon, epsilon_decay, final_epsilon,
        gamma=0.95, weights=None
    ):
        """Initialize an approximate Q-Learning RL agent with an zero 
        vector of weights.

        Args:
            env: The training environment
            alpha: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            gamma: The discount factor for computing the Q-value
        """
        super().__init__(
            env,
            alpha,
            epsilon,
            epsilon_decay,
            final_epsilon,
            gamma=0.95
        )


    def get_q_values(self, state, action):
        """
        Returns the q value for a given state and action.
        Q(s,a) = w*f(s, a)
        """
        return self.get_features(state, action) * self.weights


    def update(self, state, action, reward, terminal, next_state):
        """
        Update agent Q values and record transitions
        """
        # Record transitions
        self.transitions[tuple(state)] = {action:next_state}

        next_actions = self.action_space

        if terminal:
            next_q_value = 0
        else:
            next_q_value = max([self.get_q_values(next_state, next_action) for next_action in range(next_actions)])


        difference = (reward + (self.gamma * next_q_value)) - self.get_q_values(state, action)
        features = self.get_features(state, action)
        for feature in features.keys():
            self.weights[feature] = self.weights[feature] + (self.alpha * difference * features[feature])

        self.training_error.append(difference)


    def get_features(self, state, action):
        """
        Returns the feature vector for a given state and action.

        Feature vector is 6 elements:
        - Bias
        - Manhattan Distance to Pad
        - Lander Speed
        - Lander Tilt
        - Legs in contact with ground
        - Engines firing
        """

        pad = [0, 0]
        state_prime = self.get_next_state(state, action)

        next_x, next_y = state_prime[0], state_prime[1]
        next_x_speed, next_y_speed = state_prime[2], state_prime[3]
        next_tilt = state_prime[4]
        next_tilt_speed = state_prime[5]


        dist_to_pad = abs(next_x - pad[0]) + abs(next_y - pad[1])
        speed = abs(next_x_speed) + abs(next_y_speed)
        tilt = abs(next_tilt)
        tilt_speed = abs(next_tilt_speed)
        legs = (state_prime[6] + state_prime[7]) / 2

        if action == 0:
            engines = 1.0
        elif action == 2:
            engines = 0.0
        else:
            engines = 0.5


        features = util.Counter()
        features["bias"] = 1.0
        features["dist_to_pad"] = dist_to_pad / 3
        features["speed"] = speed / 10
        features["tilt"] = tilt / 3.14
        features["tilt_speed"] = tilt_speed / 5
        features["legs"] = legs
        features["engines"] = engines

        return features