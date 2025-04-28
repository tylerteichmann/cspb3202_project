###############################################################################
#
# Author: Tyler Teichmann
# Purpose: Approximate Q-Learning agent with features specific
# to the "Blackjack-v1" environment.
# Date created: 2025-04-18
#
#


import numpy as np
from agent import ApproximateQLearner


class ApproximateQJack(ApproximateQLearner):
    def __init__(
        self, env, alpha, epsilon, epsilon_decay, final_epsilon,
        gamma=0.95, weights=None
    ):
        """Initialize an approximate Q-Learning RL agent with an zero 
        vector of weights for the "Blackjack-v1" environment.

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
        self.weights = np.zeros(4)


    def get_features(self, state, action):
        """
        Returns the feature vector for a given state and action.

        Feature vector is 4 elements:
        - Bias
        - Player's current sum of cards
        - Dealer card showing
        - Usable Ace
        """

        state_prime = self.get_next_state(state, action)

        player_sum = state[0]
        dealer_card = state[1]
        ace = state[2]

        features = np.array([
            1.0,
            player_sum / 21,
            dealer_card / 10,
            ace,
        ])

        return features