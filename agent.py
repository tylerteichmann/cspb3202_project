###############################################################################
#
# Author: Tyler Teichmann
# Purpose: Basic agent and random agent class for various environments.
# Date created: 2025-04-18
#
#


import numpy as np
from collections import defaultdict


class Agent:
    def __init__(self, env):
        """Super class for a lander agent with a learning rate, epsilon,
        and a discount factor

        Args:
            env: The training environment
        """
        # Sets the environment and set of actions an agent can take
        self.env = env
        self.action_space = env.action_space.n

        # Table for saving transitions
        self.transitions = defaultdict(lambda: [defaultdict(lambda: 0) for action in range(self.action_space)])


    def get_next_state(self, state, action):
        possible_states = self.transitions[tuple(state)][action].keys()
        state_prime = np.array([state for state in possible_states])

        return state_prime


    def get_action(self, state) -> int:
        """
        Returns the action based on agent priorities
        Should be implimented by subclass.
        """
        raise NotImplementedError()


    def update(self, state, action, reward, terminal, state_prime):
        """
        Update agent Q values
        Should be implimented by subclass.
        """
        raise NotImplementedError()


    def decay_epsilon(self):
        """
        Decays the agent's epsilon value until it reaches the final value.
        """
        return



class RandomAgent(Agent):
    def __init__(self, env):
        """
        Random Agent
        """
        super().__init__(env)

    def get_action(self, state):
        return self.env.action_space.sample()

    def update(self, state, action, reward, terminal, state_prime):
        # Record transitions
        self.transitions[tuple(state)] = {action:{state_prime:reward}}


class MaxAgent(Agent):
    def __init__(self, env):
        """
        Agent that tries to maximize reward
        """
        super().__init__(env)


    def get_action(self, state):
        rewards = np.zeros(self.action_space)

        for action in range(self.action_space):
            next_states = self.get_next_state(state, action)
            r = np.array([self.transitions[tuple(state)][action][tuple(next_state)] for next_state in next_states])
            rewards[action] = r.mean()

        return np.argmax(rewards)

    def update(self, state, action, reward, terminal, state_prime):
        # Record transitions with rewards R(s, a, s', r)
        self.transitions[tuple(state)][action][tuple(state_prime)] = reward

class QLearner(Agent):
    def __init__(
        self, env, alpha, epsilon, epsilon_decay, final_epsilon, gamma=0.95
    ):
        """Initialize a Q-Learning RL agent with an empty dictionary
        of state-action values (q_values).

        Args:
            env: The training environment
            alpha: The learning rate
            initial_epsilon: The initial epsilon value
            epsilon_decay: The decay for epsilon
            final_epsilon: The final epsilon value
            gamma: The discount factor for computing the Q-value
        """
        super().__init__(env)

        # Alpha and Gamma values
        self.alpha = alpha
        self.gamma = gamma

        # Epsilon values
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        # Empty dictionary for storing q-values
        self.q_values = defaultdict(lambda: np.zeros(self.action_space))


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
                    [self.get_q_values(state, action)
                     for action in range(self.action_space)]
                )
            )


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
            reward + self.gamma * q_value_prime
            - self.q_values[tuple(state)][action]
        )

        # Q(s, a) <-- Q(s, a) + alpha(r + gamma(Max Q(s', a') - Q(s, a))
        self.q_values[tuple(state)][action] = (
            self.q_values[tuple(state)][action]
            + self.alpha * temporal_difference
        )


    def decay_epsilon(self):
        """
        Decays the agent's epsilon value until it reaches the final value.
        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)


class ApproximateQLearner(QLearner):
    def __init__(
        self, env, alpha, epsilon, epsilon_decay, final_epsilon,
        gamma=0.95, weights=None
    ):
        """
        Initialize an approximate Q-Learning RL agent with an zero 
        vector of weights. Shoult not be implimented alone. Use a class
        that extends this for the appropriate environment

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

        # Weight vector for each action
        self.weights = None


    def get_q_values(self, state, action):
        """
        Returns the q value for a given state and action.
        Q(s,a) = w*f(s, a)
        """
        features = self.get_features(state, action)
        q_value = np.dot(self.weights, features)

        return q_value


    def update(self, state, action, reward, terminal, state_prime):
        """
        Update agent Q values and record transitions
        """

        # Record transitions
        self.transitions[tuple(state)][action][tuple(state_prime)] = reward

        # Q(s, a)
        q_value = self.get_q_values(state, action)

        # Max Q(s', a')
        q_value_prime = (
            (not terminal) * 
            np.max(
                [self.get_q_values(state_prime, action_prime)
                 for action_prime in range(self.action_space)]
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


    def get_features(self, state, action):
        """
        Returns the feature vector for a given state and action.
        Should be implimented by subclass. Unique to environment
        """
        raise NotImplementedError()