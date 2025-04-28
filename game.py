###############################################################################
#
# Author: Tyler Teichmann
# Purpose: Simple RL agent for the lunar lander environment in gymnasium API.
# Developed for CSPB 3202: Intro to Artificial Intelligence.
# Date created: 2025-04-18
#
#


import sys
import gymnasium as gym
from tqdm import tqdm
from agent import RandomAgent, MaxAgent, QLearner
from lander_agent import ApproximateQLander
from jack_agent import ApproximateQJack
from visualize import visualize



def main(agent_type="Random", game="LunarLander-v3", render_mode=None):

    ###########################################################################
    # Environment setup
    #
    # Specify enviromnent parameters and make the evironment. Wrap the
    # environment to record episode stats and clips
    #
    n_episodes = 10000
    env = gym.make(game, render_mode=render_mode)
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)


    ###########################################################################
    # Agent Setup
    #
    # Set the initial parameters for the agent to include the learning rate, 
    # start epsilon, epsilon decay, and final epsilon.
    #
    # Then, treate the model based on the chosen option. If no option was 
    # chosen, create a random agent.
    #
    alpha = 0.001
    start_epsilon = 0.5
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.01

    if agent_type == "QLearner":
        agent = QLearner(
            env,
            alpha,
            start_epsilon,
            epsilon_decay,
            final_epsilon
        )
    elif agent_type == "Approximate":
        if game == "Blackjack-v1":
            agent = ApproximateQJack(
                env,
                alpha,
                start_epsilon,
                epsilon_decay,
                final_epsilon
            )
        else:
            agent = ApproximateQLander(
                env,
                alpha,
                start_epsilon,
                epsilon_decay,
                final_epsilon
            )
    elif agent_type == "Max":
        agent = MaxAgent(env)
    else:
        agent = RandomAgent(env)

    ###########################################################################
    # Run itterations
    #
    # 1. Reset the environment
    # 2. While episode is not terminal
    #   a. Agent retrieves an action
    #   b. Takes that action and observes new state and reward
    #   c. Updates its knowledge (q-table, features)
    # 3. Decay learning epsilon when episode ends
    #
    for episode in tqdm(range(n_episodes)):
        state, _ = env.reset()
        episode_over = False

        while not episode_over:
            action = agent.get_action(state)
            state_prime, reward, terminal, truncated, _ = env.step(action)

            agent.update(state, action, reward, terminal, state_prime)

            episode_over = terminal or truncated
            state = state_prime

        agent.decay_epsilon()

    visualize(env, agent)
    env.close()


if __name__ == "__main__":

    if len(sys.argv) == 3:
        agent_type = sys.argv[1]
        game = sys.argv[2]
        main(agent_type, game)
    else:
        main()