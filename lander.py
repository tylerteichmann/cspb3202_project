###############################################################################
#
# Author: Tyler Teichmann
# Purpose: Simple RL agent for the lunar lander environment in gymnasium API.
# Developed for CSPB 3202: Intro to Artificial Intelligence.
# Date created: 2025-04-18
#
#


import gymnasium as gym
from agent import QLanderAgent
from tqdm import tqdm
import visualize


def main():

    # Environment setup
    env = gym.make("LunarLander-v3")


    # Agent setup
    alpha = 0.001
    n_episodes = 1000000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    agent = QLanderAgent(
        env=env,
        alpha=alpha,
        initial_epsilon=start_epsilon,
        epsilon_decay=epsilon_decay,
        final_epsilon=final_epsilon,
    )


    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)


    # Run itterations
    for i in tqdm(range(n_episodes)):
        state, _ = env.reset()
        state = tuple(state)
        episode_over = False

        while not episode_over:
            action = agent.get_action(state)
            state_prime, reward, terminal, truncated, _ = env.step(action)
            state_prime = tuple(state_prime)

            agent.update_q_values(
                state=state,
                action=action,
                reward=reward,
                terminal=terminal,
                state_prime=state_prime
            )

            state = state_prime
            episode_over = terminal or truncated


    visualize.visualize(env, agent)
    env.close()


if __name__ == "__main__":
    main()