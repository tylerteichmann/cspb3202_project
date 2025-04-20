###############################################################################
#
# Author: Tyler Teichmann
# Purpose: Simple RL agent for the lunar lander environment in gymnasium API.
# Developed for CSPB 3202: Intro to Artificial Intelligence.
# Date created: 2025-04-18
#
#


import gymnasium as gym
from agent import QLander
from agent import ApproximateQLander
from tqdm import tqdm
import visualize


def main(agent_type="Approximate"):

    # Environment setup
    env = gym.make("LunarLander-v3")


    # Agent setup
    alpha = 0.001
    n_episodes = 1000000
    start_epsilon = 1.0
    epsilon_decay = start_epsilon / (n_episodes / 2)
    final_epsilon = 0.1

    if agent_type == "Q-Learner":
        agent = QLander(
            env=env,
            alpha=alpha,
            initial_epsilon=start_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
        )
    elif agent_type == "Approximate":
        agent = ApproximateQLander(
            env=env,
            alpha=alpha,
            initial_epsilon=start_epsilon,
            epsilon_decay=epsilon_decay,
            final_epsilon=final_epsilon,
        )


    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=n_episodes)


    # Run itterations
    for episode in tqdm(range(n_episodes)):
        state, _ = env.reset()
        episode_over = False

        while not episode_over:
            action = agent.get_action(state)
            state_prime, reward, terminal, truncated, _ = env.step(action)

            agent.update(
                state=state,
                action=action,
                reward=reward,
                terminal=terminal,
                state_prime=state_prime
            )

            episode_over = terminal or truncated
            state = state_prime


        agent.decay_epsilon()


    visualize.visualize(env, agent)
    env.close()


if __name__ == "__main__":
    main()