###############################################################################
#
# Author: Tyler Teichmann
# Purpose: Simple RL agent for the lunar lander environment in gymnasium API.
# Developed for CSPB 3202: Intro to Artificial Intelligence.
# Date created: 2025-04-18
#
#


import gymnasium as gym
import swig

def main():
    # Create environment
    env = gym.make("LunarLander-v3", render_mode="human")
    env.reset()

    episode_over = False

    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        episode_over = terminated or truncated

    env.close()




if __name__ == "__main__":
    main()