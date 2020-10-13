import numpy as np
import gym
import nle
import random


def run_episode(env):
    # create instance of MyAgent
    from MyAgent import MyAgent
    agent = MyAgent(env.observation_space, env.action_space)

    done = False
    episode_return = 0.0
    state = env.reset()
    while not done:
        # pass state to agent and let agent decide action
        action = agent.act(state)
        new_state, reward, done, _ = env.step(action)
        episode_return += reward
        state = new_state
    return episode_return


if __name__ == '__main__':
    # Seed
    seeds = [1,2,3,4,5]

    # Initialise environment
    env = gym.make("NetHackScore-v0")# Its the 16 cardinal points + upstairs, downstairs, chilling , reading messages, kick, eat and search
    dummy= env.reset() # The players rating and info is stored in the state array
    print(dummy)
    # The message is encoded using unicode ( the translator @ : https://www.branah.com/unicode-converter    to read the message)
    # Will be a zero array if there is no message
    # LOOK INTO WHAT SPECIALS ARE?

    # TODO : Check what is the difference between glyph and character
    # Guaranteed!: The dungeon shape will always be within (21,79)
    #We dont care about the inventory since we cannot use weapons, store food, wear robes, etc
    # When we see food, we just eat it --- no questions asked!

    # The glyph id array cannot be used as is since it is not directly suitable for machine learning. They "provide some mysterious tooling to decipher the type of glyph" since most are swallow
    # The 'glpyh' state is the current text-based environment

    # Number of times each seed will be run
    num_runs = 10

    # Run a few episodes on each seed
    rewards = []
    for seed in seeds:
        env.seed(seed)
        seed_rewards = []
        for i in range(num_runs):
            seed_rewards.append(run_episode(env))
        rewards.append(np.mean(seed_rewards))

    # Close environment and print average reward
    env.close()
    print("Average Reward: %f" %(np.mean(rewards)))
