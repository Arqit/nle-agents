import numpy as np
import gym
import nle
import random
from train import padder
from MyAgent import *

device = "cuda"

def run_episode(env):
    rewards = []
    done = False
    episode_return = 0.0
    state = padder(env.reset())
    # create instance of MyAgent
    agent = MyAgent(np.zeros((3,79,79)), env.action_space,train=False, seeds=env.get_seeds())
    while not done:
        # pass state to agent and let agent decide action
        action = agent.act(torch.unsqueeze(state, 0)).item()
        new_state, reward, done, _ = env.step(action)
        rewards.append(reward)
        new_state = padder(new_state)
        episode_return += reward
        state = new_state
    np.savetxt('/content/nle-agents/src/'+str(env.get_seeds()[0])+"DQN.csv",rewards,delimiter=',')
    del agent
    print(episode_return)
    return episode_return


# Seed
seeds = [1, 2, 3, 4, 5]

# Initialise environment
env = gym.make("NetHackScore-v0")

# Number of times each seed will be run
num_runs = 10

# Run a few episodes on each seed
rewards = []
for seed in seeds:
    seed_rewards = []
    for i in range(num_runs):
        env.seed(seed, seed, False)
        seed_rewards.append(run_episode(env))
    rewards.append(np.mean(seed_rewards))

# Close environment and print average reward
env.close()
print(np.mean(rewards))
