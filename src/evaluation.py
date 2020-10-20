import numpy as np
import gym
import nle
import random
from MyAgent import MyAgent
import wandb
wandb.init(project="nethack-le")

def run_episode(env, seed, agent):
    done = False
    episode_return = 0.0
    state = env.reset()
    env.seed(seed,seed)
    while not done:
        # pass state to agent and let agent decide action
        action = agent.act(state)
        new_state, reward, done, _ = env.step(action)
        episode_return += reward
        wandb.log({"Reward": reward, "Seed" : seed})
        state = new_state
        env.render()
        print(episode_return)
    return episode_return


if __name__ == '__main__':
    # Seed
    seeds = [1,2,3,4,5]

    # Initialise environment
    env = gym.make("NetHackScore-v0")

    # Number of times each seed will be run
    num_runs = 10
    
    # Run a few episodes on each seed
    rewards = []
    for seed in seeds:
        seed_rewards = []
        agent = MyAgent(env.observation_space, env.action_space,seed,env)
        for i in range(num_runs):
            seed_rewards.append(run_episode(env, seed,agent))
        rewards.append(np.mean(seed_rewards))

    # Close environment and print average reward
    env.close()
    print("Average Reward: %f" %(np.mean(rewards)))
