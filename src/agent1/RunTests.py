import numpy as np
import gym
import nle
import random
from MyAgent import MyAgent
import wandb
import json
import os
from numpyencoder import NumpyEncoder
from multiprocessing import Queue, Pool
#wandb.init(project="nethack-le")

def run_episode(seed, load=False):
    env = gym.make("NetHackScore-v0", savedir=None)
    agent =  MyAgent(env.observation_space, env.action_space, seeds=env.get_seeds())
    done = False
    episode_return = 0.0
    env.seed(seed,seed,False)
    env.reset()
    count = 0
    if load:
        episode_return = agent.load(directory)
        action_list = agent.actions
        for i in action_list:
            _,_,_,_ = env.step(i)
    while not done:
        # pass state to agent and let agent decide action
        action = agent.act(None)
        _, reward, done, _ = env.step(action)
        episode_return += reward
        # env.render()
        #wandb.log({"Reward": reward, "Seed" : seed})
        # state = new_state
        #env.render()
        #wandb.log({"Epsiode-Return": episode_return, "Seed" : seed})
        count+=1
        if count %10 == 0:
            env.render()
            print(count, seed, episode_return)
        #     agent.save(episode_return,directory)
        # print(action, episode_return)
    del agent
    env.close()
    return episode_return

seeds = [4703,8685,2786,8997,1240]#,2,3,4,5]
print(seeds)
# Initialise environment

# Number of times each seed will be run
num_runs = 1#0
# Run a few episodes on each seed
rewards = []
directory = os.getcwd()
seed_rewards = []
#for i in range(num_runs):
#pool = Pool(processes=4)
#processes = [pool.apply_async() for i in range(4)]
rewards = run_episode(seeds[0])
#agent.save()
print(rewards)
# Close environment and print average reward
print("Average Reward: %f" %(np.mean(rewards)))
