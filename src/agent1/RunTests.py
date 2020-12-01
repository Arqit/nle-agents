import numpy as np
import gym
import nle
import random
from MyAgent import MyAgent
import wandb
import json
import os
from numpyencoder import NumpyEncoder
from multiprocessing import Queue, Pool, Process
#wandb.init(project="nethack-le")

def run_episode(seed, load=True):
    env = gym.make("NetHackScore-v0", savedir=None)
    env.seed(seed,seed,False)
    env.reset()
    agent =  MyAgent(env.observation_space, env.action_space, seeds=env.get_seeds())
    done = False
    episode_return = 0.0
    count = 0
    rewards = []
    if load:
        episode_return = agent.load(directory)
        action_list = agent.actions
        for i in action_list:
            _,reward,done,_ = env.step(i)
            rewards.append(reward)
            count+=1
            if done:
                break
    while not done:
        # pass state to agent and let agent decide action
        action = agent.act(None)
        _, reward, done, _ = env.step(action)
        episode_return += reward
        rewards.append(reward)
        # env.render()
        #wandb.log({"Reward": reward, "Seed" : seed})
        # state = new_state
        #env.render()
        #wandb.log({"Epsiode-Return": episode_return, "Seed" : seed})
        count+=1
        if count %10 == 0:
            agent.save(episode_return,directory)
            print(count, seed, episode_return)
        #     
        # print(action, episode_return)
    del agent
    env.close()
    np.savetxt(directory+"/"+str(seed)+"MCTS.csv", rewards, delimiter=",")

seeds = [4703,8685,2786,8997,1240]#,2,3,4,5]
print(seeds)
# Initialise environment

# Number of times each seed will be run
num_runs = 1#0
# Run a few episodes on each seed
directory = os.getcwd()
#for i in range(num_runs):
processes = []
for i in range(5):
    processes.append(Process(target=run_episode, args=(seeds[i],)))

for i in range(5):
    processes[i].start()

for i in range(5):
    processes[i].join()

#agent.save()
# Close environment and print average reward
#print("Average Reward: %f" %(np.mean(rewards)))
