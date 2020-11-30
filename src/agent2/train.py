import random
import gym
from replay_buffer import PrioritizedReplayBuffer
from gym import spaces
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
from MyAgent import MyAgent
import math
import nle
import wandb
wandb.init(project="project_dqn")

def padder(observation):
    padded_world = np.zeros((3, 79, 79))
    state = torch.cat((torch.cat((torch.unsqueeze(torch.from_numpy(observation['glyphs']), 0), torch.unsqueeze(torch.from_numpy(observation['colors']), 0))),
                       torch.unsqueeze(torch.from_numpy(observation['chars']), 0)))
    padded_world[:, 29:50, :] = state  # Pad the image so that it is square!
    new_world = torch.tensor(padded_world)
    return new_world

if __name__ == '__main__':
    hyper_params = {  # Tinker around with these
        'replay-buffer-size': int(60000),  # replay buffer size
        'learning-rate': 0.00098,  # learning rate for RMSprop optimizer
        'discount-factor': 0.99,  # discount factor
        'num-steps': int(5e6),  # total number of steps to run the environment for
        'batch-size': 64,  # number of transitions to optimize at the same time
        'learning-starts': 120000,  # number of steps before learning starts
        'learning-freq': 5,  # number of iterations between every optimization step
        'use-double-dqn': True,  # use double deep Q-learning
        'target-update-freq': 1000,  # number of iterations between every target network update
        'eps-start': 1.0,  # e-greedy start threshold  -> Sort this out with the noisy layer stuff!
        'eps-end': 0.1,  # e-greedy end threshold
        'eps-fraction': 0.5,  # fraction of num-steps
        'print-freq': 10,
        'alpha': 0.2,
        'beta': 0.6,
        'prior_eps': 1e-6
    }

    print(hyper_params)
    seed = np.random.randint(0,10000)
    np.random.seed(seed)
    random.seed(seed)

    env = gym.make("NetHackScout-v0",savedir = None)  # We disable saving the ttyrec files as it is unneccesary when training
    env.penalty_step = -10
    env.penalty_time = -1
    env.seed(seed)
    action_nums = [0,1,2,3,4,9,10,11,12,17,18,20,21,22]
    env.__dict__['_actions'] = tuple([env.__dict__['_actions'][i] for i in action_nums])
    env.action_space.n = len(action_nums)



    print(env.__dict__)
    # We are used the glyphs, colors and chars stacked as input
    replay_buffer = PrioritizedReplayBuffer(hyper_params['replay-buffer-size'], batch_size=hyper_params['batch-size'], alpha=hyper_params['alpha'])
    agent = MyAgent(
        env.observation_space,  # assuming that we are taking the world as input
        env.action_space,
        train=True,
        replay_buffer=replay_buffer,
        use_double_dqn=hyper_params['use-double-dqn'],
        lr=hyper_params['learning-rate'],
        batch_size=hyper_params['batch-size'],
        discount_factor=hyper_params['discount-factor'],
        beta=hyper_params['beta'],
        prior_eps=hyper_params['prior_eps']
    )

    total_reward = 0
    epsilons = []  # Sort this out with the noisy layer stuff!
    losses = []
    scores = []

    eps_timesteps = hyper_params['eps-fraction'] * float(hyper_params['num-steps']) # This is the efficient way we use to anneal epsilon ( Analogous to the DQN lab)
    food = 5
    state = padder(env.reset())

    for t in range(1, hyper_params['num-steps'] + 1):
        fract = min(1.0, float(t) / eps_timesteps)  # Sort this out with the noisy layer stuff
        eps_threshold = hyper_params["eps-start"] + fract * (hyper_params["eps-end"] - hyper_params["eps-start"])

        if random.random() < eps_threshold:  # Will presumably be replaced by the Noisy Layer stuff
            action = np.random.choice(agent.action_space.n) # Sort this out when we constrain the action sapce
        else:
            action = agent.act(state)
        # Sort this out
        if action == 21 and food>0:  # The eating 'macro' which attempts to handle the food selection issue (the developers need to get their act together)
             (state_prime, reward1, done, _) = env.step(21)
             (state_prime, reward2, done, _) = env.step(4)
             food -= 1
             reward = reward1 + reward2
        else:
            (state_prime, reward, done, _) = env.step(action) # take a step in the environment
        state_prime = padder(state_prime)
        replay_buffer.add(state, action, reward, state_prime, float(done))
        total_reward += reward
        state = state_prime

        # This is used to adjust beta accordingly
        fraction = min(t / hyper_params['num-steps'], 1.0)
        agent.beta = agent.beta + fraction * (1.0 - agent.beta)
        if done:
            scores.append(total_reward)
            wandb.log({"Episode Reward": total_reward, "Steps": t})
            wandb.log({"Episode Reward": total_reward, "Episodes": len(scores) + 1})
            seed = np.random.randint(1,10000)
            env.seed(seed,seed,False)
            state = padder(env.reset())
            total_reward = 0

        if t%400000==0:
            agent.save_network(count)
            count+=1

        if t > hyper_params['learning-starts'] and t % hyper_params['learning-freq'] == 0:
            ans = agent.optimise_td_loss()
            wandb.log({"Loss":ans, "Steps":t})
            wandb.log({"Loss":ans, "Episodes":len(scores)+1})

        if t > hyper_params['learning-starts'] and t % hyper_params['target-update-freq'] == 0:
            agent.update_target_network()
        num_episodes = len(scores)

        if done and hyper_params['print-freq'] is not None and len(scores) % hyper_params['print-freq'] == 0:
            mean_100ep_reward = round(np.mean(scores[-101:-1]), 1)
            wandb.log({"Mean 100 Episode Reward":mean_100ep_reward, "Steps":t})
            wandb.log({"Mean 100 Episode Reward":mean_100ep_reward, "Episodes":len(scores)+1})
            print('********************************************************')
            print('steps: {}'.format(t))
            print('episodes: {}'.format(num_episodes))
            print('mean 100 episode reward: {}'.format(mean_100ep_reward))
            print('% time spent exploring: {}'.format(eps_threshold))
            print('********************************************************')
    agent.save_network(999)
