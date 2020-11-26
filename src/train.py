import random
import gym
from MyAgent import *
from replay_buffer import PrioritizedReplayBuffer
from gym import spaces
import torch
import torch.nn as nn
import numpy as np
from collections import OrderedDict
import torch.nn.functional as F
import math


class Noisy_Layer(nn.Module):
    """
    A form of linear layer that can be used to inject random noise into model that is a different exploration policy than e-greedy"""

    def __init__(self, input_size: int, output_size: int, init_std: float = 0.4):
        super(Noisy_Layer, self).__init__()

        self.in_feat = input_size
        self.out_feat = output_size
        self.init_std = init_std
        # Set up the weight and bias for each of the nodes in the layer
        self.weight_mu = nn.Parameter(torch.FloatTensor(output_size, input_size))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(output_size, input_size))
        self.register_buffer('weight_epsilon', torch.FloatTensor(output_size, input_size))

        self.bias_mu = nn.Parameter(torch.FloatTensor(output_size))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(output_size))
        self.register_buffer('bias_epsilon', torch.FloatTensor(output_size))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x: torch.Tensor):
        """
        Here is the forward pass of the network
        """

        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    def reset_parameters(self):
        """
        Reset trainable network parameters (factorized gaussian noise).
        """

        mu_range = 1 / math.sqrt(self.in_feat)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.init_std / math.sqrt(self.in_feat)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.init_std / math.sqrt(self.out_feat)
        )

    def reset_noise(self):
        """
        Make new noise.
        """

        epsilon_in = self.scale_noise(self.in_feat)
        epsilon_out = self.scale_noise(self.out_feat)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """
        Set scale to make noise (factorized gaussian noise).
        """

        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))

        return x.sign().mul(x.abs().sqrt())


class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """

    def __init__(self, observation_space, action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()
        self.conv = None
        self.fc = None

        input_shape = observation_space.shape
        # Set up the convolutional neural section
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 128, 8, stride=4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 3, stride=1),
            nn.LeakyReLU(0.2, inplace=True))

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_space.n))

        # Set up the duelling network
        self.fc_layer_initial = nn.Sequential(
            nn.Linear(conv_out_size, 1024),
            nn.ReLU(), )

        # Set up the action/advantage layer of the network
        # This has the same output dimensions ans the action space
        self.advantage_layer = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_space.n),
        )

        # set up the value layer
        # Outputs to one node as it basically gives a value associated with a certain state/action pair
        self.value_layer = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1),
        )

    def _get_conv_out(self, shape):
        """
        Get the output of the convolution layers to feed in to the fc layers
        """
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        """
        Returns the values of a forward pass of the network
        :param x: The input to feed into the network of the same size of the input layer of the network
        """
        # print(x.size())
        conv_out = self.conv(x).view(x.size()[0], -1)
        initial = self.fc_layer_initial(conv_out)
        # This is how duelling Q-networks make the predictions
        val = self.value_layer(initial)
        adv = self.advantage_layer(initial)

        q = val + adv - adv.mean(dim=-1, keepdim=True)
        return q

    def reset_noise(self):
        """
        Reset the noise of all of the parameters """
        # for layer in self.fc.children():
        # if isinstance(layer,Noisy_Layer):
        # layer.reset_noise()
        for layer in self.fc_layer_initial.children():
            if isinstance(layer, Noisy_Layer):
                layer.reset_noise()
        for layer in self.advantage_layer.children():
            if isinstance(layer, Noisy_Layer):
                layer.reset_noise()
        for layer in self.value_layer.children():
            if isinstance(layer, Noisy_Layer):
                layer.reset_noise()


def padder(observation):
    padded_world = np.zeros((3, 79, 79))
    state = torch.cat((torch.cat((torch.unsqueeze(torch.from_numpy(observation['glyphs']), 0), torch.unsqueeze(torch.from_numpy(observation['colors']), 0))),
                       torch.unsqueeze(torch.from_numpy(observation['chars']), 0)))
    padded_world[:, 29:50, :] = state  # Pad the image so that it is square!
    new_world = torch.tensor(padded_world)
    return new_world


hyper_params = {  # Tinker around with these
    'seed': 42,  # which seed to use
    'replay-buffer-size': int(20000),  # replay buffer size
    'learning-rate': 1e-3,  # learning rate for Adam optimizer --> This is the default learning rate
    'discount-factor': 0.99,  # discount factor
    'num-steps': int(1e6),  # total number of steps to run the environment for
    'batch-size': 32,  # number of transitions to optimize at the same time
    'learning-starts': 10000,  # number of steps before learning starts
    'learning-freq': 5,  # number of iterations between every optimization step
    'use-double-dqn': True,  # use double deep Q-learning
    'target-update-freq': 1000,  # number of iterations between every target network update
    'eps-start': 1.0,  # e-greedy start threshold  -> Sort this out with the noisy layer stuff!
    'eps-end': 0.01,  # e-greedy end threshold
    'eps-fraction': 0.5,  # fraction of num-steps
    'print-freq': 10,
    'alpha': 0.2,
    'beta': 0.6,
    'prior_eps': 1e-6
}

np.random.seed(hyper_params['seed'])
random.seed(hyper_params['seed'])

env = gym.make("NetHackScore-v0")  # If its automatically picking up gold, then autopickup must be enabled for everything
env.seed(hyper_params['seed'])

print(env.__dict__)
# We are used the glyphs, colors and chars stacked as input
replay_buffer = PrioritizedReplayBuffer(hyper_params['replay-buffer-size'], batch_size=hyper_params['batch-size'], alpha=hyper_params['alpha'])
agent = MyAgent(
    np.zeros((3, 79, 79)),  # assuming that we are taking the world as input
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

eps_timesteps = hyper_params['eps-fraction'] * float(hyper_params['num-steps'])

state = padder(env.reset())

for t in range(1, hyper_params['num-steps'] + 1):
    fract = min(1.0, float(t) / eps_timesteps)  # Sort this out with the noisy layer stuff
    eps_threshold = hyper_params["eps-start"] + fract * (hyper_params["eps-end"] - hyper_params["eps-start"])

    if random.random() < eps_threshold:  # Will presumably be replaced by the Noisy Layer stuff
        action = np.random.choice(agent.action_space.n)
    else:
        action = agent.act(torch.unsqueeze(state, 0)).item()

    if action == 21:  # The eating 'macro' which attempts to handle the food selection issue (the developers need to get their act together)
        action = 19  # Just get the agent to wait until we chose an action other than 'EAT'

    (state_prime, reward, done, _) = env.step(action)
    # env.render()
    state_prime = padder(state_prime)
    replay_buffer.add(state, action, reward, state_prime, float(done))
    total_reward += reward
    state = state_prime

    fraction = min(t / hyper_params['num-steps'], 1.0)
    agent.beta = agent.beta + fraction * (1.0 - agent.beta)
    if done:
        state = padder(env.reset())
        scores.append(total_reward)
        total_reward = 0

    if t > hyper_params['learning-starts'] and t % hyper_params['learning-freq'] == 0:
        ans = agent.optimise_td_loss()

    if t > hyper_params['learning-starts'] and t % hyper_params['target-update-freq'] == 0:
        agent.update_target_network()
    num_episodes = len(scores)

    if done and hyper_params['print-freq'] is not None and len(scores) % hyper_params['print-freq'] == 0:
        mean_100ep_reward = round(np.mean(scores[-101:-1]), 1)

        print('********************************************************')
        print('steps: {}'.format(t))
        print('episodes: {}'.format(num_episodes))
        print('mean 100 episode reward: {}'.format(mean_100ep_reward))
        print('% time spent exploring: {}'.format(eps_threshold))
        print('********************************************************')
agent.save_network()
