#from AbstractAgent import AbstractAgent
from gym import spaces
from replay_buffer import ReplayBuffer
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torchsummary import summary
from collections import OrderedDict
import math

device = "cuda"

# We'll probably take some inspiration from their train function ( in terms of the loss function used, the learning rates + other hyperparameters)
#If we are to implement rainbow... I think that we will need some form of noisy net that can be turned on and off
#The noisy net can be used for exploration instead of a policy such as e-greedy and is plugged as a layer directly into the network

class Noisy_Net(nn.Linear):
    def __init__(self, in_features, out_features, std_init=0.4):
        #As far as I can tell this works the same as a standard linear layer
        super(Noisy_Net, self).__init__()

        self.in_features  = in_features
        self.out_features = out_features
        self.std_init     = std_init

        self.weight_mu    = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(out_features, in_features))
        self.register_buffer('weight_epsilon', torch.FloatTensor(out_features, in_features))

        self.bias_mu    = nn.Parameter(torch.FloatTensor(out_features))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(out_features))
        self.register_buffer('bias_epsilon', torch.FloatTensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma.mul(Variable(self.weight_epsilon))
            bias   = self.bias_mu   + self.bias_sigma.mul(Variable(self.bias_epsilon))
        else:
            weight = self.weight_mu
            bias   = self.bias_mu

        return F.linear(x, weight, bias)

    def reset_parameters(self):
        mu_range = 1 / math.sqrt(self.weight_mu.size(1))

        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.weight_sigma.size(1)))

        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.bias_sigma.size(0)))

    def reset_noise(self):
        epsilon_in  = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)

        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(self._scale_noise(self.out_features))

    def _scale_noise(self, size):
        x = torch.randn(size)
        x = x.sign().mul(x.abs().sqrt())
        return x

class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """
# Just check if there's any notable difference in whether a box or nd array is used?
    def __init__(self, observation_space, action_space: spaces.Discrete, conv_parameters = None,linear_parameters=None, use_noisy = False):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        :param conv_parameters: An array of the convolutional network parameters
        :param linear_parameters: An array of the linear network parameters.


        Example of conv_parameters [input_size,output_size,kernel_size,stride], example of linear [input_size,output_size]}
        The function also checks for the transition between the conv layers and the fc layers
        """
        super().__init__()
        # Needs major repairing and consideration
        # Below is basically directly copied over from our DQN agent. Will need to be adjusted since it expects input with size (4,84,84)
        self.conv = None
        self.fc = None
        input_shape = observation_space.shape
        if conv_parameters == None:
            input_shape = observation_space.shape
            self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 64,8, stride=4),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, stride=1),
            nn.ReLU())
            #print(self.conv)

            conv_out_size = self._get_conv_out(input_shape)
            self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, action_space.n))

        else:
            conv_arr = []
            conv_counter = 1

            fc_counter = 1
            fc_arr = []

            self.conv = nn.Sequential(OrderedDict(conv_arr))
            conv_out_size = self._get_conv_out(input_shape)
            #Loop thorugh the dictionary for each layer of the network
            for i in conv_parameters:
                #make the conv network
                conv_arr.append(('conv{}'.format(i),nn.Conv2d(i[0],i[1],i[2],i[3])))
                conv_arr.append(('relu{}'.format(conv_counter),nn.ReLU()))
                conv_counter += 1


            self.conv = nn.Sequential(OrderedDict(conv_arr))
            conv_out_size = self._get_conv_out(input_shape)
            #Make the transition layer
            if not use_noisy:

                fc_arr.append(('fc{}'.format(fc_counter),nn.Linear(conv_out_size,linear_parameters[0][1])))
                fc_arr.append(('relu{}'.format(conv_counter),nn.ReLU()))
                fc_counter += 1
                conv_counter += 1

                for i in range(1,len(linear_parameters)):
                    fc_arr.append(('fc{}'.format(fc_counter),nn.Linear(linear_parameters[i][0],linear_parameters[i][1])))
                    fc_arr.append(('relu{}'.format(conv_counter),nn.ReLU()))
                    fc_counter += 1
                    conv_counter += 1
            else:
                fc_arr.append(('fc{}'.format(fc_counter),nn.Linear(conv_out_size,linear_parameters[0][1])))
                fc_arr.append(('relu{}'.format(conv_counter),nn.ReLU()))
                fc_counter += 1
                conv_counter += 1

                for i in range(1,len(linear_parameters)):
                    fc_arr.append(('fc{}'.format(fc_counter),Noisy_Net(linear_parameters[i][0],linear_parameters[i][1])))
                    fc_arr.append(('relu{}'.format(conv_counter),nn.ReLU()))
                    fc_counter += 1
                    conv_counter += 1

            self.fc = nn.Sequential(OrderedDict(fc_arr))



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
        #print(x.size())
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)



class MyAgent: # Ensure that the has the correct form when compared to our original Pong DQN implementation
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        replay_buffer: ReplayBuffer,
        use_double_dqn,
        lr,
        batch_size,
        gamma,
        beta,
        prior_eps
    ):
        """
        Initialise the DQN algorithm using the Adam optimiser
        :param action_space: the action space of the environment
        :param observation_space: the state space of the environment
        :param replay_buffer: storage for experience replay
        :param lr: the learning rate for Adam
        :param batch_size: the batch size
        :param gamma: the discount factor
        """
        self.observation_space = observation_space
        self.action_space = action_space
        self.replay_buffer = replay_buffer
        self.use_double_dqn = use_double_dqn
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        self.beta = beta
        self.prior_eps = prior_eps
        # TODO: Initialise -agent's networks-, optimiser and -replay buffer-
        #Edit this so that we can change the noisy network

        conv_params = [[self.observation_space.shape[0],32,8,4],[32,64,4,2],[64,64,3,1]]
        #Make the first one of linear params whatever as it feeds out of the conv net
        linear_params = [[1,512],[512,self.action_space.n]]




        self.Q = DQN(self.observation_space,self.action_space)
        self.Q.cuda()
        summary(self.Q,(3,79,79))
        self.Q_hat = DQN(self.observation_space, self.action_space)
        self.Q_hat.cuda()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.lr)

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        samples = self.replay_buffer.sample(self.beta)
        states = torch.tensor(samples[0]).type(torch.cuda.FloatTensor)
        actions = torch.tensor(samples[1]).type(torch.cuda.LongTensor)
        rewards = torch.tensor(samples[2]).type(torch.cuda.FloatTensor)
        next_states = torch.tensor(samples[3]).type(torch.cuda.FloatTensor)
        done = torch.tensor(samples[4]).type(torch.cuda.LongTensor)


        weights = torch.tensor(samples[5]).type(torch.cuda.FloatTensor)
        indices = samples[6]

        actual_Q = self.Q(states).gather(1, actions.unsqueeze(-1)).squeeze(-1) # Manually work through this thoroughly!
        Q_primes = self.Q_hat(next_states).max(1)[0]
        Q_primes[done] = 0.0
        Q_primes = Q_primes.detach()
        predicted_Q_values = (Q_primes * self.gamma + rewards)
        elementwise_loss = nn.MSELoss(reduction ='none')(actual_Q, predicted_Q_values)
        new_loss = torch.mean(elementwise_loss*weights)
        self.optimizer.zero_grad()
        new_loss.backward()
        self.optimizer.step()
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.replay_buffer.update_priorities(indices, new_priorities)
        return new_loss.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """

        # TODO update target_network parameters with policy_network parameters
        self.Q_hat.load_state_dict(self.Q.state_dict())

    def save_network(self):
        torch.save(self.Q.state_dict(),"The_weights.pth")

    def load_network_weights(self):
        self.Q.load_state_dict(torch.load("The_weights.pth"))

    def act(self, state: torch.Tensor):  # Correct
        """
        Select an action greedily from the Q-network given the state
        :param state: the current state
        :return: the action to take
        """
        # TODO Select action greedily from the Q-network given the state
        the_state = state.type(torch.cuda.FloatTensor)
        the_answer = self.Q.forward(the_state).cpu()
        action = torch.argmax(the_answer)
        return action
