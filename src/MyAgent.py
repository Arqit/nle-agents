from AbstractAgent import AbstractAgent
from gym import spaces
from replay_buffer import ReplayBuffer
import torch.nn as nn
import torch
import numpy as np
from torchsummary import summary

# We'll probably take some inspiration from their train function
# The architecture will probably require some experimentation

class DQN(nn.Module):
    """
    A basic implementation of a Deep Q-Network. The architecture is the same as that described in the
    Nature DQN paper.
    """
# Just check if there's any notable difference in whether a box or nd array is used?
    def __init__(self, observation_space, action_space: spaces.Discrete):
        """
        Initialise the DQN
        :param observation_space: the state space of the environment
        :param action_space: the action space of the environment
        """
        super().__init__()

        # Below is basically directly copied over from our DQN agent. Will need to be adjusted since it expects input with size (4,84,84)
        input_shape = observation_space.shape
        self.conv = nn.Sequential(
        nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU())


        #print(self.conv)

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
        nn.Linear(conv_out_size, 512),
        nn.ReLU(),
        nn.Linear(512, action_space.n))

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))


    def forward(self, x):
        conv_out = self.conv(x).view(x.size()[0], -1)
        return self.fc(conv_out)



class MyAgent:
    def __init__(
        self,
        observation_space: spaces.Box,
        action_space: spaces.Discrete,
        replay_buffer: ReplayBuffer,
        use_double_dqn,
        lr,
        batch_size,
        gamma,
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
        self.replay_buffer = replay_buffer #maybe the wrong replay buffer is being used
        self.use_double_dqn = use_double_dqn
        self.lr = lr
        self.batch_size = batch_size
        self.gamma = gamma
        # TODO: Initialise -agent's networks-, optimiser and -replay buffer-
        self.Q = DQN(self.observation_space,self.action_space)
        self.Q.cuda()
        self.Q_hat = DQN(self.observation_space, self.action_space)
        self.Q_hat.cuda()
        self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.lr)

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        # TODO
        #   Optimise the TD-error over a single minibatch of transitions
        #   Sample the minibatch from the replay-memory
        #   using done (as a float) instead of if statement
        #   return loss
        samples = self.replay_buffer.sample(self.batch_size)
        states = torch.tensor(samples[0]).type(torch.cuda.FloatTensor)
        next_states = torch.tensor(samples[3]).type(torch.cuda.FloatTensor)
        actions = torch.tensor(samples[1]).type(torch.cuda.LongTensor)
        rewards = torch.tensor(samples[2]).type(torch.cuda.FloatTensor)
        done = torch.tensor(samples[4]).type(torch.cuda.LongTensor)

        actual_Q = self.Q(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)# This is correct!!! You can google this if you want... the gather is needed so that we can index the self.Q(state_V) using a tensor
        Q_primes = self.Q_hat(next_states).max(1)[0]
        Q_primes[done] = 0.0
        Q_primes = Q_primes.detach()
        predicted_Q_values = Q_primes * self.gamma + rewards

        loss_t = nn.MSELoss()(actual_Q, predicted_Q_values)
        self.optimizer.zero_grad()
        loss_t.backward()
        self.optimizer.step()

        return loss_t.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """

        # TODO update target_network parameters with policy_network parameters
        self.Q_hat.load_state_dict(self.Q.state_dict())
        # raise NotImplementedError

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
        return torch.argmax(the_answer)
