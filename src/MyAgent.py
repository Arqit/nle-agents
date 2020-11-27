from AbstractAgent import AbstractAgent
import torch.nn.functional as F
import torch
import torch.nn as nn
from gym import spaces
import numpy as np


device = "cuda"

class MyAgent(AbstractAgent):
    def __init__(self, observation_space, action_space, **kwargs):
        global device
        self.observation_space = observation_space
        self.action_space = action_space
        if kwargs.get("train", None):
            self.replay_buffer = kwargs.get("replay_buffer",None)
            self.use_double_dqn = kwargs.get("use_double_dqn", None)
            self.lr = kwargs.get("lr",None)
            self.batch_size = kwargs.get("batch_size",None)
            self.discount_factor = kwargs.get("discount_factor",None)
            self.beta = kwargs.get("beta",None)
            self.prior_eps = kwargs.get("prior_eps",None)
            self.Q = DQN(observation_space, action_space).to(device)
            self.Q_hat = DQN(observation_space, action_space).to(device)
            self.Q_hat.load_state_dict(self.Q.state_dict())
            self.Q_hat.eval()
            self.optimizer = torch.optim.Adam(self.Q.parameters(), lr=self.lr)

        else:
            self.seeds = kwargs.get('seeds', None)
            device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
            self.Q = DQN(self.observation_space,self.action_space).to(device)
            self.Q.load_state_dict(torch.load('/content/drive/MyDrive/The_weights.pth',map_location=device))

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        samples = self.replay_buffer.sample(self.beta)
        state = torch.FloatTensor(samples[0]).to(device)
        action = torch.LongTensor(samples[1].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples[2].reshape(-1, 1)).to(device)
        next_state = torch.FloatTensor(samples[3]).to(device)
        done = torch.FloatTensor(samples[4].reshape(-1, 1)).to(device)
        weights = torch.FloatTensor(samples[5].reshape(-1, 1)).to(device)
        indices = samples[6]

        curr_q_value = self.Q(state).gather(1, action)
        next_q_value = self.Q_hat(next_state).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.discount_factor * next_q_value * mask).to(device)

        # calculate element-wise dqn loss
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.replay_buffer.update_priorities(indices, new_priorities)

        return loss.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.Q_hat.load_state_dict(self.Q.state_dict())

    def save_network(self):
        print("IM IN THE NETWORK")
        torch.save(self.Q.state_dict(), "/content/drive/MyDrive/The_weights.pth")


    def act(self, observation):
        # Select action greedily from the Q-network given the state
        the_state = observation.type(torch.cuda.FloatTensor)
        the_answer = self.Q.forward(the_state).cpu()
        action = torch.argmax(the_answer)
        return action


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
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(256, 512, 3, stride=1),
            nn.ReLU())

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 1024),
            nn.ReLU(),
            nn.Linear(1024, action_space.n))

        # Set up the duelling network
        self.fc_layer_initial = nn.Sequential(
            nn.Linear(conv_out_size, 2048),
            nn.ReLU(),
            nn.Linear(2048,1024),
            nn.ReLU()
             )

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
