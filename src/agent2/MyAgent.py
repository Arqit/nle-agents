from AbstractAgent import AbstractAgent
import torch.nn.functional as F
import torch
import torch.nn as nn
from gym import spaces
import numpy as np
import math

def padder(observation): # Embeds the world in a square ( as it is common practice for input to a CNN to be square)
    padded_world = np.zeros((3, 79, 79))
    state = torch.cat((torch.cat((torch.unsqueeze(torch.from_numpy(observation['glyphs']), 0), torch.unsqueeze(torch.from_numpy(observation['colors']), 0))),
                       torch.unsqueeze(torch.from_numpy(observation['chars']), 0))) # Stack the glyph, colors and char worlds
    padded_world[:, 29:50, :] = state  # Pad the image so that it is square! -> Embed the world
    new_world = torch.tensor(padded_world) # Convert to tensor
    return new_world

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MyAgent(AbstractAgent):
    def __init__(self, observation_space, action_space, **kwargs):
        global device
        self.observation_space = np.zeros((3,79,79)) # This is harc-docded here because this could not be done in in evaluation.py... Ensures that the input of the world is the correct size
        self.action_space = action_space
        if kwargs.get("train", False): # Because we are having to instantiate the world for training and testing, each having different requirements, we make the distinction here
            self.replay_buffer = kwargs.get("replay_buffer",None)
            self.use_double_dqn = kwargs.get("use_double_dqn", None)
            self.lr = kwargs.get("lr",None)
            self.batch_size = kwargs.get("batch_size",None)
            self.discount_factor = kwargs.get("discount_factor",None)
            self.beta = kwargs.get("beta",None)
            self.prior_eps = kwargs.get("prior_eps",None)
            self.Q = DQN(observation_space, action_space).to(device) # Usual Q netwok
            self.Q_hat = DQN(observation_space, action_space).to(device) # Target Q Network
            self.Q_hat.load_state_dict(self.Q.state_dict()) # Load the weights
            self.optimizer = torch.optim.RMSprop(self.Q.parameters(), lr=self.lr, momentum = 0.95) # We empirically discovered that RMSprop performs better than Adam

        else:
            self.seeds = kwargs.get('seeds', None)
            self.Q = DQN(self.observation_space,self.action_space).to(device) # We only need the one network when testing
            self.Q.load_state_dict(torch.load('The_weights4.pth',map_location=device)) # Load the pre-trained weights

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        samples = self.replay_buffer.sample(self.beta) # Sample the replay buffer
        state = torch.FloatTensor(samples[0]).to(device) # Extract the necessary information (self-explanatory)
        action = torch.LongTensor(samples[1].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples[2].reshape(-1, 1)).to(device)
        next_state = torch.FloatTensor(samples[3]).to(device)
        done = torch.FloatTensor(samples[4].reshape(-1, 1)).to(device)
        weights = torch.FloatTensor(samples[5].reshape(-1, 1)).to(device)
        indices = samples[6]

        curr_q_value = self.Q(state).gather(1, action) # Perform a forward pass for all the samples that we sampled from the replay buffer
        next_q_value = self.Q_hat(next_state).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.discount_factor * next_q_value * mask).to(device)

        # calculate element-wise dqn loss
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none") # This is similiar the to MSE when the loss is bigger, and acts like ___ when the loss is small
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward() # Perform backprop
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.replay_buffer.update_priorities(indices, new_priorities) # Update the priorities of the samples that we sampled from the replay buffer

        return loss.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.Q_hat.load_state_dict(self.Q.state_dict())

    def save_network(self,count):
        print("Model is saved")
        torch.save(self.Q.state_dict(), "/content/drive/MyDrive/The_weights"+str(count)+".pth")


    def act(self, observation):
        # Select action greedily from the Q-network given the state
        if torch.cuda.is_available() ==False:
            observation = (padder(observation)).type(torch.FloatTensor) # convert to a cpu float tensor if an GPU  is not available
        else:
            observation = (padder(observation)).type(torch.cuda.FloatTensor) # push the observation to the GPU and convert to a FloatTensor
        the_state = torch.unsqueeze(observation, 0).to(device)
        the_answer = self.Q.forward(the_state)
        action = torch.argmax(the_answer).item()
        return action


class Noisy_Layer(nn.Module): # JOSH, PLEASE COMMENT THIS!!!
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
        self.conv = nn.Sequential( # A sufficient large DQN network which we feel has sufficient model capacity
            nn.Conv2d(input_shape[0], 128, 8, stride=4),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(128, 256, 4, stride=2),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Conv2d(256, 512, 3, stride=1),
            nn.LeakyReLU(0.2, inplace = True))

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
        Get the output of the convolution layers to feed in to the fc layers - (Used to determine the size of the output of the first part of the network)
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
