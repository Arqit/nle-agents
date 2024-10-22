from AbstractAgent import AbstractAgent
import torch.nn.functional as F
import torch
import torch.nn as nn
from gym import spaces
import numpy as np
import math
import random
import os


# from torchsummary import summary

def padder(observation):  # Embeds the world in a square ( as it is common practice for input to a CNN to be square)
    padded_world = np.zeros((3, 79, 79))
    state = torch.cat((torch.cat((torch.unsqueeze(torch.from_numpy(observation['glyphs']), 0), torch.unsqueeze(torch.from_numpy(observation['colors']), 0))),
                       torch.unsqueeze(torch.from_numpy(observation['chars']), 0)))  # Stack the glyph, colors, and char worlds
    padded_world[:, 29:50, :] = state  # Pad the image so that it is square! -> Embeds the world
    new_world = torch.tensor(padded_world)  # Convert to tensor
    return new_world


# determines if a CUDA-enabled GPU is available, if not, the CPU is used
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class MyAgent(AbstractAgent):
    def __init__(self, observation_space, action_space, **kwargs):
        global device

        self.observation_space = np.zeros(
            (3, 79, 79))  # This is hard-coded here because we require the observation to be (3,79,79) and could not be set in evaluation.py... Ensures that the input of the world is the correct size
        self.action_space = action_space
        self.train = kwargs.get("train", False)
        if self.train:  # Because we are having to instantiate this class for training and testing, each having different requirements, we make the distinction here
            self.replay_buffer = kwargs.get("replay_buffer", None)
            self.use_double_dqn = kwargs.get("use_double_dqn", None)
            self.lr = kwargs.get("lr", None)
            self.batch_size = kwargs.get("batch_size", None)
            self.discount_factor = kwargs.get("discount_factor", None)
            self.beta = kwargs.get("beta", None)
            self.prior_eps = kwargs.get("prior_eps", None)
            self.Q = DQN(self.observation_space, action_space).to(device)  # Usual Q netwok
            self.Q_hat = DQN(self.observation_space, action_space).to(device)  # Target Q Network
            self.Q_hat.load_state_dict(self.Q.state_dict())  # Load the weights
            self.optimizer = torch.optim.RMSprop(self.Q.parameters(), lr=self.lr, momentum=0.95)  # We empirically discovered that RMSprop performs better than Adam

            # for keys,vals in self.Q.named_parameters():
            # 	print(keys,len(vals))
            # summary(self.Q,(3,79,79))

            # for keys,vals in self.Q.named_parameters():
            # 	print(keys,len(vals))
            # summary(self.Q,(3,79,79))


        else:
            self.seeds = kwargs.get('seeds', None)
            path = os.path.dirname(os.path.abspath(__file__))
            model_params = os.path.join(path,'The_weights7.pth')
            self.Q = DQN(self.observation_space, self.action_space).to(device)  # We only need the one network when testing
            self.Q.load_state_dict(torch.load(model_params, map_location=device))  # Load the pre-trained weights

    def optimise_td_loss(self):
        """
        Optimise the TD-error over a single minibatch of transitions
        :return: the loss
        """
        samples = self.replay_buffer.sample_batch(self.beta)  # Sample the replay buffer
        state = torch.FloatTensor(samples[0]).to(device)
        action = torch.LongTensor(samples[1].reshape(-1, 1)).to(device)
        reward = torch.FloatTensor(samples[2].reshape(-1, 1)).to(device)
        next_state = torch.FloatTensor(samples[3]).to(device)
        done = torch.LongTensor(samples[4].reshape(-1, 1)).to(device)
        weights = torch.FloatTensor(samples[5].reshape(-1, 1)).to(device)
        indices = samples[6]

        curr_q_value = self.Q(state).gather(1, action)
        next_q_value = self.Q_hat(next_state).max(dim=1, keepdim=True)[0].detach()
        mask = 1 - done
        target = (reward + self.discount_factor * next_q_value * mask).to(device)

        # calculate element-wise dqn loss
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target, reduction="none")  # Smooth l1 loss (Huber Loss) acts similiar to L2 loss when the loss is large, and acts like L1 loss when the loss is small
        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()  # Perform backprop
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.replay_buffer.update_priorities(indices, new_priorities)  # Update the priorities of the samples that we sampled from the replay buffer
        return loss.item()

    def update_target_network(self):
        """
        Update the target Q-network by copying the weights from the current Q-network
        """
        self.Q_hat.load_state_dict(self.Q.state_dict())

    def save_network(self, count,savedir):
        print("Model is saved")
        place = os.path.join(savedir,"The_weights" + str(count) + ".pth")
        torch.save(self.Q.state_dict(),place )

    def act(self, observation):
        if self.train == False:
            if random.random() < 0.45: # The stocastic noise we are having to add when evaluating
                return np.random.randint(0, 23)
            if not torch.cuda.is_available():
                observation = (padder(observation)).type(torch.FloatTensor)  # convert to a cpu float tensor if an GPU  is not available
            else:
                observation = (padder(observation)).type(torch.cuda.FloatTensor)  # push the observation to the GPU and convert to a FloatTensor
        else:
            if not torch.cuda.is_available():
                observation = observation.type(torch.FloatTensor)  # convert to a cpu float tensor if an GPU  is not available
            else:
                observation = observation.type(torch.cuda.FloatTensor)  # push the observation to the GPU and convert to a FloatTensor
        the_state = torch.unsqueeze(observation, 0).to(device)
        the_answer = self.Q.forward(the_state)
        action = torch.argmax(the_answer).item()
        return action


class Noisy_Layer(nn.Module):
    """
    A form of linear layer that can be used to inject random noise into model that is a different exploration policy than e-greedy"""

    def __init__(self, input_size, output_size, init_std = 0.4):
        super(Noisy_Layer, self).__init__()

        self.in_feat = input_size
        self.out_feat = output_size
        self.init_std = init_std
        # Set up the weight and bias for each of the nodes in the layer
        #This works by essentially creating two linear layers, one for the sigma and one for the mu
        #These layers are basically stacked on top of one another and the parameters are learned just like a normal liner layer
        #We have one for each parameter of the normal distribution that we sample from.
        self.weight_mu = nn.Parameter(torch.FloatTensor(output_size, input_size))
        self.weight_sigma = nn.Parameter(torch.FloatTensor(output_size, input_size))
        self.register_buffer('weight_epsilon', torch.FloatTensor(output_size, input_size))

        self.bias_mu = nn.Parameter(torch.FloatTensor(output_size))
        self.bias_sigma = nn.Parameter(torch.FloatTensor(output_size))
        self.register_buffer('bias_epsilon', torch.FloatTensor(output_size))

        self.reset_parameters()
        self.reset_noise()

    def forward(self, x):
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
        Reset trainable network parameters with the use of factorized Gaussian Noise
        """
        #This function is called when the layers are initialized
        #It basically sets the layers to have random noise in them.
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
        #This is called every time the model performs a learning step
        #It resets the noise parameters so that the model does not essentially have a one-and-done exploration policy.
        epsilon_in = self.scale_noise(self.in_feat)
        epsilon_out = self.scale_noise(self.out_feat)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    @staticmethod
    def scale_noise(size):
        """
        Set scale to make noise (factorized gaussian noise).
        """

        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size)) #This parameter can be adjusted but I think leaving it as default is okay

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
        self.conv = nn.Sequential(  # A sufficient large DQN network which we feel has sufficient model capacity
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
            nn.Linear(conv_out_size, 2048),
            nn.ReLU(),
            nn.Linear(2048, 1024),
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
