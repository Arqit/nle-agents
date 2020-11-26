from AbstractAgent import AbstractAgent
from train import DQN
import torch.nn.functional as F
import torch

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

        self.seeds = kwargs.get('seeds', None)
        device = torch.device(('cuda:0' if torch.cuda.is_available() else 'cpu'))
        self.Q = DQN(self.observation_space,self.action_space)
        self.Q.load_state_dict(torch.load('The_weights.pth',map_location=device))

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
        torch.save(self.Q.state_dict(), "The_weights.pth")


    def act(self, observation):
        # Select action greedily from the Q-network given the state
        the_state = observation.type(torch.cuda.FloatTensor)
        the_answer = self.Q.forward(the_state).cpu()
        action = torch.argmax(the_answer)
        return action