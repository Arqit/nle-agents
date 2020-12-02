import random
from typing import List
import numpy as np
from segment_tree import *


class ReplayBuffer:
    """
    Simple storage for transitions from an environment.
    """

    def __init__(self, size, batch_size):
        """
        Initialise a buffer of a given size for storing transitions
        :param size: the maximum number of transitions that can be stored
        """
        self._storage = [] # Create the storage container to store the samples
        self._maxsize = size # Max size of the replay buffer
        self._next_idx = 0
        self.batch_size = batch_size

    def __len__(self):
        return len(self._storage)

    def add(self, state, action, reward, next_state, done):
        """
        Add a transition to the buffer. Old transitions will be overwritten if the buffer is full.
        :param state: the agent's initial state
        :param action: the action taken by the agent
        :param reward: the reward the agent received
        :param next_state: the subsequent state
        :param done: whether the episode terminated
        """
        data = (state, action, reward, next_state, done)

        if self._next_idx >= len(self._storage):
            self._storage.append(data) # Add to the replay buffer
        else:
            self._storage[self._next_idx] = data # If the replay is full, overwrite samples
        self._next_idx = (self._next_idx + 1) % self._maxsize

    def _encode_sample(self, indices): # Given an array of indices, individually extract the state, actions, reward, next_state and done
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            data = self._storage[i]
            state, action, reward, next_state, done = data
            states.append(np.array(state, copy=False))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones),
        )

    def sample(self):
        """
        Randomly sample a batch of transitions from the buffer.
        :param batch_size: the number of transitions to sample
        :return: a mini-batch of sampled transitions
        """ # If a normal replay buffer is used, this randomly samples indices fro the replay buffer
        indices = np.random.randint(0, len(self._storage) - 1, size=self.batch_size)
        return self._encode_sample(indices)

class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(self ,size: int,batch_size: int = 32,alpha: float = 0.6):
        """Initialization."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha # Controls the amount of prioritization is enforced

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self._maxsize:
            tree_capacity *= 2

        # We use a segment tree that underlies the management of our prioritized ereplay buffer
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def add(self,obs,act,rew,next_obs,done):
        """Store experience and priority."""
        super().add(obs, act, rew, next_obs, done)
        # Insert the new sample in the appropriate location in the sum_tree and min_tree
        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self._maxsize

    def _encode_sample(self, indices):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices: # As before, this individually extracts the state, action, reward, next state and done
            data = self._storage[i%len(self)]
            state, action, reward, next_state, done = data
            states.append(np.array(state, copy=False))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        return np.array(states),np.array(actions),np.array(rewards),np.array(next_states),np.array(dones)

    def sample_batch(self, beta = 0.4) :
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional() # We now sample the replay buffer, relative to the priorities of the samples

        states,actions,rewards,next_states,dones = self._encode_sample(indices)
        weights = np.array([self._calculate_weight(i, beta) for i in indices]) # calculates the weights
        return (states, actions, rewards, next_states,dones, weights, indices)

    def update_priorities(self, indices, priorities):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            #assert priority > 0
            #assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx%len(self))

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight
