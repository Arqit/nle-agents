import numpy as np
from segment_tree import MinSegmentTree, SumSegmentTree
import random

# alpha controls the amount of prioritization is applied

class ReplayBuffer:  # This is all working properly!
    """
    Simple storage for transitions from an environment.
    """

    def __init__(self, size,batch_size):
        """
        Initialise a buffer of a given size for storing transitions
        :param size: the maximum number of transitions that can be stored
        """
        self._storage = []
        self._maxsize = size
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
            self._storage.append(data)
        else:
            self._storage[self._next_idx] = data
        self._next_idx = (self._next_idx + 1) % self._maxsize



    def sample(self):
        """
        Randomly sample a batch of transitions from the buffer.
        :param batch_size: the number of transitions to sample
        :return: a mini-batch of sampled transitions
        """
        indices = np.random.randint(0, len(self._storage) - 1, size=self.batch_size)
        return self._encode_sample(indices)

# They are not even using the batch_size that's being passed
class PrioritizedReplayBuffer(ReplayBuffer): # I should change my above implementation
    def __init__(self,size, batch_size,alpha = 0.2):
        assert alpha >=0
        super(PrioritizedReplayBuffer,self).__init__(size, batch_size)
        self.batch_size = batch_size
        self.max_priority = 1.0 # This will be assigned to newly added transitions since we dont have a priority for them as yet and this will guarantee that the will be sampled next
        self._next_idx = 0
        self.alpha = alpha

        # Capacity must be positive and power of 2 (but why?) # It will be rounded up to the closest power of 2
        # I presume that it has something to do with how a segment tree works (look into this)
        tree_capacity = 1
        while tree_capacity < self._maxsize:
            tree_capacity *= 2

        # These are the structures that will be used to maintain the priorities
        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def add(self, state, action, reward, next_state, done): # Try to massage / reconcile this with what we currently have
        super().add(state, action, reward, next_state, done)# what's with this fancy syntax?

        # The tree_ptr needs to be maintained properly because this tells us where we are
        # in the tree
        self.sum_tree[self._next_idx] = self.max_priority** self.alpha # This can be thought of as a default value for new additions
        self.min_tree[self._next_idx] = self.max_priority** self.alpha
        self._next_idx = (self._next_idx +1) %self._maxsize # This increments the ptr and does a wrap around (overwrite) if we attempt to exceed the max size


    def _encode_sample(self, indices):
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for i in indices:
            if i> len(self):
                i = np.random.randint(0,10000)
            data = self._storage[i]
            state, action, reward, next_state, done = data
            states.append(np.array(state, copy=False))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(next_state, copy=False))
            dones.append(done)
        return np.array(states),np.array(actions),np.array(rewards),np.array(next_states),np.array(dones)



    def sample(self, beta = 0.6):
        assert len(self)>= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()
        states,actions,rewards,next_states,dones = self._encode_sample(indices)
        weights = np.array([self._calculate_weight(i,beta) for i in indices])

        return (states,actions,rewards,next_states,dones,weights,indices)


    def update_priorities(self,indices,priorities): # So far, it actually seems reasonable to adapt their version to ours so that it doesnt look sus
        "Update priorities of sampled transitions"
        assert len(indices) == len(priorities)

        for idx,priority in zip(indices, priorities):
            #assert priority >0
            assert 0<= idx<len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    # This is actually the function that determines the indices of the transitions that we are going to samples. This is invoked by the sample_batch function
    def _sample_proportional(self):
        indices = []
        p_total = self.sum_tree.sum(0,len(self) -1 )
        segment = p_total /self.batch_size

        for i in range(self.batch_size):
            a = segment *i
            b = segment *( i+1)
            upperbound = random.uniform(a,b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx%self._maxsize)

        return indices

    def _calculate_weight(self,idx,beta):
        """Calculate the weight of the experience at idx."""
        # get the max_weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # Calc the weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) **(-beta)
        weight = weight /max_weight

        return weight

    def __len__(self):
        return len(self._storage)
