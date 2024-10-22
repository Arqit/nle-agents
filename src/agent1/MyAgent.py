from AbstractAgent import AbstractAgent
from Node import Tree
import numpy as np
from copy import deepcopy
import nle
import gym

class MyAgent(AbstractAgent):
    def __init__(self, observation_space, action_space, **kwargs):
        self.env = gym.make('NetHackScore-v0', savedir=None)
        self.observation_space = observation_space
        self.action_space = action_space
        self.seeds = kwargs.get('seeds', None)
        self.reset()
        self.tree = None
        self.actions = []
        self.depth = 100
        self.startingObservation = None
        
    def act(self,observation):
        return self.UCTS()

    def load(self,directory):
        self.tree = Tree(True)
        reward = self.tree.load(directory,self.seeds[0])
        self.actions = self.tree[self.tree.root]['actions']
        return reward

    def save(self, reward,directory):
        self.tree.save(reward,directory,self.seeds[0])
    
    def resetAgent(self):
        self.reset()
        self.actions = []
        del self.tree
        self.tree = None

    def DeletePrior(self,bc):
        acts = self.tree[bc]['actions'] 
        keys = list(self.tree.dictionary.keys())
        if -1 in keys:
            keys.remove(-1)
        if -2 in keys:
            keys.remove(-2)
        actCount = len(acts)
        for ind in keys:
            if acts[:actCount] != self.tree[ind]['actions'][:actCount]: #are you the best move or it's decendant?
                _ = self.tree.dictionary.pop(ind)
        self.tree.root = bc
        self.tree[bc]['parent'] = None

    
    def reset(self):
        self.env.seed(*self.seeds)
        self.env.reset()

    def UCTS(self):
        if self.tree == None:
            self.tree = Tree()
        #give all actions taken to arrive at curren node to the current root
        self.tree[self.tree.root]["actions"] = self.actions
        for _ in (range(self.depth)): 
            v_1 = self.treePolicy(self.tree.root)
            delta = self.defaultPolicy(v_1)
            self.tree.backup(delta, v_1)
        best_child = self.tree.bestChild(self.tree.root,0) #select best child without using exploration ?
        a = self.tree[best_child]["actions"][-1]
        self.DeletePrior(best_child)
        self.actions.append(a)
        return a



    def defaultPolicy(self, state):
        '''This function performs a rollout down a path using
        a random policy. The reward for the path is returned
        so that it can be backed up. '''
        self.StepEnvironment(self.tree[state]['parent']) 
        _,total,done,_ = self.env.step(self.tree[state]['actions'][-1])
        
        # if (np.all(np.array(obs["message"][:14]) == np.array([82, 101, 97, 108, 108, 121, 32, 97, 116, 116, 97, 99, 107]))):
        #     total -= 1000
        #done = self.tree[state]['isTerminal']
        #depth = 50
        if done:
            return total#self.tree[state]['reward']

        while not done:# and depth > 0:
            action = np.random.choice(self.action_space.n)
            state, reward, done, _ = self.env.step(action)
            total += reward
            #depth-=1
        return total
    
 
    def StepEnvironment(self, state):
        '''
        Get the state's environment to be where you would be in the
        environment if you followed all the steps from the root to the 
        current node
        '''
#         print('\nstep',state)
        self.reset()
        action_list = self.tree[state]["actions"]
        for i in action_list:
            _,_,done,_ = self.env.step(i)
            if done:
                break

            
        
    def expand(self,state):
        '''
            Adds a single child to state. The child is
            the state that is reached when taking an action a' from
            state where a' is an action that has not already been
            played. 
        '''
        actions = [self.tree[self.tree[state]["children"][i]]["actions"][-1] for i in range(len(self.tree[state]["children"]))] #actions that have been played
        allActions = self.FisherYatesShuffle(np.arange(self.action_space.n)) #ensures random all action
        temp = deepcopy(self.tree[state]["actions"])
        # iterate through current list of actions and expand an unvisited one
        for i in allActions:
            if i not in actions:
                self.StepEnvironment(state)
                _, reward, done, _= self.env.step(i)
                temp.append(i)
                child = self.tree.stateCount
                self.tree.AddState(a = deepcopy(temp)) 
                self.tree.addChild(state,child)
                self.tree[child]["isTerminal"] = done
                return child

    def treePolicy(self,state): #nxa
        """
        Expands the tree until a terminal node is reached.
        """

        while self.tree[state]["isTerminal"] == False:
            
            if len(self.tree[state]["children"]) < self.action_space.n:
                #expansion phase
                return self.expand(state)
            else:
                #selection phase
                
                state = self.tree.bestChild(state, 1)

        return state

        


    def FisherYatesShuffle(self, arr):
        for i in range(0,len(arr)-2):
            j = np.random.randint(len(arr)-i)+i
            temp = arr[i]
            arr[i] = arr[j]
            arr[j] = temp
        return arr

    def __del__(self):
        self.env.close()
        del self.tree
    

    

            
