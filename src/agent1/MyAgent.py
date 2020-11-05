from AbstractAgent import AbstractAgent
from Node import Tree
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import nle
import gym
from multiprocessing import Process, Queue, Pool, Value
import os

class MyAgent(AbstractAgent):
    def __init__(self, observation_space, action_space,seed,steps,env_name):
        self.observation_space = observation_space
        self.action_space = action_space
        self.env = gym.make(env_name)
        self.seed = seed
        self.reset()
        self.tree = None
        self.actions = []
        #self.depth = depth
        self.startingObservation = None
        self.pCount = os.cpu_count()
        runsPerActions = np.ceil(action_space.n/self.pCount)
        #stepsPerDepth = np.ceil(depth/action_space.n)
        self.steps = int(steps*runsPerActions)
        
    def act(self,observation):

        return self.UCTS()
    
    def resetAgent(self):
        self.reset()
        self.actions = []
        del self.tree
        self.tree = None

    def DeletePrior(self,bc):
        acts = self.tree[bc]['actions'] 
        keys = list(self.tree.dictionary.keys())
        actCount = len(acts)
        for ind in keys:
            if acts[:actCount] != self.tree[ind]['actions'][:actCount]: #are you the best move or it's decendant?
                _ = self.tree.dictionary.pop(ind)
        self.tree.root = bc
        self.tree[bc]['parent'] = None

    def reset(self):
        self.env.seed(self.seed)
        self.env.reset()

    def UCTS(self):
        if self.tree == None:
            self.tree = Tree()
        #give all actions taken to arrive at curren node to the current root
        self.tree[self.tree.root]["actions"] = self.actions
        for _ in tqdm(range(self.steps)): 
            v_1 = self.treePolicy(self.tree.root)
            processes = []
            returns = Queue()
            self.StepEnvironment(self.tree[v_1[0]]['parent']) 
            for i in range(len(v_1)):
                processes.append(Process(target=self.defaultPolicy, args=(v_1[i],returns, i)))
                processes[i].start()
            #delta = self.defaultPolicy(v_1)
            for i in range(len(v_1)):
                processes[i].join()
            for i in range(len(v_1)):
                val = returns.get()
                self.tree.backup(val[1], v_1[val[0]])
            returns.close()
        best_child = self.tree.bestChild(self.tree.root,0) #select best child without using exploration ?
        a = self.tree[best_child]["actions"][-1]
        self.DeletePrior(best_child)
        self.actions.append(a)
        return a



    def defaultPolicy(self, state, queue, i):
        '''This function performs a rollout down a path using
        a random policy. The reward for the path is returned
        so that it can be backed up. '''
        _,total,_,_ = self.env.step(self.tree[state]['actions'][-1])
        done = self.tree[state]['isTerminal']

        if done:
            return total#self.tree[state]['reward']

        while not done:
            action = np.random.choice(self.action_space.n)
            state, reward, done, _ = self.env.step(action)
            total += reward
            
        queue.put((i, total))
    
 
    def StepEnvironment(self, state):
        '''
        Get the state's environment to be where you would be in the
        environment if you followed all the steps from the root to the 
        current node
        '''
#         print('\nstep',state)
        self.reset()
        action_list = self.tree[state]["actions"]
        reward = 0
        for i in action_list:
            state,reward,_,_ = self.env.step(i)            
        
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
        children = []
        for i in allActions:
            if i not in actions:
                self.StepEnvironment(state)
                _, reward, done, _= self.env.step(i)
                temp.append(i)
                child = self.tree.stateCount
                self.tree.AddState(a = deepcopy(temp)) 
                self.tree.addChild(state,child)
                self.tree[child]["isTerminal"] = done
                children.append(child)
                if len(children) == self.pCount:
                    return children
        return children

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
    

    

            
