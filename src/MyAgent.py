from AbstractAgent import AbstractAgent
from Node import node, Tree
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import nle
import gym

class MyAgent(AbstractAgent):
    def __init__(self, observation_space, action_space,seed):
        self.observation_space = observation_space
        self.action_space = action_space
        self.env = gym.make("NetHackScore-v0")
        self.seed = seed
        self.reset()
        self.tree = None
        self.actions = []
        self.startingObservation = None
        # TODO Initialise your agent's models

        # for example, if your agent had a Pytorch model it must be load here
        # model.load_state_dict(torch.load( 'path_to_network_model_file', map_location=torch.device(device)))

    def act(self, observation):
        # Code for allowing for multiple runs with the same agent
        # if self.startingObservation == None:
        #     self.startingObservation = observation
        # else:
        #     if np.all(self.startingObservation == observation):
        #         self.resetAgent()
        #         print("Next Run")

        # TODO: return selected action
        return self.UCTS()
    
    def resetAgent(self):
        self.reset()
        self.actions = []
        del self.tree
        self.tree = None

    def DeletePrior(self):
        stack = [self.tree.root]
        while len(stack)>0:
            if np.all(self.tree[stack[-1]]["actions"] == self.actions):
                if self.tree[stack[-1]]["parent"] != None: 
                    self.tree[self.tree[stack[-1]]["parent"]]["children"].remove(stack[-1])
                self.tree[stack[-1]]["parent"] = None
                self.tree.root = stack[-1]
                stack.pop()
            elif len(self.tree[stack[-1]]["children"]) == 0: 
                if self.tree[stack[-1]]["parent"] != None: 
                    self.tree[self.tree[stack[-1]]["parent"]]["children"].remove(stack[-1])        
                del self.tree[stack[-1]]
                stack.pop()
            else:
                for i in range(len(self.tree[stack[-1]]["children"])):
                    stack.append(self.tree[stack[-1]]["children"][i])
                    break
    
    def reset(self):
        self.env.seed(self.seed, self.seed)
        self.env.reset()

    def UCTS(self, depth=100):
        if self.tree == None:
            self.tree = Tree()
        self.tree[self.tree.root]["actions"] = self.actions
        for _ in tqdm(range(depth)): 
            v_1 = self.treePolicy(self.tree.root)
            delta = self.defaultPolicy(v_1)
            self.tree.backup(delta, v_1)
        a = self.tree[self.tree.bestChild(self.tree.root,0)]["actions"][-1]
        self.actions.append(a)
        self.DeletePrior()
        return a



    def defaultPolicy(self, state):
        total = self.StepEnvironment(state)
        done = False
        while not done:
            action = np.random.choice(self.action_space.n)
            _, reward, done, _ = self.env.step(action)
            total += reward
        return reward
    
    '''Takes in a state and loops through the state's action list'''
    def StepEnvironment(self, state):
        self.reset()
        total = 0
        action_list = self.tree[state]["actions"]
        for i in action_list:
            state,reward,_,_ = self.env.step(i)
            total += reward
        return total
        
    def expand(self,state):
        actions = [self.tree[self.tree[state]["children"][i]]["actions"][-1] for i in range(len(self.tree[state]["children"]))] #actions that have been played
        allActions = self.FisherYatesShuffle(np.arange(self.action_space.n)) #ensures random all action
        temp = deepcopy(self.tree[state]["actions"])
        # # iterate through current list of actions
        for i in allActions:
            if i not in actions:
                self.StepEnvironment(state)
                new_state, _, done, _= self.env.step(i)
                temp.append(i)
                child = self.tree.AddState(deepcopy(temp))
                self.tree.addChild(state,child)
                self.tree[child]["isTerminal"] = done
                return child

    def treePolicy(self,state): #nxa
        """
        Expands the tree until a terminal node is reached.
        """

        '''
        while computational load allows:
            if node is leaf:
                expand and rollout first child
            else:
                find best child:
                if best child is not visited:
                    rollout and backup reward
                else:
                    expand best child
                    rollout first child
        '''
        while self.tree[state]["isTerminal"] == False:
            if len(self.tree[state]["children"]) < self.action_space.n:
                return self.expand(state)
            else:
                state = self.tree.bestChild(state, 1)
        return state
        # if len(self.tree[state]["children"])==0: #first check if it's a leaf node
        #     self.expand(state)
        #     for i in range(len(self.tree[state]["children"])):
        #         #TODO if all the children have been visited then we should
        #         #choose which one to rollout based on the best child
        #         if self.tree[self.tree[state]["children"][i]]["visits"] == 0 and not self.tree[self.tree[state]["children"][i]]["isTerminal"]:
        #             reward = self.defaultPolicy(self.tree[state]["children"][i]) #rollout
        #             self.tree.backup(reward,self.tree[state]["children"][i])
        #             return reward
        # else:
        #     new_state = self.tree.bestChild(state,1)
        #     if (self.tree[new_state]["visits"] == 0):
        #         reward = self.defaultPolicy(new_state) #rollout
        #         self.tree.backup(reward,new_state)
        #     else:
                
        #         self.expand(new_state)
        #         for i in range(len(self.tree[new_state]["children"])):
        #         #TODO if all the children have been visited then we should
        #         #choose which one to rollout based on the best child
        #             if self.tree[self.tree[new_state]["children"][i]]["visits"] == 0 and not self.tree[self.tree[stnew_stateate]["children"][i]]["isTerminal"]:
        #                 reward = self.defaultPolicy(self.tree[new_state]["children"][i]) #rollout
        #                 self.tree.backup(reward,self.tree[new_state]["children"][i])
        #                 return reward
        


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
    

# def FullUCTS(self,num_episodes=100):
#         stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards1=np.zeros(num_episodes), episode_rewards2=np.zeros(num_episodes))
#         state = self.game.reset()
#         root = node(state)
#         while state >= 0:
#             newNode = self.UCTS(state)
#             root.addChild(newNode)
#             state = newNode.state
#         return root, stats
    

    

            
