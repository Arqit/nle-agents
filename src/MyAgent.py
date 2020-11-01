from AbstractAgent import AbstractAgent
from Node import node, Tree
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import nle

class MyAgent(AbstractAgent):
    def __init__(self, observation_space, action_space,seed,env):
        self.observation_space = observation_space
        self.action_space = action_space
        self.env = env
        self.seed = seed
        self.tree = None
        self.current = None
        # TODO Initialise your agent's models

        # for example, if your agent had a Pytorch model it must be load here
        # model.load_state_dict(torch.load( 'path_to_network_model_file', map_location=torch.device(device)))

    def act(self, observation):
        # Perform processing to observation

        # TODO: return selected action
        return self.UCTS(observation)

    def save(self):
        self.tree.save()
    
    def reset(self):
        self.env.reset()
        self.env.seed(self.seed, self.seed)

    def resetAgent(self):
        self.current = self.tree.root

    def UCTS(self, state, depth=100):
        if (self.tree==None):
            # The first run will enter here
            self.tree = Tree(state)
            self.current = self.tree.root
        else:
            self.current = self.tree.getChild(self.current,state)
        for _ in tqdm(range(depth)): 
            self.treePolicy(self.current)
            self.StepEnvironment(self.current)
        return self.tree[self.tree.bestChild(self.current,0)]["actions"][-1]

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
        #actions = [self.tree[self.tree[state]["children"][i]]["actions"][-1] for i in range(len(self.tree[state]["children"]))] #actions that have been played
        allActions = self.FisherYatesShuffle(np.arange(self.action_space.n)) #ensures random all action
        temp = deepcopy(self.tree[state]["actions"])
        # # iterate through current list of actions
        for i in allActions:
            #if i not in actions:
            self.StepEnvironment(state)
            new_state, _, done, _= self.env.step(i)
            temp.append(i)
            child = self.tree.AddState(new_state,deepcopy(temp))
            temp.pop()
            self.tree.addChild(state,child)
            self.tree[child]["isTerminal"] = done

    def treePolicy(self,state): #nxa
        """
        Expands the tree until a terminal node is reached.
        """
        if len(self.tree[state]["children"])==0: #first check if it's a leaf node
            self.expand(state)
        for i in range(len(self.tree[state]["children"])):
            #TODO if all the children have been visited then we should
            #choose which one to rollout based on the best child
            if self.tree[self.tree[state]["children"][i]]["visits"] == 0 and not self.tree[self.tree[state]["children"][i]]["isTerminal"]:
                reward = self.defaultPolicy(self.tree[state]["children"][i]) #rollout
                self.tree.backup(reward,self.tree[state]["children"][i])
                return reward

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
    

    

            
