from AbstractAgent import AbstractAgent
from Node import node
import numpy as np
from copy import deepcopy
from tqdm import tqdm

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
    
    def reset(self):
        self.env.reset()
        self.env.seed(self.seed, self.seed)

    def UCTS(self, state, num_episodes=100):
        if (self.tree==None):
            # The first run will enter here
            root = node(state)
            self.tree = root
            self.current = self.tree
        else:
            self.current = self.current.getChild(state)
        for _ in tqdm(range(num_episodes)): 
            self.treePolicy(self.current)
            self.StepEnvironment(self.current.actions)
        print(self.current.bestChild(0).actions[-1])
        return self.current.bestChild(0).actions[-1]

    def defaultPolicy(self, n):
        total = self.StepEnvironment(n.actions)
        done = False
        while not done:
            action = np.random.choice(self.action_space.n)
            _, reward, done, _ = self.env.step(action)
            total += reward
        return reward
    
    '''Takes in an action list and loops through those'''
    def StepEnvironment(self, action_list):
        self.reset()
        total = 0
        for i in action_list:
            state,reward,done,_ = self.env.step(i)
            total += reward
        return total
        
    def expand(self,n):
        actions = [n.children[i].actions[-1] for i in range(len(n.children))] #actions that have been played
        allActions = self.FisherYatesShuffle(np.arange(self.action_space.n)) #ensures random all action
        newNode = None
        temp = n.actions
        #create new env
        # # iterate through current list of actions
        for i in allActions:
            if i not in actions:
                self.StepEnvironment(n.actions)
                new_state, _, done, _= self.env.step(i)
                temp.append(i)
                newNode = node(new_state,deepcopy(temp))
                temp.pop()
                n.addChild(newNode)
                newNode.isTerminal = done

    def treePolicy(self,v): #nxa
        """
        Expands the tree until a terminal node is reached.
        """
        if len(v.children)==0: #first check if it's a leaf node
            self.expand(v)
        for i in range(len(v.children)):
            #TODO if all the children have been visited then we should
            #choose which one to rollout based on the best child
            if v.children[i].visits == 0 and not v.children[i].isTerminal:
                v.children[i].visits += 1
                reward = self.defaultPolicy(v.children[i]) #rollout
                v.children[i].backup(reward)
                return reward

    def FisherYatesShuffle(self, arr):
        for i in range(0,len(arr)-2):
            j = np.random.randint(len(arr)-i)+i
            temp = arr[i]
            arr[i] = arr[j]
            arr[j] = temp
        return arr
    

# def FullUCTS(self,num_episodes=100):
#         stats = EpisodeStats(episode_lengths=np.zeros(num_episodes), episode_rewards1=np.zeros(num_episodes), episode_rewards2=np.zeros(num_episodes))
#         state = self.game.reset()
#         root = node(state)
#         while state >= 0:
#             newNode = self.UCTS(state)
#             root.addChild(newNode)
#             state = newNode.state
#         return root, stats
    

    

            
