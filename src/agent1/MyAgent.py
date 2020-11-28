from AbstractAgent import AbstractAgent
from importlib import reload
import Node
reload(Node)
from Node import Tree
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import nle
import gym
import matplotlib.pyplot as plt

class MyAgent(AbstractAgent):
    def __init__(self, observation_space, action_space, **kwargs):
        self.env = gym.make('NetHackScout-v0', savedir=None)
        self.observation_space = observation_space
        self.action_space = [1,2,3,4,9,10,11,12,18]
        self.seed = kwargs.get('seeds', 100)
        self.reset()
        self.tree = None
        self.actions = []
        self.depth = kwargs.get('depth',20)
        self.rollout_lengths = []
        self.rollout_rewards = []
        self.startingObservation = None
        self.reward_steps = []
        self.move_stack = []
        
    def act(self,observation):
        if len(self.move_stack) == 0:
            # print('Did this')
            self.UCTS()
        return self.move_stack.pop(-1)


    def load(self,directory):
        self.tree = Tree(True)
        reward = self.tree.load(directory,self.seed)
        self.actions = self.tree[self.tree.root]['actions']
        return reward

    def save(self, reward,directory):
        self.tree.save(reward,directory,self.seed)
    
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
        self.env.seed(*self.seed)
        s = self.env.reset()
        return s
    def UCTS(self):
        if self.tree == None:
            self.tree = Tree()
        #give all actions taken to arrive at curren node to the current root
        self.tree[self.tree.root]["actions"] = self.actions
        for _ in (range(self.depth)): 

            v_1 = self.treePolicy(self.tree.root)

            if self.tree[v_1]['pre_descend'] != -1:
                #initiate descend macro

                #add descend as child of moving to stairs
                temp = deepcopy(self.tree[v_1]["actions"])
                _,_,_ = self.StepEnvironment(v_1)
                _, _, done, _= self.env.step(18)
                temp.append(18)
                child = self.tree.stateCount
                self.tree.AddState(a = deepcopy(temp)) 
                self.tree.addChild(v_1,child)
                self.tree[child]["isTerminal"] = False #descending stairs won't 
                self.tree[child]['is_good'] = done
                self.tree[v_1]['good_children'].append(child)
                best_child = child
                self.DeletePrior(best_child)
                self.move_stack = [18,temp[-2]]
                self.actions.append(temp[-2])
                self.actions.append(18)
                return

                
            if self.tree[v_1]['is_good']:
                delta = self.defaultPolicy(v_1)
            else: 
                delta = 0
            self.tree.backup(delta, v_1)
        # print(self.tree.dictionary)

        best_child = self.tree.bestChild(self.tree.root,0) #select best child without using exploration ?
        a = self.tree[best_child]["actions"][-1]
        self.DeletePrior(best_child)
        self.actions.append(a)
        self.move_stack.append(a)
        return 



    def defaultPolicy(self, state):
        '''This function performs a rollout down a path using
        a random policy. The reward for the path is returned
        so that it can be backed up. '''
        self.StepEnvironment(self.tree[state]['parent']) 
        lstate,total,_,_ = self.env.step(self.tree[state]['actions'][-1])
        
        # if (np.all(np.array(obs["message"][:14]) == np.array([82, 101, 97, 108, 108, 121, 32, 97, 116, 116, 97, 99, 107]))):
        #     total -= 1000
        done = self.tree[state]['isTerminal']
        
        if done:
            return total#self.tree[state]['reward']
        counter = 0
        # print(self.tree[state]['actions'])
        while not done: #and depth > 0:
            moves,descend = self.getLegalMoves(lstate)
            '''
            If you can descend then you must descend! This action
            yields a great reward and will also stop the rollout
            '''
            if len(descend) == 0:
                action = np.random.choice(moves)
                lstate, reward, done, _ = self.env.step(action)
            else:
                # self.env.render()
                lstate, reward, done, _ = self.env.step(descend[0])
                counter+=1
                # self.env.render()
                lstate, reward, done, _ = self.env.step(18)
                # self.env.render()


            counter += 1
            if counter == 800:
                break
            # self.env.render()
            if reward >= 2:
                self.reward_steps.append(counter)
                total += reward
                break
            else: 
                total += reward - 0.2

            if total < -20 :# you've been moving around randomly a lot
                break
            # if total > 20 or total < -1:
            #     break


            # state_prev = state
            # print('action',action)
            # self.env.render()

            #depth-=1
        # plt.imshow(state_prev['glyphs'])
        # plt.show()
        # print("you died",d)
        self.rollout_lengths.append(counter)
        self.rollout_rewards.append(total)
        return total
    
 
    def StepEnvironment(self, state):
        '''
        Get the state's environment to be where you would be in the
        environment if you followed all the steps from the root to the 
        current node
        '''
#         print('\nstep',state)
        s = self.reset()
        d = None
        r = None
        action_list = self.tree[state]["actions"]
        for i in action_list:
            s,_,d,r = self.env.step(i)
        return s,d,r
            
        
    def expand(self,state):
        '''
            Adds a single child to state. The child is
            the state that is reached when taking an action a' from
            state where a' is an action that has not already been
            played. 
        '''
        actions = [self.tree[self.tree[state]["children"][i]]["actions"][-1] for i in range(len(self.tree[state]["children"]))] #actions that have been played
        allActions = self.FisherYatesShuffle(self.action_space) #ensures random all action
        temp = deepcopy(self.tree[state]["actions"])
        # iterate through current list of actions and expand an unvisited one
        for i in allActions:
            if i not in actions:
                latest_state,_,_ = self.StepEnvironment(state)
                possible_moves,descend = self.getLegalMoves(latest_state)
                '''
                If you are one move away from the stairs then save
                the direction you need to go in. This is to initiate
                the descend macro. You will not explore any other actions.
                i will be turned to the action you need to take to descend.
                
                '''
                temp.append(i)
                child = self.tree.stateCount
                if len(descend) == 0:
                    self.tree[child]['pre_descend'] = -1
                    _, _, done, _= self.env.step(i)
                else:
                    self.tree[child]['pre_descend'] = descend[0]
                    _, _, done, _= self.env.step(descend[0])
                    i = descend[0]
                

                self.tree.AddState(a = deepcopy(temp)) 
                self.tree.addChild(state,child)
                self.tree[child]["isTerminal"] = done
                is_good = i in (possible_moves )
                self.tree[child]['is_good'] = is_good



                if is_good: #so that best child is only ever chosen from this list
                    self.tree[state]['good_children'].append(child)
                return child

    def treePolicy(self,state): #nxa
        """
        Expands the tree until a terminal node is reached.
        """

        while self.tree[state]["isTerminal"] == False:
            
            if len(self.tree[state]["children"]) < len(self.action_space):
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
    
    def getSurrounding(self,state):
        col,row = state['blstats'][0],state['blstats'][1]
        vicinity = state['glyphs'][row-1:row+2,col-1:col+2]
        # plt.imshow(vicinity,cmap='inferno')
        # print(vicinity)
        return vicinity

    def getLegalMoves(self,state):
        vicinity = self.getSurrounding(state).flatten()
        stairs = np.where(vicinity==2383)
        descend = len(stairs[0])==1
        legals = [int(i==0) or int(i>2359 and i<2370) for i in vicinity]
        legals = np.ones(len(legals),dtype=np.int8) - legals
        close_move = [8,1,5,4,19,2,7,3,6] #first eight positions
        legal_close = np.unique(close_move*legals)[1:-1]
        long_move = [16,9,13,12,19,10,15,11,14] #long eight positions
        legal_long = np.unique(long_move*legals)[1:-1]
        if not descend:
            descend = []
        else:
            descend = [close_move[stairs[0][0]]]

        return list(legal_close)+list(legal_long),descend