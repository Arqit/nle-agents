import numpy as np
import json
from numpyencoder import NumpyEncoder
from collections import defaultdict
import os

class Tree:
    def __init__(self,loading=False):
        self.dictionary = defaultdict(lambda: {})
        self.stateCount = 0
        self.root = 0
        if not loading:
            self.AddState()
        
        '''
        We need the state count to add new children. We can't
        just take length of states because it gets pruned and
        states will get over written
        '''
        """ Saves to the tmp folder """
    def save(self, reward,directory,seed):
        fileName=directory+"/"+str(seed)+".json"
        f = open(fileName, "w+")
        self.dictionary[-1] = self.stateCount
        self.dictionary[-2] = reward
        json.dump(self.dictionary, f, cls=NumpyEncoder)
        print("saving nodes")
        f.close()

    def load(self,directory,seed):
        fileName=directory+"/"+str(seed)+".json"
        file = open(fileName, "r")
        inDict = json.loads(file.read())
        file.close()
        for i,j in inDict.items():
            self.dictionary[int(i)] = j
        keys = np.sort(list(self.dictionary.keys()))
        self.root = keys[2]
        self.stateCount = self.dictionary[-1]
        return self.dictionary[-2]

    '''Only keep lowest actions to get to state'''
    def AddState(self, a=[], p=None): # a child node
        index = self.stateCount
        self[index]["parent"] = p
        self[index]["actions"] = a
        self[index]["isTerminal"] = False
        self[index]["children"] = []
        self[index]["reward"] = 0
        #self[index]["solo"] = 0
        self[index]["visits"] = 0
        self.stateCount += 1

    def backup(self, delta, state):
        #backup a reward when a terminal state is reached
        current = state
        while current != None:
            self[current]["visits"] += 1
            self[current]["reward"] += delta
            current = self[current]["parent"]

    def addChild(self,parent,child):
        if self[child]["parent"] == None: #check that ch
            self[child]["parent"] = parent
            self[parent]["children"].append(child)
    
    def bestChild(self,current,c): #UCB
        
        '''
        Sometimes the best node is the terminal node i.e. the game is over. In that 
        case then that node cannot be given as "best child" and the next 'best child'
        must be found. If no best children are available then the algorithm must track back up.
        '''
        
        first = np.divide([self[self[current]["children"][i]]["reward"] for i in range(len(self[current]["children"]))],[self[self[current]["children"][i]]["visits"] for i in range(len(self[current]["children"]))])
        second = np.sqrt(np.divide(2*np.log(self[current]["visits"]),[self[self[current]["children"][i]]["visits"] for i in range(len(self[current]["children"]))]))
		return self[current]["children"][np.argmax(first+c*second)]
    
    def __del__(self):
        del self.dictionary

    def __getitem__(self, index):
        return self.dictionary[index]
