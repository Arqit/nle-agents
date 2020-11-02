import numpy as np
import json
from numpyencoder import NumpyEncoder
from collections import defaultdict
import os

class Tree:
    def __init__(self):
        self.dictionary = defaultdict(lambda: {})
        self.root = self.AddState()
        self.index = 0

    '''Only keep lowest actions to get to state'''
    def AddState(self, a=[], p=None): # a child node
        self[self.index]["parent"] = p
        self[self.index]["actions"] = a
        self[self.index]["isTerminal"] = False
        self[self.index]["children"] = []
        self[self.index]["reward"] = 0
        self[self.index]["visits"] = 0
        return self.index

    def backup(self, delta, state):
        current = state
        while current != None:
            self[current]["visits"] += 1
            self[current]["reward"] += delta
            current = self[current]["parent"]

    def addChild(self,parent,child):
        if self[child]["parent"] == None:
            self[child]["parent"] = parent
            self[parent]["children"].append(child)

    def getChild(self,current,state): #use actions
        for key in state.keys():
            if (type(state[key])==np.ndarray):
                state[key] = state[key].tolist()
        for i in range(len(self[current]["children"])):
            if np.all(self[self[current]["children"][i]]["state"] == state):
                return self[current]["children"][i]
        return None
    
    def bestChild(self,current,c): #UCB
        first = np.divide([self[self[current]["children"][i]]["reward"] for i in range(len(self[current]["children"]))],[self[self[current]["children"][i]]["visits"] for i in range(len(self[current]["children"]))])
        second = np.sqrt(np.divide(2*np.log(self[current]["visits"]),[self[self[current]["children"][i]]["visits"] for i in range(len(self[current]["children"]))]))
        return self[current]["children"][np.argmax(first+c*second)]
    
    def __del__(self):
        del self.dictionary

    def __getitem__(self, index):
        return self.dictionary[index]

# class node:
#     def __init__(self, s, a=[], p=None):
#         self.actions = a
#         self.state = s
#         self.visits = 0
#         self.reward = 0
#         self.parent = p
#         self.isTerminal = False
#         self.children = [] # all children of current node

#     def backup(self, delta):
#         current = self
#         while current != None:
#             self.visits += 1
#             self.reward += delta
#             current = current.parent

#     def addChild(self,child):
#         child.parent = self
#         self.children.append(child)
        
#     def getChild(self,state):
#         for i in range(len(self.children)):
#             if self.children[i].state == state:
#                 return self.children[i]
#         return None

#     def sampleChild(self):
#         if len(self.children>0):
#             return np.random.choice(self.children)
#         else:
#             return None

#     def deleteChildren(self):
#         if (len(self.children)) == 0:
#             del self
#         else:
#             for i in range(len(self.children)):
#                 self.children[i].deleteChildren()
    
#     def bestChild(self,c): #UCB
#         first = np.divide([self.children[i].reward for i in range(len(self.children))],[self.children[i].visits for i in range(len(self.children))])
#         second = np.sqrt(np.divide(2*np.log(self.visits),[self.children[i].visits for i in range(len(self.children))]))
#         return self.children[np.argmax(first+c*second)]