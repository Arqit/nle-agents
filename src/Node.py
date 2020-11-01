import numpy as np
import json
from numpyencoder import NumpyEncoder
from collections import defaultdict
import os

class Tree:
    def __init__(self, state):
        self.dictionary = defaultdict(lambda: {})
        self.stateList = [state]
        self.root = self.AddState(state)
        

    """ Saves to the tmp folder """
    def save(self, fileName="../data.json", fileName2="../states.json"):
        f = open(fileName, "w+")
        print(os.getcwd())
        json.dump(self.dictionary, f, cls=NumpyEncoder)
        print("saving nodes")
        file2 = open(fileName2, "w")
        json.dump(self.stateList, file2, cls=NumpyEncoder, indent=4)
        print("saving")
        f.close()
        file2.close()

    def load(self, fileName="../data.json", fileName2="../states.json"):
        file = open(fileName, "r")
        file2 = open(fileName2, "r")
        self.dictionary = file.read()
        self.stateList = file2.read()
        file.close()
        file2.close()

    '''Only keep lowest actions to get to state'''
    def AddState(self, state, a=[], p=None):
        index = len(self.stateList)
        for key in state.keys():
            if (type(state[key])==np.ndarray):
                state[key] = (state[key]).tolist()
        self.stateList.append(state)
        self[index]["state"] = state
        self[index]["parent"] = p
        self[index]["actions"] = a
        self[index]["isTerminal"] = False
        self[index]["children"] = []
        self[index]["reward"] = 0
        self[index]["visits"] = 0
        return index

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

    def getChild(self,current,state):
        for key in state.keys():
            if (type(state[key])==np.ndarray):
                state[key] = state[key].tolist()
        for i in range(len(self[current]["children"])):
            if np.all(self[self[current]["children"][i]]["state"] == state):
                return self[current]["children"][i]
        return None

    def sampleChild(self, current):
        if len(self[current]["children"]>0):
            return np.random.choice(self[current]["children"])
        else:
            return None
    
    def bestChild(self,current,c): #UCB
        first = np.divide([self[self[current]["children"][i]]["reward"] for i in range(len(self[current]["children"]))],[self[self[current]["children"][i]]["visits"] for i in range(len(self[current]["children"]))])
        second = np.sqrt(np.divide(2*np.log(self[current]["visits"]),[self[self[current]["children"][i]]["visits"] for i in range(len(self[current]["children"]))]))
        return self[current]["children"][np.argmax(first+c*second)]
    
    def __del__(self):
        #self.save()
        del self.dictionary

    def __getitem__(self, index):
        # if (type(index)==dict):
        #     for i in :
        #         if self.stateList[i] == index:
        #             return self.dictionary[i]
        # else:
        return self.dictionary[index]

class node:
    def __init__(self, s, a=[], p=None):
        self.actions = a
        self.state = s
        self.visits = 0
        self.reward = 0
        self.parent = p
        self.isTerminal = False
        self.children = [] # all children of current node

    def backup(self, delta):
        current = self
        while current != None:
            self.visits += 1
            self.reward += delta
            current = current.parent

    def addChild(self,child):
        child.parent = self
        self.children.append(child)
        
    def getChild(self,state):
        for i in range(len(self.children)):
            if self.children[i].state == state:
                return self.children[i]
        return None

    def sampleChild(self):
        if len(self.children>0):
            return np.random.choice(self.children)
        else:
            return None

    def deleteChildren(self):
        if (len(self.children)) == 0:
            del self
        else:
            for i in range(len(self.children)):
                self.children[i].deleteChildren()
    
    def bestChild(self,c): #UCB
        first = np.divide([self.children[i].reward for i in range(len(self.children))],[self.children[i].visits for i in range(len(self.children))])
        second = np.sqrt(np.divide(2*np.log(self.visits),[self.children[i].visits for i in range(len(self.children))]))
        return self.children[np.argmax(first+c*second)]