import numpy as np

class node:
    def __init__(self, s, a=None, p=None):
        self.action = a
        self.state = s
        self.visits = 0
        self.reward = 0
        self.parent = p
        self.isTerminal = False
        self.children = []

    def backup(self, delta):
        self.visits += 1
        self.reward += delta
        if self.parent != None:
            self.parent.backup(delta)

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
    
    def bestChild(self,c): #UCB
        first = np.divide([self.children[i].reward for i in range(len(self.children))],[self.children[i].visits for i in range(len(self.children))])
        second = np.sqrt(np.divide(2*np.log(self.visits),[self.children[i].visits for i in range(len(self.children))]))
        return self.children[np.argmax(first+c*second)]