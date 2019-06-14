import numpy as np
import random
import math



class desTree():
    class Node():
        def __init__(self, is_leaf=False, threshold=None, label=None):
            self.is_leaf = is_leaf
            self.threshold = threshold
            self.label = label
            self.children = []
            

    def __init__(self, max_depth, max_features='log2'):
       self.root = None
       self.max_depth = max_depth
       self.max_features = max_features


    def grow_tree(self, features, labels):
        self.features = np.copy(features)
        self.labels = np.copy(labels)
        if self.max_features == 'log2':
            def sel_attributes():
                return np.random.randint(0, self.features.shape[1], size=np.ceil(np.log2(self.features.shape[1])))
        self.sel_attributes = sel_attributes
        self.root = self.recursive_grow()
        del self.labels
        del self.features

    def infogain():
        pass

    def recursive_grow(self, cur_depth):
        if cur_depth == self.max_depth:
            # 深度最大
            return Node()
        cur_attr = self.sel_attributes()
        
        return Node(True)

    
