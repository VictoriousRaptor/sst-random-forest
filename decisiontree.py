import numpy as np
import random
import math


class Node():
    def __init__(self, is_leaf=False, attr=-1, threshold=None, label=None):
        self.is_leaf = is_leaf
        self.attr = attr
        self.threshold = threshold
        self.label = label
        self.children = []

    def has_child():
        return len(self.children) != 0


class desTree():

    def __init__(self, max_depth, label_num=2, max_features='log2'):
        self.root = None
        self.max_depth = max_depth
        self.max_features = max_features
        self.label_num = label_num
        self.v_gini = np.vectorize(self.gini)

    def grow_tree(self, features, labels):
        self.features = np.copy(features)
        self.labels = np.copy(labels)
        if self.max_features == 'log2':
            def sel_attributes():
                return np.random.randint(0, self.features.shape[1], size=np.ceil(np.log2(self.features.shape[1])))
        self.sel_attributes = sel_attributes
        self.root = self.recursive_grow(0, np.arange(0, self.features.shape[0]), self.sel_attributes())
        del self.labels
        del self.features

    def gini(self, cur_samples):
        # gini_value * |Dv| of a subset
        # counts of different classes
        _, counts = np.unique(self.labels[cur_samples], return_counts=True)
        counts = counts / len(cur_samples)
        return (1 - np.sum(counts**2)) * len(cur_samples)

    def gini_index(self, total_num, split):
        """[summary]

        Parameters
        ----------
        split : list(ndarray)
            a list of disjoint subsets of data
        """
        return np.sum(self.v_gini(split)) / total_num


    def all_same(self, cur_samples):
        return np.bincount(self.labels[cur_samples]).argmax() == len(cur_samples)

    def split_attributes(self, cur_samples, cur_attrs):
        size = len(cur_samples)
        best_split = None
        best_crit = -1*float("inf")  # 当前最大基尼系数
        best_attribute = -1
        best_threshold = None
        tmp_samples = self.features[cur_samples].copy()  # copy一份样本，仍然包括所有的属性列
        for attr in cur_attrs:
            order = tmp_samples.argsort(axis=attr)
            for i in range(len(tmp_samples) - 1):
                if tmp_samples[i, attr] != tmp_samples[i+1, attr]:
                    # 和邻居不相等
                    threshold = tmp_samples[i, attr] + tmp_samples[i+1, attr] / 2
                    split = (order[:i+1], order[i+1:])  # 小于/大于阈值的序号
                    crit = self.gini_index(size, split)
                    if crit > best_crit:
                        best_split = split
                        best_threshold = threshold
                        best_crit = crit
                        best_attribute = attr
        
        return best_attribute, best_threshold, (cur_samples[best_split[0]], cur_samples[best_split[1]])  # 阈值，(小于, 大于)


    def recursive_grow(self, cur_depth, cur_samples, cur_attrs):
        if cur_depth == self.max_depth or len(cur_attrs) == 0:
            # 深度最大，选择最多的
            return Node(True, -1, None, np.bincount(self.labels[cur_samples]).argmax())
        elif self.all_same(cur_samples):
            # 这个节点的全部一样
            return Node(True, -1, None, self.labels[cur_samples[0]])
        else:
            cur_attrs = self.sel_attributes()  # 随机选择特征
            best_attr, thresh, split = self.split_attributes(cur_samples, cur_attrs)
            node = Node(False, best_attr, thresh)
            node.children = [self.recursive_grow(cur_depth+1, s, self.sel_attributes()) for s in split]
            return node
    
    def find(self, sample):
        return recursive_find(self.root, sample)

    
    def recursive_find(self, node:Node, sample):
        if node.is_leaf:
            return node.label
        else:
            # 划分属性值
            if sample[node.attr] > node.threshold:
                return self.recursive_find(node.children[1], sample)
            else:
                return self.recursive_find(node.children[0], sample)