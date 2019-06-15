import numpy as np
import random
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


class Node():
    def __init__(self, is_leaf=False, attr=-1, threshold=None, label=None):
        self.is_leaf = is_leaf
        self.attr = attr
        self.threshold = threshold
        self.label = label
        self.children = []

    def has_child(self):
        return len(self.children) != 0


class desTree():

    def __init__(self, max_depth, label_num=2, max_features='log2'):
        self.root = None
        self.max_depth = max_depth
        self.max_features = max_features
        self.label_num = label_num
        self.v_gini = np.vectorize(self.gini)

    def grow(self, features, labels):
        self.features = np.copy(features)
        self.labels = np.copy(labels)
        if self.max_features == 'log2':
            def sel_attributes():
                # return np.random.randint(0, self.features.shape[1], size=np.ceil(np.log2(self.features.shape[1])))
                return np.random.choice(self.features.shape[1], size=int(np.ceil(np.log2(self.features.shape[1]))), replace=False)  # 不放回选择
        self.sel_attributes = sel_attributes
        self.root = self.recursive_grow(0, np.arange(
            0, self.features.shape[0]), self.sel_attributes())
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
        return np.sum([self.gini(s) for s in split]) / total_num

    def all_same(self, cur_samples):
        return np.unique(self.labels[cur_samples], return_counts=True)[1].max() == len(cur_samples)
        # return np.bincount(self.labels[cur_samples]).argmax() == len(cur_samples)

    def split_attributes(self, cur_samples, cur_attrs):
        size = len(cur_samples)
        best_split = tuple()
        best_crit = -1*float("inf")  # 当前最大基尼系数
        best_attribute = -1
        best_threshold = None
        tmp_samples = self.features[cur_samples].copy()  # copy一份样本，仍然包括所有的属性列
        for attr in cur_attrs:
            order = tmp_samples.argsort(axis=attr)  # TODO: 按照某一列来排序，而不是某个axis
            for i in range(len(tmp_samples) - 1):
                if tmp_samples[i, attr] != tmp_samples[i+1, attr]:
                    # 和邻居不相等
                    threshold = tmp_samples[i, attr] + \
                        tmp_samples[i+1, attr] / 2
                    split = (order[:i+1], order[i+1:])  # 小于/大于阈值的序号
                    crit = self.gini_index(size, split)
                    if crit > best_crit:
                        best_split = split
                        best_threshold = threshold
                        best_crit = crit
                        best_attribute = attr

        # 阈值，(小于, 大于)
        return best_attribute, best_threshold, (cur_samples[best_split[0]], cur_samples[best_split[1]]) if len(best_split) != 0 else tuple()

    def recursive_grow(self, cur_depth, cur_samples, cur_attrs):
        if cur_depth == self.max_depth or len(cur_attrs) == 0:
            # 深度最大，选择最多的
            return Node(True, -1, None, np.unique(self.labels[cur_samples], return_counts=True)[1].argmax())
        elif self.all_same(cur_samples):
            # 这个节点的全部一样
            return Node(True, -1, None, self.labels[cur_samples[0]])
        else:
            cur_attrs = self.sel_attributes()  # 随机选择特征
            best_attr, thresh, split = self.split_attributes(
                cur_samples, cur_attrs)
            if len(split) == 0:
                # 没切出来东西
                return Node(True, -1, None, np.unique(self.labels[cur_samples], return_counts=True)[1].argmax())
            node = Node(False, best_attr, thresh)
            node.children = [self.recursive_grow(
                cur_depth+1, s, self.sel_attributes()) for s in split]
            return node

    def classify(self, sample):
        return [self.recursive_find(self.root, s) for s in sample]

    def recursive_find(self, node: Node, sample):
        if node.is_leaf:
            return node.label
        else:
            # 划分属性值
            if sample[node.attr] > node.threshold:
                return self.recursive_find(node.children[1], sample)
            else:
                return self.recursive_find(node.children[0], sample)


class RandomForest():

    def __init__(self, label_num, tree_count, tree_depth):
        self.trees = [desTree(tree_depth, label_num)] * tree_count
        self.tree_count = tree_count
        self.tree_depth = tree_depth

    def grow(self, features, labels):
        num_feature = features.shape[0]
        for tree in self.trees:
            # Bagging
            indice = np.random.choice(num_feature, size=num_feature, replace=True)
            tree.grow(features[indice, :], labels[indice])

    def classify(self, features):
        candidates = [tree.classify(features) for tree in self.trees]
        return np.unique(candidates, return_counts=True)[1].argmax()
            
        
if __name__ == "__main__":
    iris = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(iris[0], iris[1], test_size=0.33, random_state=42)
    forest = RandomForest(3, 5, 2)
    forest.grow(X_train, y_train)
    print(forest.classify(X_test))