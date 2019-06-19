import math
import random
import time

import numpy as np
from sklearn.datasets import load_iris, load_wine
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
        # self.v_gini = np.vectorize(self.gini)

    def grow(self, samples, labels):
        if isinstance(samples, np.ndarray):
            self.samples = np.copy(samples)
        else:
            self.samples = np.copy(samples.toarray())
        self.labels = np.copy(labels)
        if self.max_features == 'log2':
            def sel_attributes():
                return np.random.choice(self.samples.shape[1], size=int(np.ceil(np.log2(self.samples.shape[1]))), replace=False)  # 不放回选择
        elif self.max_features == 'sqrt':
            def sel_attributes():
                return np.random.choice(self.samples.shape[1], size=int(np.ceil(math.sqrt(self.samples.shape[1]))), replace=False)  # 不放回选择
        self.sel_attributes = sel_attributes
        self.root = self.recursive_grow(0, np.arange(self.samples.shape[0], dtype=np.long), self.sel_attributes())
        del self.labels
        del self.samples

    def gini(self, cur_samples_idx):
        # gini_value * |Dv| of a subset
        # counts of different classes
        _, counts = np.unique(self.labels[cur_samples_idx], return_counts=True)
        counts = counts / len(cur_samples_idx)
        return (1 - np.sum(counts**2)) * len(cur_samples_idx)

    def gini_index(self, total_num, split):
        """[summary]

        Parameters
        ----------
        split : list(ndarray)
            a list of disjoint subsets of data
        """
        return np.sum([self.gini(s) for s in split]) / total_num

    def entropy(self, cur_samples_idx):
        _, counts = np.unique(self.labels[cur_samples_idx], return_counts=True)
        counts = counts / len(cur_samples_idx)
        return -np.sum(counts*np.log2(counts))

    def infogain(self, total_num, split):
        return self.entropy(np.concatenate((split[0], split[1]))) - np.sum([self.entropy(s) * len(s) for s in split]) / total_num

    def all_same(self, cur_samples_idx):
        tmp = np.unique(self.labels[cur_samples_idx])
        return len(tmp) == 1, tmp[0]  # 只有一个unique的元素
        # return np.bincount(self.labels[cur_samples]).argmax() == len(cur_samples_idx)

    def split_attributes(self, cur_samples_idx, cur_attrs_idx):
        size = len(cur_samples_idx)  # 当前样本数
        best_split = tuple()
        best_crit = 1 * float("inf")  # 当前最小criterion(基尼系数)
        best_attribute = -1
        best_threshold = None
        tmp_samples = self.samples[cur_samples_idx].copy()  # copy一份样本，仍然包括所有的属性列

        for attr in cur_attrs_idx:
            order = tmp_samples[:, attr].argsort()  # 返回按照attr列排序的索引, 取值[0, len(tmp_samples))
            for i in range(len(order) - 1):
                if tmp_samples[order[i], attr] != tmp_samples[order[i+1], attr]:
                    # 和邻居不相等
                    threshold = (tmp_samples[order[i], attr] + tmp_samples[order[i+1], attr]) / 2
                    split = (cur_samples_idx[order[:i+1]], cur_samples_idx[order[i+1:]])  # 对于self.samples的索引, 小于/大于阈值的序号
                    crit = self.gini_index(size, split)
                    # crit = -self.infogain(size, split)
                    if crit < best_crit:
                        best_split = split
                        best_threshold = threshold
                        best_crit = crit
                        best_attribute = attr

        # 属性，阈值，(小于, 大于)
        return best_attribute, best_threshold, best_split if len(best_split) != 0 else tuple()

    def recursive_grow(self, cur_depth, cur_samples_idx, cur_attrs):
        if cur_depth == self.max_depth or len(cur_attrs) == 0:
            # 深度最大，或没有属性可供切分，选择最多的
            unique, count = np.unique(self.labels[cur_samples_idx], return_counts=True)
            most_occur = unique[count.argmax()]
            return Node(True, -1, None, most_occur)
        elif self.all_same(cur_samples_idx)[0]:
            # 这个节点的全部一样
            return Node(True, -1, None, self.all_same(cur_samples_idx)[1])
        else:
            best_attr, thresh, split = self.split_attributes(
                cur_samples_idx, cur_attrs)
            if len(split) == 0:
                # 没切出来东西
                unique, count = np.unique(self.labels[cur_samples_idx], return_counts=True)
                most_occur = unique[count.argmax()]
                return Node(True, -1, None, most_occur)
            node = Node(False, best_attr, thresh)
            node.children = [self.recursive_grow(
                cur_depth+1, s, self.sel_attributes()) for s in split]
            return node

    def classify(self, sample):
        return [self.recursive_classify(self.root, s) for s in sample]

    def recursive_classify(self, node: Node, sample):
        if node.is_leaf:
            return node.label
        else:
            # 划分属性值
            if sample[node.attr] > node.threshold:
                return self.recursive_classify(node.children[1], sample)
            else:
                return self.recursive_classify(node.children[0], sample)


class RandomForest():

    def __init__(self, label_num, tree_count, tree_depth):
        self.trees = [desTree(tree_depth, label_num)] * tree_count
        self.tree_count = tree_count
        self.tree_depth = tree_depth


    def grow(self, samples, labels):
        # num_samples = samples.shape[0]
        # tmp = np.random.choice(samples.shape[0], size=samples.shape[0], replace=True)
        # cnt = 0
        for tree in self.trees:
            # Bagging
            indice = np.random.choice(samples.shape[0], size=samples.shape[0], replace=True)
            # if np.array_equiv(np.sort(tmp), np.sort(indice)):
            #     cnt += 1
            # tmp = indice
            tree.grow(samples[indice, :], labels[indice])
        # print(cnt)


    def classify(self, samples):
        candidates = np.array([tree.classify(samples) for tree in self.trees])
        result = np.zeros(candidates.shape[1], dtype=np.long)
        for i in range(candidates.shape[1]):
            vote = candidates[:, i]
            unique, counts = np.unique(vote, return_counts=True)  # 计票
            result[i] = unique[counts.argmax()]  # 返回票数最多的类别
        return result

    def score(self, samples, labels):
        result = self.classify(samples)
        return np.sum(result == labels) / labels.shape[0]

        
            
        
if __name__ == "__main__":
    iris = load_iris(True)
    X_train, X_test, y_train, y_test = train_test_split(iris[0], iris[1], test_size=0.33, random_state=7)
    tree = desTree(5, 3)
    tree.grow(X_train, y_train)
    print(np.sum(tree.classify(X_train) == y_train) / y_train.shape[0])
    print(np.sum(tree.classify(X_test) == y_test) / y_test.shape[0])
    forest = RandomForest(3, 100, 3)
    forest.grow(X_train, y_train)
    print(forest.score(X_train, y_train))
    print(forest.score(X_test, y_test))
    # print(np.sum(forest.classify(X_train) == y_train) / y_train.shape[0])
    # print(np.sum(forest.classify(X_test) == y_test) / y_test.shape[0])
    # for t in forest.trees:
    #     print(np.sum(t.classify(X_train) == y_train) / y_train.shape[0])

    wine = load_wine(True)
    X_train, X_test, y_train, y_test = train_test_split(wine[0], wine[1], test_size=0.33, random_state=7)
    tree = desTree(5, 3)
    tree.grow(X_train, y_train)
    print(np.sum(tree.classify(X_train) == y_train) / y_train.shape[0])
    print(np.sum(tree.classify(X_test) == y_test) / y_test.shape[0])
    forest = RandomForest(3, 50, 4)
    forest.grow(X_train, y_train)
    print(forest.score(X_train, y_train))
    print(forest.score(X_test, y_test))
