import math
import multiprocessing
import random
import time

import numpy as np
from joblib import Parallel, delayed
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

    def depth(self):
        if self.root is None:
            return 0
        return self.recur_depth(1, self.root)

    def recur_depth(self, depth, node):
        if node.is_leaf:
            return depth
        else:
            return max([self.recur_depth(depth+1, n) for n in node.children])

    def fit(self, X, y):
        self.grow(X, y)
        return self

    def predict(self, X):
        return self.classify(X)

    def grow(self, samples, labels, share_res=False, div=None):
        if not share_res:
            if isinstance(samples, np.ndarray):
                self.samples = np.copy(samples)
            else:
                self.samples = np.copy(samples.toarray())
            self.labels = np.copy(labels)
        else:
            # TODO: 现有调用方式没区别的
            self.samples = samples
            self.labels = labels

        # 负优化
        # 预先计算划分点
        # TODO：没必要每棵树都计算划分点
        # self.sorted_samples = np.argsort(samples, axis=0)  # sorted by axis 1
        # self.div = [list() for i in range(self.samples.shape[1])]
        # for j in range(self.samples.shape[1]):
        #     for i in range(self.samples.shape[0]-1):
        #         if self.samples[self.sorted_samples[i, j], j] != self.samples[self.sorted_samples[i+1, j], j]:
        #             self.div[j].append(((self.samples[self.sorted_samples[i, j], j] + self.samples[self.sorted_samples[i+1, j], j])/2, i+1))

        if self.max_features == 'log2':
            def sel_attributes():
                # 不放回选择
                return np.random.choice(self.samples.shape[1], size=int(np.ceil(np.log2(self.samples.shape[1]))), replace=False)
        elif self.max_features == 'sqrt':
            def sel_attributes():
                # 不放回选择
                return np.random.choice(self.samples.shape[1], size=int(np.ceil(math.sqrt(self.samples.shape[1]))), replace=False)
        self.sel_attributes = sel_attributes

        self.root = self.recursive_grow(1, np.arange(
            self.samples.shape[0], dtype=np.long), self.sel_attributes())
        del self.labels
        del self.samples

    def gini(self, cur_samples_idx):
        # gini_value * |Dv| of a subset
        # counts of different classes
        _, counts = np.unique(self.labels[cur_samples_idx], return_counts=True)
        counts = counts / len(cur_samples_idx)
        return (1 - np.sum(counts**2)) * len(cur_samples_idx)

    def gini_index(self, total_num, split):
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

        tmp_samples = self.samples[cur_samples_idx]  # copy一份样本，仍然包括所有的属性列
        # TODO: 没必要每次递归都进行排序二分
        # 但是实现做不到更快了
        for attr in cur_attrs_idx:
            # 返回按照attr列排序的索引, 取值[0, len(tmp_samples))
            order = tmp_samples[:, attr].argsort()
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

        # 负优化
        # for attr in cur_attrs_idx:
        #     for threshold, upper in self.div[attr]:
        #         # 不是每个threshold都在当前取值范围内
        #         smaller = np.intersect1d(self.sorted_samples[:upper, attr], cur_samples_idx, assume_unique=True)
        #         bigger = np.setdiff1d(cur_samples_idx, smaller, assume_unique=True)
        #         split = (smaller, bigger)
        #         crit = self.gini_index(size, split)
        #         # crit = -self.infogain(size, split)
        #         if crit < best_crit:
        #             best_split = split
        #             best_threshold = threshold
        #             best_crit = crit
        #             best_attribute = attr

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

    def classify(self, samples):
        if isinstance(samples, np.ndarray):
            tmp = samples
        else:
            tmp = samples.toarray()
        if self.root is not None:
            return [self.recursive_classify(self.root, s) for s in tmp]
        else:
            print('empty tree')
            return [0] * len(samples)

    def recursive_classify(self, node: Node, sample):
        if node.is_leaf:
            return node.label
        else:
            # 划分属性值
            if sample[node.attr] > node.threshold:
                return self.recursive_classify(node.children[1], sample)
            else:
                return self.recursive_classify(node.children[0], sample)

    def score(self, X, y):
        result = self.classify(X)
        return np.sum(result == y) / y.shape[0]


class RandomForest():

    def __init__(self, label_num, tree_count, tree_depth, para=True):
        if not para:
            self.trees = [desTree(tree_depth, label_num) for tree in range(tree_count)]
        self.para = para  # parallel
        self.label_num = label_num
        self.tree_count = tree_count
        self.tree_depth = tree_depth

    def grow(self, samples, labels):
        if self.para:
            self.para_grow(samples, labels)
        else:
            self.seq_grow(samples, labels)

    def seq_grow(self, samples, labels):
        for tree in self.trees:
            # Bagging
            indice = np.random.choice(samples.shape[0], size=samples.shape[0], replace=True)
            tree.grow(samples[indice, :], labels[indice])
        # print(cnt)

    def para_grow(self, samples, labels):

        def __grow(samples, labels, tree_depth, label_num):
            indice = np.random.choice(
                samples.shape[0], size=samples.shape[0], replace=True)
            tree = desTree(tree_depth, label_num)
            tree.grow(samples[indice, :], labels[indice])
            return tree

        self.trees = Parallel(n_jobs=-1)(delayed(__grow)(samples, labels,
                                                         self.tree_depth, self.label_num) for _ in range(self.tree_count))

    def classify(self, samples):
        if self.para:
            def __classify(tree, samples):
                return tree.classify(samples)

            candidates = np.array(
                Parallel(n_jobs=-1)(delayed(__classify)(tree, samples) for tree in self.trees))
        else:
            candidates = np.array([tree.classify(samples)
                                   for tree in self.trees])

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
    # check_estimator(desTree)
    print('iris')
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
    print('wine')
    wine = load_wine(True)
    X_train, X_test, y_train, y_test = train_test_split(
        wine[0], wine[1], test_size=0.33, random_state=7)
    tree = desTree(5, 3)
    tree.grow(X_train, y_train)
    print(np.sum(tree.classify(X_train) == y_train) / y_train.shape[0])
    print(np.sum(tree.classify(X_test) == y_test) / y_test.shape[0])
    forest = RandomForest(3, 50, 4)
    forest.grow(X_train, y_train)
    print(forest.score(X_train, y_train))
    print(forest.score(X_test, y_test))
    for tree in forest.trees:
        print('{:d} {:.4f}'.format(tree.depth(), tree.score(X_test, y_test)))
    forest.trees[0].depth()

