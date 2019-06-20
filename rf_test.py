# -*- coding: utf-8 -*-

import argparse
import time

import numpy as np

from sklearn.ensemble import RandomForestClassifier

import decisiontree

from RCNN import RCNN
# from RNN import myRNN, LSTMClassifier
from dataset import SSTDataset_torch, loadGloveModel
from TextCNN import TextCNN



def main():
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    parser = argparse.ArgumentParser()
    parser.add_argument('--label_num', type=int, default=2, help='Target label numbers')
    parser.add_argument('--dataset_path', type=str, default='data/dataset/', help='PATH to dataset')
    parser.add_argument('--runs', type=int, default=1, help='')
    
    args = parser.parse_args()

    
    # Random Forest
    best = (0, 0)
    avg = [0, 0]
    for i in range(args.runs):
        # clf = RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, criterion='gini', max_features='log2')
        # clf.fit(train_features, train_labels)
        # train_acc = clf.score(train_features, train_labels)
        # test_acc = clf.score(test_features, test_labels)

        # 10d features
        train_features = np.load('train_feature_CNN.npy')
        test_features = np.load('test_feature_CNN.npy')
        train_labels = np.load('train_label_for_forest.npy')
        test_labels = np.load('test_label_for_forest.npy')

        # forest = decisiontree.RandomForest(2, 9, 50)
        forest = decisiontree.RandomForest(2, 9, 10)
        print(train_features.shape)
        start = time.time()
        # forest.grow(np.vstack((train_features, dev_features)), np.hstack((train_labels, dev_labels)))
        forest.grow(train_features, train_labels)
        end = time.time() - start
        print(end)
        train_acc = forest.score(train_features, train_labels)
        test_acc = forest.score(test_features, test_labels)
        if test_acc > best[1]:
            best = (train_acc, test_acc)
        avg[0] += train_acc
        avg[1] += test_acc
        print(i, ' {:.4f}|{:.4f}'.format(train_acc, test_acc))
        for tree in forest.trees:
            print(tree.score(test_features, test_labels))

    print('best {:.4f}|{:.4f}'.format(best[0], best[1]))
    print('avg {:.4f}|{:.4f}'.format(avg[0]/args.runs, avg[1]/args.runs))
    print('\n{:.4f}|{:.4f}|{:.4f}|{:.4f}'.format(best[0], best[1], avg[0]/args.runs, avg[1]/args.runs))
        


    print("Parameters:")
    for attr, value in sorted(args.__dict__.items()):
        print("\t{}={}".format(attr.upper(), value))


if __name__ == "__main__":
    main()
