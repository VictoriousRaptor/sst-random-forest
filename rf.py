# In[]
import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
import scipy.sparse

import dataset
import decisiontree
import time


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='data/dataset/', help='PATH to dataset')
parser.add_argument('--feature', type=str, default='tfidf', choices=['tfidf', 'vector'], help='Which kind of feature to use')
parser.add_argument('--tree_count', type=int, default=300, help='Number of trees in the forest')
parser.add_argument('--tree_depth', type=int, default=64, help='Max depth of a single tree')
parser.add_argument('--emb_dim', type=int, default=50, help='Word embedding dim')
parser.add_argument('--runs', type=int, default=10, help='Number of forests')
parser.add_argument('--model', type=str, default='sklearn', choices=['sklearn', 'mine'], help='Which RF implementation to use')

args = parser.parse_args()

# In[]
if args.model == 'mine':
    args.feature = 'vector'

if args.feature == 'vector':
    wordvec = dataset.loadGloveModel('../midterm/data/glove/glove.6B.'+ str(args.emb_dim) +'d.txt')
    args.weight = wordvec

# In[]
ds = dataset.SSTDataset(args.dataset_path, 2, args)

if args.feature == 'vector':
    delattr(args, 'weight')

train_set = ds.train_set()
test_set = ds.test_set()

#%%
best = (0, 0)
avg = [0, 0]
if args.model == 'sklearn':
    for i in range(args.runs):
        clf = RandomForestClassifier(n_estimators=args.tree_count, max_depth=args.tree_depth, n_jobs=-1, criterion='gini', max_features='log2')
        # clf = RandomForestClassifier(n_estimators=args.tree_count, n_jobs=-1, criterion='gini', max_features='log2')
        clf.fit(train_set.features, train_set.labels)
        train_acc = clf.score(train_set.features, train_set.labels)
        test_acc = clf.score(test_set.features, test_set.labels)
        if test_acc > best[1]:
            best = (train_acc, test_acc)
        avg[0] += train_acc
        avg[1] += test_acc
        print(i, ' {:.4f}|{:.4f}'.format(train_acc, test_acc))
        test_prediction = clf.predict(test_set.features)
        print(confusion_matrix(test_set.labels, test_prediction))
elif args.model == 'mine':
    # pca = PCA(50)
    # train_set.features = pca.fit_transform(train_set.features.toarray())
    # test_set.features = pca.transform(test_set.features.toarray())
    forest = decisiontree.RandomForest(2, args.tree_count, args.tree_depth)
    print(train_set.features.shape)
    start = time.time()
    forest.grow(train_set.features, train_set.labels)
    end = time.time() - start
    print(end)
    print(forest.score(train_set.features, train_set.labels))
    print(forest.score(test_set.features, test_set.labels))
    for tree in forest.trees:
        print(np.count_nonzero(tree.classify(test_set.features) == test_set.labels) / len(test_set.labels))



#%%
print('best {:.4f}|{:.4f}'.format(best[0], best[1]))
print('avg {:.4f}|{:.4f}'.format(avg[0]/args.runs, avg[1]/args.runs))
print('\n{:.4f}|{:.4f}|{:.4f}|{:.4f}'.format(best[0], best[1], avg[0]/args.runs, avg[1]/args.runs))

print("Parameters:")

for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

