# In[]
import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier

import dataset


parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='data/dataset/', help='PATH to dataset')
parser.add_argument('--feature', type=str, default='tfidf', choices=['tfidf', 'vector'])

args = parser.parse_args()

# In[]
wordvec = dataset.loadGloveModel('../midterm/data/glove/glove.6B.'+ str(50) +'d.txt')
args.weight = wordvec

# In[]
ds = dataset.SSTDataset(args.dataset_path, 2, 50, args)

train_set = ds.train_set()
test_set = ds.test_set()

# In[]
clf = RandomForestClassifier(n_estimators=200, max_depth=10, n_jobs=-1)
clf.fit(train_set.features, train_set.labels)
# res = clf.predict(test_set.features)
test_acc = clf.score(test_set.features, test_set.labels)
train_acc = clf.score(train_set.features, train_set.labels)
print('{:.4f}|{:.4f}'.format(train_acc, test_acc))

print("Parameters:")
delattr(args, 'weight')
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

#%%
