from sklearn.ensemble import RandomForestClassifier
import dataset
import numpy as np


wordvec = dataset.loadGloveModel('../midterm/data/glove/glove.6B.'+ str(50) +'d.txt')

train_set = dataset.SSTDataset_vector('data/dataset/', 'train', 2, 50, wordvec)
test_set = dataset.SSTDataset_vector('data/dataset/', 'test', 2, 50, wordvec)

clf = RandomForestClassifier(n_estimators=100, max_depth=5)
clf.fit(train_set.features, train_set.labels)
res = clf.predict(test_set.features)
print(np.sum(res == test_set.labels) / len(test_set))