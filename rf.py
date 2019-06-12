# In[]
from sklearn.ensemble import RandomForestClassifier
import dataset
import numpy as np

PATH_TO_DATASET = 'data/dataset/'

# In[]
wordvec = dataset.loadGloveModel('../midterm/data/glove/glove.6B.'+ str(50) +'d.txt')

# In[]
ds = dataset.SSTDataset(PATH_TO_DATASET, 2, 50, wordvec, 'tfidf')

train_set = ds.train_set()
test_set = ds.test_set()

# In[]
clf = RandomForestClassifier(n_estimators=2000, max_depth=10, n_jobs=-1)
clf.fit(train_set.features, train_set.labels)
# res = clf.predict(test_set.features)
test_acc = clf.score(test_set.features, test_set.labels)
train_acc = clf.score(train_set.features, train_set.labels)
print(train_acc)
print(test_acc)


#%%
