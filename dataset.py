#%%
import re
import time
import copy

import numpy as np
import pandas as pd
import torch
import torchtext.data
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from torch.utils.data import Dataset, TensorDataset


#%%
def clean_data(sentence):
    # From yoonkim: https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    sentence = re.sub(r"[^A-Za-z(),!?\'\`]", " ", sentence)
    sentence = re.sub(r"\s{2,}", " ", sentence)
    return sentence.strip().lower()

#%%
def get_class(sentiment, num_classes):
    # 根据sentiment value 返回一个label
    return int(sentiment * (num_classes - 0.001))

#%%
def loadGloveModel(gloveFile):
    glove = pd.read_csv(gloveFile, sep=' ', header=None, encoding='utf-8', index_col=0, na_values=None, keep_default_na=False, quoting=3)
    return glove  # (word, embedding), 400k*dim
#%%
class holder(object):
    def __init__(self, length):
        self.labels = np.zeros((length,), dtype=np.long)
        print(self.labels.shape)
        self.features = None

    def set_feature(self, obj):
        self.features = obj

    def __getitem__(self, index):
        return self.features[index], self.labels[index]

    def __len__(self):
        return len(self.features)
#%%
class SSTDataset():

    def __init__(self, path_to_dataset, num_classes, args):
        """

        Parameters
        ----------
        path_to_dataset : str
            PATH to SST dataset

        num_classes : int
            2 or 5

        wordvec : pd.DataFrame
            GloVe embedding

        mode : str, optional
            What kind of feature to use, 'vector' or 'tfidf', by default 'vector'

        """
        set_names = ['train', 'dev', 'test']  
        phrase_ids = []
        for name in set_names:
            tmp = pd.read_csv(path_to_dataset + 'phrase_ids.' +
                                    name + '.txt', header=None, encoding='utf-8', dtype=int)
            phrase_ids.append(set(np.array(tmp).squeeze()))  # 在数据集中出现的pharse id
        self.num_classes = num_classes
        phrase_dict = [{} for i in range(len(set_names))]  # {id->phrase} 

        label_tmp = pd.read_csv(path_to_dataset + 'sentiment_labels.txt',
                     sep='|', dtype={'phrase ids': int, 'sentiment values': float})
        label_tmp = np.array(label_tmp)[:, 1:]  # sentiment value
        
        with open(path_to_dataset + 'dictionary.txt', 'r', encoding='utf-8') as f:
            for line in f:
                phrase, phrase_id = line.strip().split('|')
                for j, ids in enumerate(phrase_ids):
                    if int(phrase_id) in ids:  # 在数据集中出现
                        phrase = clean_data(phrase)  # 预处理
                        phrase_dict[j][int(phrase_id)] = phrase
        

        phrase_dict[0].update(phrase_dict[1])  # 验证集用于训练！
        print(len(phrase_dict[0]))
        self.sets = [holder(len(i)) for i in phrase_dict]

        if args.feature == 'vector':
            for i, s in enumerate(self.sets):
                features = []

                missing_count = 0
                # 查找每个句子中词的词向量
                for i, (idx, p) in enumerate(phrase_dict[i].items()):
                    tmp1 = []  # 暂存句子中单词的id
                    # 分词
                    for w in p.split(' '):
                        try:
                            tmp1.append(args.weight.index.get_loc(w))  # 单词w在glove中的index
                        except KeyError:
                            missing_count += 1

                    # self.phrase_vec.append(np.array(tmp1, dtype=np.long))  # 包含句子中每个词的glove index
                    features.append(np.average(np.array(args.weight.iloc[tmp1, :]), axis=0))
                    # features.append(np.average(np.array([args.weight.iloc[j, :] for j in tmp1]), axis=0))
                    s.labels[i] = get_class(label_tmp[idx], self.num_classes) # pos i 的句子的label
                    
                s.features = np.array(features)
                print(s.features.shape)

        elif args.feature == 'tfidf':
            # 预置的stopwords列表，忽略出现少于10次的单词和出现99%以上的
            # self.tfv = TfidfVectorizer(stop_words='english', min_df=3, max_df=0.99) 
            # self.tfv = TfidfVectorizer(stop_words='english', max_df=0.99) 
            nltk = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]
            self.tfv = TfidfVectorizer(stop_words=None, ngram_range=(1, 1), norm=None, min_df=2) 
            # self.tfv = TfidfVectorizer(stop_words=None, ngram_range=(1,2), max_df=0.1) 
            for i, s in enumerate(self.sets):
                for j, (idx, p) in enumerate(phrase_dict[i].items()):
                    s.labels[j] = get_class(label_tmp[idx], self.num_classes) # pos i 的句子的label

                if i == 0:
                    # train
                    s.features = self.tfv.fit_transform(phrase_dict[i].values())
                else:
                    s.features = self.tfv.transform(phrase_dict[i].values())
                print(s.features.shape)

            print('vocab:', len(self.tfv.vocabulary_))
            print('rare words:', len(self.tfv.stop_words_))


    def train_set(self):
        return self.sets[0]
    
    def dev_set(self):
        return self.sets[1]

    def test_set(self):
        return self.sets[2]

#%%
class SSTDataset_torch(Dataset):
    label_tmp = None

    def __init__(self, path_to_dataset, name, num_classes, wordvec_dim, wordvec, device='cpu'):
        """SST dataset
        
        Args:
            path_to_dataset (str): path_to_dataset
            name (str): train, dev or test
            num_classes (int): 2 or 5
            wordvec_dim (int): Dimension of word embedding
            wordvec (array): word embedding
            device (str, optional): torch.device. Defaults to 'cpu'.
        """
        phrase_ids = pd.read_csv(path_to_dataset + 'phrase_ids.' +
                                 name + '.txt', header=None, encoding='utf-8', dtype=int)
        phrase_ids = set(np.array(phrase_ids).squeeze())  # phrase_id in this dataset
        self.num_classes = num_classes
        phrase_dict = {}  # {id->phrase} 


        if SSTDataset_torch.label_tmp is None:
            # Read label/sentiment first
            # Share 1 array on train/dev/test set. No need to do this 3 times.
            SSTDataset_torch.label_tmp = pd.read_csv(path_to_dataset + 'sentiment_labels.txt',
                                    sep='|', dtype={'phrase ids': int, 'sentiment values': float})
            SSTDataset_torch.label_tmp = np.array(SSTDataset_torch.label_tmp)[:, 1:]  # sentiment value
        
        with open(path_to_dataset + 'dictionary.txt', 'r', encoding='utf-8') as f:
            i = 0
            for line in f:
                phrase, phrase_id = line.strip().split('|')
                if int(phrase_id) in phrase_ids:  # phrase in this dataset
                    phrase = clean_data(phrase)  # preprocessing
                    phrase_dict[int(phrase_id)] = phrase
                    i += 1
        f.close()
  

        # 记录每个句子中单词在glove中的index
        self.phrase_vec = []  # word index in glove

        # 每个句子的label
        # label of each sentence
        self.labels = torch.zeros((len(phrase_dict),), dtype=torch.long)

        missing_count = 0
        # 查找每个句子中词的词向量
        for i, (idx, p) in enumerate(phrase_dict.items()):
            tmp1 = []  # 暂存句子中单词的id
            # 分词
            for w in p.split(' '):
                try:
                    tmp1.append(wordvec.index.get_loc(w))  # 单词w在glove中的index
                except KeyError:
                    missing_count += 1

            self.phrase_vec.append(torch.tensor(tmp1, dtype=torch.long))  # 包含句子中每个词的glove index
            self.labels[i] = get_class(SSTDataset_torch.label_tmp[idx], self.num_classes) # pos i 的句子的label

        print(missing_count)

    def __getitem__(self, index):
        return self.phrase_vec[index], self.labels[index]

    def __len__(self):
        return len(self.phrase_vec)




# #%%
# if __name__ == "__main__":
#     # test

#     wordvec = loadGloveModel('../midterm/data/glove/glove.6B.'+ str(50) +'d.txt')

#     # test = SSTDataset('data/dataset/', 'test', 2)
# #%%
#     test_vec = SSTDataset('data/dataset/', 2, 50, wordvec, 'vector')
#     test_tfidf = SSTDataset('data/dataset/', 2, 50, wordvec, 'tfidf')
#     # print(SSTDataset.label_tmp)


# #%%
