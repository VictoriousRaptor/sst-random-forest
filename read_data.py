import pandas as pd
import numpy as np

def clean_data(sentence):
    # From yoonkim: https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)
    sentence = re.sub(r"\s{2,}", " ", sentence)
    return sentence.strip().lower()


def get_class(sentiment, num_classes):
    # 根据sentiment value 返回一个label
    return int(sentiment * (num_classes - 0.001))


def read_sst(path_to_dataset, name, num_classes):
        """SST dataset
        
        Args:
            path_to_dataset (str): 路径
            name (str): train, dev or test
            num_classes (int): 2 or 5
        """
        phrase_ids = pd.read_csv(path_to_dataset + 'phrase_ids.' +
                                 name + '.txt', header=None, encoding='utf-8', dtype=int)
        phrase_ids = set(np.array(phrase_ids).squeeze())  # 在数据集中出现的pharse id
        phrase_dict = {}  # {phrase: id} 

            # 先读label (sentiment)
            # 训练/测试/验证集共享一个，没必要读3次
        label_tmp = pd.read_csv(path_to_dataset + 'sentiment_labels.txt',
                                sep='|', dtype={'phrase ids': int, 'sentiment values': float})
        label_tmp = np.array(label_tmp)[:, 1:]  # sentiment value
        
        with open(path_to_dataset + 'dictionary.txt', 'r', encoding='utf-8') as f:
            i = 0
            for line in f:
                phrase, phrase_id = line.strip().split('|')
                if int(phrase_id) in phrase_ids:  # 在数据集中出现
                    phrase = clean_data(phrase)  # 预处理
                    phrase_dict[int(phrase_id)] = phrase
                    i += 1
        f.close()
  

        # 记录每个句子中单词在glove中的index
        self.phrase_vec = []

        # 每个句子的label
        self.labels = torch.zeros((len(phrase_dict),), dtype=torch.long)

        i = 0
        missing_count = 0
        # 查找每个句子中词的词向量
        for idx, p in phrase_dict.items():
            tmp1 = []  # 暂存句子中单词的id
            # 分词
            for w in p.split(' '):
                try:
                    tmp1.append(wordvec.index.get_loc(w))  # 单词w在glove中的index
                except KeyError:
                    missing_count += 1

            self.phrase_vec.append(torch.tensor(tmp1, dtype=torch.long))  # 包含句子中每个词的glove index
            self.labels[i] = get_class(label_tmp[idx], self.num_classes) # pos i 的句子的label
            i += 1

        print(missing_count)