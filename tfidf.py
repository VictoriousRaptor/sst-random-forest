import numpy as np
from sklearn.preprocessing import normalize

def count_freq(sentences):
    """Calculate word freq and word count for word in these sentences
    
    Args:
        sentences (list(str)): 
    """
    wordfreq = dict()  # word->freq
    wordcount = dict()  # word->count
    for s in sentences:
        tmp = s.split(' ')
        word_in_sentence = set(tmp)
        for w in word_in_sentence:
            if wordcount.get(w) is None:
                wordcount[w] = 1
            else:
                wordcount[w] += 1
        for w in tmp:
            if wordfreq.get(w) is None:
                wordfreq[w] = 1
            else:
                wordfreq[w] += 1
    return wordfreq, wordcount

def count_freq(categories):
    wordfreq, wordcount = [], []
    for category in categories:
        t1, t2 = count_freq(category)
        wordfreq.append(t1)
        wordcount.append(t2)
    return wordfreq, wordcount

def tfidf(sentence, label_num, cat_len, wordfreq, wordcount):
    
    

    score = np.zeros(label_num)
    for i, (wf, wc) in enumerate(zip(wordfreq, wordcount)):
        tmp = []
        for w in sentence.split(' '):
            # TODO: if w not in wf (not in this category)?
            tf = wf[w] / len(wf)
            idf = np.log10(len(cat_len[i]) / (1 + wc[w]))
            tmp.append(tf * idf)
        score[i] = np.average(normalize(tmp, axis=-1))  # sum?

    return score

    
        