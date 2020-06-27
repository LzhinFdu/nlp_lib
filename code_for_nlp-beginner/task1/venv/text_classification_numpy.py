import numpy as np
import pandas as pd
import os

from collections import Counter
from scipy.sparse import csr_matrix
#max_tf：解决高频词问题，后期需要调试

train_times = 10000

class Word2vec():
    def __init__(self, max_tf=0.8):
        self.m_wordlist = Counter()
        self.m_ngramlist = Counter()
        self.max_tf = max_tf
        self.m_id = {}
        self.m_dict = {}
        self.ngram = []

    def text_process(self, data):
        return data.lower().split(' ')

    def bag_of_words(self, datas):
        for data in datas:
            data = self.text_process(data)
            self.m_wordlist.update(data)
        self.m_dict = dict(filter(lambda x: x[1] < self.max_tf * len(datas), self.m_wordlist.items()))
        self.m_id = dict([(k, i) for i, k in enumerate(self.m_dict.keys())])
        num = []
        indences = []
        indptr = [0]
        for data in datas:
            data = self.text_process(data)
            temp_list = Counter(data)
            for k,v in temp_list.items():
                if k in self.m_id:
                    num.append(v)
                    indences.append(self.m_id[k])
            indptr.append(len(indences))
        return csr_matrix((num, indences, indptr), dtype=int, shape=(len(datas), len(self.m_id)))

    def get_n_gram_str(self, data, num=2):
        self.ngram.clear()
        data = self.text_process(data)
        data_lenth = len(data)
        for j, word in enumerate(data):
            #print(np.shape(j))
            #print(type(j))
            if(j + num >= data_lenth):
                self.ngram.append(' '.join(data[j:]))
            else:
                self.ngram.append(' '.join(data[j:j+num]))

    def N_gram(self, datas, num=2):
        for data in datas:
            self.get_n_gram_str(data, num)
            self.m_ngramlist.update(self.ngram)
        self.m_dict = dict(filter(lambda x: x[1] < self.max_tf * len(datas), self.m_ngramlist.items()))
        self.m_id = dict([(k, i) for i, k in enumerate(self.m_dict.keys())])
        #m_id为100000的数量级
        num = []
        indences = []
        indptr = [0]
        for data in datas:
            self.get_n_gram_str(data)
            temp_list = Counter(self.ngram)
            for k, v in temp_list.items():
                if k in self.m_id:
                    num.append(v)
                    indences.append(self.m_id[k])
            indptr.append(len(indences))
        return csr_matrix((num, indences, indptr), dtype=int, shape=(len(datas), len(self.m_id)))

class SoftmaxRegression():
    def __init__(self, input, target, class_num, learning_rate = 0.000005, batch_size = 1024):
        self.input = input
        self.target = target
        self.class_num = class_num
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.W = np.random.uniform(0.0, 1.0, size=(self.input.shape[1], self.class_num))

    def dataset_split(self):
        len_ = self.input.shape[0]
        train_len = int(0.9 * len_)
        test_len = len_ - train_len
        return self.input[:train_len, :], self.input[train_len:, :], self.target[:train_len], self.target[train_len:]

    def softmax(self, x):
        return np.exp(x.dot(self.W) - 500 * np.ones((x.shape[0], self.class_num))) \
               /np.exp(x.dot(self.W) - 500 * np.ones((x.shape[0], self.class_num))).sum()
    #这里实际训练时出现exp溢出问题
    def predict(self, x):
        return np.argmax(self.softmax(x), -1)

    def train_with_batch(self,time = 0):
        train_X, test_X, train_Y, test_Y = self.dataset_split()
        probs = self.softmax(train_X)
        gradient = train_X.transpose().dot(np.eye(self.class_num)[train_Y] - probs) - self.W
        self.W += self.learning_rate * gradient
        print('time %d: test accuracy is %.3f' % (time, ((self.predict(test_X) == test_Y).sum()) / len(test_Y)))

    def train_with_SGD(self, time = 0):
        train_X, test_X, train_Y, test_Y = self.dataset_split()
        train_len = train_X.shape[0]
        index = np.arange(train_len)
        np.random.shuffle(index)
        for start_ in np.arange(train_len):
            batch_X = train_X[index[start_]]
            batch_Y = train_Y[index[start_]]
            probs = self.softmax(batch_X)
            gradient = batch_X.transpose().dot(np.eye(self.class_num)[batch_Y] - probs) - self.W
            self.W += self.learning_rate * gradient
        print('time %d: test accuracy is %.3f'% (time, ((self.predict(test_X)==test_Y).sum())/len(test_Y)))

    def train_with_minibatch(self, time = 0):
        train_X, test_X, train_Y, test_Y = self.dataset_split()
        train_len = train_X.shape[0]
        index = np.arange(train_len)
        np.random.shuffle(index)
        for start_ in np.arange(0, train_len, self.batch_size):
            batch_X = train_X[index[start_:start_+self.batch_size]]
            batch_Y = train_Y[index[start_:start_+self.batch_size]]
            probs = self.softmax(batch_X)
            gradient = batch_X.transpose().dot(np.eye(self.class_num)[batch_Y] - probs) - self.W
            self.W += self.learning_rate * gradient
        print('time %d: test accuracy is %.3f'% (time, ((self.predict(test_X)==test_Y).sum())/len(test_Y)))

if __name__ == '__main__':
    data_path = 'data'
    train_ = pd.read_csv(os.path.join(data_path, 'train.tsv'), sep='\t')
    datas = train_['Phrase']
    vec_ = Word2vec()
    X = vec_.bag_of_words(datas)
    Y = train_['Sentiment']
    assert X.shape[0] == len(Y)
    m_regression = SoftmaxRegression(X, Y, 5)
    for time in np.arange(train_times):
        m_regression.train_with_batch(time)


