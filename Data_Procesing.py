import torch
from torch.utils.data import Dataset
from Config import Config
import re
import gensim
import numpy as np


class My_Dataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        # 考虑到测试机没有label，但是还要用这个数据结构
        if label is not None:
            self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = torch.from_numpy(self.data[index])
        if self.label is not None:
            label = torch.from_numpy(self.label[index])
            return data, label
        else:
            return data


def get_stopwords_list():
    # 创建停用词表
    stopwords_list = [line.strip() for line in open(Config.stopwords_path, encoding='utf-8').readlines()]
    return stopwords_list


def build_word2id(save_path):
    """
    :param save_path: word2id的保存地址
    :return: word2id矩阵

    strip(): 只能删除开头或是结尾的字符或是字符串。
    split(): 分割字符串
    """
    stopwords = get_stopwords_list()
    word2id = {'_PAD_': 0}
    paths = [Config.train_path, Config.val_path]

    for path in paths:
        # 打开文件
        with open(path, encoding='utf-8') as f:
            # 读取每一行
            for line in f.readlines():
                # 去除开头结尾的空字符串，并且根据空格分割成一个个词
                words = line.strip().split()
                available_words = []
                # 遍历所有词
                for word in words:
                    # 如果不是停用词,英文单词和制表符，就是可用词
                    if word not in stopwords and len(re.findall('[a-zA-Z]+', word)) == 0 and word != '\t':
                        available_words.append(word)
                # 遍历所有可用词，加入word2id字典
                for word in available_words:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)
    # 把word2id存到文件中
    with open(save_path, 'w', encoding='utf-8') as f:
        for word in word2id:
            f.write(word + '\t' + str(word2id[word]) + '\n')
    return word2id


def build_word2vec(pretrained_word2vec_path, word2id, save_word2vec_path=None):
    """
    构建word2vec
    :param save_word2vec_path: 保存word2vec
    :param pretrained_word2vec_path: 预训练的word2vec文件
    :param word2id: word2id
    :return: word2vec size:[单词数量，每个词的维度]
    """
    # 所有单词数量
    n_word = max(word2id.values()) + 1
    # 加载预训练的word2vec
    pretrained_word2vec = gensim.models.KeyedVectors.load_word2vec_format(fname=pretrained_word2vec_path, binary=True)
    # 初始化word2vec，范围[-1, 1], size[n_word, pretrained_word2vec.vector_size],即[单词数量，每个词的维度]
    word2vec = np.array(np.random.uniform(low=-1., high=1., size=[n_word, pretrained_word2vec.vector_size]))
    # 遍历所有单词
    for word in word2id.keys():
        # 如果预训练文件中有对应的单词，就将对应单词的向量赋值给word2vec
        try:
            word2vec[word2id[word]] = pretrained_word2vec[word]
        # 如果没有，就用初始化的向量
        except KeyError:
            pass
    # 如果目录不为空，保存word2vec
    if save_word2vec_path:
        with open(save_word2vec_path, 'w', encoding='utf-8') as f:
            for vec in word2vec:
                # 把List[int] -> List[str], 因为join函数的参数类型是Iterable[str], List[str]符合要求
                vec = [str(v) for v in vec]
                f.write(' '.join(vec) + '\n')
    return word2vec


def text2array(path, word2id, seq_len, no_label):
    """
    :param no_label: 有无label
    :param path: 文件地址
    :param word2id: word2id
    :param seq_len: 固定句子长度
    :return: array size:[句子个数， 句子固定长度], label size:[句子个数, 1]
    """
    i = 0
    label_array = []
    # 获取句子个数
    n_sentence = len(open(path, encoding='utf-8').readlines())
    # 初始化句子矩阵,[句子个数， 句子固定长度]
    sentence_array = np.zeros(shape=(n_sentence, seq_len))
    with open(path, encoding='utf-8') as f:
        for line in f.readlines():
            l_s = line.strip().split()
            sentence = l_s[1:]
            # 将句子中的单词变为对应的id，若单词找不到则为0
            new_sentence = [word2id.get(word, 0) for word in sentence]
            new_sentence_np = np.array(new_sentence).reshape(1, -1)
            # 如果句子长度小于固定长度，将句子右对齐，左边用0补上; 否则截断
            if np.size(new_sentence_np, 1) < seq_len:
                sentence_array[i, seq_len - np.size(new_sentence_np, 1):] = new_sentence_np[0, :]
            else:
                sentence_array[i, 0:seq_len] = new_sentence_np[0, 0:seq_len]
            i = i + 1
            if not no_label:
                label_array.append(int(l_s[0]))
    if no_label:
        return np.array(sentence_array)
    return np.array(sentence_array), np.array([label_array]).T


# def prepare_data(word2id, seq_len, train_path, test_path, val_path):
#     """
#     :param word2id:
#     :param seq_len:
#     :param train_path:
#     :param test_path:
#     :param val_path:
#     :return: array size:[句子个数， 句子固定长度], label size:[句子个数, 1]
#     """
#     # text -> array & label
#     train_array, train_label = text2array(train_path, word2id, seq_len, False)
#     val_array, val_label = text2array(val_path, word2id, seq_len, False)
#     test_array, test_label = text2array(test_path, word2id, seq_len, False)
#     return train_array, train_label, val_array, val_label, test_array, test_label


if __name__ == '__main__':
    print('Data_Processing main')

