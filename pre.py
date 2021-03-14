import torch
from Model import LSTM_Model, LSTM_ATTENTION_Model
from Config import Config
import numpy as np
from Data_Procesing import build_word2vec
import jieba


def predict_sentence(model, sentence):
    """
    预测单个句子
    :param model:
    :param sentence:
    :return:
    """
    seq_len = Config.max_sen_len
    sentence_array = np.zeros(shape=(1, seq_len))
    l_s = sentence.strip().split()
    sentence = l_s[1:]
    # 将句子中的单词变为对应的id，若单词找不到则为0
    new_sentence = [word2id.get(word, 0) for word in sentence]
    new_sentence_np = np.array(new_sentence).reshape(1, -1)
    # 如果句子长度小于固定长度，将句子右对齐，左边用0补上; 否则截断
    if np.size(new_sentence_np, 1) < seq_len:
        sentence_array[0, seq_len - np.size(new_sentence_np, 1):] = new_sentence_np[0:]
    else:
        sentence_array[0, 0:seq_len] = new_sentence_np[0:seq_len]
    sentence_array = torch.from_numpy(sentence_array)
    sentence_array = sentence_array.type(torch.LongTensor)
    output = model(sentence_array)
    max_value, max_index = torch.max(output, 1)
    cla = max_index.numpy()[0]
    if cla == 0:
        print('good')
    else:
        print('bad')


if __name__ == '__main__':
    # 构建word2id
    split = []
    with open(Config.word2id_path, encoding='utf-8') as f:
        for line in f.readlines():
            sp = line.strip().split()  # 去掉\n \t 等
            split.append(sp)
        word2id = dict(split)  # 转成字典
    for key in word2id:  # 将字典的值，从str转成int
        word2id[key] = int(word2id[key])
    # 构建word2vec
    word2vec = build_word2vec(Config.pre_word2vec_path, word2id, None)
    word2vec = torch.from_numpy(word2vec)
    word2vec = word2vec.float()
    # 构建模型
    model = LSTM_ATTENTION_Model(
        input_size=Config.embedding_dim,
        hidden_size=Config.hidden_dim,
        num_layers=Config.num_layers,
        bidirectional=Config.bidirectional,
        dropout=Config.drop_keep_prob,
        n_class=Config.n_class,
        pretrained_weight=word2vec,
        update_w2v=Config.update_w2v
    )
    # 加载模型
    model.load_state_dict(torch.load(Config.model_state_dict_path))
    # 预测的句子
    sentence = '这个电影一般般'
    # jiaba分词
    sentence = ' '.join(jieba.lcut(sentence))
    print(sentence)
    predict_sentence(model, sentence)
