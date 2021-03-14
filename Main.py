import torch
import numpy as np
from Config import Config
from Data_Procesing import build_word2id, build_word2vec, text2array, My_Dataset
from torch.utils.data import DataLoader
from Model import LSTM_Model, LSTM_ATTENTION_Model
import tqdm


def prepare_data():
    # 句子固定长度，65
    seq_len = Config.max_sen_len

    # word2id
    word2id = build_word2id(save_path=Config.word2id_path)

    # word2vec
    word2vec = build_word2vec(Config.pre_word2vec_path, word2id)
    word2vec = torch.from_numpy(word2vec)
    word2vec = word2vec.float()

    # text -> array & label
    # array size:[], label size:[]
    # array size:[句子个数， 句子固定长度], label size:[句子个数, 1]
    train_array, train_label = text2array(Config.train_path, word2id, seq_len, False)
    val_array, val_label = text2array(Config.val_path, word2id, seq_len, False)
    test_array, test_label = text2array(Config.test_path, word2id, seq_len, False)

    # 配置autoloader
    train_dataset = My_Dataset(train_array, train_label)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_dataset = My_Dataset(val_array, val_label)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0
    )
    test_dataset = My_Dataset(test_array, test_label)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=0
    )
    return word2vec, train_loader


def train(train_loader, model, device, epochs, lr):
    """
    :param train_loader:
    :param model:
    :param device:
    :param epochs:
    :param lr:
    :return:

    torch.max(input, dim)
        dim是max函数索引的维度0/1，0是每列的最大值，1是每行的最大值
        函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引。

    nn.CrossEntropyLoss()函数计算交叉熵损失，用法：
        不需要现将输出经过softmax层，否则计算的损失会有误，即直接将网络输出用来计算损失即可
        output是网络的输出，size=[batch_size, class]
        target是数据的真实标签，是标量，size=[batch_size]
        loss=nn.CrossEntropyLoss()
        loss_output=loss(output,target)

    """
    # 在使用pytorch构建神经网络的时候，训练过程中会在程序上方添加一句model.train()，作用是启用batch normalization和drop out。
    model.train()

    # 将模型放入GPU
    model = model.to(device)

    # 优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # 损失函数
    loss_fun = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        train_loss = 0.0
        total = 0
        correct = 0
        train_loader = tqdm.tqdm(train_loader)
        for i, data in (enumerate(train_loader)):
            # 清除梯度
            optimizer.zero_grad()
            # input size=[batch, seq_len]=[64, 65], target size=[batch, 1]=[64, 1]
            input, target = data[0], data[1]
            # 改变数据类型
            input = input.type(torch.LongTensor)
            target = target.type(torch.LongTensor)
            # 放入GPU
            input = input.to(device)
            target = target.to(device)
            # 把input放入模型得到output
            # output size=[batch, n_class]=[64, 2], target size=[batch]=[64]
            output = model(input)
            target = target.squeeze(1)
            # loss计算
            loss = loss_fun(output, target)
            # 反向传播，计算梯度
            loss.backward()
            # 根据梯度更新参数模型
            optimizer.step()
            # 累加loss值
            train_loss += loss.item()
            # 对output按列求最大值
            max_value, max_index = torch.max(output, 1)
            # 我们需要的预测值是最大值的索引，用来看这句话属于哪个类别（no.0 or no.1）
            predict = max_index
            # target总数量，即句子个数
            total += target.size(0)
            # 预测正确的数量
            correct += (predict == target).sum().item()

            # 进度条显示信息
            postfix = {
                'train_loss:{:.5f}, train_acc:{:.3f}%'.format(train_loss/(i+1), 100*correct/total)
            }
            train_loader.set_postfix(log=postfix)


if __name__ == '__main__':
    # 开启gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 准备数据
    word2vec, train_loader = prepare_data()
    # 创建模型
    # LSTM_Model(
    #   (embedding): Embedding(54848, 50)
    #   (lstm): LSTM(50, 100, num_layers=2, dropout=0.2, bidirectional=True)
    #   (decoder1): Linear(in_features=400, out_features=100, bias=True)
    #   (decoder2): Linear(in_features=100, out_features=2, bias=True)
    # )
    # model = LSTM_Model(
    #     input_size=Config.embedding_dim,
    #     hidden_size=Config.hidden_dim,
    #     num_layers=Config.num_layers,
    #     bidirectional=Config.bidirectional,
    #     dropout=Config.drop_keep_prob,
    #     n_class=Config.n_class,
    #     pretrained_weight=word2vec,
    #     update_w2v=Config.update_w2v
    # )

    # LSTM_ATTENTION_Model(
    #   (embedding): Embedding(54848, 50)
    #   (lstm): LSTM(50, 100, num_layers=2, dropout=0.2, bidirectional=True)
    #   (decoder1): Linear(in_features=200, out_features=100, bias=True)
    #   (decoder2): Linear(in_features=100, out_features=2, bias=True)
    # )
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
    print(model)
    # 训练
    train(train_loader, model, device, Config.n_epoch, Config.lr)
    torch.save(model.state_dict(), Config.model_state_dict_path)
