import torch


class LSTM_Model(torch.nn.Module):
    '''
    input_size:输入数据大小，即单个词的维度，即词向量的维度（embed_dim）
    hidden_size:隐藏层数量
    num_layer:lstm层数
    bidirectional:是否为双层lstm
    dropout:dropout比例
    n_class:分类数量
    pretrained_weight:预训练权重
    update_w2v:是否在训练中更新w2v参数
    '''

    def __init__(self, input_size, hidden_size, num_layers, bidirectional, dropout, n_class, pretrained_weight,
                 update_w2v):
        super(LSTM_Model, self).__init__()
        # 加载预训练词向量（权重）
        self.embedding = torch.nn.Embedding.from_pretrained(pretrained_weight)
        self.embedding.weight.requires_grad = update_w2v
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                  bidirectional=bidirectional, dropout=dropout)
        if bidirectional:
            self.decoder1 = torch.nn.Linear(in_features=hidden_size * 4, out_features=hidden_size)
        else:
            self.decoder1 = torch.nn.Linear(in_features=hidden_size * 2, out_features=hidden_size)
        self.decoder2 = torch.nn.Linear(in_features=hidden_size, out_features=n_class)

    def forward(self, inputs):
        # [batch, seq_len] -> [batch,  seq_len, embed_dim], seq_len表示句子长度，每个句子的长度是固定的
        embeddings = self.embedding(inputs)
        # [batch, seq_len, embed_dim] -> [seq_len, batch, embed_dim]
        embeddings = embeddings.permute([1, 0, 2])
        # states:[seq_len, batch, hidden_dim*bidirectional], 此处若是单层lstm，bidirectional=1，否则为2
        states, hidden = self.lstm(embeddings)
        # encoding:[batch, 2*hidden_dim*bidirectional],因为将states的第一个和最后一个元素拼接，所以*2
        encoding = torch.cat([states[0], states[-1]], dim=1)
        # [batch, hidden_dim]
        outputs = self.decoder1(encoding)
        # [batch, n_class]
        outputs = self.decoder2(outputs)
        return outputs


class LSTM_ATTENTION_Model(torch.nn.Module):
    '''
        input_size:输入数据大小，即单个词的维度，即词向量的维度（embed_dim）
        hidden_size:隐藏层数量
        num_layer:lstm层数
        bidirectional:是否为双层lstm
        dropout:dropout比例
        n_class:分类数量
        pretrained_weight:预训练权重
        update_w2v:是否在训练中更新w2v参数

        矩阵乘法：
            1) a * b，即两个矩阵对应元素相乘，要求两个矩阵维度完全一致
            2) 二维矩阵时，torch.mm(a,b)和 torch.matmul(a,b)是一样的
    '''

    def __init__(self, input_size, hidden_size, num_layers, bidirectional, dropout, n_class, pretrained_weight,
                 update_w2v):
        super(LSTM_ATTENTION_Model, self).__init__()
        # 加载预训练词向量（权重）
        self.embedding = torch.nn.Embedding.from_pretrained(pretrained_weight)
        self.embedding.weight.requires_grad = update_w2v
        self.lstm = torch.nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                                  bidirectional=bidirectional, dropout=dropout)
        if bidirectional:
            self.attention_weight_w = torch.nn.Parameter(torch.Tensor(2*hidden_size, 2*hidden_size))
            self.attention_weight_proj = torch.nn.Parameter(torch.Tensor(2*hidden_size, 1))
            self.decoder1 = torch.nn.Linear(in_features=2*hidden_size, out_features=hidden_size)
        else:
            self.attention_weight_w = torch.nn.Parameter(torch.Tensor(hidden_size, hidden_size))
            self.attention_weight_proj = torch.nn.Parameter(torch.Tensor(hidden_size, 1))
            self.decoder1 = torch.nn.Linear(in_features=hidden_size, out_features=hidden_size)
        self.decoder2 = torch.nn.Linear(in_features=hidden_size, out_features=n_class)

        torch.nn.init.uniform_(self.attention_weight_w, -0.1, 0.1)
        torch.nn.init.uniform_(self.attention_weight_proj, -0.1, 0.1)

    def forward(self, inputs):
        # [batch, seq_len] -> [batch, seq_len, embed_dim]
        embeddings = self.embedding(inputs)
        # [batch, seq_len, embed_dim] -> [batch, seq_len, embed_dim]， 这个permute就没有意义了
        # embeddings = embeddings.permute([0, 1, 2])
        # states : [batch, seq_len, hidden_dim*bidirectional]
        states, hidden = self.lstm(embeddings)

        # attention
        # u:[batch, seq_len, hidden_dim*bidirectional]
        u = torch.tanh(torch.matmul(states, self.attention_weight_w))
        # att:[batch, seq_len, 1]
        att = torch.matmul(u, self.attention_weight_proj)
        # att_score:[batch, seq_len, 1]
        att_score = torch.softmax(att, dim=1)
        # scored_x:[batch, seq_len, hidden_dim*bidirectional]
        scored_x = states * att_score
        # encoding:[batch, hidden_dim*bidirectional], 把句子里每个词的分数相加得到整个句子的分数？
        encoding = torch.sum(scored_x, dim=1)

        # out:[batch, hidden_dim]
        out = self.decoder1(encoding)
        # out:[batch, n_class]
        out = self.decoder2(out)
        return out
