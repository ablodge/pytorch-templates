
import torch
import torch. nn as nn
import torch.nn.functional as F


class FNN(nn.Module):

    def __init__(self, sizes, dropout=0.3):
        super().__init__()
        self.dropout = dropout
        self.binary = sizes[-1] == 1
        self.layers = nn.ModuleList()
        for i in range(len(sizes) - 1):
            self.layers.append(nn.Linear(sizes[i], sizes[i+1]))

        for layer in self.modules():
            for name, param in layer.named_parameters():
                if 'bias' not in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.constant_(param, 0.0)

    def forward(self, x_in):
        output = x_in
        for layer in self.layers:
            output = F.dropout(F.relu(layer(output)), self.dropout)
        if self.binary:
            output = torch.sigmoid(output)
        else:
            output = F.softmax(output, len(output))
        return output



class RNNClassifier(nn.Module):

    def __init__(self, vocab_size, emb_dim=4, hidden_size=4, num_layers=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_size, bidirectional=True, batch_first=True)
        self.out = FNN(hidden_size* 2, 1, hidden_sizes=[hidden_size for _ in range(num_layers)], binary=True)

        # torch.nn.init.uniform_(self.emb.weight, a=0, b=1)
        for net in [self.emb,self.rnn,self.out]:
            for name, param in net.named_parameters():
                # print('initializing',name)
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                else:
                    nn.init.xavier_uniform_(param)

    def forward(self, x):
        x = self.emb(x)
        s, hn = self.rnn(x)
        x = hn.permute(1, 0, 2)
        x = x.view(-1, x.size()[1]*x.size()[2])
        x = self.out(x)
        return x


class RNNSeq(nn.Module):

    def __init__(self, vocab_size, emb_dim=4, hidden_size=4, num_layers=1):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_size, num_layers, bidirectional=True, batch_first=True)
        self.out = nn.GRU(hidden_size * 2, 1, 1, batch_first=True)
        self.act = nn.Sigmoid()

        # torch.nn.init.uniform_(self.emb.weight, a=0, b=1)
        for net in [self.emb,self.rnn,self.out]:
            for name, param in net.named_parameters():
                # print('initializing',name)
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                else:
                    nn.init.xavier_uniform_(param)

    def forward(self, x):
        x = self.emb(x)
        x, hn = self.rnn(x)
        x, hn = self.out(x)
        x = self.act(x)
        x = x.view(len(x), -1)
        return x
