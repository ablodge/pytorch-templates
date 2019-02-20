import torch.nn as nn
import torch.optim as optim
from datasets import JapaneseMorphData, batch_iter
from model import NN
from train_and_eval import train, eval, cross_val
import torch
import time


class RNNSeqClassifier(NN):

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
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)
                # print(param)

        self.loss = nn.BCELoss()
        self.opt = optim.Adadelta(self.parameters())
        # print(self)

    def forward(self, x):
        x = self.emb(x)
        x, hn = self.rnn(x)
        x, hn = self.out(x)
        x = self.act(x)
        x = x.view(len(x), -1)
        return x

    def backward(self, batch_x, batch_y):
        loss_val = 0
        output = self(batch_x)
        # print(output.size())
        for y1, y2 in zip(output, batch_y):
            self.opt.zero_grad()
            loss_output = self.loss(y1, y2)
            loss_output.backward(retain_graph=True)
            loss_val += loss_output.item()
            self.opt.step()
        return loss_val/len(batch_x), output


def main():
    train_data = JapaneseMorphData('train', max_size=10000)
    test_data = JapaneseMorphData('test', train_data.vocab, max_size=10000)

    rnn = RNNSeqClassifier(len(train_data.vocab) + 1,
                           emb_dim=10,
                           hidden_size=10,
                           num_layers=4)

    # train(rnn, batch_iter(train_data, batch_size=1), size=1000)
    # eval(rnn, batch_iter(test_data, batch_size=1), size=1000)

    cross_val(rnn, batch_iter(train_data, batch_size=1), size=1000)


if __name__ == '__main__':
    main()
