import argparse
import random

import torch
import torch. nn as nn
import time
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from torch.autograd import Variable
from utils import ProgressBar, describe, dir_iter, SequenceVocabulary
from model_templates import FNN

from nltk.tokenize import MWETokenizer, TweetTokenizer

# default args
args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-d','--device', default='cpu', help='Either "cpu" or "cuda"')
args.add_argument('-e','--epochs', default=10, type=int, help='Number of epochs')
args.add_argument('-lr','--learning-rate', default=0.1, type=float, help='Learning rate')
args.add_argument('-b','--batch-size', default=10, type=int, help='Batch size')
args.add_argument('-do','--dropout', default=0.3, type=int, help='Dropout rate')
args.add_argument('-ea','--early-stopping', default=-1, type=int, help='Early stopping criteria')
args.add_argument('-em','--embedding-size', default=10, type=int, help='Embedding dimension size')
args.add_argument('--hidden-size', default=10, type=int, help='Hidden layer size')
args.add_argument('-s','--seed', default=int(time.time()/1000), type=int, help='Random seed')
args.add_argument('-i','--input-file', default=10, type=int, help='Input file')
args.add_argument('-o','--output-file', default=10, type=int, help='Output file')
args.add_argument('-m','--model-output-file', default=10, type=int, help='Model output file')
args = args.parse_args()


# data ===============================================================================================================

class ReviewsData(Dataset):
    TRAIN_POS_DIR = 'reviews-imdb/train/pos'
    TRAIN_NEG_DIR = 'reviews-imdb/train/neg'
    TEST_POS_DIR = 'reviews-imdb/test/pos'
    TEST_NEG_DIR = 'reviews-imdb/test/neg'

    vocab = SequenceVocabulary()
    tokenizer1 = TweetTokenizer()
    tokenizer2 = MWETokenizer()

    def __init__(self, train_size=-1, test_size=-1, dev_size=-1):
        super().__init__()
        self.split = 'train'
        self.split_dict = {'train': [], 'dev': [], 'test': []}

        for doc in dir_iter(self.TRAIN_POS_DIR):
            self.split_dict['train'].append((self.tokenize(doc),'pos'))
        for doc in dir_iter(self.TRAIN_NEG_DIR):
            self.split_dict['train'].append((self.tokenize(doc),'neg'))
        for doc in dir_iter(self.TEST_POS_DIR):
            self.split_dict['test'].append((self.tokenize(doc),'pos'))
        for doc in dir_iter(self.TEST_NEG_DIR):
            self.split_dict['test'].append((self.tokenize(doc),'neg'))

        random.shuffle(self.split_dict['train'])
        random.shuffle(self.split_dict['test'])

        if dev_size>0:
            tr = self.split_dict['train']
            self.split_dict['dev'] = tr[:dev_size]
            self.split_dict['train'] = tr[dev_size:]
        if train_size>0:
            self.split_dict['train'] = self.split_dict['train'][:train_size]
        if test_size>0:
            self.split_dict['test'] = self.split_dict['test'][:test_size]

        for doc, sentiment in self.split_dict['train']:
            for tok in doc:
                self.vocab.add(tok)

    def tokenize(self, doc):
        tokens = self.tokenizer1.tokenize(doc)
        tokens = self.tokenizer2.tokenize(tokens)
        return tokens

    def __getitem__(self, index):
        doc, sentiment = self.split_dict[self.split][index]
        doc = [self.vocab.index(t) for t in doc]
        doc = doc[:512]
        self.vocab.pad(doc, 512)
        sentiment = 1. if sentiment=='pos' else 0.
        return torch.Tensor(doc).long(), torch.Tensor([sentiment])

    def __len__(self):
        return len(self.split_dict[self.split])

    def set_split(self, str):
        self.split = str

    def readable(self, index):
        return self.split_dict[self.split][index]

    def vocab_length(self):
        return len(self.vocab)


def batch_generator(dataset, batch_size, shuffle=True, device='cpu'):
    loader = DataLoader(dataset, batch_size, shuffle, drop_last=True)
    for x, y in loader:
        batch = {'x': x.to(device), 'y': y.to(device)}
        yield batch

data = ReviewsData(train_size=1000, dev_size=100, test_size=100)


# model ===========================================================================================================

model_state = {'epochs':0,
               'test_loss': -1,
               'test_acc': -1,
               'train_loss':[],
               'train_acc':[],
               'dev_loss':[],
               'dev_acc':[],
               'errors':[]}

if not torch.cuda.is_available():
    args.cuda = False
args.device = torch.device("cuda" if args.cuda else "cpu")


class ReviewsRNN(nn.Module):
    def __init__(self, vocab_size, num_classes=1, emb_dim=16, hidden_size=8, num_layers_rnn=1, num_layers_fnn=2):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.GRU(emb_dim, hidden_size, num_layers=num_layers_rnn, bidirectional=True, batch_first=True)
        self.fnn = FNN([2*hidden_size*num_layers_rnn, num_classes])

        for layer in self.modules():
            for name, param in layer.named_parameters():
                if 'bias' not in name:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.constant_(param, 0.0)

    def forward(self, x):
        x = self.emb(x)
        output, hidden = self.rnn(x)
        x = hidden.permute(1, 0, 2)
        x = x.reshape(-1, x.size()[1] * x.size()[2])
        x = self.fnn(x)
        return x


rnn = ReviewsRNN(data.vocab_length())
rnn = rnn.to(args.device)

loss_f = nn.BCELoss()
opt = torch.optim.Adam(rnn.parameters(), lr=args.learning_rate)


# train ===============================================================================================================
rnn.train()

progress = ProgressBar('Training', args.epochs*len(data))
early_stopping = 0

for epoch in range(args.epochs):
    running_acc = 0
    running_loss = 0
    i = 0

    # Part 1: Training
    data.set_split('train')
    for batch in batch_generator(data, args.batch_size):
        x, y = batch['x'], batch['y']
        # step 1 zero the gradient
        opt.zero_grad()
        # step 2 compute the output
        y_pred = rnn(x)
        # step 3 compute the loss
        loss = loss_f(y_pred, y)
        loss_batch = loss.item()
        # step 4 use the loss to produce gradients
        loss.backward()
        # step 5 use the optimizer to take gradient step
        opt.step()
        # compute the accuracy
        running_loss += loss_batch
        running_acc += y.round().eq(y_pred.round()).sum().item()

        i += len(y)
        if i % 100 == 0:
            progress.report(i+len(data)*epoch, f'Loss {running_loss/i} Acc {running_acc/i}')

    # Part 2: Update loss and accuracy
    model_state['epochs'] += 1
    model_state['train_loss'].append(running_loss / i)
    model_state['train_acc'].append(running_acc / i)

    # Part 3: Development set
    data.set_split('dev')
    i = 0
    running_loss = running_acc = 0
    for batch in batch_generator(data, args.batch_size):
        x, y = batch['x'], batch['y']
        y_pred = rnn(x)
        loss = loss_f(y_pred, y)
        running_loss += loss.item()
        running_acc += y.round().eq(y_pred.round()).sum().item()
        i += len(y)
    model_state['dev_loss'].append(running_loss / i)
    model_state['dev_acc'].append(running_acc / i)

    # Part 4: Early Stopping
    if args.early_stopping > 0:
        dev_acc = model_state['dev_acc']
        if len(dev_acc) > 1 and dev_acc[-1] >= dev_acc[-2]:
            early_stopping += 1
        else:
            early_stopping = 0
        if early_stopping >= args.early_stopping:
            break

progress.end_timer(f'Loss {running_loss/i} Acc {running_acc/i}')

# test =================================================================================================================
data.set_split('test')
rnn.eval()
total_size = len(data)

progress = ProgressBar('Evaluation', total_size)

i = 0
running_acc = 0
running_loss = 0
for batch in batch_generator(data, args.batch_size):
    x, y = batch['x'], batch['y']
    # step 1 compute the output
    y_pred = rnn(x)
    # step 2 compute the loss
    loss = loss_f(y_pred, y)
    loss_batch = loss.item()
    # compute the accuracy
    running_loss += loss_batch
    running_acc += y.round().eq(y_pred.round()).sum().item()

    for j in range(len(y)):
        if y[j].round().item() != y_pred[j].round().item():
            model_state['errors'].append((data.vocab.read(list(x[j])), y[j], y_pred[j]))

    i += len(y)
    if i%100 == 0:
        progress.report(i, f'Loss {running_loss/i} Acc {running_acc/i}')
# update model state
model_state['test_loss'] = running_loss/i
model_state['test_acc'] = running_acc/i
progress.end_timer(f'Loss {running_loss/i} Acc {running_acc/i}')

# error analysis =======================================================================================================
for name in model_state:
    if name=='errors':
        print('Errors:')
        print('\n'.join(str(x) for x in model_state[name]))
    else:
        print(name, ':', model_state[name])

import matplotlib.pyplot as plt

plt.plot(model_state['train_loss'])
plt.show()
