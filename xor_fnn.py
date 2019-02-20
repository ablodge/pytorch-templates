import argparse
import random

import torch
import torch. nn as nn
import time
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
from utils import ProgressBar, describe

# default args
args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-d','--device', default='cpu', help='Either "cpu" or "cuda"')
args.add_argument('-e','--epochs', default=100, type=int, help='Number of epochs')
args.add_argument('-lr','--learning-rate', default=0.1, type=float, help='Learning rate')
args.add_argument('-b','--batch-size', default=10, type=int, help='Batch size')
args.add_argument('-do','--dropout', default=0.3, type=int, help='Dropout rate')
args.add_argument('--early-stopping', default=-1, type=int, help='Early stopping criteria')
# args.add_argument('--embedding-size', default=10, type=int, help='Embedding dimension size')
# args.add_argument('--hidden-size', default=10, type=int, help='Hidden layer size')
# args.add_argument('-s','--seed', default=int(time.time()/1000), type=int, help='Random seed')
# args.add_argument('-i','--input-file', default=10, type=int, help='Input file')
# args.add_argument('-o','--output-file', default=10, type=int, help='Output file')
# args.add_argument('-m','--model-output-file', default=10, type=int, help='Model output file')
args = args.parse_args()

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


class FNN(nn.Module):

    def __init__(self, sizes):
        super().__init__()

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
            output = F.dropout(F.relu(layer(output)),args.dropout)
        # if self.binary:
        #     output = torch.sigmoid(output)
        # else:
        #     output = F.softmax(output, len(output))
        return output


fnn = FNN(sizes=[2, 4, 4, 4, 1])
fnn = fnn.to(args.device)

loss_f = nn.MSELoss()
opt = torch.optim.SGD(fnn.parameters(), lr=args.learning_rate)


# data ===============================================================================================================

class XORData(Dataset):

    def __init__(self):
        super().__init__()
        self.split = 'train'
        self.split_dict = {'train' : list(range(0, 1000)),
                           'dev': list(range(1000, 2000)),
                           'test': list(range(2000, 3000))}

    def __getitem__(self, index):
        random.seed(self.split_dict[self.split][index])
        x1 = round(random.uniform(0, 1))
        x2 = round(random.uniform(0, 1))
        y = 1. if bool(x1<0.5) != bool(x2<0.5) else 0.
        return torch.Tensor([x1,x2]), torch.Tensor([y])

    def __len__(self):
        return len(self.split_dict[self.split])

    def set_split(self, str):
        self.split = str

    @classmethod
    def readable(cls, x, y):
        return f'{x[0]>=0.5} xor {x[1]>=0.5} => {y[0]>=0.5}'


def batch_generator(dataset, batch_size, shuffle=True, device='cpu'):
    loader = DataLoader(dataset, batch_size, shuffle, drop_last=True)
    for x, y in loader:
        batch = {'x': x.to(device), 'y': y.to(device)}
        yield batch


data = XORData()

# train ===============================================================================================================
data.set_split('train')
fnn.train()
progress = ProgressBar('Training', args.epochs)
early_stopping = 0

for epoch in range(args.epochs):
    running_acc = 0
    running_loss = 0
    i = 0
    # Part 1: Training
    for batch in batch_generator(data, args.batch_size):
        x, y = batch['x'], batch['y']
        # step 1 zero the gradient
        opt.zero_grad()
        # step 2 compute the output
        y_pred = fnn(x)
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
        if i%100 == 0:
            progress.report(epoch+1, f'Loss {running_loss/i} Acc {running_acc/i}')
    # Part 2: Update loss and accuracy
    model_state['epochs'] += 1
    model_state['train_loss'].append(running_loss/i)
    model_state['train_acc'].append(running_acc/i)
    # Part 3: Development set
    data.set_split('dev')
    i = 0
    running_loss = running_acc = 0
    for batch in batch_generator(data, args.batch_size):
        x, y = batch['x'], batch['y']
        y_pred = fnn(x)
        loss = loss_f(y_pred, y)
        running_loss += loss.item()
        running_acc += y.round().eq(y_pred.round()).sum().item()
        i += len(y)
    model_state['dev_loss'].append(running_loss/i)
    model_state['dev_acc'].append(running_acc/i)
    # Part 4: Early Stopping
    if args.early_stopping>0:
        dev_acc = model_state['dev_acc']
        if len(dev_acc)>1 and dev_acc[-1] >= dev_acc[-2]:
            early_stopping += 1
        else:
            early_stopping = 0
        if early_stopping >= args.early_stopping:
            break

progress.end_timer(f'Loss {running_loss/i} Acc {running_acc/i}')

# test =================================================================================================================
data.set_split('test')
fnn.eval()
total_size = len(data)

progress = ProgressBar('Evaluation', total_size)

i = 0
running_acc = 0
running_loss = 0
for batch in batch_generator(data, args.batch_size):
    x, y = batch['x'], batch['y']
    # step 1 compute the output
    y_pred = fnn(x)
    # step 2 compute the loss
    loss = loss_f(y_pred, y)
    loss_batch = loss.item()
    # compute the accuracy
    running_loss += loss_batch
    running_acc += y.round().eq(y_pred.round()).sum().item()

    for j in range(len(y)):
        if y[j].round().item() != y_pred[j].round().item():
            model_state['errors'].append((x[j], y[j], y_pred[j]))

    i += len(y)
    if i%100 == 0:
        progress.report(i, f'Loss {running_loss/i} Acc {running_acc/i}')
# update model state
model_state['test_loss'] = running_loss/i
model_state['test_acc'] = running_acc/i
progress.end_timer(f'Loss {running_loss/i} Acc {running_acc/i}')

# error analysis =======================================================================================================
print(model_state)

import matplotlib.pyplot as plt

plt.plot(model_state['dev_loss'])
plt.show()
