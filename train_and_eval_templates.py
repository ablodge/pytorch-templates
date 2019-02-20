import torch
import time
import math
from sklearn.model_selection import ParameterGrid
from utils import ProgressBar


def train(net, data, model_state=None, epochs=10, early_stopping_criterion=-1):
    progress = ProgressBar('Training', epochs)
    net.train()
    early_stopping = 0

    for epoch in range(epochs):
        data.set_split('train')
        running_acc = running_loss = 0
        i = 0
        # Part 1: Training
        for batch in batch_generator(data, args.batch_size):
            x, y = batch['x'], batch['y']
            # step 1 zero the gradient
            opt.zero_grad()
            # step 2 compute the output
            y_pred = net(x)
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
                progress.report(epoch + 1, f'Loss {running_loss/i} Acc {running_acc/i}')
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
            y_pred = net(x)
            loss = loss_f(y_pred, y)
            running_loss += loss.item()
            running_acc += y.round().eq(y_pred.round()).sum().item()
            i += len(y)
        model_state['dev_loss'].append(running_loss / i)
        model_state['dev_acc'].append(running_acc / i)
        # Part 4: Early Stopping
        if early_stopping_criterion > 0:
            dev_acc = model_state['dev_acc']
            if len(dev_acc) > 1 and dev_acc[-1] >= dev_acc[-2]:
                early_stopping += 1
            else:
                early_stopping = 0
            if early_stopping >= early_stopping_criterion:
                break

    progress.end_timer(f'Loss {running_loss/i} Acc {running_acc/i}')


def eval(net, data, model_state=None):
    data.set_split('test')
    progress = ProgressBar('Evaluation', len(data))
    net.eval()

    i = 0
    running_acc = running_loss = 0
    for batch in batch_generator(data, args.batch_size):
        x, y = batch['x'], batch['y']
        # step 1 compute the output
        y_pred = net(x)
        # step 2 compute the loss
        loss = loss_f(y_pred, y)
        loss_batch = loss.item()
        # compute the accuracy
        running_loss += loss_batch
        running_acc += y.round().eq(y_pred.round()).sum().item()

        for j in range(len(y)):
            if y[j].round().item() != y_pred[j].round().item():
                model_state['errors'].append((x[j], y[j], y_pred[j]))
        if i % 100 == 0:
            progress.report(i, f'Loss {running_loss/i} Acc {running_acc/i}')
        i += len(y)
    # update model state
    model_state['test_loss'] = running_loss / i
    model_state['test_acc'] = running_acc / i
    progress.end_timer(f'Loss {running_loss/i} Acc {running_acc/i}')
    print(model_state)


# def cross_val(net, data, model_state=None, epochs=10, early_stopping_criterion=-1):
#     ...
#     for i in range(folds):
#         train(net, data, model_state, epochs, early_stopping_criterion)
#         eval(net, data, model_state)




