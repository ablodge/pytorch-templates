import torch

def train(data, model, loss, opt, epochs, batch_size):
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_accuracy = 0.0
        for i,batch in enumerate(data):
            x, y = batch

            loss = loss(model)
            loss = loss(x, y)
            loss.backward()
            opt.zero_grad()
            opt.step()

            # print stats
            running_loss += loss.item()
            for y_pred, y_true in zip(model(x), y):
                running_accuracy += accuracy(y_pred,y_true)
            if i % 1000 == 0 and i > 0:
                print(f'epoch: {epoch+1} item: {i} loss: {running_loss/2000} acc: {running_accuracy/2000}')
                running_loss = running_accuracy = 0.0

def cross_eval(data, model, metric, folds=10):
    ...