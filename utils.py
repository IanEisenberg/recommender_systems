import numpy as np
import torch

def make_train_step(model, loss_fun, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(users, items, ratings):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(users, items)
        # Computes loss
        loss = loss_fun(ratings, yhat)
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()
    return train_step

def run_one_epoch(model, data, loss_fun, optimizer):
    stats = {}
    tracked_loss = 0
    train_step = make_train_step(model, loss_fun, optimizer)
    for users, items, ratings in data.loaders['train']:
        # calculate loss
        tracked_loss += train_step(users, items, ratings)

    train_perf = tracked_loss / len(data.loaders['train'])
    # get validation performance
    val_pred, val_true = get_pred_true(model, data, 'val')
    val_perf = loss_fun(val_pred, val_true).item()
    # record
    stats['val_loss'] = val_perf
    stats['train_loss'] = train_perf
    return stats

def train(model, data, loss_fun, optimizer, 
          n_epochs=50, print_epochs=10, history=None):
    if history is None:
        history = []
        last_epoch = 0
    else:
        last_epoch = history[-1]['epoch']
    # for early stopping
    patience = 10
    epochs_since_improvement = 0
    prev_val_loss = np.float('inf')
    for epoch in range(n_epochs):
        stats = {'epoch': last_epoch + 1}
        last_epoch += 1
        perf = run_one_epoch(model, data, loss_fun, optimizer)
        stats.update(perf)
        history.append(stats)
        # handle early stopping
        if stats['val_loss'] < prev_val_loss:
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1
        prev_val_loss = stats['val_loss']
        if epochs_since_improvement > patience:
            pass
        # report out
        if epoch % print_epochs == (print_epochs-1):
            print('[Epoch %d] loss: %.3f validation loss: %.3f' %
                  (last_epoch,
                   stats['train_loss'],
                   stats['val_loss']))
    return history
    
def predict(model, data_loader):
    preds = torch.FloatTensor()
    true = torch.FloatTensor()
    for users, items, ratings in data_loader:
        model.eval()
        preds = torch.cat([preds, model(users, items)])
        true = torch.cat([true, ratings])
    return preds.detach(), true

def get_pred_true(model, data, key):
    loader = data.loaders[key]
    pred, true = predict(model, loader)
    return pred, true

def evaluate(model, data, metric_funs):
    metrics = {}
    for key, val in data.loaders.items():
        pred, true = get_pred_true(model, data, key)
        for name, fun in metric_funs.items():
            metrics['%s_%s' % (key, name)] = fun(pred, true)
    return metrics
