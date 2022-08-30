import numpy as np
import torch
import copy
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import pickle
import os


plt.switch_backend('agg')


def adjust_learning_rate(optimizer, epoch, args, warmup=0):
    epoch = epoch - 1
    if args.lradj == 'type1':
        if epoch < warmup:
            lr_adjust = {epoch: epoch*(args.learning_rate - args.learning_rate / 100)/warmup + args.learning_rate / 100}
        else:
            lr_adjust = {epoch: args.learning_rate * (args.lr_decay_rate ** ((epoch - warmup) // 1))}   # 0.5
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
    elif args.lradj == 'type3':
        period = 5
        decay_rate1 = args.lr_decay_rate + (1 - args.lr_decay_rate) / 2
        decay_rate2 = args.lr_decay_rate
        lr_start = args.learning_rate * decay_rate1**((epoch + period) // period)/decay_rate1
        lr_end = args.learning_rate * decay_rate2 ** ((epoch + period * 2) // period) / decay_rate2
        lr_adjust = {epoch: (0.5 + 0.5*np.cos(np.pi/period*(epoch % period)))*(lr_start - lr_end) + lr_end}
    elif args.lradj == 'type4':
        epoch += 1
        lr_adjust = {epoch: args.learning_rate*min(epoch**-0.5, epoch*50**-1.5)}

    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        if args.lradj != 'type4':
            print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0, checkpoint=True, model_setup=None):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.checkpoint = checkpoint
        self.model_setup = model_setup
        self.val_losses = []

    def __call__(self, val_loss, model, path, epoch):
        score = -val_loss
        self.val_losses.append(val_loss)
        if self.best_score is None:
            self.best_score = score
            if self.checkpoint:
                self.save_checkpoint(val_loss, model, path, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if self.checkpoint:
                self.save_checkpoint(val_loss, model, path, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path, epoch):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss

        # if not os.path.exists(path + '/' + 'model_setup.pickle'):
        pickle.dump(self.model_setup.__dict__, open(path + '/' + 'model_setup.pickle', 'wb'))
        pickle.dump({'epoch': epoch, 'val_loss_min': val_loss, 'val_losses': self.val_losses},
                    open(path + '/' + 'epoch_loss.pickle', 'wb'))
        with open(path + '/' + 'model_setup.txt', 'w') as f:
            f.write('Epoch: ' + str(epoch + 1) + '\nValLoss: ' + str(val_loss))
            f.write('\n\n__________________________________________________________\n\n')
            for key, value in self.model_setup.__dict__.items():
                f.write('%s \t%s\n' % (key, value))


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.png'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')


class PlotLossesSame:
    def __init__(self, start_epoch, **kwargs):
        self.epochs = [int(start_epoch)]
        self.metrics = list(kwargs.keys())
        plt.ion()

        self.fig, self.axs = plt.subplots(figsize=(8, 5))
        # self.axs.yaxis.set_major_formatter(FormatStrFormatter('%.3f'))
        for i, (metric, values) in enumerate(kwargs.items()):
            self.__dict__[metric] = [values]
            self.axs.plot([], [], label=metric, alpha=0.7)
        self.axs.grid()
        self.axs.legend()

    def on_epoch_end(self, **kwargs):
        if list(kwargs.keys()) != self.metrics:
            raise ValueError('Need to pass the same arguments as were initialised')

        self.epochs.append(self.epochs[-1] + 1)

        for i, metric in enumerate(self.metrics):
            self.__dict__[metric].append(kwargs[metric])
            self.axs.lines[i].set_data(self.epochs, self.__dict__[metric])
        self.axs.relim()  # recompute the data limits
        self.axs.autoscale_view()  # automatic axis scaling
        self.fig.canvas.flush_events()
