import layers.graphs
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import Informer, Autoformer, Transformer, LogSparseTransformer, FFTransformer, \
    LSTM, MLP, persistence, GraphTransformer, GraphLSTM, GraphFFTransformer, \
    GraphInformer, GraphLogSparse, GraphMLP, GraphAutoformer, GraphPersistence
from utils.tools import EarlyStopping, adjust_learning_rate, visual, PlotLossesSame
from utils.metrics import metric
from utils.graph_utils import data_dicts_to_graphs_tuple, split_torch_graph
from utils.CustomDataParallel import DataParallelGraph

import pickle
import torch
import torch.nn as nn
from torch import optim

import os
import time

import warnings
import numpy as np


warnings.filterwarnings('ignore')


class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        if self.args.data == 'WindGraph':
            self.args.seq_len = self.args.label_len

    def _build_model(self):
        model_dict = {
            'Autoformer': Autoformer,
            'Transformer': Transformer,
            'Informer': Informer,
            'LogSparse': LogSparseTransformer,
            'FFTransformer': FFTransformer,
            'LSTM': LSTM,
            'MLP': MLP,
            'persistence': persistence,
            'GraphTransformer': GraphTransformer,
            'GraphLSTM': GraphLSTM,
            'GraphFFTransformer': GraphFFTransformer,
            'GraphInformer': GraphInformer,
            'GraphLogSparse': GraphLogSparse,
            'GraphMLP': GraphMLP,
            'GraphAutoformer': GraphAutoformer,
            'GraphPersistence': GraphPersistence,
        }

        model = model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            if self.args.data == 'WindGraph':
                model = DataParallelGraph(model, device_ids=self.args.device_ids)
            else:
                model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def MAPE(self, pred, tar, eps=1e-07):
        loss = torch.mean(torch.abs(pred - tar) / (tar + eps))
        return loss

    def vali(self, setting, vali_data, vali_loader, criterion, epoch=0, plot_res=1, save_path=None):
        total_loss = []
        total_mse = []
        total_mape = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):

                # Graph Data (i.e. Spatio-temporal)
                if self.args.data == 'WindGraph':
                    batch_x = data_dicts_to_graphs_tuple(batch_x, device=self.device)
                    batch_y = data_dicts_to_graphs_tuple(batch_y, device=self.device)
                    dec_inp = batch_y

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_zeros = dec_inp.nodes[:, (self.args.label_len - 1):self.args.label_len, :]        # Select last value
                    dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()                        # Repeat for pred_len
                    dec_zeros = torch.cat([dec_inp.nodes[:, :self.args.label_len, :], dec_zeros], dim=1)  # Add Placeholders
                    dec_zeros = dec_zeros.float().to(self.device)
                    dec_inp = dec_inp.replace(nodes=dec_zeros)

                    dec_inp = dec_inp.replace(nodes=dec_inp.nodes[:, :, -self.args.dec_in:])
                    batch_x = batch_x.replace(nodes=batch_x.nodes[:, :, -self.args.enc_in:])
                # Non-Graph Data (i.e. just temporal)
                else:
                    dec_inp = batch_y

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float()

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_zeros = dec_inp[:, (self.args.label_len-1):self.args.label_len, :]         # Select last value
                    dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()                 # Repeat for pred_len
                    dec_inp = torch.cat([dec_inp[:, :self.args.label_len, :], dec_zeros], dim=1)   # Add Placeholders
                    dec_inp = dec_inp.float().to(self.device)

                    dec_inp = dec_inp[:, :, -self.args.dec_in:]
                    batch_x = batch_x[:, :, -self.args.enc_in:]

                if self.args.data == 'WindGraph' and self.args.use_multi_gpu:
                    batch_x, sub_bs_x, target_gpus = split_torch_graph(batch_x, self.args.devices.split(','))
                    dec_inp, sub_bs_y, _ = split_torch_graph(dec_inp, self.args.devices.split(','))
                    assert np.array([sum(sub_i) == len(sub_i) for sub_i in [sub_bs_x[j] == sub_bs_y[j] for j in range(len(sub_bs_x))]]).all()

                    batch_x_mark = [batch_x_mark[indx_i, ...].to(gpu_i) for gpu_i, indx_i in zip(target_gpus, sub_bs_x)]
                    batch_y_mark = [batch_y_mark[indx_i, ...].to(gpu_i) for gpu_i, indx_i in zip(target_gpus, sub_bs_x)]

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if 'M' in self.args.features:
                    f_dim = -self.args.c_out
                else:
                    f_dim = 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                if self.args.data == 'WindGraph':
                    batch_y = batch_y.nodes[:, -self.args.pred_len:, f_dim:].to(self.device)
                else:
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss)
        total_loss = np.average(total_loss)

        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')     # self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        if 'sistence' in self.args.model:       # Check if the model is the persistence model
            criterion = self._select_criterion()
            vali_loss = self.vali(setting, vali_data, vali_loader, criterion)
            test_loss = self.vali(setting, test_data, test_loader, criterion)

            self.test('persistence_' + str(self.args.pred_len), test=0, base_dir='', save_dir='results/' + self.args.model + '/', save_flag=True)

            print('vali_loss: ', vali_loss)
            print('test_loss: ', test_loss)
            assert False

        self.vali_losses = []  # Store validation losses

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path) and self.args.checkpoint_flag:
            os.makedirs(path)

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True,
                                       checkpoint=self.args.checkpoint_flag, model_setup=self.args)

        if self.args.checkpoint_flag:
            load_path = os.path.normpath(os.path.join(path, 'checkpoint.pth'))
            if os.path.exists(load_path) and self.load_check(path=os.path.normpath(os.path.join(path, 'model_setup.pickle'))):
                self.model.load_state_dict(torch.load(load_path))
                epoch_info = pickle.load(
                    open(os.path.normpath(os.path.join('./checkpoints/' + setting, 'epoch_loss.pickle')), 'rb'))
                start_epoch = epoch_info['epoch']
                early_stopping.val_losses = epoch_info['val_losses']
                early_stopping.val_loss_min = epoch_info['val_loss_min']
                self.vali_losses = epoch_info['val_losses']
                del epoch_info
            else:
                start_epoch = 0
                print('Could not load best model')
        else:
            start_epoch = 0

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        teacher_forcing_ratio = 0.8     # For LSTM Enc-Dec training (not used for others).
        total_num_iter = 0
        time_now = time.time()
        for epoch in range(start_epoch, self.args.train_epochs):

            # Reduce the tearcher forcing ration every epoch
            if self.args.model == 'LSTM':
                teacher_forcing_ratio -= 0.08
                teacher_forcing_ratio = max(0., teacher_forcing_ratio)
                print('teacher_forcing_ratio: ', teacher_forcing_ratio)

            # type4 lr scheduling is updated more frequently
            if self.args.lradj != 'type4':
                adjust_learning_rate(model_optim, epoch + 1, self.args)

            train_loss = []

            self.model.train()
            epoch_time = time.time()
            num_iters = len(train_loader)
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                if self.args.lradj == 'type4':
                    adjust_learning_rate(model_optim, total_num_iter + 1, self.args)
                    total_num_iter += 1
                if isinstance(batch_y, dict):
                    batch_x = data_dicts_to_graphs_tuple(batch_x, device=self.device)
                    batch_y = data_dicts_to_graphs_tuple(batch_y, device=self.device)
                    dec_inp = batch_y

                    model_optim.zero_grad()
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_zeros = dec_inp.nodes[:, (self.args.label_len - 1):self.args.label_len, :]        # Select last value
                    dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()                        # Repeat for pred_len
                    dec_zeros = torch.cat([dec_inp.nodes[:, :self.args.label_len, :], dec_zeros], dim=1)  # Add Placeholders
                    dec_zeros = dec_zeros.float().to(self.device)
                    dec_inp = dec_inp.replace(nodes=dec_zeros)

                    dec_inp = dec_inp.replace(nodes=dec_inp.nodes[:, :, -self.args.dec_in:])
                    batch_x = batch_x.replace(nodes=batch_x.nodes[:, :, -self.args.enc_in:])
                else:
                    dec_inp = batch_y

                    model_optim.zero_grad()
                    batch_x = batch_x.float().to(self.device)

                    batch_y = batch_y.float().to(self.device)
                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_zeros = dec_inp[:, (self.args.label_len-1):self.args.label_len, :]         # Select last value
                    dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()                 # Repeat for pred_len
                    dec_inp = torch.cat([dec_inp[:, :self.args.label_len, :], dec_zeros], dim=1)   # Add Placeholders
                    dec_inp = dec_inp.float().to(self.device)

                    dec_inp = dec_inp[:, :, -self.args.dec_in:]
                    batch_x = batch_x[:, :, -self.args.enc_in:]

                # Note that the collate function is not optimised and might have some potential errors
                if self.args.data == 'WindGraph' and self.args.use_multi_gpu:
                    batch_x, sub_bs_x, target_gpus = split_torch_graph(batch_x, self.args.devices.split(','))
                    dec_inp, sub_bs_y, _ = split_torch_graph(dec_inp, self.args.devices.split(','))
                    assert np.array([sum(sub_i) == len(sub_i) for sub_i in [sub_bs_x[j] == sub_bs_y[j] for j in range(len(sub_bs_x))]]).all()

                    batch_x_mark = [batch_x_mark[indx_i, ...].to(gpu_i) for gpu_i, indx_i in zip(target_gpus, sub_bs_x)]
                    batch_y_mark = [batch_y_mark[indx_i, ...].to(gpu_i) for gpu_i, indx_i in zip(target_gpus, sub_bs_x)]
                    teacher_forcing_ratio = [teacher_forcing_ratio for gpu_i, indx_i in zip(target_gpus, sub_bs_x)]

                # encoder - decoder
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                         teacher_forcing_ratio=teacher_forcing_ratio, batch_y=batch_y)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark,
                                         teacher_forcing_ratio=teacher_forcing_ratio, batch_y=batch_y)

                if 'M' in self.args.features:
                    f_dim = -self.args.c_out
                else:
                    f_dim = 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                if isinstance(batch_y, layers.graphs.GraphsTuple):
                    batch_y = batch_y.nodes[:, -self.args.pred_len:, f_dim:].to(self.device)
                else:
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                loss.backward()
                model_optim.step()

                if (i + 1) % 100 == 0 and self.args.verbose == 1:
                    print("\titers: {0}/{1}, epoch: {2} | loss: {3:.7f}".format(i + 1, num_iters, epoch + 1, np.average(train_loss)))

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(setting, vali_data, vali_loader, criterion, epoch=epoch, save_path=path)
            test_flag = False
            if test_flag:
                test_loss = self.vali(setting, test_data, test_loader, criterion, epoch=epoch, save_path=path)

            # Plot the losses:
            if self.args.plot_flag and self.args.checkpoint_flag:
                loss_save_dir = path + '/pic/train_loss.png'
                loss_save_dir_pkl = path + '/train_loss.pickle'
                if os.path.exists(loss_save_dir_pkl):
                    fig_progress = pickle.load(open(loss_save_dir_pkl, 'rb'))

                if 'fig_progress' not in locals():
                    fig_progress = PlotLossesSame(epoch + 1,
                                                  Training=train_loss,
                                                  Validation=vali_loss)
                else:
                    fig_progress.on_epoch_end(Training=train_loss,
                                              Validation=vali_loss)

                if not os.path.exists(os.path.dirname(loss_save_dir)):
                    os.makedirs(os.path.dirname(loss_save_dir))
                fig_progress.fig.savefig(loss_save_dir)
                pickle.dump(fig_progress, open(loss_save_dir_pkl, 'wb'))    # To load figure that we can append to

            if test_flag:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss {4:.7f}".format(
                    epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            else:
                print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))
            early_stopping(vali_loss, self.model, path, epoch)
            self.vali_losses += [vali_loss]       # Append validation loss
            if early_stopping.early_stop:
                print("Early stopping")
                break

        # After Training, load the best model.
        if self.args.checkpoint_flag:
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def load_check(self, path, ignore_vars=None, ignore_paths=False):
        # Function to check that the checkpointed and current settings are compatible.
        if ignore_vars is None:
            ignore_vars = [
                'is_training',
                'train_epochs',
                'plot_flag',
                'root_path',
                'data_path',
                'data_path',
                'checkpoints',
                'checkpoint_flag',
                'output_attention',
                'do_predict',
                'des',
                'n_closest',
                'verbose',
                'data_step',
                'itr',
                'patience',
                'des',
                'gpu',
                'use_gpu',
                'use_multi-gpu',
                'devices',
            ]
        if ignore_paths:
            ignore_vars += [
                'model_id',
                'test_dir',
            ]

        setting2 = pickle.load(open(path, 'rb'))
        for key, val in self.args.__dict__.items():
            if key in ignore_vars:
                continue
            if val != setting2[key]:
                print(val, ' is not equal to ', setting2[key], ' for ', key)
                return False

        return True

    def test(self, setting, test=1, base_dir='', save_dir=None, ignore_paths=False, save_flag=True):
        test_data, test_loader = self._get_data(flag='test')
        if save_dir is None:
            save_dir = base_dir
        if test:
            print('loading model')
            if len(base_dir) == 0:
                load_path = os.path.normpath(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'))
            else:
                load_path = os.path.normpath(os.path.join(base_dir + 'checkpoints/' + setting, 'checkpoint.pth'))
            load_check_flag = self.load_check(path=os.path.normpath(os.path.join(os.path.dirname(load_path),
                                                                                 'model_setup.pickle')),
                                              ignore_paths=ignore_paths)
            if os.path.exists(load_path) and load_check_flag:
                self.model.load_state_dict(torch.load(load_path))
            else:
                print('Could not load best model')

        preds = []
        trues = []
        station_ids = []
        if save_flag:
            if len(save_dir) == 0:
                folder_path = './test_results/' + setting + '/'
            else:
                folder_path = save_dir + 'test_results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                if self.args.data == 'WindGraph':
                    batch_x = data_dicts_to_graphs_tuple(batch_x, device=self.device)
                    batch_y = data_dicts_to_graphs_tuple(batch_y, device=self.device)
                    dec_inp = batch_y

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_zeros = dec_inp.nodes[:, (self.args.label_len - 1):self.args.label_len, :]        # Select last value
                    dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()                        # Repeat for pred_len
                    dec_zeros = torch.cat([dec_inp.nodes[:, :self.args.label_len, :], dec_zeros], dim=1)  # Add Placeholders
                    dec_zeros = dec_zeros.float().to(self.device)
                    dec_inp = dec_inp.replace(nodes=dec_zeros)

                    dec_inp = dec_inp.replace(nodes=dec_inp.nodes[:, :, -self.args.dec_in:])
                    batch_x = batch_x.replace(nodes=batch_x.nodes[:, :, -self.args.enc_in:])
                else:
                    dec_inp = batch_y

                    batch_x = batch_x.float().to(self.device)
                    batch_y = batch_y.float().to(self.device)

                    batch_x_mark = batch_x_mark.float().to(self.device)
                    batch_y_mark = batch_y_mark.float().to(self.device)

                    # decoder input
                    dec_zeros = dec_inp[:, (self.args.label_len-1):self.args.label_len, :]
                    dec_zeros = dec_zeros.repeat(1, self.args.pred_len, 1).float()
                    dec_inp = torch.cat([dec_inp[:, :self.args.label_len, :], dec_zeros], dim=1)
                    dec_inp = dec_inp.float().to(self.device)

                    dec_inp = dec_inp[:, :, -self.args.dec_in:]
                    batch_x = batch_x[:, :, -self.args.enc_in:]

                if self.args.data == 'WindGraph' and self.args.use_multi_gpu:
                    batch_x, sub_bs_x, target_gpus = split_torch_graph(batch_x, self.args.devices.split(','))
                    dec_inp, sub_bs_y, _ = split_torch_graph(dec_inp, self.args.devices.split(','))
                    assert np.array([sum(sub_i) == len(sub_i) for sub_i in [sub_bs_x[j] == sub_bs_y[j] for j in range(len(sub_bs_x))]]).all()

                    batch_x_mark = [batch_x_mark[indx_i, ...].to(gpu_i) for gpu_i, indx_i in zip(target_gpus, sub_bs_x)]
                    batch_y_mark = [batch_y_mark[indx_i, ...].to(gpu_i) for gpu_i, indx_i in zip(target_gpus, sub_bs_x)]

                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                if 'M' in self.args.features:
                    f_dim = -self.args.c_out
                else:
                    f_dim = 0

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                if self.args.data == 'WindGraph':
                    batch_y = batch_y.nodes[:, -self.args.pred_len:, f_dim:].to(self.device)
                else:
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y

                preds.append(pred)
                trues.append(true)
                if self.args.data == 'WindGraph':
                    station_ids.append(batch_x.station_names)
                if i % 20 == 0:
                    if self.args.data == 'WindGraph':
                        input = batch_x.nodes.detach().cpu().numpy()
                    else:
                        input = batch_x.detach().cpu().numpy()

                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    if save_flag:
                        visual(gt, pd, os.path.join(folder_path, str(i) + '.png'))

        preds = np.vstack(preds)
        trues = np.vstack(trues)
        if self.args.data == 'WindGraph':
            station_ids = np.concatenate(station_ids)

        print('test shape:', preds.shape, trues.shape)
        print('test shape:', preds.shape, trues.shape)

        # save results
        if save_flag:
            if len(save_dir) == 0:
                folder_path = './results/' + setting + '/'
            else:
                folder_path = save_dir + 'results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)

        preds_un = test_data.inverse_transform(preds)
        trues_un = test_data.inverse_transform(trues)
        mae_un, mse_un, rmse_un, mape_un, mspe_un = metric(preds_un, trues_un)

        losses = {
            'mae_sc': mae,
            'mse_sc': mse,
            'rmse_sc': rmse,
            'mape_sc': mape,
            'mspe_sc': mspe,
            '': '\n\n',
            'mae_un': mae_un,
            'mse_un': mse_un,
            'rmse_un': rmse_un,
            'mape_un': mape_un,
            'mspe_un': mspe_un,
        }

        if self.args.data == 'WindGraph':
            for stat in np.unique(station_ids):
                indxs_i = np.where(station_ids == stat)[0]

                mae_i, mse_i, rmse_i, mape_i, mspe_i = metric(preds[indxs_i], trues[indxs_i])

                preds_un_i = test_data.inverse_transform(preds[indxs_i])
                trues_un_i = test_data.inverse_transform(trues[indxs_i])

                mae_un_i, mse_un_i, rmse_un_i, mape_un_i, mspe_un_i = metric(preds_un_i, trues_un_i)

                losses_i = {
                    '': '\n\n',
                    'mae_sc_' + stat: mae_i,
                    'mse_sc_' + stat: mse_i,
                    'rmse_sc_' + stat: rmse_i,
                    'mape_sc_' + stat: mape_i,
                    'mspe_sc_' + stat: mspe_i,
                    'mae_un_' + stat: mae_un_i,
                    'mse_un_' + stat: mse_un_i,
                    'rmse_un_' + stat: rmse_un_i,
                    'mape_un_' + stat: mape_un_i,
                    'mspe_un_' + stat: mspe_un_i,
                }
                losses.update(losses_i)

        if not save_flag:
            return losses

        with open(folder_path + "results_loss.txt", 'w') as f:
            for key, value in losses.items():
                f.write('%s:%s\n' % (key, value))

        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)
        np.save(folder_path + 'pred_un.npy', preds_un)
        np.save(folder_path + 'true_un.npy', trues_un)
        if self.args.data == 'WindGraph':
            np.save(folder_path + 'station_ids.npy', station_ids)

        with open(folder_path + 'metrics.txt', 'w') as f:
            f.write('mse: ' + str(mse) + '\n')
            f.write('mae: ' + str(mae) + '\n')
            f.write('rmse: ' + str(rmse) + '\n')
            f.write('mape: ' + str(mape) + '\n')
            f.write('mspe: ' + str(mspe) + '\n')

        return losses
