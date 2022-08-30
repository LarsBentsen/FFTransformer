import argparse
import torch
from exp.exp_main import Exp_Main
import random
import numpy as np


def main():
    fix_seed = 2022
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='FFTransformer, Transformer family, LSTM and MLP for Wind Forecasting')

    # basic config
    parser.add_argument('--is_training', type=int, required=False, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=False, default='test', help='model id for saving')
    parser.add_argument('--model', type=str, required=False, default='FFTransformer',
                        help='model name, options: [FFTransformer, Autoformer, Informer, Transformer, LogSparse, LSTM, MLP, persistence (and same with GraphXxxx)]')
    parser.add_argument('--plot_flag', type=int, default=1, help='Whether to save loss plots or not')
    parser.add_argument('--test_dir', type=str, default='', help='Base dir to save test results')
    parser.add_argument('--verbose', type=int, default=1, help='Whether to print inter-epoch losses.')

    # data loader
    parser.add_argument('--data', type=str, required=False, default='Wind', help='dataset type, Wind or WindGraph')
    parser.add_argument('--root_path', type=str, default='./dataset_example/WindData/dataset/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='wind_data.csv', help='data file')
    parser.add_argument('--target', type=str, default='KVITEBJØRNFELTET', help='optional target station for non-graph models')
    parser.add_argument('--freq', type=str, default='10min', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--checkpoint_flag', type=int, default=1, help='Whether to checkpoint or not')
    parser.add_argument('--n_closest', type=int, default=None, help='number of closest nodes for graph connectivity, None --> complete graph')
    parser.add_argument('--all_stations', type=int, default=0, help='Whether to use all stations or just target for non-spatial models.')
    parser.add_argument('--data_step', type=int, default=1, help='Only use every nth point. Set data_step = 1 for full dataset.')
    parser.add_argument('--min_num_nodes', type=int, default=2, help='Minimum number of nodes in a graph')

    # forecasting task
    parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S]; M:multivariate input, S:univariate input')
    parser.add_argument('--seq_len', type=int, default=64, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length. Note that Graph models only use label_len and pred_len')
    parser.add_argument('--pred_len', type=int, default=6, help='prediction sequence length')
    parser.add_argument('--enc_in', type=int, default=8, help='Number of encoder input features')
    parser.add_argument('--dec_in', type=int, default=8, help='Number of decoder input features')
    parser.add_argument('--c_out', type=int, default=1, help='output size, note that it is assumed that the target features are placed last')

    # model define
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='number of encoder layers for non-spatial and number of LSTM or MLP layers for GraphLSTM and GraphMLP')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--gnn_layers', type=int, default=2, help='Number of sequential graph blocks in GNN')
    parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average for Autoformer')
    parser.add_argument('--factor', type=int, default=3, help='attn factor')
    parser.add_argument('--distil', action='store_false', default=True, help='whether to use distilling in encoder, using this argument means not using distilling, not used for GNN models')
    parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF', help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder', default=False)
    parser.add_argument('--win_len', type=int, default=6, help='Local attention length for LogSparse Transformer')
    parser.add_argument('--res_len', type=int, default=None, help='Restart attention length for LogSparse Transformer')
    parser.add_argument('--qk_ker', type=int, default=4, help='Key/Query convolution kernel length for LogSparse Transformer')
    parser.add_argument('--v_conv', type=int, default=0, help='Weather to apply ConvAttn for values (in addition to K/Q for LogSparseAttn')
    parser.add_argument('--sparse_flag', type=int, default=1, help='Weather to apply logsparse mask for LogSparse Transformer')
    parser.add_argument('--top_keys', type=int, default=0, help='Weather to find top keys instead of queries in Informer')
    parser.add_argument('--kernel_size', type=int, default=3, help='Kernel size for the 1DConv value embedding')
    parser.add_argument('--train_strat_lstm', type=str, default='recursive', help='The training strategy to use for the LSTM model. recursive or mixed_teacher_forcing')
    parser.add_argument('--norm_out', type=int, default=1, help='Whether to apply laynorm to outputs of Enc or Dec in FFTransformer')
    parser.add_argument('--num_decomp', type=int, default=4, help='Number of wavelet decompositions for FFTransformer')
    parser.add_argument('--mlp_out', type=int, default=0, help='Whether to apply MLP to GNN outputs.')

    # Optimization
    parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
    parser.add_argument('--lr_decay_rate', type=float, default=0.8, help='Rate for which to decay lr with')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='mse', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multiple gpus')

    args = parser.parse_args()

    if args.features == 'S':
        assert (np.array([args.c_out, args.enc_in, args.dec_in]) == 1).all(), "c_out, enc_in and dec_in should be 1 for univariate"

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    Exp = Exp_Main

    if args.is_training:
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}'.format(
                args.model_id,
                args.model,
                args.data,
                args.features,
                args.seq_len,
                args.label_len,
                args.pred_len,
                ii)

            exp = Exp(args)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)

            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting, base_dir=args.test_dir)

            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_{}'.format(
            args.model_id,
            args.model,
            args.data,
            args.features,
            args.seq_len,
            args.label_len,
            args.pred_len,
            ii)

        exp = Exp(args)  # set experiments
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
