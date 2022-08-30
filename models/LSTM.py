import torch
import torch.nn as nn
from layers.LSTM_EncDec import Encoder, Decoder
from layers.Embed import DataEmbedding
import random


class Model(nn.Module):
    """
    LSTM in Encoder-Decoder
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.d_model = configs.d_model
        self.enc_layers = configs.e_layers
        self.dec_layers = configs.d_layers

        self.train_strat_lstm = configs.train_strat_lstm

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.label_len = configs.label_len
        self.output_size = configs.c_out
        assert configs.label_len >= 1

        self.enc_embedding = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, kernel_size=configs.kernel_size, pos_embed=False)
        self.dec_embedding = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                           configs.dropout, kernel_size=configs.kernel_size, pos_embed=False)

        self.encoder = Encoder(d_model=self.d_model, num_layers=self.enc_layers, dropout=configs.dropout)
        self.decoder = Decoder(output_size=configs.c_out, d_model=self.d_model,
                               dropout=configs.dropout, num_layers=self.dec_layers)

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, teacher_forcing_ratio=None, **_):
        if self.train_strat_lstm == 'mixed_teacher_forcing' and self.training:
            assert teacher_forcing_ratio is not None

        target = x_dec[:, -self.pred_len:, -self.output_size:]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)

        enc_out, enc_hid = self.encoder(enc_out)

        if self.enc_layers != self.dec_layers:
            assert self.dec_layers <= self.enc_layers
            enc_hid = [hid[-self.dec_layers:, ...] for hid in enc_hid]

        dec_inp = x_dec[:, -(self.pred_len + 1), -self.output_size:]
        dec_hid = enc_hid

        outputs = torch.zeros_like(target)

        if not self.training or self.train_strat_lstm == 'recursive':
            for t in range(self.pred_len):
                dec_out, dec_hid = self.decoder(dec_inp, dec_hid)
                outputs[:, t, :] = dec_out
                dec_inp = dec_out
        else:
            if self.train_strat_lstm == 'mixed_teacher_forcing':
                for t in range(self.pred_len):
                    dec_out, dec_hid = self.decoder(dec_inp, dec_hid)
                    outputs[:, t, :] = dec_out
                    if random.random() < teacher_forcing_ratio:
                        dec_inp = target[:, t, :]
                    else:
                        dec_inp = dec_out

        return outputs  # [B, L, D]
