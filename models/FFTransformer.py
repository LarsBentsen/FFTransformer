import torch
import torch.nn as nn
from layers import FFTransformer_EncDec as FFT
from layers.FFT_SelfAttention import FFTAttention
from layers.SelfAttention_Family import LogSparseAttentionLayer, ProbAttention
from layers.Embed import DataEmbedding
from layers.WaveletTransform import get_wt
from layers.Functionality import MLPLayer


class Model(nn.Module):
    """
    FFTransformer Encoder-Decoder with Convolutional ProbSparse Attn for Trend and ProbSparse for Freq Strean
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.output_attention = configs.output_attention
        self.num_decomp = configs.num_decomp

        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # Frequency Embeddings:
        self.enc_embeddingF = DataEmbedding(self.num_decomp * configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout, kernel_size=configs.kernel_size,
                                            temp_embed=False, pos_embed=False)
        self.dec_embeddingF = DataEmbedding(self.num_decomp * configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout, kernel_size=configs.kernel_size,
                                            temp_embed=False, pos_embed=False)

        # Trend Embeddings:
        self.enc_embeddingT = DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout, kernel_size=configs.kernel_size)
        self.dec_embeddingT = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                            configs.dropout, kernel_size=configs.kernel_size)

        self.encoder = FFT.Encoder(
            [
                FFT.EncoderLayer(
                    FFTAttention(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, top_keys=False, context_zero=True),
                        configs.d_model, configs.n_heads
                    ),
                    LogSparseAttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, top_keys=configs.top_keys),
                        d_model=configs.d_model, n_heads=configs.n_heads,
                        qk_ker=configs.qk_ker, v_conv=configs.v_conv
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
            norm_freq=torch.nn.LayerNorm(configs.d_model) if configs.norm_out else None,
            norm_trend=torch.nn.LayerNorm(configs.d_model) if configs.norm_out else None,
        )
        # Decoder
        self.decoder = FFT.Decoder(
            [
                FFT.DecoderLayer(
                    FFTAttention(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, top_keys=False, context_zero=True),
                        configs.d_model, configs.n_heads
                    ),
                    LogSparseAttentionLayer(
                        ProbAttention(True, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, top_keys=configs.top_keys),
                        d_model=configs.d_model, n_heads=configs.n_heads, qk_ker=configs.qk_ker, v_conv=configs.v_conv
                    ),
                    FFTAttention(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, top_keys=False, context_zero=True),
                        configs.d_model, configs.n_heads
                    ),
                    LogSparseAttentionLayer(
                        ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=configs.output_attention, top_keys=configs.top_keys),
                        d_model=configs.d_model, n_heads=configs.n_heads, qk_ker=configs.qk_ker, v_conv=configs.v_conv
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.d_layers)
            ],
            norm_freq=nn.LayerNorm(configs.d_model) if configs.norm_out else None,
            norm_trend=nn.LayerNorm(configs.d_model) if configs.norm_out else None,
            mlp_out=MLPLayer(d_model=configs.d_model, d_ff=configs.d_ff, kernel_size=1,
                             dropout=configs.dropout, activation=configs.activation) if configs.mlp_out else None,
            out_projection=nn.Linear(configs.d_model, configs.c_out),
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None, **_):
        # Wavelet Decomposition
        x_enc_freq, x_enc_trend = get_wt(x_enc, num_decomp=self.num_decomp)
        x_dec_freq, x_dec_trend = get_wt(x_dec[:, :-self.pred_len, :], num_decomp=self.num_decomp)  # Remove PHs first

        # Add placeholders after decomposition:
        dec_trend_place = torch.mean(x_dec_trend, 1).unsqueeze(1).repeat(1, self.pred_len, 1)
        x_dec_trend = torch.cat([x_dec_trend, dec_trend_place], dim=1)

        dec_freq_place = torch.zeros([x_dec_freq.shape[0], self.pred_len, x_dec_freq.shape[2]], device=x_dec_freq.device)
        x_dec_freq = torch.cat([x_dec_freq, dec_freq_place], dim=1)

        # Embed the inputs:
        x_enc_freq = self.enc_embeddingF(x_enc_freq, x_mark_enc)
        x_dec_freq = self.dec_embeddingF(x_dec_freq, x_mark_dec)

        x_enc_trend = self.enc_embeddingT(x_enc_trend, x_mark_enc)
        x_dec_trend = self.dec_embeddingT(x_dec_trend, x_mark_dec)

        attns = []

        enc_freq, enc_trend, a = self.encoder([x_enc_freq, x_enc_trend], attn_mask=enc_self_mask)
        attns.append(a)

        dec_out, a = self.decoder([x_dec_freq, x_dec_trend], enc_freq, enc_trend, x_mask=dec_self_mask, cross_mask=dec_enc_mask)
        attns.append(a)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
