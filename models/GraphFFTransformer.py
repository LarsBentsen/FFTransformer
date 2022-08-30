import torch
import torch.nn as nn
import layers.FFTransformer_EncDec as FFT
from layers.FFT_SelfAttention import FFTAttention
from layers.Transformer_EncDec import EncoderLayer
from layers.SelfAttention_Family import AttentionLayer, FullAttention, LogSparseAttentionLayer, ProbAttention
from layers.graph_modules import GraphLayer, GraphNetwork
from layers.graph_blocks import unsorted_mean_agg
from layers.Embed import EdgeDataEmbedding, DataEmbedding
from layers.WaveletTransform import get_wt
from layers.Functionality import MLPLayer


class Model(nn.Module):
    """
    GNN with FFTTransformer Encoder as node-update function and Transformer Encoder as edge-update function
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len

        self.output_attention = configs.output_attention
        self.num_decomp = configs.num_decomp

        # Embedding
        self._embedding_freq = DataEmbedding(self.num_decomp * configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                             configs.dropout, kernel_size=configs.kernel_size,
                                             temp_embed=False, pos_embed=False)
        self._embedding_trend = DataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                              configs.dropout, kernel_size=configs.kernel_size)
        self._embedding_edge = EdgeDataEmbedding(d_model=configs.d_model, dropout=configs.dropout)

        # Decoder
        self._network = GraphNetwork(
            [
                GraphLayer(
                    edge_update_fn=EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, attention_dropout=configs.dropout,
                                          output_attention=configs.output_attention),
                            configs.d_model, configs.n_heads),
                        d_model=configs.d_model,
                        d_ff=configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    ),
                    node_update_fn=FFT.EncoderLayer(
                        FFTAttention(
                            ProbAttention(False, configs.factor, attention_dropout=configs.dropout,
                                          output_attention=configs.output_attention, top_keys=False, context_zero=True),
                            configs.d_model, configs.n_heads),
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
                    ),
                    d_model=configs.d_model,
                    edges_agg=unsorted_mean_agg,    # Could use different edge aggregation such as sum or attention
                    num_node_series=2,
                    n_closest=configs.n_closest
                )
                for l in range(configs.gnn_layers)
            ],
            mlp_out=MLPLayer(d_model=configs.d_model, d_ff=configs.d_ff, kernel_size=1,
                             dropout=configs.dropout, activation=configs.activation) if configs.mlp_out else None,
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, **_):
        #     x_enc and x_mark_enc are not used, but kept to keep the signatures similar for spatial and
        #     non-spatial models.

        x_inp_freq, x_inp_trend = get_wt(x_dec.nodes[:, :-self.pred_len, :], num_decomp=self.num_decomp)

        # Add placeholders after the decomposition:
        trend_place = torch.mean(x_inp_trend, 1).unsqueeze(1).repeat(1, self.pred_len, 1)
        x_inp_trend = torch.cat([x_inp_trend, trend_place], dim=1)

        freq_place = torch.zeros([x_inp_freq.shape[0], self.pred_len, x_inp_freq.shape[2]], device=x_inp_freq.device)
        x_inp_freq = torch.cat([x_inp_freq, freq_place], dim=1)

        # Embed the inputs
        x_inp_freq = self._embedding_freq(x_inp_freq, x_mark_dec, n_node=x_dec.n_node)
        x_inp_trend = self._embedding_trend(x_inp_trend, x_mark_dec, n_node=x_dec.n_node)
        x_inp_edge = self._embedding_edge(x_dec.edges)

        x_inp = x_dec.replace(
            nodes=torch.stack([x_inp_freq, x_inp_trend], -1),
            edges=x_inp_edge,
        )

        # Map static edge variables to constant temporal features:
        num_edges, dim = x_inp.edges.shape
        x_inp = x_inp.replace(edges=x_inp.edges.unsqueeze(1).expand(num_edges, x_inp_trend.shape[1], dim))

        attns = []
        outputs, a = self._network(x_inp, attn_mask=enc_self_mask)
        attns.append(a)

        if self.output_attention:
            return outputs.nodes[:, -self.pred_len:, :], attns
        else:
            return outputs.nodes[:, -self.pred_len:, :]  # [B, L, D]
