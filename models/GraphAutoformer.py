import torch
import torch.nn as nn
from layers.Autoformer_EncDec import DecoderLayer as AutoDecoderLayer
from layers.Autoformer_EncDec import GraphWrapperLayer as GraphWrapper
from layers.Autoformer_EncDec import series_decomp
from layers.Functionality import MLPLayer
from layers.Transformer_EncDec import EncoderLayer as TransEncoderLayer
from layers.SelfAttention_Family import AttentionLayer, FullAttention
from layers.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from layers.graph_modules import GraphLayer, GraphNetwork
from layers.graph_blocks import unsorted_mean_agg
from layers.Embed import GraphDataEmbedding


class Model(nn.Module):
    """
    GNN with Autoformer Decoder (without cross-attn) as node update function: https://arxiv.org/abs/2106.13008
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.c_out = configs.c_out

        self.output_attention = configs.output_attention
        kernel_size = configs.moving_avg
        self.decomp = series_decomp(kernel_size)

        # Embedding
        self._embedding = GraphDataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                             configs.dropout, kernel_size=configs.kernel_size, pos_enc=False)

        # Decoder
        self._network = GraphNetwork(
            [
                GraphLayer(
                    edge_update_fn=TransEncoderLayer(
                        AttentionLayer(
                            FullAttention(False, attention_dropout=configs.dropout,
                                          output_attention=configs.output_attention),
                            configs.d_model, configs.n_heads),
                        d_model=configs.d_model,
                        d_ff=configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ),
                    node_update_fn=GraphWrapper(
                        AutoDecoderLayer(
                            AutoCorrelationLayer(
                                AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                                output_attention=configs.output_attention),
                                configs.d_model, configs.n_heads),
                            None,
                            d_model=configs.d_model,
                            c_out=configs.c_out,
                            d_ff=configs.d_ff,
                            moving_avg=configs.moving_avg,
                            dropout=configs.dropout,
                            activation=configs.activation
                        )
                    ),
                    d_model=configs.d_model,
                    edges_agg=unsorted_mean_agg,    # Could use different edge aggregation such as sum or attention
                    num_node_series=1
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

        mean = torch.mean(x_dec.nodes[:, :-self.pred_len, :], dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)
        zeros = torch.zeros([x_dec.nodes.shape[0], self.pred_len, x_dec.nodes.shape[2]], device=x_dec.nodes.device)
        seasonal_init, trend_init = self.decomp(x_dec.nodes[:, :-self.pred_len, :])

        trend_init = torch.cat([trend_init[..., -self.c_out:], mean[..., -self.c_out:]], dim=1)
        seasonal_init = torch.cat([seasonal_init, zeros], dim=1)

        x_inp = self._embedding(x_dec.replace(nodes=seasonal_init), x_mark_dec)

        # Map static edge variables to constant temporal features:
        num_edges, dim = x_inp.edges.shape
        x_inp = x_inp.replace(edges=x_inp.edges.unsqueeze(1).expand(num_edges, x_inp.nodes.shape[1], dim))

        attns = []
        outputs, a = self._network(x_inp, trend=trend_init, attn_mask=enc_self_mask)
        attns.append(a)

        if self.output_attention:
            return outputs.nodes[:, -self.pred_len:, :], attns
        else:
            return outputs.nodes[:, -self.pred_len:, :]  # [B, L, D]
