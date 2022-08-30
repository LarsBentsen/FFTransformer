import torch.nn as nn
from layers.Transformer_EncDec import EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.graph_modules import GraphLayer, GraphNetwork
from layers.graph_blocks import unsorted_mean_agg
from layers.Embed import GraphDataEmbedding
from layers.Functionality import MLPLayer


class Model(nn.Module):
    """
    GNN with Vanilla Transformer Encoder as update function:  https://arxiv.org/abs/1706.03762
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention

        # Embedding
        self._embedding = GraphDataEmbedding(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                             configs.dropout, kernel_size=configs.kernel_size)

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
                        activation=configs.activation
                    ),
                    node_update_fn=EncoderLayer(
                        AttentionLayer(
                            FullAttention(False, attention_dropout=configs.dropout,
                                          output_attention=configs.output_attention),
                            configs.d_model, configs.n_heads),
                        d_model=configs.d_model,
                        d_ff=configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation
                    ),
                    d_model=configs.d_model,
                    edges_agg=unsorted_mean_agg,
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

        # Embed the inputs
        x_inp = self._embedding(x_dec, x_mark_dec)

        # Map static edge variables to constant temporal features:
        num_edges, dim = x_inp.edges.shape
        x_inp = x_inp.replace(edges=x_inp.edges.unsqueeze(1).expand(num_edges, x_inp.nodes.shape[1], dim))

        attns = []
        outputs, a = self._network(x_inp, attn_mask=enc_self_mask)
        attns.append(a)

        if self.output_attention:
            return outputs.nodes[:, -self.pred_len:, :], attns
        else:
            return outputs.nodes[:, -self.pred_len:, :]  # [B, L, D]
