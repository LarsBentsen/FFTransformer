import torch
import torch.nn as nn
from layers.MLP_Layer import MLPLayer
from layers.graph_modules import GraphLayer, GraphNetwork
from layers.graph_blocks import unsorted_mean_agg


class MLP(nn.Module):
    def __init__(self, layers):
        super(MLP, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, outputs, **_):
        for layer in self.layers:
            outputs = layer(outputs)

        return outputs, None


class Model(nn.Module):
    """
    GNN with MLP as update functions
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pred_len = configs.pred_len
        self.seq_len = configs.label_len
        self.output_features = configs.c_out
        self.d_model = configs.d_model
        self.layers = configs.e_layers
        self.output_attention = configs.output_attention

        # Embedding
        self._node_embedding = nn.Linear((configs.label_len + configs.pred_len) * configs.dec_in, configs.d_model)
        self._edge_embedding = nn.Linear(2, configs.d_model)

        # Decoder
        self._network = GraphNetwork(
            [
                GraphLayer(
                    edge_update_fn=MLP(
                            [
                                MLPLayer(
                                    input_size=self.d_model,
                                    output_size=self.d_model,
                                    dropout=configs.dropout,
                                    activation=configs.activation,
                                    norm_layer='layer'
                                )
                                for i in range(configs.e_layers)
                            ]
                    ),
                    node_update_fn=MLP(
                            [
                                MLPLayer(
                                    input_size=self.d_model,
                                    output_size=self.d_model,
                                    dropout=configs.dropout,
                                    activation=configs.activation,
                                    norm_layer='layer'
                                )
                                for i in range(configs.e_layers)
                            ]
                    ),
                    d_model=configs.d_model,
                    edges_agg=unsorted_mean_agg,
                )
                for l in range(configs.gnn_layers)
            ],
            mlp_out=MLPLayer(
                input_size=self.d_model,
                output_size=self.d_model,
                dropout=configs.dropout,
                activation=configs.activation,
                norm_layer='layer'
            ) if configs.mlp_out else None,
            projection=nn.Linear(configs.d_model, self.output_features * self.pred_len, bias=True)
        )

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, enc_self_mask=None, **_):
        #     x_enc and x_mark_enc are not used, but kept to keep the signatures similar for spatial and
        #     non-spatial models.

        # Map static edge variables to constant temporal features:
        x_inp = x_dec.replace(nodes=x_dec.nodes.reshape(x_dec.nodes.shape[0], -1))
        x_inp = x_inp.replace(
            nodes=self._node_embedding(x_inp.nodes),
            edges=self._edge_embedding(x_inp.edges)
        )

        outputs, _ = self._network(x_inp, attn_mask=enc_self_mask)
        outputs = outputs.nodes.view(outputs.nodes.shape[0], self.pred_len, self.output_features)

        if self.output_attention:
            return outputs, None
        else:
            return outputs  # [B, L, D]
