import torch.nn as nn
from layers.MLP_Layer import MLPLayer


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.input_features = configs.enc_in
        self.output_features = configs.c_out
        self.d_model = configs.d_model
        self.layers = configs.e_layers
        self.output_attention = configs.output_attention

        # Encoder
        self.mlp = nn.ModuleList(
            [
                MLPLayer(
                    input_size=self.d_model if i != 0 else (self.input_features * self.seq_len),
                    output_size=self.d_model,
                    dropout=configs.dropout,
                    activation=configs.activation,
                    norm_layer='layer'
                )
                for i in range(configs.e_layers)
            ]
        )
        self.projection = nn.Linear(configs.d_model, (self.output_features * self.pred_len), bias=True)

    def forward(self, x, *_, **__):

        # Reshape input
        outputs = x.reshape(x.shape[0], -1)

        # Pass through MLP
        for layer in self.mlp:
            outputs = layer(outputs)

        # Project
        outputs = self.projection(outputs)

        # Reshape to correct output
        outputs = outputs.view(outputs.shape[0], self.pred_len, self.output_features)

        if self.output_attention:
            return outputs, None
        else:
            return outputs

