import torch.nn as nn


class Model(nn.Module):
    """
    Persistence model
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.output_size = configs.c_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, **_):
        #     x_enc and x_mark_enc are not used, but kept to keep the signatures similar for spatial and
        #     non-spatial models.

        outputs = x_dec.nodes[:, -1:, -self.output_size:]
        outputs = outputs.repeat(1, self.pred_len, 1)

        return outputs
