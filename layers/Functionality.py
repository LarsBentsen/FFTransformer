import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPLayer(nn.Module):
    def __init__(self, d_model, d_ff, kernel_size=1, dropout=0., activation='relu', res_con=True):
        super(MLPLayer, self).__init__()
        self.kernel_size = kernel_size
        if self.kernel_size != 1:
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=kernel_size, padding=(kernel_size-1))
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=kernel_size, padding=(kernel_size-1))
        else:
            self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=kernel_size)
            self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=kernel_size)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.res_con = res_con

    def forward(self, x):
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        if self.kernel_size != 1:
            y = y[..., 0:-(self.kernel_size - 1)]
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        if self.kernel_size != 1:
            y = y[:, 0:-(self.kernel_size - 1), :]
        if self.res_con:
            return self.norm2(x + y)
        else:
            return y


class ConvLayer(nn.Module):
    def __init__(self, c_in):
        super(ConvLayer, self).__init__()
        self.downConv = nn.Conv1d(in_channels=c_in,
                                  out_channels=c_in,
                                  kernel_size=3,
                                  padding=2,
                                  padding_mode='circular')
        self.norm = nn.BatchNorm1d(c_in)
        self.activation = nn.ELU()
        self.maxPool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.downConv(x.permute(0, 2, 1))
        x = self.norm(x)
        x = self.activation(x)
        x = self.maxPool(x)
        x = x.transpose(1, 2)
        return x