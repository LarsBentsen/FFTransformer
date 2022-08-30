import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, d_model, num_layers, dropout=0., output_hidden=True):
        super(Encoder, self).__init__()

        self.d_model = d_model
        self.num_layers = num_layers
        self.output_hidden = output_hidden

        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=num_layers,
                            dropout=dropout, batch_first=True)

    def forward(self, x, **_):
        lstm_out, self.hidden = self.lstm(x)    # Assumes input as [B, L, d]

        if self.output_hidden:
            return lstm_out, self.hidden
        else:
            return lstm_out, None


class Decoder(nn.Module):
    def __init__(self, output_size, d_model, num_layers, dropout=0.):
        super(Decoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size=output_size, hidden_size=d_model, num_layers=num_layers,
                            dropout=dropout, batch_first=True)
        self.linear = nn.Linear(d_model, output_size)

    def forward(self, x, encoder_hidden_states, **_):
        lstm_out, self.hidden = self.lstm(x.unsqueeze(1), encoder_hidden_states)
        output = self.linear(lstm_out.squeeze(1))

        return output, self.hidden