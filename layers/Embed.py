import torch
import torch.nn as nn
import math


class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model, kernel_size=3, spatial=False):
        super(TokenEmbedding, self).__init__()
        assert torch.__version__ >= '1.5.0'
        padding = kernel_size - 1
        self.spatial = spatial
        self.d_model = d_model
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model,
                                   kernel_size=kernel_size, padding=padding, padding_mode='zeros', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        B, L, d = x.shape[:3]
        if self.spatial:
            x = x.permute(0, 3, 1, 2).reshape(-1, L, d)
            x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)[:, :L, :]
            x = x.reshape(B, -1, L, self.d_model).permute(0, 2, 3, 1)
        else:
            x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)[:, :L, :]
        return x


class TokenEmbedding_edges(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding_edges, self).__init__()
        assert torch.__version__ >= '1.5.0'
        self.d_model = d_model
        self.tokenConv = nn.Linear(c_in, d_model, bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')

    def forward(self, x):
        x = self.tokenConv(x)
        return x


class FixedEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(FixedEmbedding, self).__init__()

        w = torch.zeros(c_in, d_model).float()
        w.require_grad = False

        position = torch.arange(0, c_in).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        w[:, 0::2] = torch.sin(position * div_term)
        w[:, 1::2] = torch.cos(position * div_term)

        self.emb = nn.Embedding(c_in, d_model)
        self.emb.weight = nn.Parameter(w, requires_grad=False)

    def forward(self, x):
        return self.emb(x).detach()


class TemporalEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='fixed', freq='h'):
        super(TemporalEmbedding, self).__init__()

        minute_size = 6
        hour_size = 24
        weekday_size = 7
        day_size = 32
        month_size = 13

        Embed = FixedEmbedding if embed_type == 'fixed' else nn.Embedding
        if freq == 't' or freq == '10min':
            self.minute_embed = Embed(minute_size, d_model)
        self.hour_embed = Embed(hour_size, d_model)
        self.weekday_embed = Embed(weekday_size, d_model)
        self.day_embed = Embed(day_size, d_model)
        self.month_embed = Embed(month_size, d_model)

    def forward(self, x):
        x = x.long()

        minute_x = self.minute_embed(x[:, :, 4]) if hasattr(self, 'minute_embed') else 0.
        hour_x = self.hour_embed(x[:, :, 3])
        weekday_x = self.weekday_embed(x[:, :, 2])
        day_x = self.day_embed(x[:, :, 1])
        month_x = self.month_embed(x[:, :, 0])

        return hour_x + weekday_x + day_x + month_x + minute_x


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model, embed_type='timeF', freq='h'):
        super(TimeFeatureEmbedding, self).__init__()

        freq_map = {'h': 4, 't': 5, 's': 6, 'm': 1, 'a': 1, 'w': 2, 'd': 3, 'b': 3, '10min': 5}
        d_inp = freq_map[freq]
        self.embed = nn.Linear(d_inp, d_model, bias=False)

    def forward(self, x):
        return self.embed(x)


class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, kernel_size=3,
                 spatial=False, temp_embed=True, d_pos=None, pos_embed=True):
        super(DataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model, kernel_size=kernel_size, spatial=spatial)
        self.d_model = d_model
        if d_pos is None:
            self.d_pos = d_model
        else:
            self.d_pos = d_pos
        self.position_embedding = PositionalEmbedding(d_model=self.d_pos) if pos_embed else None
        if temp_embed:
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                        freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq)
        else:
            self.temporal_embedding = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark, n_node=None):
        val_embed = self.value_embedding(x)
        temp_embed = self.temporal_embedding(x_mark) if self.temporal_embedding is not None else None
        pos_embed = self.position_embedding(x) if self.position_embedding is not None else None
        if self.d_pos != self.d_model and pos_embed is not None:
            pos_embed = pos_embed.repeat_interleave(2, dim=-1)
        if temp_embed is not None:
            if not (len(val_embed.shape) == len(temp_embed.shape)):  # == len(pos_embed.shape)
                temp_embed = torch.unsqueeze(temp_embed, -1)
                pos_embed = torch.unsqueeze(pos_embed, -1) if pos_embed is not None else None
        if n_node is not None and temp_embed is not None:
            temp_embed = torch.repeat_interleave(temp_embed, n_node, 0)
        if pos_embed is not None:
            x = val_embed + temp_embed + pos_embed if temp_embed is not None else val_embed + pos_embed
        else:
            x = val_embed + temp_embed if temp_embed is not None else val_embed
        return self.dropout(x)


class EdgeDataEmbedding(nn.Module):
    def __init__(self, d_model, dropout=0.1, edge_feats=2):
        super(EdgeDataEmbedding, self).__init__()

        self.value_embedding_edges = TokenEmbedding_edges(c_in=edge_feats, d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.value_embedding_edges(x)

        return self.dropout(x)


class GraphDataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1, kernel_size=3,
                 temp_embed=True, edge_feats=2, pos_enc=True):
        super(GraphDataEmbedding, self).__init__()

        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model, kernel_size=kernel_size)
        self.value_embedding_edges = TokenEmbedding_edges(c_in=edge_feats, d_model=d_model)
        self.pos_enc = pos_enc
        if self.pos_enc:
            self.position_embedding = PositionalEmbedding(d_model=d_model)
        if temp_embed:
            self.temporal_embedding = TemporalEmbedding(d_model=d_model, embed_type=embed_type,
                                                        freq=freq) if embed_type != 'timeF' else TimeFeatureEmbedding(
                d_model=d_model, embed_type=embed_type, freq=freq)
        else:
            self.temporal_embedding = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x_node = x.nodes
        x_edges = x.edges
        val_embed = self.value_embedding(x_node)
        temp_embed = self.temporal_embedding(x_mark) if self.temporal_embedding is not None else None
        if self.pos_enc:
            pos_embed = self.position_embedding(x_node)
        else:
            pos_embed = None
        x_edges = self.value_embedding_edges(x_edges)
        if temp_embed is not None:
            if pos_embed is None:
                if not (len(val_embed.shape) == len(temp_embed.shape)):
                    temp_embed = torch.unsqueeze(temp_embed, -1)
            else:
                if not (len(val_embed.shape) == len(temp_embed.shape) == len(pos_embed.shape)):
                    temp_embed = torch.unsqueeze(temp_embed, -1)
                    pos_embed = torch.unsqueeze(pos_embed, -1) if self.pos_enc else None

        temp_embed = torch.repeat_interleave(temp_embed, x.n_node, 0)
        if self.pos_enc:
            x_node = val_embed + temp_embed + pos_embed if temp_embed is not None else val_embed + pos_embed
        else:
            x_node = val_embed + temp_embed if temp_embed is not None else val_embed

        x = x.replace(nodes=self.dropout(x_node), edges=self.dropout(x_edges))
        return x
