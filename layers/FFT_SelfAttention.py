import torch
import torch.fft
import torch.nn as nn


#  Here we use ProbSparse attention, but this could be changed if desirable
class FFTAttention(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, scale=None, **_):
        super(FFTAttention, self).__init__()

        self.scale = scale

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.query_projection = nn.Conv1d(d_model, d_keys * n_heads, kernel_size=3, stride=3)
        self.key_projection = nn.Conv1d(d_model, d_keys * n_heads, kernel_size=3, stride=3)
        self.value_projection = nn.Conv1d(d_model, d_values * n_heads, kernel_size=3, stride=3)
        self.out_projection = nn.Linear(d_values * n_heads, d_model * 2)
        self.n_heads = n_heads
        self.attn = attention

    def forward(self, queries, keys, values):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries_fft = torch.fft.rfft(queries.permute(0, 2, 1))
        keys_fft = torch.fft.rfft(keys.permute(0, 2, 1))
        values_fft = torch.fft.rfft(values.permute(0, 2, 1))

        L_fft = queries_fft.shape[-1]
        S_fft = keys_fft.shape[-1]

        freqs_L = torch.fft.rfftfreq(L).unsqueeze(0).to(queries_fft.device)
        freqs_S = torch.fft.rfftfreq(S).unsqueeze(0).to(queries_fft.device)

        # (BS, L, D) --> (BS, 3L, D)    i.e. perform 1dConv with kernel=stride=3 to obtain (BS, L, D)
        # 3L corresponds to the real and imaginary components + the frequency values (as some sort of positional enc).
        queries_fft = torch.stack([queries_fft.real, queries_fft.imag, freqs_L.unsqueeze(0).expand(queries_fft.shape)], -1)
        queries_fft = queries_fft.reshape(B, queries_fft.shape[1], -1)
        queries_fft = self.query_projection(queries_fft).permute(0, 2, 1).view(B, L_fft, H, -1)

        keys_fft = torch.stack([keys_fft.real, keys_fft.imag, freqs_S.unsqueeze(0).expand(keys_fft.shape)], -1)
        keys_fft = keys_fft.reshape(B, keys_fft.shape[1], -1)
        keys_fft = self.key_projection(keys_fft).permute(0, 2, 1).view(B, S_fft, H, -1)

        values_fft = torch.stack([values_fft.real, values_fft.imag, freqs_S.unsqueeze(0).expand(values_fft.shape)], -1)
        values_fft = values_fft.reshape(B, values_fft.shape[1], -1)
        values_fft = self.value_projection(values_fft).permute(0, 2, 1).view(B, S_fft, H, -1)

        V, attn = self.attn(
            queries_fft, keys_fft, values_fft,
            attn_mask=None
        )
        V = V.transpose(2, 1)
        V = V.contiguous().view(B, L_fft, -1)

        V = self.out_projection(V)
        V = V.view(B, L_fft, -1, 2)

        V = torch.complex(V[..., 0], V[..., 1]).permute(0, 2, 1)

        V = torch.fft.irfft(V, n=L).permute(0, 2, 1)

        return V.contiguous(), attn

