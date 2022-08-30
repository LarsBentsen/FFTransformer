import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask, LogSparseMask


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, scale=None, attention_dropout=0.1, output_attention=False,
                 sparse_flag=False, win_len=0, res_len=None, fft_flag=False, **_):
        super(FullAttention, self).__init__()
        self.sparse_flag = sparse_flag
        self.win_len = win_len
        self.res_len = res_len
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.fft_flag = fft_flag

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape[:4]

        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

        if self.sparse_flag:
            sparse_mask = LogSparseMask(B, L, S, self.win_len, self.res_len, device=queries.device)
            if self.mask_flag:
                attn_mask._mask = attn_mask._mask.logical_or(sparse_mask._mask)
            else:
                attn_mask = sparse_mask

        if self.sparse_flag or self.mask_flag:
            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        if self.fft_flag:
            V = torch.einsum("bhls,bshdc->blhdc", A, values)
        else:
            V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)


class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False,
                 top_keys=False, context_zero=False, **_):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.top_keys = top_keys
        self.context_zero = context_zero

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        if self.top_keys:
            Q_expand = Q.unsqueeze(-2).expand(B, H, L_Q, L_K, E)  # Note that expanding does not allocate new memory
            index_sample = torch.randint(L_Q, (sample_k, L_K))  # real U = U_part(factor*ln(L_k))*L_q
            Q_sample = Q_expand[:, :, index_sample, torch.arange(L_K).unsqueeze(0), :]
            Q_K_sample = torch.einsum('bhsld, bhdlr -> bhslr', Q_sample, K.unsqueeze(-3).transpose(-3, -1)).squeeze()
        else:
            K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)    # Note that expanding does not allocate new memory
            index_sample = torch.randint(L_K, (L_Q, sample_k))  # real U = U_part(factor*ln(L_k))*L_q
            # Select sample_k number of samples from the keys
            K_sample = K_expand[:, :, torch.arange(L_Q).unsqueeze(1), index_sample, :]
            Q_K_sample = torch.matmul(Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        if self.top_keys:
            M = Q_K_sample.max(-2)[0] - torch.div(Q_K_sample.sum(-2), L_Q)
        else:
            M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        if self.top_keys:
            K_reduce = K[torch.arange(B)[:, None, None],
                       torch.arange(H)[None, :, None],
                       M_top, :]  # factor*ln(L_q)
            Q_K = torch.matmul(Q, K_reduce.transpose(-2, -1))  # factor*ln(L_q)*L_k
        else:
            Q_reduce = Q[torch.arange(B)[:, None, None],
                       torch.arange(H)[None, :, None],
                       M_top, :]  # factor*ln(L_q)
            Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.context_zero:
            if not self.mask_flag:
                V_sum = V.mean(dim=-2)
                contex = V_sum.unsqueeze(-2).expand(B, H, L_Q, V_sum.shape[-1]).clone()
            else:  # use mask
                assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
                contex = V.cumsum(dim=-2)
        else:
            contex = torch.zeros(B, H, L_Q, D).to(V.device)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, L_K=L_V, top_keys=self.top_keys, device=V.device)
            scores.masked_fill_(attn_mask.mask, -1e20)      # np.inf)
            no_indxs = torch.where(torch.all(attn_mask._mask, -1))

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        if self.top_keys:
            V_sub = V[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :]
            context_in = torch.matmul(attn, V_sub).type_as(context_in)
            if self.mask_flag:
                context_in[no_indxs] = V[no_indxs]
        else:
            context_in[torch.arange(B)[:, None, None],
            torch.arange(H)[None, :, None],
            index, :] = torch.matmul(attn, V).type_as(context_in)

        if self.output_attention:
            if self.top_keys:
                attns = torch.zeros([B, H, L_Q, L_V]).type_as(attn).to(attn.device)
                attns[torch.arange(B)[:, None, None, None],
                      torch.arange(H)[None, :, None, None],
                      torch.arange(L_Q)[None, None, :, None],
                      index[:,:, None,:]] = attn
            else:
                attns = (torch.ones([B, H, L_Q, L_V]) / L_V).type_as(attn).to(attn.device)
                attns[torch.arange(B)[:, None, None], torch.arange(H)[None, :, None], index, :] = attn
            return (context_in, attns)
        else:
            return (context_in, None)

    def forward(self, queries, keys, values, *_, **__):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        # Finding top keys instead of top queries
        if self.top_keys:
            u = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
            U_part = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

            U_part = U_part if U_part < L_Q else L_Q
            u = u if u < L_K else L_K
        else:
            U_part = self.factor * np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
            u = self.factor * np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

            U_part = U_part if U_part < L_K else L_K
            u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(context, values, scores_top, index, L_Q)

        return context.contiguous(), attn


# Convolutional Attention from the LogSparse Transformer
class LogSparseAttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, qk_ker, d_keys=None, d_values=None, v_conv=False, **_):
        super(LogSparseAttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.qk_ker = qk_ker
        self.query_projection = nn.Conv1d(d_model, d_keys * n_heads, self.qk_ker)
        self.key_projection = nn.Conv1d(d_model, d_keys * n_heads, self.qk_ker)
        self.v_conv = v_conv
        if v_conv:
            self.value_projection = nn.Conv1d(d_model, d_values * n_heads, self.qk_ker)
        else:
            self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, **_):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = nn.functional.pad(queries.permute(0, 2, 1), pad=(self.qk_ker - 1, 0))
        queries = self.query_projection(queries).permute(0, 2, 1).view(B, L, H, -1)

        keys = nn.functional.pad(keys.permute(0, 2, 1), pad=(self.qk_ker - 1, 0))
        keys = self.key_projection(keys).permute(0, 2, 1).view(B, S, H, -1)

        if self.v_conv:
            values = nn.functional.pad(values.permute(0, 2, 1), pad=(self.qk_ker - 1, 0))
            values = self.value_projection(values).permute(0, 2, 1).view(B, S, H, -1)
        else:
            values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )

        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None, output_size=None):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model) if output_size is None else nn.Linear(d_values * n_heads, output_size)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask, **_):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
