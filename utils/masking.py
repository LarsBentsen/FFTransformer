import torch
import numpy as np


class TriangularCausalMask():
    def __init__(self, B, L, device="cpu"):
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class LogSparseMask():
    def __init__(self, B, Q_L, K_L, win_len=0, res_len=None, device="cpu"):
        mask_shape = [B, 1, Q_L, K_L]
        with torch.no_grad():
            if res_len is None:
                L_ls = K_L - 1 - win_len
                n_reps = 1
            else:
                n_reps = np.ceil(K_L / res_len)
                L_ls = res_len

            l_floor = int(np.log2(L_ls))
            indxs = np.array([int(2 ** (l_floor - i)) for i in range(l_floor + 1)])
            reps_array = np.expand_dims(np.arange(n_reps, dtype='int')*L_ls, 1)
            indxs = (indxs + reps_array + win_len).flatten()
            indxs = indxs[indxs < (K_L - 1)]
            my_mask = np.ones(K_L, dtype='int')
            my_mask[indxs] = 0
            my_mask[:(win_len + 1)] = 0
            my_mask = np.concatenate([np.flip(my_mask[1:]), my_mask])
            my_mask = np.array([my_mask[(K_L - i):(K_L * 2 - i)] for i in range(1, Q_L + 1)], dtype='bool')
            self._mask = torch.from_numpy(my_mask).to(device).unsqueeze(0).unsqueeze(1).expand(mask_shape)

    @property
    def mask(self):
        return self._mask


class ProbMask():
    def __init__(self, B, H, L_Q, index, scores, L_K, top_keys=False, device="cpu"):
        _mask = torch.ones(L_Q, L_K, dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L_Q, L_K)
        if top_keys:
            indicator = _mask_ex[torch.arange(B)[:, None, None, None],
                        torch.arange(H)[None, :, None, None],
                        torch.arange(L_Q)[None, None, :, None],
                        index[:, :, None,:]].to(device)
        else:
            indicator = _mask_ex[torch.arange(B)[:, None, None],
                        torch.arange(H)[None, :, None],
                        index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask


class ProbMask_OLD():
    def __init__(self, B, H, L, index, scores, device="cpu"):
        _mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).to(device).triu(1)
        _mask_ex = _mask[None, None, :].expand(B, H, L, scores.shape[-1])
        indicator = _mask_ex[torch.arange(B)[:, None, None],
                    torch.arange(H)[None, :, None],
                    index, :].to(device)
        self._mask = indicator.view(scores.shape).to(device)

    @property
    def mask(self):
        return self._mask
