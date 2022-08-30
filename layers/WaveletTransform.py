import torch
from pytorch_wavelets import DWT1DForward, DWT1DInverse


def get_wt(x, num_decomp=4, sep_out=True):
    dwt_for = DWT1DForward(J=num_decomp, mode='symmetric', wave='db4').to(x.device)
    dwt_inv = DWT1DInverse(mode='symmetric', wave='db4').to(x.device)
    approx, detail = dwt_for(x.permute(0, 2, 1))     # [A3, [D0, D1, D2, D3]]
    coefs = [approx, *detail[::-1]]
    coefs_res = []
    sizes = []
    additional_zeros = []
    for coef in coefs[::-1][:-1]:
        params = [torch.zeros_like(coef).to(x.device), coef]
        if len(sizes) != 0:
            additional_zeros = [torch.zeros(s).to(x.device) for s in sizes[::-1]]
            params += additional_zeros
        cr = dwt_inv((params[0], params[1:][::-1]))
        sizes.append(coef.shape)
        coefs_res.append(cr)
    params = [coefs[0], torch.zeros_like(coefs[0]).to(x.device)] + additional_zeros
    cr = dwt_inv((params[0], params[1:][::-1]))
    coefs_res.append(cr)
    coefs_res = coefs_res[::-1]
    x_freq = torch.stack(coefs_res[1:], -1)[:, :, :x.shape[1], :]
    x_trend = coefs_res[0][..., None][:, :, :x.shape[1], :]

    x_freq = x_freq.permute(0, 2, 1, 3).reshape(*x.shape[:2], -1)       # Concatenate all the series
    x_trend = x_trend.permute(0, 2, 1, 3).reshape(*x.shape[:2], -1)

    if sep_out:
        return x_freq, x_trend
    else:
        return torch.cat([x_freq, x_trend], -1)
