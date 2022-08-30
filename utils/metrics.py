import numpy as np


def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))


def CORR(pred, true):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / d).mean(-1)


def MAE(pred, true, return_mean=True):
    _logits = np.abs(pred - true)
    if return_mean:
        return np.mean(_logits)
    else:
        return _logits


def MSE(pred, true, return_mean=True):
    _logits = (pred - true) ** 2
    if return_mean:
        return np.mean(_logits)
    else:
        return _logits


def RMSE(pred, true, return_mean=True):
    return np.sqrt(MSE(pred, true, return_mean=return_mean))


def MAPE(pred, true, eps=1e-07, return_mean=True):
    _logits = np.abs((pred - true) / (true + eps))
    if return_mean:
        return np.mean(_logits)
    else:
        return _logits


def MSPE(pred, true, eps=1e-07, return_mean=True):
    _logits = np.square((pred - true) / (true + eps))
    if return_mean:
        return np.mean(_logits)
    else:
        return _logits


def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)

    return mae, mse, rmse, mape, mspe
