from typing import Dict

import numpy as np
from .logger import logger


def mae(y_pred, y_true, verbose=False):
    metric = np.abs(y_pred - y_true).mean()
    if verbose:
        logger.info("mae: %.4f", metric)
    return metric


def mse(y_pred, y_true, verbose=False):
    metric = np.mean((y_pred - y_true) ** 2)
    if verbose:
        logger.info("mse: %.4f", metric)
    return metric


def rmse(y_pred, y_true, verbose=False):
    metric = np.sqrt(mse(y_pred, y_true))
    if verbose:
        logger.info("rmse: %.4f", metric)
    return metric


def tweedie_for_series(y, y_hat, p):
    a = y.values() * np.power(y_hat.values(), 1 - p) / (1 - p)
    b = np.power(y_hat.values(), 2 - p) / (2 - p)
    return np.mean(-a + b)


def tweedie_for_lists(y_pred, y_true, p):
    return [tweedie_for_series(y, y_hat, p) for y, y_hat in zip(y_pred, y_true)]


def tweedie(y_pred, y_true, p=1.5, verbose=False):
    if isinstance(y_pred, list) and isinstance(y_true, list):
        return tweedie_for_lists(y_pred, y_true, p)
    metric = tweedie_for_series(y_pred, y_true, p)
    if verbose:
        logger.info("tweedie: %.4f", metric)
    return metric
