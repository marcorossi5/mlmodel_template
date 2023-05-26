from typing import Dict

import mlflow
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


METRICS_FN = {
    "mae": mae,
    "rmse": rmse,
}


def compute_metrics(y_pred, y_true) -> Dict[str, float]:
    metrics = {k: fn(y_pred, y_true, verbose=True) for k, fn in METRICS_FN.items()}
    mlflow.log_metrics(metrics)
    return metrics
