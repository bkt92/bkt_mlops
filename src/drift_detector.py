import numpy as np
from libs.ks_2samp import ks_2samp
import asyncio

def ks_drift_detect(x_ref: np.ndarray, x: np.ndarray, p_val: np.float32 = 0.05):
    """
    Check data drift by using K-S score (code from alibi_detect).

    Parameters
    ----------
    x_ref
        Reference instances to compare distribution with.
    x
        Batch of instances.

    p_val
        Threshold for detection.

    Returns
    -------
    is drift: 0 no drift or 1 drift.
    """
    x = x.reshape(x.shape[0], -1)
    x_ref = x_ref.reshape(x_ref.shape[0], -1)
    n_features = x_ref.reshape(x_ref.shape[0], -1).shape[-1]
    p_vals = np.zeros(n_features, dtype=np.float32)
    dist = np.zeros_like(p_vals)
    for f in range(n_features):
        dist[f], p_vals[f] = ks_2samp(x_ref[:, f], x[:, f], alternative= 'two-sided')
    threshold = p_val / n_features
    drift_pred = int((p_vals < threshold).any()) 
    return drift_pred, p_vals

async def ks_2samp_async(x1, x2):
    return ks_2samp(x1, x2, alternative= 'two-sided')[1]

async def ks_drift_detect_async(x_ref: np.ndarray, x: np.ndarray, p_val: np.float32 = 0.05):
    x = x.reshape(x.shape[0], -1)
    x_ref = x_ref.reshape(x_ref.shape[0], -1)
    n_features = x_ref.reshape(x_ref.shape[0], -1).shape[-1]
    tasks = [ks_2samp_async(x_ref[:, i], x[:, i]) for i in range(n_features)]
    p_vals = np.array(await asyncio.gather(*tasks), dtype=np.float32)
    threshold = p_val / n_features
    drift_pred = int((p_vals < threshold).any())
    return drift_pred