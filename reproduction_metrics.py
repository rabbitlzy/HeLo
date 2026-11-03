"""Six distribution metrics, evaluated on EVERY sample before subject averaging."""
import numpy as np
from scipy.special import xlogy

NAMES = ["Chebyshev", "Clark", "Canberra", "KL", "Cosine", "Intersection"]
PAPER_DMER_SD = np.array([0.0446, 0.3256, 0.8664, 0.0323, 0.9714, 0.9128])
PAPER_DMER_LOSO = np.array([0.0882, 0.6289, 1.7603, 0.1027, 0.9148, 0.8194])


def distribution_metrics(prediction, target):
    p, q = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.ndim != 2 or p.shape != q.shape or not len(p):
        raise ValueError("Expected matching nonempty sample-by-label matrices")
    if not np.isfinite(p).all() or not np.isfinite(q).all() or (p < 0).any() or (q < 0).any():
        raise ValueError("Distributions must be finite and nonnegative")
    if not np.allclose(p.sum(1), 1, atol=1e-5) or not np.allclose(q.sum(1), 1, atol=1e-5):
        raise ValueError("Distribution rows must sum to one")
    diff = np.abs(p - q)
    ratio = np.divide(diff, p + q, out=np.zeros_like(diff), where=(p + q) > 0)
    kl = np.sum(xlogy(q, q) - xlogy(q, p), axis=1)
    cosine = (p * q).sum(1) / (np.linalg.norm(p, axis=1) * np.linalg.norm(q, axis=1))
    return np.stack([diff.max(1), np.sqrt((ratio**2).sum(1)), ratio.sum(1), kl,
                     cosine, np.minimum(p, q).sum(1)], axis=1)
