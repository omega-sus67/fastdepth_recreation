"""Accumulate valid-pixel depth metrics with explicit invalid prediction handling."""
import numpy as np

class DepthMetrics:
    def __init__(self, epsilon=1e-6):
        if epsilon <= 0:
            raise ValueError('epsilon must be positive')
        self.epsilon = epsilon
        self.n = self.invalid = 0
        self.abs_sum = self.sq_sum = self.rel_sum = self.delta_sum = 0.0
        self.target_sum = self.target_sq_sum = 0.0

    def update(self, prediction, target):
        p = np.asarray(prediction, dtype=np.float64)
        d = np.asarray(target, dtype=np.float64)
        if p.shape != d.shape:
            raise ValueError('prediction and target shapes differ')
        mask = np.isfinite(d) & (d > 0)
        p, d = p[mask], d[mask]
        if not d.size:
            return
        if not np.isfinite(p).all():
            raise ValueError('nonfinite predictions at valid target pixels')
        self.invalid += int((p <= 0).sum())
        error = p - d
        ratio_prediction = np.maximum(p, self.epsilon)
        self.n += d.size
        self.abs_sum += float(np.abs(error).sum())
        self.sq_sum += float(np.square(error).sum())
        self.rel_sum += float((np.abs(error) / d).sum())
        self.delta_sum += int((np.maximum(ratio_prediction / d, d / ratio_prediction) < 1.25).sum())
        self.target_sum += float(d.sum())
        self.target_sq_sum += float(np.square(d).sum())

    def result(self):
        if not self.n:
            raise ValueError('no valid ground-truth pixels')
        denominator = self.target_sq_sum - self.target_sum ** 2 / self.n
        return {'valid_pixels': self.n, 'nonpositive_predictions': self.invalid,
                'rmse': (self.sq_sum / self.n) ** 0.5, 'mae': self.abs_sum / self.n,
                'rel': self.rel_sum / self.n, 'delta1': self.delta_sum / self.n,
                'r2': 1 - self.sq_sum / denominator if denominator > 0 else None,
                'aggregation': 'pooled valid pixels', 'ratio_epsilon': self.epsilon}
