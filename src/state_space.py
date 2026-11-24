"""Local linear trend with a retained initial level for endpoint-change inference."""

import numpy as np


def state_trend(days, weights, sigma, robust=True, process_sd=0.004, cutoff=2.5):
    days, weights = np.asarray(days, float), np.asarray(weights, float)
    # [current level, velocity, initial level]. Start with a diffuse level prior,
    # then assimilate the first observation exactly once.
    state = np.array([weights[0], 0.0, weights[0]])
    cov = np.array([[100.0, 0.0, 100.0], [0.0, 0.12**2, 0.0], [100.0, 0.0, 100.0]])
    for dt, y in zip(np.diff(days, prepend=days[0]), weights):
        F = np.array([[1.0, dt, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        Q = process_sd**2 * np.array(
            [[dt**3 / 3, dt**2 / 2, 0.0], [dt**2 / 2, dt, 0.0], [0.0, 0.0, 0.0]]
        )
        state = F @ state
        cov = F @ cov @ F.T + Q
        residual = y - state[0]
        variance = sigma**2
        if robust:
            z = abs(residual) / np.sqrt(cov[0, 0] + variance)
            variance *= max(1.0, z / cutoff) ** 2
        gain = cov[:, 0] / (cov[0, 0] + variance)
        state += gain * residual
        cov -= np.outer(gain, cov[0, :])
        cov = (cov + cov.T) / 2
    contrast = np.array([1.0, 0.0, -1.0]) / (days[-1] - days[0])
    return (
        float(contrast @ state),
        float(np.sqrt(max(0, contrast @ cov @ contrast))),
        float(state[2]),
    )
