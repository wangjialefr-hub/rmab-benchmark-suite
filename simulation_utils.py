"""Integer arm counts and LP-policy completion shared by the simulators."""

import numpy as np


SIMULATION_VERSION = "integer-actions-ftva-v2"
TOLERANCE = 1e-7


def integer_budget(alpha, N):
    """Use floor(alpha*N), snapping floating-point near-integers first."""
    if not isinstance(N, (int, np.integer)) or isinstance(N, bool) or N <= 0:
        raise ValueError("N must be a positive integer.")
    if not np.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0, 1].")
    target = float(alpha) * N
    nearest = round(target)
    return int(nearest if abs(target - nearest) < TOLERANCE else np.floor(target))


def state_counts(distribution, N):
    """Largest-remainder rounding of a distribution to exactly N arm states.

    Exact empirical counts are preserved. Ties go to the lower state index.
    """
    integer_budget(0, N)
    x = np.asarray(distribution, dtype=float)
    if (x.ndim != 1 or not np.all(np.isfinite(x)) or np.any(x < -TOLERANCE)
            or abs(x.sum() - 1) > TOLERANCE):
        raise ValueError("State distribution must be finite, nonnegative and sum to 1.")
    x = np.maximum(x, 0)
    desired = N * x / x.sum()
    counts = np.floor(desired).astype(int)
    remaining = int(N - counts.sum())
    order = np.argsort(-(desired - counts), kind="stable")
    counts[order[:remaining]] += 1
    return counts


def integer_action_counts(y, N, budget=None):
    """Round an S-by-2 allocation, preserving state counts and total budget.

    Each state first receives floor(N*y[s,1]) active arms; remaining activations
    go to the largest remainders. Reward and transitions must use these SAME
    counts. This is deterministic rounding, not independent Bernoulli sampling.
    """
    y = np.asarray(y, dtype=float)
    if (y.ndim != 2 or y.shape[1] != 2 or not np.all(np.isfinite(y))
            or np.any(y < -TOLERANCE)):
        raise ValueError("Action allocation must be a finite nonnegative S-by-2 array.")
    counts = state_counts(y.sum(axis=1), N)
    if budget is None:
        budget = integer_budget(float(np.clip(y[:, 1].sum(), 0, 1)), N)
    if not isinstance(budget, (int, np.integer)) or not 0 <= budget <= N:
        raise ValueError("Active budget must be an integer between 0 and N.")
    if abs(N * y[:, 1].sum() - budget) > 1 + N * TOLERANCE:
        raise ValueError("Proposed action allocation does not match the active budget.")
    desired = np.clip(N * y[:, 1], 0, counts)
    active = np.floor(desired).astype(int)
    remaining = int(budget - active.sum())
    if remaining > 0:
        candidates = np.flatnonzero(active < counts)
        order = candidates[np.argsort(-(desired - active)[candidates], kind="stable")]
        if remaining > len(order):
            raise ValueError("Cannot round the proposed allocation within one arm per state.")
        active[order[:remaining]] += 1
    elif remaining < 0:
        candidates = np.flatnonzero(active > 0)
        order = candidates[np.argsort((desired - active)[candidates], kind="stable")]
        if -remaining > len(order):
            raise ValueError("Cannot round the proposed allocation within one arm per state.")
        active[order[:-remaining]] -= 1
    return np.column_stack((counts - active, active))


def activation_probabilities(y):
    """Complete an LP policy with probability 1/2 outside its support.

    Hong et al., arXiv:2306.00196, equation (8). Zero-occupancy states still
    need a defined action distribution even though their stationary mass is 0.
    """
    y = np.asarray(y, dtype=float)
    if (y.ndim != 2 or y.shape[1] != 2 or not np.all(np.isfinite(y))
            or np.any(y < -TOLERANCE)):
        raise ValueError("Invalid LP occupation measure.")
    y = np.maximum(y, 0)
    mass = y.sum(axis=1)
    return np.divide(y[:, 1], mass, out=np.full_like(mass, 0.5), where=mass > 0)
