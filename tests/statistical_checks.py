# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Statistical assertions shared by the tests that check sampled quantities.
#
# Each check compares an estimate with its expected value and accepts a deviation of up
# to Z standard errors of the estimate, where the standard error is computed from the
# expected distribution (the null hypothesis), not from the sample. With Z = 4 a single
# check is two-sided at a confidence level of 99.994 % (false-failure probability
# 6.3e-5). The suite contains a few hundred such checks, so for a new seed the
# probability that any of them fails by chance stays below about 2 %; with the fixed
# seeds of conftest.py the outcome is deterministic.
import math

import numpy as np

Z = 4.0


def _as_float64(samples):
    if hasattr(samples, "detach"):
        samples = samples.detach().cpu().numpy()
    return np.asarray(samples, dtype=np.float64).ravel()


def assert_mean(samples, expected_mean, sigma, label=""):
    """Sample mean of i.i.d. samples with known standard deviation ``sigma``."""
    x = _as_float64(samples)
    n = x.size
    assert n >= 2, f"{label}: too few samples ({n})"
    se = sigma / math.sqrt(n)
    mean = float(np.mean(x))
    assert abs(mean - expected_mean) <= Z * se, (
        f"{label}: mean {mean:.4f}, expected {expected_mean:.4f} "
        f"+- {Z * se:.4f} ({Z:g} standard errors, n = {n})")


def assert_std(samples, expected_sigma, label=""):
    """Sample standard deviation of i.i.d. normal samples.

    For normal samples the standard error of the sample standard deviation is
    approximately sigma / sqrt(2 (n - 1)).
    """
    x = _as_float64(samples)
    n = x.size
    assert n >= 2, f"{label}: too few samples ({n})"
    se = expected_sigma / math.sqrt(2.0 * (n - 1))
    std = float(np.std(x, ddof=1))
    assert abs(std - expected_sigma) <= Z * se, (
        f"{label}: std {std:.4f}, expected {expected_sigma:.4f} "
        f"+- {Z * se:.4f} ({Z:g} standard errors, n = {n})")


def assert_proportion(successes, n, expected_p, label=""):
    """Fraction of successes in ``n`` independent Bernoulli trials."""
    assert n >= 1, f"{label}: no trials"
    se = math.sqrt(expected_p * (1.0 - expected_p) / n)
    p = successes / n
    assert abs(p - expected_p) <= Z * se, (
        f"{label}: proportion {p:.4f}, expected {expected_p:.4f} "
        f"+- {Z * se:.4f} ({Z:g} standard errors, n = {n})")


def assert_uniform_mean(samples, low, high, label=""):
    """Sample mean of i.i.d. samples from the uniform distribution on [low, high]."""
    sigma = (high - low) / math.sqrt(12.0)
    assert_mean(samples, 0.5 * (low + high), sigma, label)
