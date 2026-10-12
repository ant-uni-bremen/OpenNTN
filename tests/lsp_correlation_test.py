# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the correlations of the large scale parameters (TR 38.901 V16.1.0, clause
# 7.5, step 4, with the procedure of clause 3.3.1 of WINNER II D1.1.2, its reference
# [14]) against values computed in the test from the tables of TR 38.811 V15.4.0.
#
# Step 4 generates the LSPs "taking into account cross correlation according to Table
# 7.5-6 and using the procedure described in clause 3.3.1 of [14] with the square root
# matrix C_MxM(0) being generated using the Cholesky decomposition and the following
# order of the large scale parameter vector: s_M = [s_SF, s_K, s_DS, s_ASD, s_ASA,
# s_ZSD, s_ZSA]^T". Clause 3.3.1 of [14] first filters one spatially correlated field
# per LSP, with the correlation exp(-d/d_X) of the correlation distance d_X, and then
# applies the square root of the cross-correlation matrix per link (its eqs. (3.12)
# and (3.13)). With M the Cholesky factor of the cross-correlation matrix C in the
# order s_M, the LSP X of one link then has the correlations C with the other LSPs of
# the link, and the correlation sum_k M_Xk^2 exp(-d/d_k) with the same LSP of a UT at
# the distance d in the same state. Step 4 then limits ASA to 104 degrees and ZSA to 52
# degrees ("ASA = min(ASA, 104°)", "ZSA = min(ZSA, 52°)"), which the expected
# correlations of the logarithms take into account.
import json
import math
import os

import numpy as np
import pytest
import torch
from sionna.phy import config

from openntn import Antenna, DenseUrban
from openntn.utils import gen_single_sector_topology
from statistical_checks import assert_correlation

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_tables",
                       "tr38811_v15.4.0.json"), encoding="ascii") as _f:
    TABLES = json.load(_f)["tables"]

# Order s_M of TR 38.901 V16.1.0, clause 7.5, step 4
STANDARD_ORDER = ["SF", "K", "DS", "ASD", "ASA", "ZSD", "ZSA"]
ELEVATION_ANGLE = 30.0
NUM_DROPS = 20000
# Limits of step 4 [deg]
CAPS = {"ASA": 104.0, "ZSA": 52.0}
_NODES, _WEIGHTS = np.polynomial.legendre.leggauss(400)


def _phi(x):
    return np.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _cdf(x):
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def _capped_moments(a):
    """Mean and second moment of min(Z, a) for a standard normal Z."""
    if math.isinf(a):
        return 0.0, 1.0
    tail = 1.0 - float(_cdf(np.array(a)))
    mean = -float(_phi(a)) + a * tail
    second = float(_cdf(np.array(a))) - a * float(_phi(a)) + a * a * tail
    return mean, second


def _capped_correlation(rho, a, b):
    """Correlation of min(Z1, a) and min(Z2, b) for standard normal Z1, Z2 with the
    correlation rho (a, b: the limits in standard units, inf for no limit). The inner
    expectation over Z2 given Z1 = u is in closed form; the outer integral over u is
    split at a and evaluated with Gauss-Legendre quadrature."""
    s = math.sqrt(max(1.0 - rho * rho, 0.0))

    def inner(u):
        m = rho * u
        if math.isinf(b):
            return m
        if s == 0.0:
            return np.minimum(m, b)
        z = (m - b) / s
        return m - ((m - b) * _cdf(z) + s * _phi(z))

    def integral(lo, hi, g):
        u = 0.5 * (hi - lo) * _NODES + 0.5 * (hi + lo)
        return 0.5 * (hi - lo) * float(np.sum(_WEIGHTS * g(u) * inner(u) * _phi(u)))

    upper = 12.0
    if math.isinf(a) or a >= upper:
        cross = integral(-upper, upper, lambda u: u)
    else:
        cross = integral(-upper, a, lambda u: u) + integral(a, upper, lambda u: a + 0 * u)
    mean_a, second_a = _capped_moments(a)
    mean_b, second_b = _capped_moments(b)
    return (cross - mean_a * mean_b) / math.sqrt(
        (second_a - mean_a ** 2) * (second_b - mean_b ** 2))


def _table_correlations(table, los, satellite):
    """Cross-correlation matrix in the order s_M and the correlation distances [m] of
    Table 6.7.2-1a (LOS) or 6.7.2-2a (NLOS), dense urban, S band, at 30 degrees. For a
    satellite the cross-correlations of ASD and ZSD are 0 in both directions: NOTE 8 of
    the tables makes these spreads zero at the satellite, and the scenario applies it
    (get_param), with zero cross-correlations; the parameter files hold the tables as
    printed. NLOS links have no K-factor (the tables give N/A)."""
    rows = TABLES[table]["rows"]
    k = int(ELEVATION_ANGLE / 10) - 1
    n = len(STANDARD_ORDER)
    c = np.eye(n)
    for key, values in rows.items():
        if not key.startswith("corr_") or key.startswith("corr_dist_"):
            continue
        a, b = key[len("corr_"):].split("_vs_")
        a, b = ("K" if a == "Κ" else a), ("K" if b == "Κ" else b)
        if values[k] == "N/A":
            continue
        if satellite and ({a, b} & {"ASD", "ZSD"}):
            continue
        i, j = STANDARD_ORDER.index(a), STANDARD_ORDER.index(b)
        c[i, j] = c[j, i] = float(values[k])
    distances = np.array([float(rows["corr_dist_" + x][k]) if rows["corr_dist_" + x][k]
                          != "N/A" else 1.0 for x in STANDARD_ORDER])
    used = [x for x in STANDARD_ORDER
            if not (satellite and x in ("ASD", "ZSD")) and not (not los and x == "K")]
    # The limits of ASA and ZSA in units of the standard deviation of lg X
    limits = {x: math.inf for x in STANDARD_ORDER}
    for x, cap in CAPS.items():
        limits[x] = ((math.log10(cap) - float(rows["mu_lg" + x][k]))
                     / float(rows["sigma_lg" + x][k]))
    return c, distances, used, limits


def _drop(direction, los, distance):
    """LSPs of NUM_DROPS drops of two UTs at the given horizontal distance, both in the
    forced link state, dense urban, S band, 30 degrees."""
    antenna = Antenna(polarization="single", polarization_type="V",
                      antenna_pattern="omni", carrier_frequency=2.2e9)
    model = DenseUrban(carrier_frequency=2.2e9, ut_array=antenna, bs_array=antenna,
                       direction=direction, elevation_angle=ELEVATION_ANGLE,
                       enable_pathloss=False, enable_shadow_fading=False)
    topology = list(gen_single_sector_topology(batch_size=NUM_DROPS, num_ut=2,
                                               scenario="dur",
                                               elevation_angle=ELEVATION_ANGLE,
                                               bs_height=600000.0))
    ut_loc = topology[0].clone()
    ut_loc[:, 1, :] = ut_loc[:, 0, :] + torch.tensor([distance, 0.0, 0.0],
                                                     dtype=ut_loc.dtype,
                                                     device=ut_loc.device)
    topology[0] = ut_loc
    topology[5] = torch.zeros_like(topology[5])        # outdoor
    model.set_topology(*topology, los=los)
    lsp = model._lsp_sampler()
    return {"SF": lsp.sf, "K": lsp.k_factor, "DS": lsp.ds, "ASA": lsp.asa,
            "ZSA": lsp.zsa, "ASD": lsp.asd, "ZSD": lsp.zsd}


def _log_lsp(values, ut):
    return np.log10(values[:, 0, ut].cpu().numpy().astype(np.float64))


@pytest.mark.parametrize("los,table", [(True, "6.7.2-1a"), (False, "6.7.2-2a")])
@pytest.mark.parametrize("distance", [5.0, 20.0, 100.0])
def test_lsp_correlations(los, table, distance):
    config.precision = "double"
    c, distances, used, limits = _table_correlations(table, los, satellite=True)
    # Cholesky factor in the order s_M of the LSPs of the link
    index = [STANDARD_ORDER.index(x) for x in used]
    m = np.linalg.cholesky(c[np.ix_(index, index)])
    rho = np.exp(-distance / distances[index])
    lsp = _drop("downlink", los, distance)
    label = f"LOS {los}, {distance:g} m"
    for a, x in enumerate(used):
        # Per link: the cross-correlations of the table, for both UTs
        for b in range(a + 1, len(used)):
            y = used[b]
            for ut in (0, 1):
                expected = _capped_correlation(c[index[a], index[b]], limits[x],
                                               limits[y])
                assert_correlation(_log_lsp(lsp[x], ut), _log_lsp(lsp[y], ut),
                                   expected, f"{label}, UT {ut}: {x} vs {y}")
        # Across the two UTs: sum_k M_xk^2 exp(-d/d_k)
        expected = _capped_correlation(float(np.sum(m[a] ** 2 * rho)), limits[x],
                                       limits[x])
        assert_correlation(_log_lsp(lsp[x], 0), _log_lsp(lsp[x], 1), expected,
                           f"{label}: {x} of UT 0 vs {x} of UT 1")


def test_expected_values_of_the_procedure():
    # The closed form for LOS links at 20 m: SF keeps exp(-20/37), K exp(-20/12)
    # (corr(SF, K) is 0); ASA and ZSA mix the correlation functions of the LSPs before
    # them in s_M. The superseded order (cross-correlation first) gives exp(-20/15) =
    # 0.264 for both.
    c, distances, used, _ = _table_correlations("6.7.2-1a", True, satellite=True)
    index = [STANDARD_ORDER.index(x) for x in used]
    m = np.linalg.cholesky(c[np.ix_(index, index)])
    values = {x: float(np.sum(m[a] ** 2 * np.exp(-20.0 / distances[index])))
              for a, x in enumerate(used)}
    assert math.isclose(values["SF"], math.exp(-20.0 / 37.0), rel_tol=1e-12)
    assert math.isclose(values["K"], math.exp(-20.0 / 12.0), rel_tol=1e-12)
    assert round(values["DS"], 3) == 0.473
    assert round(values["ASA"], 3) == 0.440
    assert round(values["ZSA"], 3) == 0.505


@pytest.mark.parametrize("rho", [-0.8, -0.2, 0.0, 0.3, 0.9])
def test_capped_correlation_against_monte_carlo(rho):
    # A check of the test itself: the quadrature against a large sample.
    rng = np.random.default_rng(1)
    z1 = rng.standard_normal(2_000_000)
    z2 = rho * z1 + math.sqrt(1.0 - rho * rho) * rng.standard_normal(z1.size)
    for a, b in ((math.inf, math.inf), (1.0, math.inf), (0.5, -0.3)):
        sample = float(np.corrcoef(np.minimum(z1, a), np.minimum(z2, b))[0, 1])
        assert abs(_capped_correlation(rho, a, b) - sample) < 3e-3, (rho, a, b)
