# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the spatial correlation of the large scale parameters with UTs at the same
# location.
#
# TR 38.901 V16.1.0, clause 7.5, step 4, generates the large scale parameters "using
# the procedure described in clause 3.3.1 of [14]" (WINNER II D1.1.2). That procedure
# filters a grid of independent normal numbers with exp(-d/D_X) (its eq. (3.12)) and
# keeps the values at the grid nodes of the UT locations, so each parameter X of two
# UTs in the same state has the correlation exp(-d/D_X), with d their distance in the
# horizontal plane and D_X the correlation distance of TR 38.811 V15.4.0, Tables
# 6.7.2-1a to 6.7.2-8b. UTs at the same location read the same grid node: their
# correlation is exp(0) = 1 and they get the same values. The models apply a square
# root F of the correlation matrix C of each parameter, F F^T = C, to independent
# normal draws, so that every UT has the specified distribution and UTs at one location
# share their draws. With two UTs at one location C is singular; F is then the Cholesky
# factor of C without the UTs that repeat an earlier UT, which take the row of the first
# UT they repeat.
import json
import os

import pytest
import torch
from sionna.phy import config

from openntn import Antenna, DenseUrban, lsp
from openntn.utils import gen_single_sector_topology
from statistical_checks import assert_std

# The large scale parameters in the order of their spatial factors in the models
NAMES = ("DS", "ASD", "ASA", "SF", "K", "ZSA", "ZSD")
LSPS = ("ds", "asd", "asa", "sf", "k_factor", "zsa", "zsd")
ELEVATION_ANGLE = 30
# UT 2 at the location of UT 1 and UT 3 7.6 m from it in x, closer than every
# correlation distance in LOS (12 m to 37 m in Table 6.7.2-1a)
OFFSETS = ((0.0, 0.0), (0.0, 0.0), (7.6, 0.0))
_MODELS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "openntn", "models")


def _parameters(state):
    # Dense urban, S band, downlink: Tables 6.7.2-1a (LOS) and 6.7.2-2a (NLOS)
    with open(os.path.join(_MODELS, f"Dense_Urban_{state}_S_band_DL.json"),
              encoding="utf-8") as f:
        return json.load(f)


LOS_PARAMETERS = _parameters("LOS")
NLOS_PARAMETERS = _parameters("NLOS")


def _correlation_distances(parameters):
    return [parameters[f"corrDist{name}_{ELEVATION_ANGLE}"] for name in NAMES]


def _model():
    antenna = Antenna(polarization="single", polarization_type="V",
                      antenna_pattern="omni", carrier_frequency=2.2e9)
    return DenseUrban(carrier_frequency=2.2e9, ut_array=antenna, bs_array=antenna,
                      direction="downlink", elevation_angle=float(ELEVATION_ANGLE))


def _topology(batch_size=2):
    return list(gen_single_sector_topology(batch_size=batch_size, num_ut=3,
                                           scenario="dur",
                                           elevation_angle=float(ELEVATION_ANGLE),
                                           bs_height=600e3))


def _colocated(ut_loc, examples=slice(0, 1)):
    """The UTs of the given batch examples moved to OFFSETS from their first UT."""
    ut_loc = ut_loc.clone()
    offsets = torch.tensor(OFFSETS, dtype=ut_loc.dtype)
    ut_loc[examples, :, :2] = ut_loc[examples, :1, :2] + offsets
    return ut_loc


def _built_correlation(points, states, dtype):
    """The correlation matrices of the seven large scale parameters, [7, n, n], of UTs
    at the given locations and states (LOS `True`): exp(-d/D_X) with the correlation
    distance of their state for two UTs in the same state, 0 for different states."""
    points = torch.tensor(points, dtype=dtype)
    distance = torch.linalg.vector_norm(points.unsqueeze(1) - points.unsqueeze(0),
                                        dim=-1)
    states = torch.tensor(states)
    same = (states.unsqueeze(1) == states.unsqueeze(0)).to(dtype)
    matrices = []
    for los_distance, nlos_distance in zip(_correlation_distances(LOS_PARAMETERS),
                                           _correlation_distances(NLOS_PARAMETERS)):
        scale = torch.where(states, -1.0/torch.tensor(los_distance, dtype=dtype),
                            -1.0/torch.tensor(nlos_distance, dtype=dtype))
        matrices.append(torch.exp(distance*scale.unsqueeze(-1))*same)
    return torch.stack(matrices)


def _bits(tensor):
    width = {torch.float64: torch.int64, torch.float32: torch.int32}[tensor.dtype]
    return tensor.contiguous().view(width)


def _equal_bits(a, b):
    return a.shape == b.shape and torch.equal(_bits(a), _bits(b))


def test_factor_with_colocated_uts():
    # The stored factor F of each large scale parameter in the first batch example is
    # lower triangular, F F^T equals the correlation computed here from the 2D
    # distances and the correlation distances of Table 6.7.2-1a, and the rows of the
    # two UTs at one location are equal, so that they get the same draws.
    config.precision = "double"
    model = _model()
    topology = _topology()
    ut_loc = _colocated(topology[0])
    model.set_topology(ut_loc, *topology[1:], los=True)
    factors = model._lsp_sampler._spatial_lsp_correlation_matrix_sqrt[0, 0]
    xy = ut_loc[0, :, :2]
    distance = torch.linalg.vector_norm(xy.unsqueeze(1) - xy.unsqueeze(0), dim=-1)
    failures = []
    for name, factor, correlation_distance in zip(
            NAMES, factors, _correlation_distances(LOS_PARAMETERS)):
        correlation = torch.exp(-distance/correlation_distance)
        product = factor @ factor.T
        largest = float((product - correlation).abs().max())
        equal_rows = _equal_bits(factor[0], factor[1])
        if (largest > 1e-12 or not torch.equal(factor, torch.tril(factor))
                or not equal_rows):
            diagonal = [round(x, 4) for x in torch.diagonal(product).tolist()]
            failures.append(f"{name}: |F F^T - C| up to {largest:.3g}, diagonal of "
                            f"F F^T {diagonal}, rows of UTs 1 and 2 equal {equal_rows}")
    assert not failures, "; ".join(failures)


def test_draws_with_colocated_uts():
    # 4000 batch examples with UT 2 at the location of UT 1 and UT 3 7.6 m away, LOS
    # forced: the seven large scale parameters of UTs 1 and 2 are equal in every draw,
    # and the logarithms of the delay spread and of the shadow fading of UT 3, which
    # step 4 does not limit, have the standard deviations sigma_lgDS of Table 6.7.2-1a
    # and sigma_SF/10 of Table 6.6.2-1 (the shadow fading in dB, as a logarithm) within
    # 4 standard errors.
    config.precision = "double"
    model = _model()
    topology = _topology(batch_size=4000)
    model.set_topology(_colocated(topology[0], slice(None)), *topology[1:], los=True)
    draws = model._lsp_sampler()
    for name in LSPS:
        values = getattr(draws, name)[:, 0]
        assert _equal_bits(values[:, 0], values[:, 1]), f"{name}: UTs 1 and 2 differ"
    failures = []
    for label, values, sigma in (
            ("lg DS of UT 3", draws.ds[:, 0, 2],
             LOS_PARAMETERS[f"sigmaDS_{ELEVATION_ANGLE}"]),
            ("lg SF of UT 3", draws.sf[:, 0, 2],
             LOS_PARAMETERS[f"sigmaSF_{ELEVATION_ANGLE}"]/10.0)):
        try:
            assert_std(torch.log10(values), sigma, label)
        except AssertionError as error:
            failures.append(str(error))
    assert not failures, "; ".join(failures)


@pytest.mark.parametrize("dtype,tolerance",
                         [(torch.float64, 1e-12), (torch.float32, 1e-6)])
def test_factor_of_built_matrices(dtype, tolerance):
    # Correlation matrices built here, factored by the function of the models. Three
    # UTs at one location (1, 3 and 6) and two at another (2 and 5), all LOS: F F^T
    # equals C, F is lower triangular, and the rows of the UTs at one location equal
    # that of the first of them.
    points = [(0.0, 0.0), (7.6, 0.0), (0.0, 0.0), (0.0, 9.0), (7.6, 0.0), (0.0, 0.0)]
    correlation = _built_correlation(points, [True]*6, dtype)
    factor = lsp._spatial_correlation_sqrt(correlation)
    product = factor @ factor.transpose(-1, -2)
    largest = float((product - correlation).abs().max())
    assert largest <= tolerance, f"F F^T differs from C by up to {largest:.3g}"
    assert torch.equal(factor, torch.tril(factor)), "not lower triangular"
    for ut, first in ((2, 0), (5, 0), (4, 1)):
        assert _equal_bits(factor[:, ut], factor[:, first]), f"UT {ut + 1}: row differs"
    # A correlation that rounds to 1 between neighbours only: UT 2 repeats UT 1 and
    # UT 3 repeats UT 2, so UT 3 takes the row of UT 1
    below = 1.0 - torch.finfo(dtype).eps/2.0
    chain = torch.tensor([[1.0, 1.0, below], [1.0, 1.0, 1.0], [below, 1.0, 1.0]],
                         dtype=dtype)
    factor = lsp._spatial_correlation_sqrt(chain)
    largest = float((factor @ factor.T - chain).abs().max())
    assert largest <= tolerance, f"chain: F F^T differs from C by up to {largest:.3g}"
    assert _equal_bits(factor[1], factor[0]) and _equal_bits(factor[2], factor[0]), \
        f"chain: {factor.tolist()}"
    # Without UTs that repeat an earlier UT the factor is that of cholesky_ex: two UTs
    # at one location in different states, whose correlation is 0, and UTs at
    # different locations
    for points, states in (([(0.0, 0.0), (0.0, 0.0), (7.6, 0.0)], [True, False, True]),
                           ([(0.0, 0.0), (5.0, 3.0), (-2.0, 8.0), (30.0, -4.0)],
                            [True, True, False, True])):
        correlation = _built_correlation(points, states, dtype)
        expected, _ = torch.linalg.cholesky_ex(correlation, check_errors=False)
        assert _equal_bits(lsp._spatial_correlation_sqrt(correlation), expected), \
            f"{points}, {states}: not the factor of cholesky_ex"
    # No UT at all: an empty factor, as cholesky_ex gives it
    empty = torch.zeros(7, 0, 0, dtype=dtype)
    assert lsp._spatial_correlation_sqrt(empty).shape == empty.shape, "no UT"


def test_gradients_with_colocated_uts():
    # With UT 2 at the location of UT 1, the large scale parameters stay
    # differentiable with respect to the UT locations: the gradients of their sums are
    # finite.
    config.precision = "double"
    model = _model()
    topology = _topology()
    ut_loc = _colocated(topology[0]).requires_grad_(True)
    model.set_topology(ut_loc, *topology[1:], los=True)
    not_finite = {}
    for name in LSPS:
        grad, = torch.autograd.grad(getattr(model._lsp, name).sum(), ut_loc,
                                    retain_graph=True)
        not_finite[name] = int((~torch.isfinite(grad)).sum())
    assert not any(not_finite.values()), \
        f"entries of the 18 UT coordinates with a gradient that is not finite: " \
        f"{not_finite}"


@pytest.mark.parametrize("los", [True, None])
def test_factor_without_colocated_uts(los):
    # The topology as generated, without UTs at one location, LOS forced and drawn with
    # the LOS probability: the stored factor equals that of cholesky_ex, bit for bit,
    # for the correlation computed here from the distances between the UTs of the
    # scenario and the correlation distances of Tables 6.7.2-1a and 6.7.2-2a.
    config.precision = "double"
    model = _model()
    model.set_topology(*_topology(), los=los)
    scenario = model._scenario
    states = scenario.los[:, 0]                                     # [batch, ut]
    distance = scenario.matrix_ut_distance_2d                       # [batch, ut, ut]
    same = (states.unsqueeze(2) == states.unsqueeze(1)).to(distance.dtype)
    factors = model._lsp_sampler._spatial_lsp_correlation_matrix_sqrt[:, 0]
    for name, factor, los_distance, nlos_distance in zip(
            NAMES, factors.unbind(1), _correlation_distances(LOS_PARAMETERS),
            _correlation_distances(NLOS_PARAMETERS)):
        scale = torch.where(states,
                            -1.0/torch.tensor(los_distance, dtype=distance.dtype),
                            -1.0/torch.tensor(nlos_distance, dtype=distance.dtype))
        correlation = torch.exp(distance*scale.unsqueeze(-1))*same
        expected, _ = torch.linalg.cholesky_ex(correlation, check_errors=False)
        assert _equal_bits(factor, expected), f"{name}: not the factor of cholesky_ex"
