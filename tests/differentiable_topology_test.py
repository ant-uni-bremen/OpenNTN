# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of set_topology with topology tensors that require gradients.
#
# The scenario stores copies of the topology tensors through which gradients flow, so
# that its quantities stay differentiable with respect to the tensors passed; the path
# coefficients and delays of the channel models are detached, as in Sionna. This has to
# hold in every call, not only in the first: a stored tensor that carries the graph of
# an earlier call is replaced by a new one, because writing into it in place would chain
# the graphs of the two calls. Without gradients the stored tensors are written in place,
# so that their addresses stay fixed, as torch.compile and CUDA graphs expect.
#
# The distances between the UTs, from which the spatial correlation of the large scale
# parameters is computed, have finite gradients as well: the distance of a UT to itself
# is 0 for every UT location, so its derivative is 0. A UT location that is NaN gives
# NaN distances.
import pytest
import torch
from sionna.phy import config

from openntn import Antenna, DenseUrban, SubUrban, Urban
from openntn.utils import gen_single_sector_topology

LSPS = ("ds", "asd", "asa", "zsa", "zsd", "k_factor", "sf")
# UTs 6 m to 10 m apart in each batch example, relative to its first UT: closer than the
# correlation distances of the large scale parameters in LOS (12 m to 37 m, TR 38.811
# V15.4.0, Tables 6.7.2-1a, 6.7.2-3a and 6.7.2-5a and their Ka-band tables), so that
# their spatial correlation, and with it their derivatives with respect to the UT
# locations, are far from 0
CLOSE_UT_OFFSETS = (((0.0, 0.0), (6.0, 2.0), (-3.0, 7.0)),
                    ((0.0, 0.0), (-5.0, -4.0), (4.0, -6.0)))


def _model(scenario=DenseUrban):
    antenna = Antenna(polarization="single", polarization_type="V", antenna_pattern="omni",
                      carrier_frequency=2.2e9)
    return scenario(carrier_frequency=2.2e9, ut_array=antenna, bs_array=antenna,
                    direction="downlink", elevation_angle=30.0)


def _topology(requires_grad=False):
    topology = list(gen_single_sector_topology(batch_size=2, num_ut=3, scenario="dur",
                                               elevation_angle=30.0, bs_height=600e3))
    if requires_grad:
        topology[0] = topology[0].clone().requires_grad_(True)
    return topology


def test_repeated_differentiable_topologies():
    # Two calls with UT locations that require gradients. After each, the gradient of the
    # sum of the 2D distances with respect to the UT locations is (x, y, 0) / d_2D, with
    # x and y the horizontal offsets of the UT from the satellite, computed here from the
    # positions.
    config.precision = "double"
    model = _model()
    for call in (1, 2):
        topology = _topology(requires_grad=True)
        ut_loc = topology[0]
        model.set_topology(*topology)
        model._scenario.distance_2d.sum().backward()
        offset = (ut_loc.detach() - topology[1])[..., :2]          # one satellite
        d_2d = torch.linalg.vector_norm(offset, dim=-1, keepdim=True)
        expected = torch.cat([offset / d_2d, torch.zeros_like(d_2d)], dim=-1)
        assert ut_loc.grad is not None, f"call {call}: no gradient"
        torch.testing.assert_close(ut_loc.grad, expected, rtol=1e-12, atol=0.0,
                                   msg=lambda m: f"call {call}: {m}")
        # The path coefficients and delays are detached
        h, tau = model(1, 1.0)
        assert not h.requires_grad and not tau.requires_grad, f"call {call}"


def test_stored_tensors_without_gradients():
    config.precision = "double"
    model = _model()
    scenario, lsp = model._scenario, model._lsp_sampler

    def stored():
        return {"ut_loc": scenario._ut_loc, "distance_2d": scenario._distance_2d,
                "spatial LSP correlation": lsp._spatial_lsp_correlation_matrix_sqrt}

    # A call without gradients after a call without gradients writes in place
    model.set_topology(*_topology())
    first = stored()
    model.set_topology(*_topology())
    for name, tensor in stored().items():
        assert tensor is first[name] and tensor.data_ptr() == first[name].data_ptr(), name
    # A differentiable call stores tensors that require gradients
    model.set_topology(*_topology(requires_grad=True))
    for name, tensor in stored().items():
        assert tensor.requires_grad, name
    # The first call without gradients after it stores new tensors without gradients,
    # and the next one writes into them in place again
    model.set_topology(*_topology())
    after = stored()
    for name, tensor in after.items():
        assert not tensor.requires_grad, name
    model.set_topology(*_topology())
    for name, tensor in stored().items():
        assert tensor is after[name] and tensor.data_ptr() == after[name].data_ptr(), name


def test_ut_distance_gradient():
    # The gradient of the sum of all distances between the UTs with respect to the
    # location of UT i is 2 sum_{j != i} (p_i - p_j) / d_ij in x and y, with p the
    # horizontal position and d_ij the distance, and 0 in z, on which the distances do
    # not depend. The distance of a UT to itself is 0 for every UT location, so its
    # derivative is 0.
    config.precision = "double"
    model = _model()
    topology = _topology(requires_grad=True)
    ut_loc = topology[0]
    model.set_topology(*topology)
    model._scenario.matrix_ut_distance_2d.sum().backward()
    grad = ut_loc.grad
    not_finite = int((~torch.isfinite(grad)).sum())
    assert not_finite == 0, f"{not_finite} of {grad.numel()} entries not finite"
    xy = ut_loc.detach()[..., :2]
    offset = xy.unsqueeze(2) - xy.unsqueeze(1)                       # p_i - p_j
    distance = torch.linalg.vector_norm(offset, dim=-1, keepdim=True)
    other = ~torch.eye(xy.shape[1], dtype=torch.bool,
                       device=xy.device).unsqueeze(-1)               # j != i
    expected_xy = 2.0 * torch.where(other, offset / torch.where(other, distance, 1.0),
                                    0.0).sum(dim=2)
    expected = torch.cat([expected_xy, torch.zeros_like(expected_xy[..., :1])], dim=-1)
    # The backward pass and the closed form add the same unit vectors with different
    # rounding: entries of at most 4 agree to a few units of 1e-16 in double precision.
    # rtol and atol 1e-12 are at the rounding level, with a wide margin.
    torch.testing.assert_close(grad, expected, rtol=1e-12, atol=1e-12)


def test_nan_ut_location():
    # A UT location that is NaN gives NaN distances in the row and the column of that
    # UT, as the square root of a NaN squared distance does, and not 0, which would
    # place the UT at the location of every other UT of its batch example. The
    # distances between the other UTs, and those of the other batch example, stay
    # finite.
    config.precision = "double"
    model = _model()
    topology = _topology()
    ut_loc = topology[0].clone()
    ut_loc[0, 1, 0] = float("nan")          # x of the second UT of the first example
    model.set_topology(ut_loc, *topology[1:])
    distance = model._scenario.matrix_ut_distance_2d
    of_nan_ut = torch.zeros_like(distance, dtype=torch.bool)
    of_nan_ut[0, 1, :] = True
    of_nan_ut[0, :, 1] = True
    not_nan = int((~torch.isnan(distance[of_nan_ut])).sum())
    assert not_nan == 0, (f"{not_nan} of {int(of_nan_ut.sum())} distances of the UT "
                          f"with the NaN location are not NaN")
    not_finite = int((~torch.isfinite(distance[~of_nan_ut])).sum())
    assert not_finite == 0, f"{not_finite} distances between the other UTs not finite"


@pytest.mark.parametrize("scenario", [DenseUrban, Urban, SubUrban])
def test_large_scale_parameter_gradients(scenario):
    # The large scale parameters depend on the UT locations through their spatial
    # correlation exp(-d/D) (TR 38.901 V16.1.0, eq. (7.4-5) and clause 7.5, step 4).
    # Their derivatives with respect to the UT locations are finite and equal central
    # differences, with the seed of the test set again before each evaluation, so that
    # every evaluation draws the same random numbers. The UTs are close to each other
    # and LOS is forced, so that the UTs of a batch example are correlated.
    config.precision = "double"
    seed = config.seed
    model = _model(scenario)
    topology = _topology()
    ut_loc = topology[0].clone()
    ut_loc[..., :2] = ut_loc[:, :1, :2] + torch.tensor(CLOSE_UT_OFFSETS,
                                                         dtype=ut_loc.dtype,
                                                         device=ut_loc.device)
    entries = ((0, 0, 0), (0, 1, 1), (1, 2, 0))       # (batch example, UT, coordinate)

    def large_scale_parameters(location):
        config.seed = seed
        model.set_topology(location, *topology[1:], los=True)
        # Without gradients set_topology writes the large scale parameters into the same
        # tensors, so the values are cloned to keep them across the next evaluation
        return {name: getattr(model._lsp, name).flatten().clone() for name in LSPS}

    # The derivative of every link value, [entry, link]; a sum over the links could mix
    # values limited to 104 deg or 52 deg with derivatives orders of magnitude smaller
    location = ut_loc.clone().requires_grad_(True)
    values = large_scale_parameters(location)
    derivatives, not_finite = {}, {}
    for name in LSPS:
        rows, bad = [], torch.zeros_like(ut_loc, dtype=torch.bool)
        for link in range(values[name].numel()):
            grad, = torch.autograd.grad(values[name][link], location, retain_graph=True)
            bad |= ~torch.isfinite(grad)
            rows.append(torch.stack([grad[e] for e in entries]))
        derivatives[name] = torch.stack(rows, dim=1)
        not_finite[name] = int(bad.sum())
    assert not any(not_finite.values()), \
        f"entries of the UT locations with a derivative that is not finite: {not_finite}"
    # Central differences with h = 2^-13 m (0.12 mm): a power of two, so that x + h and
    # x - h are exact in double precision at UT coordinates near 1e6 m, where a step of
    # 1 mm is rounded to a relative error of up to 6e-8
    h = 2.0 ** -13
    differences = {name: [] for name in LSPS}
    for entry in entries:
        shifted = []
        for sign in (1.0, -1.0):
            location = ut_loc.clone()
            location[entry] += sign * h
            shifted.append(large_scale_parameters(location))
        for name in LSPS:
            differences[name].append((shifted[0][name] - shifted[1][name]) / (2.0 * h))
    for name in LSPS:
        expected = torch.stack(differences[name])
        # The truncation error of the central differences is of order (h/d)^2 relative
        # to the derivative, larger where an extreme draw makes a parameter steep, and
        # their rounding error of order 1e-16 |value| / h. Over 400 seeds and the three
        # scenarios the largest difference was 1.4e-8 of the largest derivative of the
        # same parameter. A derivative can be small by cancellation, so the absolute
        # tolerance is relative to the largest derivative; ASD and ZSD are 0 for
        # satellites (TR 38.811 V15.4.0, NOTE 8 of Tables 6.7.2-1a to 6.7.2-8b) and
        # compared exactly.
        torch.testing.assert_close(derivatives[name], expected, rtol=1e-6,
                                   atol=1e-6 * float(expected.abs().max()),
                                   msg=lambda m: f"{name}: {m}")
