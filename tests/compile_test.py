# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the channel models under torch.compile.
#
# After set_topology, a call of a channel model runs without graph breaks:
# torch._dynamo.explain finds none, and torch.compile with fullgraph=True captures the
# call in one graph, and the compiled call gives the eager call bit for bit under the
# same seed. Both checks are needed: fullgraph=True fails at an operation that it cannot
# capture, but it captures Tensor.item() in the graph (a conversion of a tensor to a
# Python number, which synchronizes with the device), where torch.compile without
# fullgraph, the default, breaks the graph. The backend "eager" runs the captured
# operations without generating code, so the comparison checks the capture. In compiled
# code Sionna's random number helpers draw from PyTorch's default generator of the
# device, which sionna.phy.config.seed seeds with the seed of the generator that they use
# in eager mode.
#
# The Faraday rotation of TR 38.811 V15.4.0, clause 6.8.2, turns the polarization by the
# angle psi = 108 / f^2 degrees, f in GHz (eq. (6.8-2)). The models compute psi once,
# when they are built, from their carrier frequency in their precision. The number of
# rays per cluster is a Python int, which torch.compile takes as a constant where it
# bounds a slice or gives a size.
#
# The call of the tapped delay line model TDL runs without graph breaks as well, with
# the Doppler shift of the satellite or without it, a desired K-factor and spatial
# correlation matrices; the shift is a Python float of the model, its phase computed in
# double precision in the call.
#
# So does the call of the clustered delay line model CDL, in the downlink and in the
# uplink: with the Doppler shift of the satellite and the Faraday rotation, desired
# angular spreads and a desired K-factor, and orientations and velocities given for
# each batch example.
import math
import os

import pytest
import torch
import torch._dynamo
from sionna.phy import config

from openntn import CDL, TDL, Antenna, AntennaArray, DenseUrban, SubUrban, Urban
from openntn.utils import exp_corr_mat, gen_single_sector_topology

# S band in the downlink and Ka band in the uplink, 50 degrees, a satellite at 600 km,
# 4 batch examples of 3 UTs
LINKS = (("downlink", 2.0e9), ("uplink", 30.0e9))
ELEVATION_ANGLE = 50.0
SATELLITE_HEIGHT = 600e3
BATCH_SIZE = 4
NUM_UT = 3
NUM_TIME_SAMPLES = 14
SAMPLING_FREQUENCY = 30e3
SCENARIOS = ((DenseUrban, "dur"), (Urban, "urb"), (SubUrban, "sur"))
SEED = 5


def _model(scenario, direction, carrier_frequency, precision, always_generate_lsp=False):
    # A single-polarized UT antenna and a 1 x 4 dual-polarized array at the satellite,
    # both with the element pattern of TR 38.901
    ut_array = Antenna(polarization="single", polarization_type="V",
                       antenna_pattern="38.901", carrier_frequency=carrier_frequency,
                       precision=precision)
    bs_array = AntennaArray(num_rows=1, num_cols=4, polarization="dual",
                            polarization_type="VH", antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency, precision=precision)
    return scenario(carrier_frequency=carrier_frequency, ut_array=ut_array,
                    bs_array=bs_array, direction=direction,
                    elevation_angle=ELEVATION_ANGLE,
                    always_generate_lsp=always_generate_lsp, precision=precision)


def _bits(tensor):
    """The bits of a real or complex floating point tensor, as integers."""
    if tensor.is_complex():
        tensor = torch.view_as_real(tensor)
    width = {torch.float64: torch.int64, torch.float32: torch.int32}[tensor.dtype]
    return tensor.contiguous().view(width)


CASES = [pytest.param(scenario, key, direction, carrier_frequency, precision, per_call,
                      id=f"{scenario.__name__}-{direction}-{precision}-"
                         f"lsp-{'per-call' if per_call else 'cached'}")
         for scenario, key in SCENARIOS
         for direction, carrier_frequency in LINKS
         for precision in ("single", "double")
         for per_call in (False, True)]


@pytest.mark.slow
@pytest.mark.skipif(not torch._dynamo.is_dynamo_supported(),
                    reason="torch.compile is not supported with this Python and PyTorch")
@pytest.mark.parametrize(
    "scenario,key,direction,carrier_frequency,precision,per_call", CASES)
def test_call_without_graph_breaks(scenario, key, direction, carrier_frequency,
                                   precision, per_call):
    # The large scale parameters are drawn once in set_topology (per_call False) or in
    # every call (per_call True, always_generate_lsp)
    model = _model(scenario, direction, carrier_frequency, precision, per_call)
    model.set_topology(*gen_single_sector_topology(
        batch_size=BATCH_SIZE, num_ut=NUM_UT, scenario=key,
        elevation_angle=ELEVATION_ANGLE, bs_height=SATELLITE_HEIGHT,
        precision=precision))

    def call():
        return model(NUM_TIME_SAMPLES, SAMPLING_FREQUENCY)

    _assert_compiles(call)


def _assert_compiles(call):
    """No graph break in the call, and the call compiled with fullgraph=True on the
    backend "eager" equal to the eager call bit for bit under the same seed"""
    torch._dynamo.reset()
    try:
        explanation = torch._dynamo.explain(call)()
    finally:
        torch._dynamo.reset()
    sites = sorted({f"{os.path.basename(frame.filename)}:{frame.lineno}"
                    for reason in explanation.break_reasons
                    for frame in (reason.user_stack or [])[-1:]})
    assert explanation.graph_break_count == 0, \
        f"{explanation.graph_break_count} graph breaks, at {', '.join(sites)}"

    config.seed = SEED
    a_eager, tau_eager = call()
    try:
        compiled = torch.compile(call, fullgraph=True, backend="eager")
        config.seed = SEED
        a_compiled, tau_compiled = compiled()
    finally:
        torch._dynamo.reset()
    for name, eager, compiled_ in (("a", a_eager, a_compiled),
                                   ("tau", tau_eager, tau_compiled)):
        assert compiled_.shape == eager.shape, name
        differing = int((_bits(compiled_) != _bits(eager)).sum())
        assert differing == 0, (f"{name}: {differing} of {eager.numel()} entries differ "
                                f"from the eager call")


def _tdl_options(precision):
    # The shift of the satellite at 600 km and 50 degrees (29.6 kHz at 2 GHz), none with
    # a desired K-factor, 2 x 2 antennas with receive and transmit correlation matrices,
    # and a spatial correlation matrix of the four antenna pairs
    rx = exp_corr_mat(0.5, 2, precision=precision)
    tx = exp_corr_mat(0.3, 2, precision=precision)
    return {
        "shift": {},
        "k-factor": {"k_factor": 7.95, "doppler_shift": 0.0},
        "rx-tx-correlation": {"num_rx_ant": 2, "num_tx_ant": 2, "rx_corr_mat": rx,
                              "tx_corr_mat": tx},
        "spatial-correlation": {"num_rx_ant": 2, "num_tx_ant": 2,
                                "spatial_corr_mat": torch.kron(rx, tx)},
    }


TDL_CASES = [pytest.param(model, precision, options,
                          id=f"{model}-{precision}-{options}")
             for model, options in (("A", "shift"), ("C", "k-factor"),
                                    ("D", "rx-tx-correlation"),
                                    ("B", "spatial-correlation"))
             for precision in ("single", "double")]


@pytest.mark.slow
@pytest.mark.skipif(not torch._dynamo.is_dynamo_supported(),
                    reason="torch.compile is not supported with this Python and PyTorch")
@pytest.mark.parametrize("model,precision,options", TDL_CASES)
def test_tdl_call_without_graph_breaks(model, precision, options):
    tdl = TDL(model, 100e-9, 2.0e9, ELEVATION_ANGLE, SATELLITE_HEIGHT, max_speed=3.0,
              precision=precision, **_tdl_options(precision)[options])

    def call():
        return tdl(BATCH_SIZE, NUM_TIME_SAMPLES, SAMPLING_FREQUENCY)

    _assert_compiles(call)


# The shift of the satellite and the Faraday rotation at 600 km and 50 degrees with a
# UT at up to 3 m/s; desired angular spreads and a desired K-factor at 30 degrees with
# a UT at up to 30 m/s; no shift and the velocities of the batch examples given; the
# orientations of the batch examples given
CDL_CASES = [
    pytest.param("A", "downlink", "single", {"max_speed": 3.0}, id="A-downlink-single"),
    pytest.param("D", "uplink", "double",
                 {"elevation_angle": 30.0, "angular_spreads": (30.0, 10.0),
                  "k_factor": 3.0, "max_speed": 30.0}, id="D-uplink-double"),
    pytest.param("B", "uplink", "single",
                 {"doppler_shift": 0.0,
                  "ut_velocity": [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0],
                                  [2.0, 2.0, 1.0]]}, id="B-uplink-single"),
    pytest.param("C", "downlink", "double",
                 {"max_speed": 3.0,
                  "ut_orientation": [[math.pi, 0.0, 0.0]] * BATCH_SIZE,
                  "bs_orientation": [[0.0, 0.5, 0.0]] * BATCH_SIZE},
                 id="C-downlink-double"),
]


@pytest.mark.slow
@pytest.mark.skipif(not torch._dynamo.is_dynamo_supported(),
                    reason="torch.compile is not supported with this Python and PyTorch")
@pytest.mark.parametrize("model,direction,precision,options", CDL_CASES)
def test_cdl_call_without_graph_breaks(model, direction, precision, options):
    # A dual-polarized UT antenna and a 1 x 2 dual-polarized array at the satellite
    options = {"elevation_angle": ELEVATION_ANGLE, **options}
    ut_array = Antenna("dual", "cross", "omni", 2.0e9, precision=precision)
    bs_array = AntennaArray(1, 2, "dual", "VH", "38.901", 2.0e9, precision=precision)
    cdl = CDL(model, 100e-9, 2.0e9, options.pop("elevation_angle"), SATELLITE_HEIGHT,
              ut_array=ut_array, bs_array=bs_array, direction=direction,
              precision=precision, **options)

    def call():
        return cdl(BATCH_SIZE, NUM_TIME_SAMPLES, SAMPLING_FREQUENCY)

    _assert_compiles(call)


@pytest.mark.parametrize("scenario", [DenseUrban, Urban, SubUrban])
def test_rays_per_cluster(scenario):
    # TR 38.811 V15.4.0, Tables 6.7.2-1a to 6.7.2-6b: 20 rays per cluster in every
    # scenario, band, link state and elevation angle
    rays = _model(scenario, "downlink", 2.0e9, "single")._scenario.rays_per_cluster
    assert type(rays) is int and rays == 20, repr(rays)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("carrier_frequency", [2.0e9, 30.0e9])
def test_faraday_rotation_factor(carrier_frequency, precision):
    # The factor [[cos psi, -sin psi], [sin psi, cos psi]] of eq. (6.8-1), with psi of
    # eq. (6.8-2) computed here in double precision. In double precision the model
    # computes psi with the same operations, and the factor agrees within a few units in
    # the last place. In single precision the model computes psi in float32 from its
    # carrier frequency in float32 (30 GHz is held as 30000001024 Hz): with the rounding
    # of the frequency, of the constant pi/180 and of four operations, psi carries a
    # relative error of at most 7 times 2^-24, and cos psi and sin psi are rounded to
    # float32, so that the entries of the factor differ by at most 2.1e-7, within 2^-21
    # (4.8e-7).
    model = _model(Urban, "downlink", carrier_frequency, precision)
    sampler = model._cir_sampler
    shape = (2, 1, 3, 4, 20)
    factor = sampler._step_11_faraday_rotation(sampler._carrier_frequency, shape)
    assert factor.dtype == sampler.cdtype
    psi = math.radians(108.0 / (carrier_frequency / 1e9) ** 2)
    expected = torch.tensor([[math.cos(psi), -math.sin(psi)],
                             [math.sin(psi), math.cos(psi)]], dtype=torch.complex128,
                            device=factor.device)
    atol = 2.0 ** -21 if precision == "single" else 2.0 ** -50
    torch.testing.assert_close(
        torch.broadcast_to(factor, shape + (2, 2)).to(torch.complex128),
        expected.expand(shape + (2, 2)), rtol=0.0, atol=atol)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("carrier_frequency", [2.0e9, 30.0e9])
def test_faraday_rotation_built_once(carrier_frequency, precision):
    # The 2 x 2 rotation is built once, when the model is built, and broadcast against
    # the phase matrices of the rays, instead of four tensors of their full size in
    # every call. Broadcast to the full size, it is bit for bit the factor built entry by
    # entry from cos psi and sin psi.
    model = _model(Urban, "downlink", carrier_frequency, precision)
    sampler = model._cir_sampler
    shape = (2, 1, 3, 4, 20)
    factor = sampler._step_11_faraday_rotation(sampler._carrier_frequency, shape)
    assert factor.numel() == 4, tuple(factor.shape)
    cos_val = complex(sampler._faraday_cos, 0.0)
    sin_val = complex(sampler._faraday_sin, 0.0)
    full = torch.stack([torch.full(shape, value, dtype=sampler.cdtype, device=factor.device)
                        for value in (cos_val, -sin_val, sin_val, cos_val)],
                       dim=-1).reshape(*shape, 2, 2)
    assert torch.equal(_bits(torch.broadcast_to(factor, full.shape)), _bits(full))
