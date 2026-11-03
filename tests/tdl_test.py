# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the tapped delay line models NTN-TDL-A to NTN-TDL-D (openntn.TDL) against
# values computed in the test from 3GPP TR 38.811 V15.4.0, clause 6.9.2, TR 38.901
# V16.1.0, clause 7.7, and TS 38.101-5 V19.6.0, Annex B.2.
#
# - The tables: Tables 6.9.2-1 to 6.9.2-4 (NTN-TDL) equal Tables 6.9.1-1 to 6.9.1-4
#   (NTN-CDL) in normalized delays and powers, as clause 6.9.2 filters the TDL models
#   from the CDL models with an isotropic UE antenna; the model holds the delays and the
#   powers of its table, normalized to a total power of one with the specular path. The
#   table values are those of spec_tables/tr38811_v15.4.0.json.
# - The normalized RMS delay spread of each table is 1 with the specular path, up to the
#   rounding of its printed values; the K-factor K1 of the first tap of NTN-TDL-C and
#   NTN-TDL-D is the difference of the first two powers, 10.224 dB and 11.707 dB as the
#   NOTEs of Tables 6.9.2-3 and 6.9.2-4 give it, and the K-factor of TR 38.901
#   eq. (7.7.6-2), the power of the specular path over the sum of all Rayleigh taps, is
#   10.000 dB and 7.000 dB; with a desired K-factor (clause 7.7.6) the powers follow
#   eq. (7.7.6-1) and the RMS delay spread of the delays, the specular path included, is
#   the desired delay spread.
# - The statistics of generated taps, within 4 standard errors of the expected
#   distribution (statistical_checks.py): the mean power per tap, the K-factor of the
#   first tap of NTN-TDL-C and NTN-TDL-D from the second and fourth moments of its
#   magnitude, the RMS delay spread of the mean power delay profile, the time
#   correlation of every Rayleigh tap against J0(2 pi f_D tau) of the classical spectrum
#   of clause 7.7.2, and the time correlation of the first tap of NTN-TDL-C, whose
#   imaginary part comes only from the specular path at 0.7 f_D. The standard errors
#   are those of the sum of N sinusoids with independent uniform phases, whose angles of
#   arrival lie one in each of N equal sectors of the circle: the expected moments are
#   computed below from this distribution.
# - The Doppler shift due to the motion of the satellite of clause 6.9.2, f_d = (v/c)
#   (R/(R+h)) cos(alpha) f_c with the speed v = sqrt(G M / (R + h)) of a circular orbit
#   and the constants of openntn.utils.compute_satellite_speed, computed in the test:
#   with the UT at rest every tap of every antenna advances by 2 pi f_d / f_s per
#   sample; 0 for a geostationary satellite and an aerial platform; a given
#   doppler_shift is used as it is; with a moving UT the time correlation of every tap
#   over a lag tau turns by 2 pi f_d tau; in single precision the phase stays exact
#   after 10^6 samples.
# - The test profiles of TS 38.101-5, Annex B.2.1, derived from the NTN-TDL models: the
#   delays scaled to the RMS delay spread and rounded to the nearest multiple of 5 ns,
#   NTN-TDLA100 from NTN-TDL-A exactly, with the powers rounded to 0.1 dB, and NTN-TDLC5
#   from NTN-TDL-C with the desired K-factor 7.95 dB within 0.08 dB, which the document
#   does not state.
# - The checks of the arguments, the arguments taken by keyword only, the outputs in
#   single and double precision, and the ValueError of a model moved or cast after its
#   construction.
import inspect
import json
import math
import os
import re
import warnings

import mpmath
import numpy as np
import pytest
import torch
from sionna.phy import SPEED_OF_LIGHT, config

from openntn import TDL
from statistical_checks import Z, assert_mean

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_tables",
                       "tr38811_v15.4.0.json"), encoding="ascii") as _f:
    TABLES = json.load(_f)["tables"]
TDL_TABLES = {"A": "6.9.2-1", "B": "6.9.2-2", "C": "6.9.2-3", "D": "6.9.2-4"}
CDL_TABLES = {"A": "6.9.1-1", "B": "6.9.1-2", "C": "6.9.1-3", "D": "6.9.1-4"}
MODELS = "ABCD"
NUM_PATHS = {"A": 3, "B": 4, "C": 2, "D": 3}

R_EARTH = 6371e3
# G and the mass of the Earth of openntn.utils.compute_satellite_speed
GM = 6.6743e-11 * 5.972e24
NUM_SINUSOIDS = 20
DELAY_SPREAD = 100e-9
CARRIER_FREQUENCY = 2e9
ELEVATION_ANGLE = 50.0
SATELLITE_HEIGHT = 600e3
SAMPLING_FREQUENCY = 1e6


def _tdl(model, delay_spread=DELAY_SPREAD, carrier_frequency=CARRIER_FREQUENCY,
         elevation_angle=ELEVATION_ANGLE, satellite_height=SATELLITE_HEIGHT, **kwargs):
    return TDL(model, delay_spread, carrier_frequency, elevation_angle,
               satellite_height, **kwargs)


def _table(table):
    """Normalized delays, powers [dB] and whether the first entry is a specular path"""
    rows = TABLES[table]["rows"]
    # Tables 6.9.1-1 and 6.9.1-2 have no column "Cluster PAS": NLOS models
    first = (rows.get("cluster_pas") or rows.get("fading_distribution") or [""])[0]
    return (np.array([float(v) for v in rows["normalized_delay"]]),
            np.array([float(v) for v in rows["power_dB"]]),
            first in ("LOS path", "Specular(LOS path)"))


def _profile(model, k_factor=None):
    """The power delay profile of the table of a model, in the test's own computation:
    the delays, the powers normalized to a total of one with the first path holding the
    specular path, and the power of the specular path (0 for NLOS models). With
    k_factor [dB], TR 38.901 clause 7.7.6: the Rayleigh taps by eq. (7.7.6-1) with the
    K-factor of eq. (7.7.6-2), then the delays divided by the RMS delay spread of the
    scaled profile with the specular path."""
    delays, powers, los = _table(TDL_TABLES[model])
    if k_factor is not None:
        k_model = powers[0] - 10 * math.log10(np.sum(10 ** (powers[1:] / 10)))
        powers = np.concatenate([powers[:1], powers[1:] - k_factor + k_model])
        delays = delays / _rms_delay_spread(delays, 10 ** (powers / 10))
    linear = 10 ** (powers / 10)
    linear = linear / linear.sum()
    if los:
        first = linear[0] + linear[1]
        return delays[1:], np.concatenate([[first], linear[2:]]), linear[0]
    return delays, linear, 0.0


def _rms_delay_spread(delays, powers):
    mean = np.sum(powers * delays) / np.sum(powers)
    return math.sqrt(np.sum(powers * (delays - mean) ** 2) / np.sum(powers))


def _satellite_shift(height, elevation_angle, carrier_frequency):
    """f_d of clause 6.9.2; 0 for a geostationary satellite and an aerial platform"""
    if height >= 35786e3 or height <= 50e3:
        return 0.0
    v_sat = math.sqrt(GM / (R_EARTH + height))
    return (v_sat / SPEED_OF_LIGHT * R_EARTH / (R_EARTH + height)
            * math.cos(math.radians(elevation_angle)) * carrier_frequency)


# The tables

@pytest.mark.parametrize("model", MODELS)
def test_tdl_tables_equal_cdl_tables(model):
    # TR 38.811 clause 6.9.2: "The Tapped Delay Line (TDL) models are filtered from the
    # CDL models according to the section 7.7.4 of TR 38.901 by assuming isotropic UE
    # antenna": with an isotropic antenna eq. (7.7-4) keeps the powers and the delays of
    # the clusters, the specular path of the LOS models included
    tdl_rows = TABLES[TDL_TABLES[model]]["rows"]
    cdl_rows = TABLES[CDL_TABLES[model]]["rows"]
    assert tdl_rows["normalized_delay"] == cdl_rows["normalized_delay"]
    assert tdl_rows["power_dB"] == cdl_rows["power_dB"]
    assert _table(TDL_TABLES[model])[2] == _table(CDL_TABLES[model])[2]


@pytest.mark.parametrize("model", MODELS)
def test_model_holds_its_table(model):
    delays, powers, specular = _profile(model)
    tdl = _tdl(model, precision="double")
    assert tdl.los == (model in "CD")
    assert tdl.num_clusters == NUM_PATHS[model] == len(delays)
    np.testing.assert_allclose(tdl.delays.cpu().numpy(), delays * DELAY_SPREAD,
                               rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(tdl.mean_powers.cpu().numpy(), powers, rtol=1e-12,
                               atol=0.0)
    if tdl.los:
        assert float(tdl.mean_power_los) == pytest.approx(specular, rel=1e-12)


def _half_unit(text):
    """Half a unit of the last printed digit of a table value; 0, the delay of the first
    tap and the power of the reference tap, is exact"""
    if float(text) == 0.0:
        return 0.0
    return 0.5 * 10.0 ** -(len(text.split(".")[1]) if "." in text else 0)


@pytest.mark.parametrize("model", MODELS)
def test_normalized_delay_spread(model):
    # The tables are normalized to an RMS delay spread of 1 with the specular path (A
    # 0.99997, B 0.99994, C 0.99996, D 0.99999), up to the rounding of their printed
    # values: the bound is the change of the RMS delay spread, to first order, by half a
    # unit of the last printed digit of every normalized delay (four decimals) and power
    # (three decimals), 6.2e-5 for A, 8.7e-5 for B and 1.2e-4 for C and D
    rows = TABLES[TDL_TABLES[model]]["rows"]
    delays, powers_db, _ = _table(TDL_TABLES[model])
    powers = 10 ** (powers_db / 10)
    spread = _rms_delay_spread(delays, powers)
    mean_delay = np.sum(powers * delays) / np.sum(powers)
    by_delay = powers * (delays - mean_delay) / (spread * np.sum(powers))
    by_power_db = (((delays - mean_delay) ** 2 - spread ** 2)
                   / (2 * spread * np.sum(powers)) * powers * math.log(10) / 10)
    bound = (np.sum(np.abs(by_delay) * [_half_unit(v)
                                        for v in rows["normalized_delay"]])
             + np.sum(np.abs(by_power_db) * [_half_unit(v) for v in rows["power_dB"]]))
    assert bound <= 1.2e-4
    assert abs(spread - 1.0) <= bound
    tdl = _tdl(model, precision="double")
    spread = _rms_delay_spread(tdl.delays.cpu().numpy(), tdl.mean_powers.cpu().numpy())
    assert abs(spread / DELAY_SPREAD - 1.0) <= bound


@pytest.mark.parametrize("model,k_model", [("C", 10.000), ("D", 7.000)])
def test_k_factors_of_the_tables(model, k_model):
    # K1 of the NOTE, the difference of the first two powers; the K-factor of
    # eq. (7.7.6-2), which the property k_factor gives in linear scale
    _, powers, los = _table(TDL_TABLES[model])
    assert los
    note = TABLES[TDL_TABLES[model]]["note"]
    note = float(re.search(r"K1 = (-?\d+\.\d+) dB", note).group(1))
    assert powers[0] - powers[1] == pytest.approx(note, abs=1e-9)
    k_table = powers[0] - 10 * math.log10(np.sum(10 ** (powers[1:] / 10)))
    assert round(k_table, 3) == k_model
    tdl = _tdl(model, precision="double")
    assert 10 * math.log10(float(tdl.k_factor)) == pytest.approx(k_table, abs=1e-9)


@pytest.mark.parametrize("model", "AB")
def test_k_factor_of_nlos_models(model):
    tdl = _tdl(model)
    with pytest.raises(RuntimeError, match="k_factor"):
        tdl.k_factor
    with pytest.raises(RuntimeError, match="mean_power_los"):
        tdl.mean_power_los


@pytest.mark.parametrize("k_factor", [-3.0, 3.0, 7.95, 13.0, 31.83])
@pytest.mark.parametrize("model", "CD")
def test_desired_k_factor(model, k_factor):
    # TR 38.901 clause 7.7.6: the powers of eq. (7.7.6-1), the K-factor of
    # eq. (7.7.6-2) equal to the desired one, and the delays renormalized so that the
    # RMS delay spread of the output delays, with the specular path, is delay_spread
    delays, powers, specular = _profile(model, k_factor)
    tdl = _tdl(model, k_factor=k_factor, precision="double")
    np.testing.assert_allclose(tdl.mean_powers.cpu().numpy(), powers, rtol=1e-12,
                               atol=0.0)
    assert float(tdl.mean_power_los) == pytest.approx(specular, rel=1e-12)
    assert 10 * math.log10(float(tdl.k_factor)) == pytest.approx(k_factor, abs=1e-9)
    _, tau = tdl(2, 1, SAMPLING_FREQUENCY)
    for example in tau.cpu().numpy()[:, 0, 0]:
        np.testing.assert_allclose(example, delays * DELAY_SPREAD, rtol=1e-12, atol=0.0)
        assert _rms_delay_spread(example, powers) == pytest.approx(DELAY_SPREAD,
                                                                   rel=1e-12)


# The test profiles of TS 38.101-5, Annex B.2.1

def test_ntn_tdla100():
    # Table B.2.1-2, NTN-TDLA100: NTN-TDL-A scaled to 100 ns, the delays rounded to the
    # nearest multiple of 5 ns and the powers, relative to the first tap, to 0.1 dB
    tdl = _tdl("A", delay_spread=100e-9, precision="double")
    delays_ns = tdl.delays.cpu().numpy() * 1e9
    powers_db = 10 * np.log10(tdl.mean_powers.cpu().numpy())
    assert [5 * round(d / 5) for d in delays_ns] == [0, 110, 285]
    assert [round(p - powers_db[0], 1) for p in powers_db] == [0.0, -4.7, -6.5]


def test_ntn_tdlc5():
    # Table B.2.1-3, NTN-TDLC5: the specular path -0.6 dB and the Rayleigh part -8.9 dB
    # at 0 ns, the second tap -21.5 dB at 60 ns. NTN-TDL-C with the desired K-factor
    # 7.95 dB of clause 7.7.6 and the delay spread 5 ns gives -0.646 dB, -8.820 dB and
    # -21.575 dB, the total power 0 dB, and the delays 0 and 60.15 ns
    tdl = _tdl("C", delay_spread=5e-9, k_factor=7.95, precision="double")
    specular = float(tdl.mean_power_los)
    powers = tdl.mean_powers.cpu().numpy()
    powers_db = 10 * np.log10([specular, powers[0] - specular, powers[1]])
    assert np.max(np.abs(powers_db - np.array([-0.6, -8.9, -21.5]))) <= 0.08
    assert [5 * round(d / 5) for d in tdl.delays.cpu().numpy() * 1e9] == [0, 60]


# The arguments

ARGUMENTS = dict(model="A", delay_spread=DELAY_SPREAD,
                 carrier_frequency=CARRIER_FREQUENCY, elevation_angle=ELEVATION_ANGLE,
                 satellite_height=SATELLITE_HEIGHT)
INVALID = [
    (dict(model="E"), ValueError, "model"),
    (dict(model=1), TypeError, "model"),
    (dict(delay_spread=0.0), ValueError, "delay_spread"),
    (dict(delay_spread=math.inf), ValueError, "delay_spread"),
    (dict(delay_spread="100e-9"), TypeError, "delay_spread"),
    (dict(carrier_frequency=10e9), ValueError, "carrier_frequency"),
    (dict(carrier_frequency=1.8e9), ValueError, "carrier_frequency"),
    (dict(carrier_frequency=None), TypeError, "carrier_frequency"),
    (dict(elevation_angle=9.9), ValueError, "elevation_angle"),
    (dict(elevation_angle=True), TypeError, "elevation_angle"),
    (dict(satellite_height=100e3), ValueError, "satellite_height"),
    (dict(satellite_height=7e3), ValueError, "satellite_height"),
    (dict(satellite_height="600e3"), TypeError, "satellite_height"),
    (dict(doppler_shift=math.nan), ValueError, "doppler_shift"),
    (dict(doppler_shift="0"), TypeError, "doppler_shift"),
    (dict(k_factor=3.0), ValueError, "k_factor"),
    (dict(model="C", k_factor=math.inf), ValueError, "k_factor"),
    (dict(model="C", k_factor="3"), TypeError, "k_factor"),
    (dict(num_sinusoids=0), ValueError, "num_sinusoids"),
    (dict(num_sinusoids=20.0), TypeError, "num_sinusoids"),
    (dict(num_rx_ant=0), ValueError, "num_rx_ant"),
    (dict(num_tx_ant=True), TypeError, "num_tx_ant"),
    (dict(min_speed=-1.0), ValueError, "min_speed"),
    (dict(min_speed=math.inf), ValueError, "min_speed"),
    (dict(min_speed="0"), TypeError, "min_speed"),
    (dict(min_speed=2.0, max_speed=1.0), ValueError, "max_speed"),
    (dict(max_speed="3"), TypeError, "max_speed"),
]


@pytest.mark.parametrize("change,error,name", INVALID,
                         ids=[f"{list(c)[-1]}-{c[list(c)[-1]]!r}"
                              for c, _, _ in INVALID])
def test_invalid_argument(change, error, name):
    with pytest.raises(error, match=name):
        TDL(**{**ARGUMENTS, **change})


@pytest.mark.parametrize("change", [
    dict(carrier_frequency=1.9e9), dict(carrier_frequency=4e9),
    dict(carrier_frequency=19e9), dict(carrier_frequency=40e9),
    dict(elevation_angle=10), dict(elevation_angle=90.0),
    dict(satellite_height=160e3), dict(satellite_height=35786e3),
    dict(satellite_height=8e3), dict(satellite_height=50e3),
    dict(num_sinusoids=1), dict(min_speed=0, max_speed=0.0),
    dict(model="D", k_factor=-10), dict(doppler_shift=0),
])
def test_accepted_boundaries(change):
    a, _ = TDL(**{**ARGUMENTS, **change})(2, 3, SAMPLING_FREQUENCY)
    assert bool(torch.isfinite(a).all())


def test_delay_spread_setter():
    tdl = _tdl("B", precision="double")
    tdl.delay_spread = 300e-9
    np.testing.assert_allclose(tdl.delays.cpu().numpy(), _profile("B")[0] * 300e-9,
                               rtol=1e-12, atol=0.0)
    with pytest.raises(ValueError, match="delay_spread"):
        tdl.delay_spread = -1e-9
    with pytest.raises(TypeError, match="delay_spread"):
        tdl.delay_spread = None


def test_keyword_only_arguments():
    # The arguments after satellite_height are taken by keyword only, so that a call in
    # the order of Sionna's TR 38.901 TDL, which takes num_sinusoids,
    # los_angle_of_arrival, min_speed and max_speed after carrier_frequency, raises
    # instead of taking its numbers for the elevation angle and the height
    parameters = inspect.signature(TDL).parameters
    assert [name for name, p in parameters.items()
            if p.kind == p.POSITIONAL_OR_KEYWORD] == ["model", "delay_spread",
                                                       "carrier_frequency",
                                                       "elevation_angle",
                                                       "satellite_height"]
    assert all(p.kind == p.KEYWORD_ONLY for name, p in parameters.items()
               if name not in ("model", "delay_spread", "carrier_frequency",
                               "elevation_angle", "satellite_height"))
    with pytest.raises(TypeError):
        TDL("A", DELAY_SPREAD, CARRIER_FREQUENCY, 20, math.pi / 4, 0.0, 3.0)
    with pytest.raises(TypeError):
        TDL("A", DELAY_SPREAD, CARRIER_FREQUENCY, 20)
    with pytest.raises(ValueError, match="satellite_height"):
        TDL("A", DELAY_SPREAD, CARRIER_FREQUENCY, 20, math.pi / 4)


@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("model", MODELS)
def test_outputs(model, precision):
    tdl = _tdl(model, max_speed=3.0, num_rx_ant=2, num_tx_ant=3, precision=precision)
    a, tau = tdl(4, 5, SAMPLING_FREQUENCY)
    num_paths = NUM_PATHS[model]
    assert a.shape == (4, 1, 2, 1, 3, num_paths, 5)
    assert tau.shape == (4, 1, 1, num_paths)
    real, complex_ = ((torch.float32, torch.complex64) if precision == "single"
                      else (torch.float64, torch.complex128))
    assert (a.dtype, tau.dtype) == (complex_, real)
    assert a.device == tau.device == torch.device(tdl.device)
    assert not a.requires_grad and not tau.requires_grad
    assert bool(torch.isfinite(a).all())
    assert bool((tau == tdl.delays).all())


# A cast or a move to the device meta (it stands in for a GPU), with the precision of
# the model and the dtype and the device of its tensors after the move or the cast
MOVES = (("single", torch.float64, r"torch\.float64 on cpu"),
         ("double", torch.float32, r"torch\.float32 on cpu"),
         ("single", "meta", r"torch\.float32 on meta"))


@pytest.mark.parametrize("precision,move,tensors", MOVES,
                         ids=["single-float64", "double-float32", "single-meta"])
def test_moved_or_cast_model(precision, move, tensors):
    # A model stays on the device and in the precision of its construction: .to() moves
    # or casts its registered tensors, but not its device and its precision, so the next
    # call raises ValueError
    config.device = "cpu"
    tdl = _tdl("C", precision=precision)
    tdl(2, 3, SAMPLING_FREQUENCY)
    with warnings.catch_warnings():
        # A cast to another floating point dtype also casts the complex tensors to real,
        # which PyTorch warns about
        warnings.simplefilter("ignore")
        tdl.to(move)
    with pytest.raises(ValueError, match=rf"was built on device 'cpu' in {precision} "
                                         rf"precision, but its tensors are {tensors}"):
        tdl(2, 3, SAMPLING_FREQUENCY)


# The Doppler shift due to the motion of the satellite

@pytest.mark.parametrize("height,elevation_angle,carrier_frequency", [
    (600e3, 50.0, 2e9), (600e3, 10.0, 30e9), (1200e3, 30.0, 30e9), (600e3, 70.0, 2e9),
    (10000e3, 30.0, 20e9), (160e3, 90.0, 3.5e9),
])
def test_phase_advance_at_rest(height, elevation_angle, carrier_frequency):
    # With the UT at rest, every tap of every antenna pair advances per sample by
    # 2 pi f_d / f_s, in double precision to 1e-9 rad: 29 634.973 Hz at 600 km,
    # 50 degrees and 2 GHz; 681 051.185 Hz at 600 km, 10 degrees and 30 GHz
    f_d = _satellite_shift(height, elevation_angle, carrier_frequency)
    tdl = TDL("C", DELAY_SPREAD, carrier_frequency, elevation_angle, height,
              num_rx_ant=2, num_tx_ant=2, precision="double")
    assert tdl.doppler_shift == pytest.approx(f_d, rel=1e-12, abs=1e-6)
    a = tdl(2, 6, SAMPLING_FREQUENCY)[0].cpu().numpy()
    step = a[..., 1:] * np.conj(a[..., :-1])
    expected = np.exp(2j * np.pi * f_d / SAMPLING_FREQUENCY)
    worst = float(np.abs(np.angle(step * np.conj(expected))).max())
    assert worst <= 1e-9, f"phase step off by {worst:.3g} rad"


@pytest.mark.parametrize("height", [35786e3, 20e3, 8e3, 50e3])
def test_no_shift_without_orbital_motion(height):
    # No Doppler shift for a geostationary satellite (TR 38.811 clause 5.3.2.3) and for
    # an aerial platform: with the UT at rest the channel is constant
    tdl = _tdl("D", satellite_height=height, precision="double")
    assert tdl.doppler_shift == 0.0
    a = tdl(2, 4, SAMPLING_FREQUENCY)[0]
    assert torch.equal(a[..., 1:], a[..., :1].expand_as(a[..., 1:]))


@pytest.mark.parametrize("doppler_shift", [-500.0, 0.0, 1234.5, 70e3])
def test_given_doppler_shift(doppler_shift):
    # A given shift is used as it is, also for an aerial platform: a receding satellite,
    # a terminal that pre-compensates the shift, the motion of an aerial platform
    for height in (600e3, 20e3):
        tdl = _tdl("A", satellite_height=height, doppler_shift=doppler_shift,
                   precision="double")
        assert tdl.doppler_shift == doppler_shift
        a = tdl(2, 6, SAMPLING_FREQUENCY)[0].cpu().numpy()
        step = a[..., 1:] * np.conj(a[..., :-1])
        expected = np.exp(2j * np.pi * doppler_shift / SAMPLING_FREQUENCY)
        assert float(np.abs(np.angle(step * np.conj(expected))).max()) <= 1e-9


def test_phase_in_single_precision():
    # In single precision 2 pi f_d n / f_s itself is not resolved over long sequences:
    # at 600 km, 10 degrees and 30 GHz, f_d = 681 051.185 Hz, it reaches 4.28e6 rad
    # after 10^6 samples at 1 MHz, where single-precision numbers are 0.5 rad apart. The
    # phase of a single-precision model at the sample 10^6 equals the phase 2 pi f_d n /
    # f_s computed here in double precision within 1e-6 rad, a few times the rounding of
    # a single-precision factor. One sinusoid per tap and the UT at rest keep the call
    # small.
    num_samples = 10 ** 6
    f_d = _satellite_shift(600e3, 10.0, 30e9)
    tdl = TDL("A", DELAY_SPREAD, 30e9, 10.0, 600e3, num_sinusoids=1, precision="single")
    a = tdl(1, num_samples + 1, SAMPLING_FREQUENCY)[0][0, 0, 0, 0, 0]
    a = a.cpu().to(torch.complex128).numpy()
    measured = np.angle(a[:, -1] * np.conj(a[:, 0]))
    cycles = num_samples * (f_d / SAMPLING_FREQUENCY)
    expected = 2 * np.pi * (cycles - math.floor(cycles))
    assert float(np.abs(np.angle(np.exp(1j * (measured - expected)))).max()) <= 1e-6


# Statistics of generated taps

STATISTICS_BATCH = 50000
DOPPLER_BATCH = 20000
UT_SPEED = 30.0                                   # f_D = 200.138 Hz at 2 GHz
DOPPLER_SAMPLING = 2000.0
DOPPLER_STEPS = 4                                 # lags 0.5, 1.0 and 1.5 ms


def _first_samples(model):
    """|h|^2 [path, batch] of every tap at the first sample of STATISTICS_BATCH batch
    examples, which are independent"""
    a, _ = _tdl(model, precision="double")(STATISTICS_BATCH, 1, SAMPLING_FREQUENCY)
    return (np.abs(a[:, 0, 0, 0, 0, :, 0].cpu().numpy()) ** 2).T


def _time_samples(model, doppler_shift=0.0):
    """Path coefficients [batch, path, time] of DOPPLER_BATCH batch examples with the UT
    at UT_SPEED, DOPPLER_STEPS samples at DOPPLER_SAMPLING"""
    tdl = _tdl(model, min_speed=UT_SPEED, doppler_shift=doppler_shift,
               precision="double")
    a, _ = tdl(DOPPLER_BATCH, DOPPLER_STEPS, DOPPLER_SAMPLING)
    return a[:, 0, 0, 0, 0].cpu().numpy()


def _moments(p_specular, p_rayleigh, order):
    """E|h|^(2m), m = 0 to order, of h = sqrt(p_specular) exp(j phi_0) + sqrt(p_rayleigh
    / N) sum_n exp(j psi_n) with N = NUM_SINUSOIDS and independent uniform phases: only
    the products of m phasors and m conjugate phasors with the same multiset of indices
    keep their mean, so E|h|^(2m) is (m!)^2 times the coefficient of z^m of the product
    of the series sum_k (b z)^k / (k!)^2 of the N + 1 terms of power b. For a Rayleigh
    tap E|h|^4 = (2 - 1/N) p^2."""
    def series(power):
        return np.array([power ** k / math.factorial(k) ** 2 for k in range(order + 1)])
    product = series(p_specular)
    term = series(p_rayleigh / NUM_SINUSOIDS)
    for _ in range(NUM_SINUSOIDS):
        product = np.convolve(product, term)[:order + 1]
    return [math.factorial(m) ** 2 * product[m] for m in range(order + 1)]


def _assert_within(estimate, expected, standard_error, label):
    """An estimate within Z standard errors of the expected distribution"""
    assert abs(estimate - expected) <= Z * standard_error, (
        f"{label}: {estimate:.6g}, expected {expected:.6g} +- {Z * standard_error:.3g} "
        f"({Z:g} standard errors)")


def _k_factor_from_moments(m2, m4):
    """K1 = P_L / P_1 of h = specular path plus a Rayleigh sum of N sinusoids, from
    E|h|^2 = P_L + P_1 and E|h|^4 = P_L^2 + 4 P_L P_1 + (2 - 1/N) P_1^2 solved for
    P_1"""
    c = 1.0 + 1.0 / NUM_SINUSOIDS
    p_1 = (m2 - math.sqrt(m2 * m2 - c * (m4 - m2 * m2))) / c
    return (m2 - p_1) / p_1


@pytest.mark.parametrize("model", MODELS)
def test_tap_statistics(model):
    # At the first sample of independent batch examples: the mean power of every tap;
    # the RMS delay spread of the mean power delay profile against that of the table
    # (0.99997 times 100 ns for NTN-TDL-A), its standard error from the variances of the
    # mean powers, which are independent across the taps; for NTN-TDL-C and NTN-TDL-D
    # the K-factor K1 of the first tap from the second and fourth moments of its
    # magnitude, its standard error from the covariance of |h|^2 and |h|^4 through the
    # solution
    delays, powers, specular = _profile(model)
    delays = delays * DELAY_SPREAD
    samples = _first_samples(model)
    moments = [_moments(specular if k == 0 else 0.0,
                        powers[k] - (specular if k == 0 else 0.0), 4)
               for k in range(len(powers))]
    variances = np.array([m[2] - m[1] ** 2 for m in moments])
    for k in range(len(powers)):
        assert_mean(samples[k], powers[k], math.sqrt(variances[k]),
                    f"NTN-TDL-{model} mean power of tap {k + 1}")

    expected = _rms_delay_spread(delays, powers)
    mean_delay = np.sum(powers * delays) / np.sum(powers)
    gradient = (((delays - mean_delay) ** 2 - expected ** 2)
                / (2 * expected * np.sum(powers)))
    _assert_within(_rms_delay_spread(delays, samples.mean(axis=1)), expected,
                   math.sqrt(np.sum(gradient ** 2 * variances) / samples.shape[1]),
                   f"NTN-TDL-{model} RMS delay spread of the mean power delay profile")

    if specular:
        k1 = specular / (powers[0] - specular)
        m = moments[0]
        assert _k_factor_from_moments(m[1], m[2]) == pytest.approx(k1, rel=1e-9)
        step = 1e-6 * np.array([m[1], m[2]])
        gradient = np.array([
            (_k_factor_from_moments(m[1] + step[0], m[2])
             - _k_factor_from_moments(m[1] - step[0], m[2])) / (2 * step[0]),
            (_k_factor_from_moments(m[1], m[2] + step[1])
             - _k_factor_from_moments(m[1], m[2] - step[1])) / (2 * step[1])])
        covariance = np.array([[m[2] - m[1] ** 2, m[3] - m[1] * m[2]],
                               [m[3] - m[1] * m[2], m[4] - m[2] ** 2]])
        estimate = _k_factor_from_moments(samples[0].mean(), (samples[0] ** 2).mean())
        _assert_within(estimate, k1,
                       math.sqrt(gradient @ covariance @ gradient / samples.shape[1]),
                       f"NTN-TDL-{model} K-factor K1 of the first tap")


def _sector_means(x):
    """E[exp(j x cos(alpha_n))], n = 1 to N, for the angles of arrival of the sum of
    sinusoids, alpha_n uniform in [2 pi n / N - pi / N, 2 pi n / N + pi / N), by
    Gauss-Legendre quadrature"""
    nodes, weights = np.polynomial.legendre.leggauss(32)
    n = np.arange(1, NUM_SINUSOIDS + 1)[:, None]
    alpha = 2 * np.pi * n / NUM_SINUSOIDS + np.pi / NUM_SINUSOIDS * nodes[None, :]
    return 0.5 * np.sum(weights * np.exp(1j * x * np.cos(alpha)), axis=1)


def _correlation_moments(p_specular, p_rayleigh, omega_d, lag, shift=0.0):
    """E[W], E|W|^2 and E[W^2] of W = h(t + lag) h*(t) for a tap whose Rayleigh part is
    a sum of N sinusoids with the maximum Doppler frequency omega_d [rad/s] and whose
    specular path (power p_specular) turns at 0.7 omega_d, the whole tap turned by the
    shift [Hz]. The Rayleigh product r(t + lag) r*(t) / p_rayleigh has the mean J0 and
    the moments of its N^2 terms, the specular path an independent uniform phase."""
    c = _sector_means(omega_d * lag)
    d = _sector_means(2 * omega_d * lag)
    n = NUM_SINUSOIDS
    j0 = float(mpmath.besselj(0, omega_d * lag))
    assert abs(c.sum() / n - j0) <= 1e-12          # the quadrature against J0
    rayleigh_abs2 = (abs(c.sum()) ** 2 - np.sum(np.abs(c) ** 2) + n ** 2) / n ** 2
    rayleigh_square = (2 * (c.sum() ** 2 - np.sum(c ** 2)) + d.sum()) / n ** 2
    specular = p_specular * np.exp(0.7j * omega_d * lag)
    rest_mean = p_rayleigh * j0
    rest_abs2 = 2 * p_specular * p_rayleigh + p_rayleigh ** 2 * rayleigh_abs2
    rest_square = 2 * specular * p_rayleigh * j0 + p_rayleigh ** 2 * rayleigh_square
    turn = np.exp(2j * np.pi * shift * lag)
    mean = (specular + rest_mean) * turn
    abs2 = abs(specular) ** 2 + 2 * (np.conj(specular) * rest_mean).real + rest_abs2
    square = (specular ** 2 + 2 * specular * rest_mean + rest_square) * turn ** 2
    return mean, abs2, square


def _assert_correlation(samples, moments, label):
    """The real and the imaginary part of the mean of the samples of W"""
    mean, abs2, square = moments
    sigma_re = math.sqrt((abs2 + square.real) / 2 - mean.real ** 2)
    sigma_im = math.sqrt((abs2 - square.real) / 2 - mean.imag ** 2)
    assert_mean(samples.real, mean.real, sigma_re, label + ", real part")
    assert_mean(samples.imag, mean.imag, sigma_im, label + ", imaginary part")


@pytest.mark.parametrize("model", "AC")
def test_doppler_spectrum(model):
    # The classical spectrum of TR 38.901 clause 7.7.2 for a UT at 30 m/s, f_D = 200.138
    # Hz at 2 GHz: E[h(t + tau) h*(t)] = P J0(2 pi f_D tau) for every Rayleigh tap, and
    # P_L exp(j 2 pi 0.7 f_D tau) + P_1 J0(2 pi f_D tau) for the first tap of NTN-TDL-C,
    # whose imaginary part comes only from the specular path at f_S = 0.7 f_D: the mean
    # frequency of the tap is the share P_L / (P_L + P_1) of 0.7 f_D. At the lags 0.5,
    # 1.0 and 1.5 ms, without the shift of the satellite.
    _, powers, specular = _profile(model)
    omega_d = 2 * np.pi * UT_SPEED / SPEED_OF_LIGHT * CARRIER_FREQUENCY
    h = _time_samples(model)
    for k in range(len(powers)):
        p_specular = specular if k == 0 else 0.0
        for step in range(1, DOPPLER_STEPS):
            lag = step / DOPPLER_SAMPLING
            _assert_correlation(h[:, k, step] * np.conj(h[:, k, 0]),
                                _correlation_moments(p_specular, powers[k] - p_specular,
                                                     omega_d, lag),
                                f"NTN-TDL-{model} tap {k + 1}, lag {lag * 1e3:g} ms")


@pytest.mark.parametrize("model", "AC")
def test_shift_with_moving_ut(model):
    # With a moving UT the shift of the satellite (600 km, 50 degrees, 2 GHz) turns the
    # correlation of every tap over a lag tau by 2 pi f_d tau, so that the mean
    # frequency of every tap moves by f_d. The shift is applied after the random draws:
    # the same draws without it give the same taps up to the factor
    # exp(j 2 pi f_d n / f_s).
    f_d = _satellite_shift(SATELLITE_HEIGHT, ELEVATION_ANGLE, CARRIER_FREQUENCY)
    _, powers, specular = _profile(model)
    omega_d = 2 * np.pi * UT_SPEED / SPEED_OF_LIGHT * CARRIER_FREQUENCY
    seed = config.seed
    unshifted = _time_samples(model)
    config.seed = seed
    h = _time_samples(model, None)
    factor = np.exp(2j * np.pi * f_d * np.arange(DOPPLER_STEPS) / DOPPLER_SAMPLING)
    np.testing.assert_allclose(h, unshifted * factor, rtol=0.0, atol=1e-12)
    for k in range(len(powers)):
        p_specular = specular if k == 0 else 0.0
        for step in range(1, DOPPLER_STEPS):
            lag = step / DOPPLER_SAMPLING
            _assert_correlation(h[:, k, step] * np.conj(h[:, k, 0]),
                                _correlation_moments(p_specular, powers[k] - p_specular,
                                                     omega_d, lag, f_d),
                                f"NTN-TDL-{model} tap {k + 1} with the shift, lag "
                                f"{lag * 1e3:g} ms")
