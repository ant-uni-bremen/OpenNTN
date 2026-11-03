# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of openntn.utils.link_level_parameters, the delay spread, the angular spreads
# and the K-factor of 3GPP TR 38.811 V15.4.0, clause 6.7.2, for the link-level models
# TDL and CDL: clause 6.9.1 determines "suitable DS, ASA, ZSA, and Rician K-factor in
# case of LOS" from the channel model parameters of clause 6.7.2. The expected values
# are computed in the test from the tables of spec_tables/tr38811_v15.4.0.json.
#
# - The values of Tables 6.7.2-1a to 6.7.2-6b (a: S band, b: Ka band) at the nine
#   reference elevation angles: the medians 10 ** mu_lgDS [s], 10 ** mu_lgASA [deg] and
#   10 ** mu_lgZSA [deg] of the log-normal spreads, as TR 38.901 V16.1.0, Table 7.5-6,
#   defines lgDS = log10(DS / 1 s), lgASA = log10(ASA / 1 deg) and lgZSA = log10(ZSA /
#   1 deg), and the mean mu_K of the K-factor [dB]; None for the NLOS tables, which give
#   no K-factor.
# - The reference elevation angle by the test's own rule: the angle of the tables
#   nearest to the elevation angle (clauses 6.6.1 and 6.6.2), the higher one when the
#   elevation angle lies midway between two.
# - The edges of the S band and the Ka band, the checks of the arguments, the argument
#   los taken by keyword only, and the types of the values.
# - The values accepted by the models: TDL and CDL built with them and called once,
#   with finite outputs and the K-factor of the LOS models equal to the returned one.
import inspect
import json
import math
import os
import re

import numpy as np
import pytest
import torch

from openntn import CDL, TDL, Antenna
from openntn.utils import link_level_parameters

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_tables",
                       "tr38811_v15.4.0.json"), encoding="ascii") as _f:
    SPEC = json.load(_f)
TABLES = SPEC["tables"]
ELEVATIONS = SPEC["elevation_angles_deg"]

# The tables of each scenario and link state, with the suffix "a" in the S band and "b"
# in the Ka band
SCENARIO_TABLES = {("dur", True): "6.7.2-1", ("dur", False): "6.7.2-2",
                   ("urb", True): "6.7.2-3", ("urb", False): "6.7.2-4",
                   ("sur", True): "6.7.2-5", ("sur", False): "6.7.2-6"}
BANDS = {"a": 2e9, "b": 30e9}
CASES = [(scenario, los, suffix)
         for (scenario, los) in SCENARIO_TABLES for suffix in BANDS]
CASE_IDS = [SCENARIO_TABLES[(scenario, los)] + suffix
            for scenario, los, suffix in CASES]
SATELLITE_HEIGHT = 600e3
SAMPLING_FREQUENCY = 1e6


def _expected(scenario, los, suffix, reference):
    """The medians of the delay spread [s], the ASA and the ZSA [deg] and the mean
    K-factor [dB] of the table at a reference elevation angle; None for the K-factor of
    an NLOS table, which has no row mu_K or holds N/A in it"""
    rows = TABLES[SCENARIO_TABLES[(scenario, los)] + suffix]["rows"]
    index = ELEVATIONS.index(reference)
    k_row = rows.get("mu_K")
    if los:
        k_factor = float(k_row[index])
    else:
        assert k_row is None or k_row[index] == "N/A"
        k_factor = None
    return (10.0 ** float(rows["mu_lgDS"][index]),
            (10.0 ** float(rows["mu_lgASA"][index]),
             10.0 ** float(rows["mu_lgZSA"][index])),
            k_factor)


def _assert_values(returned, expected):
    delay_spread, (asa, zsa), k_factor = returned
    expected_delay_spread, (expected_asa, expected_zsa), expected_k_factor = expected
    assert delay_spread == pytest.approx(expected_delay_spread, rel=1e-12, abs=0.0)
    assert asa == pytest.approx(expected_asa, rel=1e-12, abs=0.0)
    assert zsa == pytest.approx(expected_zsa, rel=1e-12, abs=0.0)
    if expected_k_factor is None:
        assert k_factor is None
    else:
        assert k_factor == pytest.approx(expected_k_factor, rel=1e-12, abs=0.0)


def _reference_elevation(elevation_angle):
    """The reference elevation angle of the tables nearest to the elevation angle, the
    higher one at a tie"""
    return min(range(10, 100, 10), key=lambda r: (abs(r - elevation_angle), -r))


@pytest.mark.parametrize("scenario,los,suffix", CASES, ids=CASE_IDS)
def test_tables_of_the_scenarios(scenario, los, suffix):
    # The captions of the tables name the scenario, the link state and the band
    caption = TABLES[SCENARIO_TABLES[(scenario, los)] + suffix]["caption"]
    match = re.search(r"for (Dense Urban|Urban|Suburban) Scenario \((LOS|NLOS)\) "
                      r"(?:in|at) (S|Ka) band$", caption)
    assert match, caption
    assert match.groups() == ({"dur": "Dense Urban", "urb": "Urban",
                               "sur": "Suburban"}[scenario], "LOS" if los else "NLOS",
                              {"a": "S", "b": "Ka"}[suffix])


@pytest.mark.parametrize("elevation_angle", ELEVATIONS)
@pytest.mark.parametrize("scenario,los,suffix", CASES, ids=CASE_IDS)
def test_values_of_the_tables(scenario, los, suffix, elevation_angle):
    returned = link_level_parameters(scenario, BANDS[suffix], float(elevation_angle),
                                     los=los)
    _assert_values(returned, _expected(scenario, los, suffix, elevation_angle))


@pytest.mark.parametrize("elevation_angle", [10.0, 14.999, 15.0, 15.001, 25.0, 44.999,
                                             45.0, 85.0, 89.999, 90.0])
def test_reference_elevation_angle(elevation_angle):
    reference = _reference_elevation(elevation_angle)
    # The second nearest reference angle, whose values differ from those of the nearest
    # in some table, so that the choice between the two shows in the values
    other = min((r for r in range(10, 100, 10) if r != reference),
                key=lambda r: abs(r - elevation_angle))
    assert any(_expected(*case, reference) != _expected(*case, other) for case in CASES)
    for scenario, los, suffix in CASES:
        returned = link_level_parameters(scenario, BANDS[suffix], elevation_angle,
                                         los=los)
        _assert_values(returned, _expected(scenario, los, suffix, reference))


@pytest.mark.parametrize("carrier_frequency,suffix", [(1.9e9, "a"), (4e9, "a"),
                                                      (19e9, "b"), (40e9, "b")])
def test_band_edges(carrier_frequency, suffix):
    for scenario, los, _ in CASES:
        returned = link_level_parameters(scenario, carrier_frequency, 50.0, los=los)
        _assert_values(returned, _expected(scenario, los, suffix, 50))


@pytest.mark.parametrize("carrier_frequency", [1.89e9, 4.01e9, 10e9, 18.9e9, 40.1e9])
def test_outside_the_bands(carrier_frequency):
    with pytest.raises(ValueError, match="carrier_frequency must be in the S band"):
        link_level_parameters("urb", carrier_frequency, 50.0, los=True)


ARGUMENTS = dict(scenario="urb", carrier_frequency=2e9, elevation_angle=30.0, los=True)
# The error of a wrong scenario names the three scenarios
SCENARIO_ERROR = "scenario must be one of 'dur', 'urb' and 'sur'"
INVALID = [
    (dict(scenario=1), TypeError, SCENARIO_ERROR),
    (dict(scenario="rma"), ValueError, SCENARIO_ERROR),
    (dict(carrier_frequency=True), TypeError, "carrier_frequency"),
    (dict(carrier_frequency="2e9"), TypeError, "carrier_frequency"),
    (dict(carrier_frequency=math.nan), ValueError, "carrier_frequency"),
    (dict(elevation_angle=True), TypeError, "elevation_angle"),
    (dict(elevation_angle="30"), TypeError, "elevation_angle"),
    (dict(elevation_angle=9.99), ValueError, "elevation_angle"),
    (dict(elevation_angle=90.01), ValueError, "elevation_angle"),
    (dict(los=1), TypeError, "los"),
    (dict(los=0), TypeError, "los"),
]


@pytest.mark.parametrize("change,error,message", INVALID,
                         ids=[f"{list(c)[-1]}-{c[list(c)[-1]]!r}"
                              for c, _, _ in INVALID])
def test_invalid_argument(change, error, message):
    with pytest.raises(error, match=message):
        link_level_parameters(**{**ARGUMENTS, **change})


def test_los_by_keyword_only():
    parameters = inspect.signature(link_level_parameters).parameters
    assert list(parameters) == ["scenario", "carrier_frequency", "elevation_angle",
                                "los"]
    assert all(parameters[name].kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
               for name in ("scenario", "carrier_frequency", "elevation_angle"))
    assert parameters["los"].kind == inspect.Parameter.KEYWORD_ONLY
    assert parameters["los"].default is inspect.Parameter.empty
    with pytest.raises(TypeError):
        link_level_parameters("urb", 2e9, 30.0, True)
    with pytest.raises(TypeError):
        link_level_parameters("urb", 2e9, 30.0)


@pytest.mark.parametrize("los", [True, False])
@pytest.mark.parametrize("carrier_frequency,elevation_angle",
                         [(2e9, 30.0), (np.float32(20e9), np.int64(40)),
                          (19_000_000_000, 90)])
def test_types(carrier_frequency, elevation_angle, los):
    # Python floats, ready for the arguments of TDL and CDL, also for NumPy numbers and
    # integers as inputs
    delay_spread, angular_spreads, k_factor = link_level_parameters(
        "dur", carrier_frequency, elevation_angle, los=los)
    assert type(delay_spread) is float
    assert type(angular_spreads) is tuple and len(angular_spreads) == 2
    assert all(type(value) is float for value in angular_spreads)
    if los:
        assert type(k_factor) is float
    else:
        assert k_factor is None


# Every table at 30 degrees, and the extremes of the tables at 10 degrees: the smallest
# ZSA, 0.0126 deg in the suburban NLOS table of the S band, and the largest K-factors,
# 31.83 dB and 40.18 dB in the urban LOS tables
MODEL_CASES = ([(case, 30.0) for case in CASES]
               + [(("sur", False, "a"), 10.0), (("urb", True, "a"), 10.0),
                  (("urb", True, "b"), 10.0)])


@pytest.mark.parametrize("case,elevation_angle", MODEL_CASES,
                         ids=[f"{SCENARIO_TABLES[case[:2]]}{case[2]}-{e:g}"
                              for case, e in MODEL_CASES])
def test_values_accepted_by_the_models(case, elevation_angle):
    # The LOS models C and D with the values of a LOS table, the NLOS models A and B
    # with those of an NLOS table, in double precision with a satellite at 600 km
    scenario, los, suffix = case
    carrier_frequency = BANDS[suffix]
    delay_spread, angular_spreads, k_factor = link_level_parameters(
        scenario, carrier_frequency, elevation_angle, los=los)
    for model in ("C", "D") if los else ("A", "B"):
        tdl = TDL(model, delay_spread, carrier_frequency, elevation_angle,
                  SATELLITE_HEIGHT, k_factor=k_factor, precision="double")
        antenna = Antenna("single", "V", "omni", carrier_frequency, precision="double")
        cdl = CDL(model, delay_spread, carrier_frequency, elevation_angle,
                  SATELLITE_HEIGHT, ut_array=antenna, bs_array=antenna,
                  angular_spreads=angular_spreads, k_factor=k_factor,
                  precision="double")
        for channel_model in (tdl, cdl):
            a, tau = channel_model(2, 3, SAMPLING_FREQUENCY)
            assert bool(torch.isfinite(a).all()) and bool(torch.isfinite(tau).all())
            if los:
                k_model = 10.0 * math.log10(float(channel_model.k_factor))
                assert k_model == pytest.approx(k_factor, abs=1e-9)
