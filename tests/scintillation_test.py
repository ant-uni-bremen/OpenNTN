# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the scintillation loss of TR 38.811 V15.4.0, clause 6.6.6.
#
# Tropospheric scintillation, against Table 6.6.6.2.1-1, read from
# tests/spec_tables/tr38811_v15.4.0.json. Clause 6.6.6.2.1: "As baseline for
# system-level simulations above 6 GHz, the fading due to scintillation PL_S is equal to
# the tropospheric attenuation level at 99% of the time derived from Figure 6.6.6.2.1-1
# and summarized in Table 6.6.6.2.1-1". The table is given for 20 GHz in Toulouse; TR
# 38.821 V16.2.0 uses it at 20 GHz (downlink) and 30 GHz (uplink) alike (Tables
# 6.1.3.2-1 and 6.1.3.3-1). The tables of TR 38.811 are given at the reference elevation
# angles 10 to 90 degrees, and a UE takes the values of the reference angle nearest to
# its elevation angle (clauses 6.6.1 and 6.6.2); an angle midway between two reference
# angles takes the higher one.
#
# Ionospheric scintillation, clause 6.6.6.1.4: "For system-level simulations below
# 6 GHz, PL_S is equal to A_IS from Equation (6.6-13) for latitudes of maximum +-20 deg.
# For latitudes between +-20 deg and +-60 deg of latitude, PL_S = 0. [...] the choice of
# PL_S = 0 is also applied for latitude above +-60 deg." The clause places the region
# of the model around the magnetic equator, so the latitude is the latitude relative to
# the magnetic equator. A_IS = P_fluc / sqrt(2) (eq. (6.6-13)), with P_fluc at 4 GHz
# multiplied by (f/4)^-1.5 (step 2); the example of the clause at 2 GHz: A_IS =
# P_fluc(4 GHz) x 0.5^-1.5 / sqrt(2) = 1.1 x 0.5^-1.5 / sqrt(2) = 2.2 dB.
import json
import math
import os

import numpy as np
import pytest
import torch
from sionna.phy import config

from openntn import Antenna, DenseUrban, SubUrban, Urban
from openntn.utils import gen_single_sector_topology

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_tables",
                       "tr38811_v15.4.0.json"), encoding="ascii") as _f:
    _DATA = json.load(_f)
ELEVATION_ANGLES = _DATA["elevation_angles_deg"]
# Values as printed, for example "1.08 dB"
TABLE = dict(zip(ELEVATION_ANGLES, (float(value.split()[0]) for value in
                                    _DATA["tables"]["6.6.6.2.1-1"]["rows"]
                                    ["tropospheric_attenuation"])))

SCENARIOS = {"dur": DenseUrban, "urb": Urban, "sur": SubUrban}
KA_BAND = [("downlink", 20e9), ("uplink", 30e9)]
DIRECTIONS = ["downlink", "uplink"]
# P_fluc at 4 GHz [dB], the value of the example of clause 6.6.6.1.4
P_FLUC_4GHZ = 1.1


def _model(scenario, direction, carrier_frequency, elevation_angle, latitude=None,
           bs_height=600e3):
    config.precision = "double"
    antenna = Antenna(polarization="single", polarization_type="V",
                      antenna_pattern="38.901", carrier_frequency=carrier_frequency)
    model = SCENARIOS[scenario](carrier_frequency=carrier_frequency, ut_array=antenna,
                                bs_array=antenna, direction=direction,
                                elevation_angle=elevation_angle)
    model.set_topology(*gen_single_sector_topology(batch_size=2, num_ut=3,
                                                   scenario=scenario,
                                                   elevation_angle=elevation_angle,
                                                   bs_height=bs_height),
                       latitude=latitude)
    return model


def _scintillation_loss(scenario, direction, carrier_frequency, elevation_angle,
                        latitude=None, bs_height=600e3):
    model = _model(scenario, direction, carrier_frequency, elevation_angle, latitude,
                   bs_height)
    return model._scenario.scintillation_pathloss.cpu().numpy()


def _ionospheric_loss(carrier_frequency):
    """A_IS [dB] of eq. (6.6-13) at the carrier frequency [Hz]."""
    return P_FLUC_4GHZ * (carrier_frequency / 4e9) ** -1.5 / math.sqrt(2.0)


def test_table():
    assert ELEVATION_ANGLES == list(range(10, 91, 10))
    assert TABLE[10] == 1.08 and TABLE[90] == 0.12


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("direction,carrier_frequency", KA_BAND)
@pytest.mark.parametrize("elevation_angle", ELEVATION_ANGLES)
def test_reference_elevation_angles(scenario, direction, carrier_frequency,
                                    elevation_angle):
    # Every link has the value of the table at its reference elevation angle, exactly
    loss = _scintillation_loss(scenario, direction, carrier_frequency,
                               float(elevation_angle))
    expected = TABLE[elevation_angle]
    assert np.all(loss == expected), (f"{scenario} {direction} {elevation_angle} deg: "
                                      f"{np.unique(loss)} dB, Table 6.6.6.2.1-1 gives "
                                      f"{expected} dB")


@pytest.mark.parametrize("direction,carrier_frequency", KA_BAND)
@pytest.mark.parametrize("elevation_angle,reference_angle", [
    (12.5, 10), (24.0, 20), (25.0, 30), (26.0, 30), (45.0, 50), (65.0, 70), (85.0, 90)])
def test_nearest_reference_elevation_angle(direction, carrier_frequency, elevation_angle,
                                           reference_angle):
    # Between two reference angles the nearer one applies; at 25, 45, 65 and 85 degrees
    # the higher one
    loss = _scintillation_loss("urb", direction, carrier_frequency, elevation_angle)
    expected = TABLE[reference_angle]
    assert np.all(loss == expected), (f"{direction} {elevation_angle} deg: "
                                      f"{np.unique(loss)} dB, Table 6.6.6.2.1-1 gives "
                                      f"{expected} dB at {reference_angle} deg")


@pytest.mark.parametrize("direction,carrier_frequency", KA_BAND)
def test_tropospheric_scintillation_independent_of_latitude(direction, carrier_frequency):
    for latitude in (0.0, 47.0, -70.0):
        loss = _scintillation_loss("urb", direction, carrier_frequency, 30.0, latitude)
        assert np.all(loss == TABLE[30]), (latitude, np.unique(loss))


def test_ionospheric_example():
    # The example of clause 6.6.6.1.4: 2.2 dB at 2 GHz
    assert abs(_ionospheric_loss(2e9) - 2.2) <= 1e-12 * 2.2


@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("carrier_frequency", [2e9, 2.2e9])
@pytest.mark.parametrize("latitude", [0.0, 20.0, -20.0, 21.0, 47.0, 60.0, 61.0, 90.0,
                                      -70.0, None])
def test_ionospheric_scintillation(scenario, direction, carrier_frequency, latitude):
    # A_IS within 20 degrees of the magnetic equator, 0 at higher latitudes; None is the
    # default latitude, 47 degrees
    loss = _scintillation_loss(scenario, direction, carrier_frequency, 30.0, latitude)
    if latitude is not None and abs(latitude) <= 20.0:
        expected = _ionospheric_loss(carrier_frequency)
        assert np.all(np.abs(loss - expected) <= 1e-12 * expected), (
            f"{scenario} {direction} {carrier_frequency / 1e9:g} GHz, latitude "
            f"{latitude} deg: {np.unique(loss)} dB, clause 6.6.6.1.4 gives {expected} dB")
    else:
        assert np.all(loss == 0.0), (
            f"{scenario} {direction} {carrier_frequency / 1e9:g} GHz, latitude "
            f"{latitude} deg: {np.unique(loss)} dB, clause 6.6.6.1.4 gives 0 dB")


# Aerial platforms (8 km to 50 km) and satellites (from 160 km) that the models accept
AERIAL_HEIGHTS = [8e3, 20e3, 50e3]
SATELLITE_HEIGHTS = [160e3, 600e3, 35786e3]


@pytest.mark.parametrize("bs_height", AERIAL_HEIGHTS + SATELLITE_HEIGHTS)
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("carrier_frequency", [1.995e9, 2e9, 2.185e9])
@pytest.mark.parametrize("latitude", [0.0, 20.0, -20.0, 21.0, 47.0])
def test_ionospheric_scintillation_by_platform(bs_height, direction, carrier_frequency,
                                               latitude):
    # Table 6.10.1-1 gives the scintillation as negligible for the deployment of UAS
    # including HAPS below 6 GHz: links to an aerial platform stay below the ionosphere
    # and get no ionospheric scintillation loss at any latitude. Links to a satellite get
    # A_IS within 20 degrees of the magnetic equator, and 0 at higher latitudes.
    loss = _scintillation_loss("urb", direction, carrier_frequency, 30.0, latitude,
                               bs_height)
    label = (f"{bs_height / 1e3:g} km, {direction}, {carrier_frequency / 1e9:g} GHz, "
             f"latitude {latitude} deg: {np.unique(loss)} dB")
    if bs_height in SATELLITE_HEIGHTS and abs(latitude) <= 20.0:
        expected = _ionospheric_loss(carrier_frequency)
        assert np.all(np.abs(loss - expected) <= 1e-12 * expected), (
            f"{label}, clause 6.6.6.1.4 gives {expected} dB")
    else:
        assert np.all(loss == 0.0), f"{label}, expected 0 dB"


@pytest.mark.parametrize("bs_height", AERIAL_HEIGHTS)
@pytest.mark.parametrize("direction,carrier_frequency", KA_BAND)
def test_tropospheric_scintillation_of_aerial_platforms(bs_height, direction,
                                                        carrier_frequency):
    # From 6 GHz on, aerial platforms get the tropospheric scintillation of Table
    # 6.6.6.2.1-1 as satellites do, at every latitude
    for latitude in (0.0, 47.0):
        loss = _scintillation_loss("urb", direction, carrier_frequency, 30.0, latitude,
                                   bs_height)
        assert np.all(loss == TABLE[30]), (bs_height, latitude, np.unique(loss))


@pytest.mark.parametrize("latitude", [91.0, -91.0, 180.0, float("nan")])
def test_latitude_validated(latitude):
    model = _model("urb", "downlink", 2e9, 30.0)
    with pytest.raises(ValueError):
        model.set_topology(latitude=latitude)
    assert model._scenario.latitude == 47


@pytest.mark.parametrize("latitude", [90.0, -90.0])
def test_latitude_limits_accepted(latitude):
    model = _model("urb", "downlink", 2e9, 30.0)
    model.set_topology(latitude=latitude)
    assert model._scenario.latitude == latitude


def test_latitude_alone_updates_scintillation_loss():
    # A call that passes only the latitude recomputes the loss, and draws nothing again
    model = _model("urb", "downlink", 2e9, 30.0)
    sc = model._scenario
    lsp_names = ("ds", "asd", "asa", "sf", "k_factor", "zsa", "zsd")

    def state():
        return ([sc.los.clone(), sc.basic_pathloss.clone(), sc.gas_pathloss.clone()]
                + [getattr(model._lsp, n).clone() for n in lsp_names])

    before = state()
    assert np.all(sc.scintillation_pathloss.cpu().numpy() == 0.0)
    model.set_topology(latitude=0.0)
    expected = _ionospheric_loss(2e9)
    loss = sc.scintillation_pathloss.cpu().numpy()
    assert np.all(np.abs(loss - expected) <= 1e-12 * expected), np.unique(loss)
    model.set_topology(latitude=47.0)
    assert np.all(sc.scintillation_pathloss.cpu().numpy() == 0.0)
    for b, a in zip(before, state()):
        assert torch.equal(b, a)
