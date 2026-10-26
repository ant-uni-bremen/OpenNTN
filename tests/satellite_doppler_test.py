# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the Doppler shift due to the motion of the satellite against values computed
# in the test from TR 38.811 V15.4.0 and TR 38.821 V16.2.0.
#
# TR 38.811, clauses 6.7.1 and 6.9.2: "Additional Doppler shift due to satellite motion
# should be taken into account according to the following formula: f_d,shift =
# (v_sat/c) x (R/(R+h) cos alpha_model) x f_c". That is the largest shift, for a UT in
# the orbital plane (clause 5.3.4.4). The channel applies f_d cos(2 pi U) with U uniform
# per batch example, the direction of the orbit relative to the UTs, and the speed v_sat
# of a circular orbit, sqrt(G M / (R + h)) with the constants of
# openntn.utils.compute_satellite_speed. Clause 5.3.2.3: "In principle, the
# Geostationary satellite is fixed and therefore no Doppler shift is induced except that
# due to possible UE motion." TR 38.821, Table 6.1.1.1-8, gives the largest Doppler
# shift for elevation angles of at least 10 degrees: 24 ppm at 600 km, 20 ppm at 1200
# km and 0.15 ppm for GEO.
import math

import numpy as np
import pytest
import torch
from sionna.phy import SPEED_OF_LIGHT, config

import openntn.channel_coefficients as channel_coefficients
from openntn import Antenna, DenseUrban
from openntn.utils import gen_single_sector_topology
from statistical_checks import assert_mean

R_EARTH = 6371e3
GM = 6.6743e-11 * 5.972e24       # G and the mass of the Earth of compute_satellite_speed
CARRIER_FREQUENCY = 2.2e9
SAMPLING_FREQUENCY = 1e6


def _formula(height, elevation_angle):
    """(v_sat/c) (R/(R+h)) cos(alpha): the Doppler shift over f_c of clauses 6.7.1, 6.9.2."""
    v_sat = math.sqrt(GM / (R_EARTH + height))
    return v_sat / SPEED_OF_LIGHT * R_EARTH / (R_EARTH + height) \
        * math.cos(math.radians(elevation_angle))


def _topology_speeds(height, elevation_angle, batch_size):
    """Speed of the satellite along the line of sight of a Topology, per batch example."""
    zeros = torch.zeros(batch_size, 1, 1, dtype=torch.float64)
    topology = channel_coefficients.Topology(
        velocities=torch.zeros(batch_size, 1, 3, dtype=torch.float64), moving_end="rx",
        los_aoa=zeros, los_aod=zeros, los_zoa=zeros, los_zod=zeros,
        los=torch.ones(batch_size, 1, 1, dtype=torch.bool), distance_3d=zeros,
        tx_orientations=torch.zeros(batch_size, 1, 3, dtype=torch.float64),
        rx_orientations=torch.zeros(batch_size, 1, 3, dtype=torch.float64),
        bs_height=torch.tensor([height], dtype=torch.float64),
        elevation_angle=elevation_angle, doppler_enabled=True, los_phase=zeros)
    return topology.sat_speed.cpu().numpy()


def _channel(height, elevation_angle, direction, monkeypatch, batch_size=8):
    """Channel coefficients at two samples with the UTs at rest, and the uniform numbers
    U of the orbit directions that the channel drew."""
    antenna = Antenna(polarization="single", polarization_type="V", antenna_pattern="omni",
                      carrier_frequency=CARRIER_FREQUENCY)
    model = DenseUrban(carrier_frequency=CARRIER_FREQUENCY, ut_array=antenna,
                       bs_array=antenna, direction=direction,
                       elevation_angle=elevation_angle, enable_pathloss=False,
                       enable_shadow_fading=False)
    topology = list(gen_single_sector_topology(batch_size=batch_size, num_ut=3,
                                               scenario="dur",
                                               elevation_angle=elevation_angle,
                                               bs_height=height))
    topology[4] = torch.zeros_like(topology[4])                  # UT velocities
    model.set_topology(*topology)
    draws = []
    original = channel_coefficients.rand

    def record(shape, *args, **kwargs):
        values = original(shape, *args, **kwargs)
        if list(shape) == [batch_size]:                          # the orbit directions
            draws.append(values.detach().cpu().numpy())
        return values

    monkeypatch.setattr(channel_coefficients, "rand", record)
    h, _ = model(2, SAMPLING_FREQUENCY)
    assert len(draws) == 1
    return h.cpu().numpy(), draws[0]


@pytest.mark.parametrize("direction", ["downlink", "uplink"])
@pytest.mark.parametrize("height", [600e3, 1200e3, 10000e3])
@pytest.mark.parametrize("elevation_angle", [10.0, 30.0, 60.0])
def test_phase_advance(height, elevation_angle, direction, monkeypatch):
    # With the UTs at rest, the phase of every path, antenna pair and link of a batch
    # example advances per sample by 2 pi f_d / f_s, f_d = f_c (v_sat/c) (R/(R+h))
    # cos(alpha) cos(2 pi U).
    config.precision = "double"
    h, u = _channel(height, elevation_angle, direction, monkeypatch)
    f_d = CARRIER_FREQUENCY * _formula(height, elevation_angle) * np.cos(2 * np.pi * u)
    worst = 0.0
    for b in range(h.shape[0]):
        used = np.abs(h[b, ..., 0]) > 0
        step = h[b, ..., 1][used] * np.conj(h[b, ..., 0][used])
        expected = np.exp(2j * np.pi * f_d[b] / SAMPLING_FREQUENCY)
        worst = max(worst, float(np.abs(np.angle(step * np.conj(expected))).max()))
    assert worst <= 1e-9, (f"{height / 1e3:g} km, {elevation_angle:g} deg, {direction}: "
                           f"phase step off by {worst:.3g} rad")


@pytest.mark.parametrize("height,limit_ppm", [(600e3, 24.0), (1200e3, 20.0 * 1.005)])
def test_largest_shift(height, limit_ppm):
    # Over 20,000 batch examples the largest |f_d| / f_c reaches (v_sat/c) (R/(R+h))
    # cos(alpha) (within 0.1 %), and at 10 degrees it stays within TR 38.821 Table
    # 6.1.1.1-8 (24 ppm at 600 km, 20 ppm at 1200 km; +0.5 %, the table is rounded).
    config.precision = "double"
    speeds = _topology_speeds(height, 10.0, 20000)
    largest = float(np.abs(speeds).max()) / SPEED_OF_LIGHT
    expected = _formula(height, 10.0)
    assert abs(largest / expected - 1.0) <= 1e-3, (largest * 1e6, expected * 1e6)
    assert largest * 1e6 <= limit_ppm, f"{height / 1e3:g} km: {largest * 1e6:.2f} ppm"


def test_geostationary(monkeypatch):
    # No Doppler shift from a geostationary satellite (TR 38.811 clause 5.3.2.3; TR 38.821
    # Table 6.1.1.1-8: at most 0.15 ppm): with the UTs at rest the channel is constant.
    config.precision = "double"
    speeds = _topology_speeds(35786e3, 10.0, 1000)
    assert float(np.abs(speeds).max()) / SPEED_OF_LIGHT * 1e6 <= 0.15
    h, _ = _channel(35786e3, 10.0, "downlink", monkeypatch)
    assert np.array_equal(h[..., 1], h[..., 0])


def test_direction_of_the_orbit():
    # The direction cosine cos(2 pi U), U uniform: the mean of |cos(2 pi U)| is 2/pi, its
    # standard deviation sqrt(1/2 - 4/pi^2).
    config.precision = "double"
    speeds = _topology_speeds(600e3, 30.0, 20000)
    cosine = np.abs(speeds) / (SPEED_OF_LIGHT * _formula(600e3, 30.0))
    assert_mean(cosine, 2.0 / math.pi, math.sqrt(0.5 - 4.0 / math.pi ** 2),
                "|cos(2 pi U)| of the orbit direction")
