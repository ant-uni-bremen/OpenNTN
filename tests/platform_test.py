# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the platform that one call of the channel models simulates, and of the parts
# of the channel that apply to satellites only, against values computed in the test.
#
# TR 38.811 V15.4.0, clause 3.1, places aerial platforms "typically at an altitude
# between 8 to 50 km" and LEO satellites "typically at an altitude between 500 km to
# 2000 km"; Table 4.5-1 gives typical altitudes (LEO 600 km to 1500 km, MEO 7000 km to
# 20 000 km, GEO 35 786 km), Table 5.1-1 the reference deployments (GEO, non-GEO down to
# 600 km, UAS 8 km to 50 km). The models simulate one platform per call, at one height
# in every batch example, and accept satellites from 160 km (very low orbits included)
# to 35 786 km and aerial platforms from 8 km to 50 km. For a satellite the departure
# spreads at the satellite are zero (NOTE 8 of Tables 6.7.2-1a to 6.7.2-8b) and the
# Doppler shift due to the motion of the satellite applies. The Faraday rotation of
# clause 6.8.2 ("In case of propagation for spaceborne BS above the ionosphere") applies
# from 600 km, the lowest satellite altitude of TR 38.811, and to no satellite below,
# which flies within the ionosphere.
import json
import math
import os

import numpy as np
import pytest
import torch
from sionna.phy import config

from openntn import Antenna, DenseUrban
from openntn.utils import gen_single_sector_topology
from statistical_checks import assert_mean, assert_std

R_EARTH = 6371e3
ELEVATION_ANGLE = 30.0
CARRIER_FREQUENCY = 2.2e9


def _model(direction="downlink"):
    antenna = Antenna(polarization="single", polarization_type="V", antenna_pattern="omni",
                      carrier_frequency=CARRIER_FREQUENCY)
    return DenseUrban(carrier_frequency=CARRIER_FREQUENCY, ut_array=antenna,
                      bs_array=antenna, direction=direction,
                      elevation_angle=ELEVATION_ANGLE, enable_pathloss=False,
                      enable_shadow_fading=False)


def _topology(height, batch_size=4, num_ut=3):
    return list(gen_single_sector_topology(batch_size=batch_size, num_ut=num_ut,
                                           scenario="dur",
                                           elevation_angle=ELEVATION_ANGLE,
                                           bs_height=height))


def test_one_platform_per_call():
    config.precision = "double"
    model = _model()
    topology = _topology(600e3)
    # Two satellites, at 600 km and 1200 km
    two = list(topology)
    second = topology[1].clone()
    second[..., 2] = 1200e3
    two[1] = torch.cat([topology[1], second], dim=1)
    two[3] = torch.cat([topology[3], topology[3]], dim=1)
    with pytest.raises(ValueError):
        model.set_topology(*two)
    # One satellite at heights that differ across the batch
    mixed = list(topology)
    mixed[1] = topology[1].clone()
    mixed[1][1:, :, 2] = 1200e3
    with pytest.raises(ValueError):
        model.set_topology(*mixed)


@pytest.mark.parametrize("height", [5e3, 60e3, 100e3, 159e3, 36000e3])
def test_rejected_heights(height):
    # Neither an aerial platform (8 km to 50 km) nor a satellite (160 km to 35 786 km)
    config.precision = "double"
    with pytest.raises(ValueError):
        _model().set_topology(*_topology(height))


@pytest.mark.parametrize("height", [8e3, 20e3, 50e3, 160e3, 300e3, 600e3, 1200e3, 35786e3])
def test_distance_3d(height):
    # TR 38.811 V15.4.0, eq. (6.6-3): d = sqrt(R^2 sin^2(alpha) + h^2 + 2 h R) - R sin(alpha)
    config.precision = "double"
    model = _model()
    model.set_topology(*_topology(height))
    s = math.sin(math.radians(ELEVATION_ANGLE))
    expected = math.sqrt(R_EARTH ** 2 * s ** 2 + height ** 2 + 2 * height * R_EARTH) \
        - R_EARTH * s
    np.testing.assert_allclose(model._scenario.distance_3d.cpu().numpy(), expected,
                               rtol=1e-12)


@pytest.mark.parametrize("height,rotated", [
    (20e3, False), (160e3, False), (300e3, False), (599e3, False),
    (600e3, True), (1200e3, True), (35786e3, True)])
def test_faraday_rotation(height, rotated):
    # TR 38.811 V15.4.0, eqs. (6.8-1) and (6.8-1b): the polarization matrix of the LOS ray
    # of a satellite from 600 km on is [[1, 0], [0, -1]] F_r, with the rotation F_r by the
    # angle psi = 108 / f^2 degrees, f in GHz (eq. (6.8-2)); clause 6.8.2 gives it for a
    # "spaceborne BS above the ionosphere", and 600 km is the lowest satellite altitude of
    # TR 38.811 (Tables 4.5-1 and 5.1-1). Below 600 km, and for an aerial platform, the
    # matrix is [[1, 0], [0, -1]].
    config.precision = "double"
    model = _model()
    model.set_topology(*_topology(height), los=True)
    sampler = model._cir_sampler
    original = sampler._step_11_field_matrix
    captured = []

    def record(topology, aoa, aod, zoa, zod, h_phase):
        if aoa.shape[-1] == 1:                                  # the LOS ray
            captured.append(h_phase.detach().cpu().numpy())
        return original(topology, aoa, aod, zoa, zod, h_phase)

    sampler._step_11_field_matrix = record
    try:
        model(1, 1.0)
    finally:
        del sampler._step_11_field_matrix
    psi = math.radians(108.0 / (CARRIER_FREQUENCY / 1e9) ** 2)
    rotation = np.array([[math.cos(psi), -math.sin(psi)], [math.sin(psi), math.cos(psi)]])
    expected = np.array([[1.0, 0.0], [0.0, -1.0]])
    if rotated:
        expected = expected @ rotation
    actual = np.broadcast_to(captured[0], captured[0].shape[:-2] + (2, 2))
    np.testing.assert_allclose(actual, np.broadcast_to(expected, actual.shape), atol=1e-12,
                               err_msg=f"{height / 1e3:g} km")


@pytest.mark.parametrize("height,satellite", [(20e3, False), (300e3, True), (600e3, True),
                                              (1200e3, True)])
def test_satellite_doppler(height, satellite):
    # With the UTs at rest only the satellite moves. Every ray leaves the satellite in the
    # LOS direction, so its Doppler shift turns the phase of every path and antenna pair
    # of a batch example by the same angle per sample (TR 38.811 V15.4.0, clause 6.8.1).
    # An aerial platform gets no satellite Doppler shift: the channel stays constant.
    config.precision = "double"
    model = _model()
    topology = _topology(height, batch_size=16)
    topology[4] = torch.zeros_like(topology[4])                  # UT velocities
    model.set_topology(*topology)
    h, _ = model(2, 1e3)
    h = h.cpu().numpy()
    first, second = h[..., 0], h[..., 1]
    if not satellite:
        assert np.array_equal(first, second), f"{height / 1e3:g} km: the channel changes"
        return
    for b in range(h.shape[0]):
        used = np.abs(first[b]) > 0
        step = second[b][used] * np.conj(first[b][used])
        common = np.abs(np.angle(step * np.conj(step[0]))).max()
        assert common < 1e-9, f"example {b}: the phase steps differ by {common:.3g} rad"
        assert abs(np.angle(step[0])) > 1e-6, f"example {b}: no satellite Doppler shift"


def _departure_deviations(model):
    """Largest deviation [deg] of the ray AODs and of the cluster ZODs (mean of the
    rays) at the satellite from the LOS departure direction, from the positions."""
    rays = model._ray_sampler(model._lsp)
    aod = np.degrees(rays.aod.cpu().numpy())
    zod = np.degrees(rays.zod.cpu().numpy())
    sc = model._scenario
    delta = (sc.ut_loc.cpu().numpy()[:, None, :, :] - sc.bs_loc.cpu().numpy()[:, :, None, :])
    los_aod = np.degrees(np.arctan2(delta[..., 1], delta[..., 0]))
    los_zod = np.degrees(np.arctan2(np.hypot(delta[..., 0], delta[..., 1]), delta[..., 2]))
    d_aod = np.abs(np.remainder(aod - los_aod[..., None, None] + 180.0, 360.0) - 180.0)
    d_zod = np.abs(zod.mean(axis=-1) - los_zod[..., None])
    return d_aod.max(), d_zod.max()


@pytest.mark.parametrize("height,direction,satellite", [
    (20e3, "downlink", False), (20e3, "uplink", False),
    (160e3, "downlink", True), (160e3, "uplink", True),
    (600e3, "downlink", True), (600e3, "uplink", True),
    (35786e3, "downlink", True), (35786e3, "uplink", True)])
def test_departure_spreads_at_the_satellite(height, direction, satellite):
    # The satellite is the departure end of the rays as the generator draws them, in both
    # directions.
    # A satellite has zero departure spreads (NOTE 8): the ray AODs (dense urban: cluster
    # ASD 0) and the cluster ZODs equal the LOS departure direction (zero ZOD offset). An
    # aerial platform has the spreads of the tables in both directions.
    config.precision = "double"
    model = _model(direction)
    model.set_topology(*_topology(height), los=False)
    d_aod, d_zod = _departure_deviations(model)
    if satellite:
        assert d_aod <= 1e-9 and d_zod <= 1e-9, (
            f"{height / 1e3:g} km {direction}: departure off the LOS direction by "
            f"{d_aod:.3g} deg (AOD), {d_zod:.3g} deg (ZOD)")
    else:
        assert d_aod > 1e-6 and d_zod > 1e-6, (
            f"{height / 1e3:g} km {direction}: no departure spread "
            f"({d_aod:.3g} deg AOD, {d_zod:.3g} deg ZOD)")


with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_tables",
                       "tr38811_v15.4.0.json"), encoding="ascii") as _f:
    TABLES = json.load(_f)["tables"]


@pytest.mark.parametrize("direction", ["downlink", "uplink"])
@pytest.mark.parametrize("los,table", [(True, "6.7.2-1a"), (False, "6.7.2-2a")])
def test_aerial_platform_spreads(direction, los, table):
    # An aerial platform keeps the departure spreads of the tables (NOTE 8 concerns
    # satellites): lg ASD and lg ZSD of its links have the mean and the standard
    # deviation of Table 6.7.2-1a (LOS) or 6.7.2-2a (NLOS), dense urban, S band, 30
    # degrees, in both directions. One UT per batch example, so the links are independent.
    config.precision = "double"
    model = _model(direction)
    model.set_topology(*_topology(20e3, batch_size=2000, num_ut=1), los=los)
    rows = TABLES[table]["rows"]
    k = int(ELEVATION_ANGLE / 10) - 1
    for name in ("ASD", "ZSD"):
        values = np.log10(getattr(model._lsp, name.lower()).cpu().numpy())
        mean, std = float(rows["mu_lg" + name][k]), float(rows["sigma_lg" + name][k])
        label = f"20 km, {direction}, LOS {los}: lg {name}"
        assert_mean(values, mean, std, label)
        assert_std(values, std, label)
