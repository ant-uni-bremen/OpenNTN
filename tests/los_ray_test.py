# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the LOS ray of step 11 (TR 38.901 V16.1.0, eq. (7.5-29); TR 38.811 V15.4.0,
# eq. (6.8-1b)) against values computed in the test from the positions and the
# specification, without the implementation's angle code.
import mpmath
import numpy as np
import pytest
import torch
from sionna.phy import config

from openntn import Antenna, DenseUrban, SubUrban, Urban
from openntn.utils import gen_single_sector_topology

BS_HEIGHT = 600000.0
# Earth radius R_E of TR 38.811 V15.4.0, eq. (6.6-3), as used by the model
EARTH_RADIUS = 6371000.0
SPEED_OF_LIGHT = 299792458.0
# S band and Ka band, from 2 to 30 GHz
CARRIER_FREQUENCIES = [2.0e9, 2.2e9, 2.5e9, 3.0e9, 3.5e9, 4.0e9,
                       20.0e9, 22.5e9, 25.0e9, 27.5e9, 30.0e9]
ELEVATION_ANGLES = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
SCENARIOS = [(DenseUrban, "dur"), (Urban, "urb"), (SubUrban, "sur")]
DIRECTIONS = ["downlink", "uplink"]


def _model(model_class, direction, elevation_angle, carrier_frequency=2.2e9):
    # One vertically polarized element with an isotropic pattern at each end
    antenna = Antenna(polarization="single",
                      polarization_type="V",
                      antenna_pattern="omni",
                      carrier_frequency=carrier_frequency)
    return model_class(carrier_frequency=carrier_frequency,
                       ut_array=antenna,
                       bs_array=antenna,
                       direction=direction,
                       elevation_angle=elevation_angle,
                       enable_pathloss=False,
                       enable_shadow_fading=False)


def _run_los(model, scenario, elevation_angle, batch_size=2, num_ut=4,
             zero_orientations=False):
    """Runs the channel with forced LOS links and returns the topology and the LOS term
    that step 11 receives and computes."""
    topology = list(gen_single_sector_topology(batch_size=batch_size,
                                               num_ut=num_ut,
                                               scenario=scenario,
                                               elevation_angle=elevation_angle,
                                               bs_height=BS_HEIGHT))
    if zero_orientations:
        # Global and local coordinates coincide: the field patterns of TR 38.901
        # V16.1.0, eq. (7.1-11) are then the element patterns, without a polarization
        # rotation
        topology[2] = torch.zeros_like(topology[2])
        topology[3] = torch.zeros_like(topology[3])
    model.set_topology(*topology, los=True)
    sampler = model._cir_sampler
    original = sampler._step_11_los
    captured = {}

    def record(topology, t, carrier_frequency):
        h_los = original(topology, t, carrier_frequency)
        captured["topology"] = topology
        captured["h_los"] = h_los
        return h_los

    sampler._step_11_los = record
    try:
        # The first time sample is t = 0, where the Doppler factors are 1
        model(1, 1.0)
    finally:
        del sampler._step_11_los
    return captured["topology"], captured["h_los"]


def _unit_vector(theta, phi):
    """TR 38.901 V16.1.0, eq. (7.5-23), angles in radians."""
    return np.stack([np.sin(theta) * np.cos(phi),
                     np.sin(theta) * np.sin(phi),
                     np.cos(theta)], axis=-1)


@pytest.mark.parametrize("model_class,scenario", SCENARIOS)
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_los_arrival_direction_reverses_departure(model_class, scenario, direction):
    # TR 38.901 V16.1.0, clause 7.5, step 1c and clause 7.1: the LOS arrival direction
    # is the reversed LOS departure direction, so r(ZOA, AOA) = -r(ZOD, AOD), with the
    # zenith angle in [0, 180] degrees. At the UT, the LOS ZOA is 90 degrees minus the
    # elevation of the satellite seen from the UT.
    config.precision = "double"
    for elevation_angle in ELEVATION_ANGLES:
        model = _model(model_class, direction, elevation_angle)
        topology, _ = _run_los(model, scenario, elevation_angle)
        arrival = _unit_vector(topology.los_zoa.cpu().numpy(),
                               topology.los_aoa.cpu().numpy())
        departure = _unit_vector(topology.los_zod.cpu().numpy(),
                                 topology.los_aod.cpu().numpy())
        np.testing.assert_allclose(arrival, -departure, rtol=0, atol=1e-12,
                                   err_msg=f"{direction}, {elevation_angle} deg")

        sc = model._scenario
        zoa_ut = sc.los_zoa.cpu().numpy()
        for zoa in (zoa_ut, np.degrees(topology.los_zoa.cpu().numpy())):
            assert np.all((zoa >= 0.0) & (zoa <= 180.0)), \
                f"{direction}, {elevation_angle} deg: LOS ZOA outside [0, 180] deg"

        ut_loc = sc.ut_loc.cpu().numpy()          # [batch, num UTs, 3]
        bs_loc = sc.bs_loc.cpu().numpy()          # [batch, num BSs, 3]
        delta = bs_loc[:, :, None, :] - ut_loc[:, None, :, :]
        elevation = np.degrees(np.arctan2(delta[..., 2],
                                          np.hypot(delta[..., 0], delta[..., 1])))
        np.testing.assert_allclose(zoa_ut, 90.0 - elevation, rtol=0, atol=1e-10,
                                   err_msg=f"{direction}, {elevation_angle} deg")


def _los_phase_reference(elevation_angle, carrier_frequency):
    """-2 pi d3D f_c / c modulo 2 pi, with d3D of TR 38.811 V15.4.0, eq. (6.6-3), at 50
    digits."""
    with mpmath.workdps(50):
        sin_a = mpmath.sin(mpmath.radians(mpmath.mpf(elevation_angle)))
        r_e = mpmath.mpf(EARTH_RADIUS)
        h = mpmath.mpf(BS_HEIGHT)
        d3d = mpmath.sqrt(r_e**2 * sin_a**2 + h**2 + 2 * h * r_e) - r_e * sin_a
        phase = -2 * mpmath.pi * d3d * mpmath.mpf(carrier_frequency) / SPEED_OF_LIGHT
        return float(mpmath.fmod(phase, 2 * mpmath.pi))


def _los_phase_errors(direction):
    """Largest |angle of the LOS term - reference phase| [rad] and where it occurs."""
    worst = (0.0, None)
    for carrier_frequency in CARRIER_FREQUENCIES:
        for elevation_angle in ELEVATION_ANGLES:
            model = _model(DenseUrban, direction, elevation_angle, carrier_frequency)
            _, h_los = _run_los(model, "dur", elevation_angle, batch_size=1, num_ut=1,
                                zero_orientations=True)
            # [batch, tx, rx, 1, rx antennas, tx antennas, time], first time sample
            h = h_los[..., 0].cpu().numpy().astype(np.complex128).flatten()
            reference = _los_phase_reference(elevation_angle, carrier_frequency)
            error = np.abs(np.angle(h * np.exp(-1j * reference))).max()
            if error > worst[0]:
                worst = (error, (carrier_frequency, elevation_angle))
    return worst


@pytest.mark.slow
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_los_phase_double(direction):
    # TR 38.901 V16.1.0, eq. (7.5-29), and TR 38.811 V15.4.0, eq. (6.8-1b): the LOS term
    # contains exp(-j 2 pi d3D / lambda_0), with d3D of TR 38.811 V15.4.0, eq. (6.6-3).
    # With one vertically polarized element with an isotropic pattern at each end, zero
    # orientations and t = 0, the other factors are real and positive (the Faraday
    # rotation contributes cos(psi) > 0), so the angle of the LOS term is the
    # propagation phase.
    config.precision = "double"
    error, where = _los_phase_errors(direction)
    assert error <= 1e-6, (f"{direction}: LOS phase off by {error:.3g} rad at "
                           f"{where[0] / 1e9:g} GHz, {where[1]:g} deg")


@pytest.mark.slow
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_los_phase_single(direction):
    # As test_los_phase_double. In single precision one unit in the last place of d3D
    # is 6.25 cm or more at satellite distances, several wavelengths in the Ka band, so
    # the phase has to be reduced modulo 2 pi before the cast to the model precision.
    # At 30 GHz the carrier frequency itself is not a single-precision number.
    config.precision = "single"
    error, where = _los_phase_errors(direction)
    assert error <= 1e-6, (f"{direction}: LOS phase off by {error:.3g} rad at "
                           f"{where[0] / 1e9:g} GHz, {where[1]:g} deg")


def _distance_errors():
    """Largest |distance_3d - d3D of TR 38.811 V15.4.0, eq. (6.6-3)| in units in the
    last place of the model precision, and the elevation angle where it occurs."""
    worst = (0.0, None)
    for elevation_angle in ELEVATION_ANGLES:
        model = _model(DenseUrban, "downlink", elevation_angle)
        _run_los(model, "dur", elevation_angle, batch_size=1, num_ut=1)
        distance_3d = model._scenario.distance_3d.cpu().numpy().flatten()
        with mpmath.workdps(50):
            sin_a = mpmath.sin(mpmath.radians(mpmath.mpf(elevation_angle)))
            r_e = mpmath.mpf(EARTH_RADIUS)
            h = mpmath.mpf(BS_HEIGHT)
            reference = mpmath.sqrt(r_e**2 * sin_a**2 + h**2 + 2 * h * r_e) - r_e * sin_a
            ulp = float(np.spacing(distance_3d.dtype.type(float(reference))))
            error = max(float(abs(mpmath.mpf(float(d)) - reference)) for d in distance_3d)
        if error / ulp > worst[0]:
            worst = (error / ulp, elevation_angle)
    return worst


@pytest.mark.parametrize("precision,max_ulp", [("single", 0.5), ("double", 4.0)])
def test_distance_3d(precision, max_ulp):
    # TR 38.811 V15.4.0, eq. (6.6-3), evaluated in double precision in a form without
    # cancellation: within half a unit in the last place in single precision (the cast
    # of the double value), and within 4 units in double precision.
    config.precision = precision
    error, elevation_angle = _distance_errors()
    assert error <= max_ulp, (f"{precision}: distance_3d off by {error:.3g} ulp at "
                              f"{elevation_angle:g} deg")
