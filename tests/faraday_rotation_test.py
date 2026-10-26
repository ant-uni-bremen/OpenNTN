# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the direction of the Faraday rotation in the uplink.
#
# TR 38.811 V15.4.0, clause 6.8.2, post-multiplies the polarization matrices of the NLOS
# rays and of the LOS ray of TR 38.901 V16.1.0, eqs. (7.5-28) and (7.5-29), by the
# Faraday rotation F_r(psi) = [[cos(psi), -sin(psi)], [sin(psi), cos(psi)]] (eqs.
# (6.8-1a) and (6.8-1b)), with psi = 108 / f^2 degrees, f in GHz (eq. (6.8-2)). The
# equations are written for the downlink, where the satellite transmits: the rotation
# acts on the field of the satellite before the scatterers near the UT, between them and
# the satellite, where the ionosphere is. ITU-R P.618-13, section 4, which TR 38.811
# cites for psi: "As seen from the earth station, the planes of polarization rotate in
# the same direction on the up- and down-links." In the uplink, where the UT transmits,
# the rotation is pre-multiplied, between the scatterers and the satellite, so that:
#
# - on the LOS path the field that arrives at the satellite is turned against the field
#   of the UT about the direction from the UT to the satellite by the same angle as the
#   field that arrives at the UT against the field of the satellite in the downlink,
#   -psi (psi about the direction of propagation of the downlink);
# - on the NLOS paths the power that the satellite collects over two orthogonal elements
#   does not depend on psi, as in the downlink, since F_r is a rotation.
#
# The rotation applies to satellites from 600 km on. The fields come from one element
# with an isotropic pattern at each end in the global orientation, a vertically
# polarized one at the transmitter and a vertically and a horizontally polarized one at
# the receiver.
import math

import numpy as np
import pytest
import torch
from sionna.phy import config

from openntn import Antenna, DenseUrban, Urban
from openntn.utils import gen_single_sector_topology

SATELLITE_HEIGHT = 600e3


def _unit(theta, phi):
    return np.array([math.sin(theta) * math.cos(phi), math.sin(theta) * math.sin(phi),
                     math.cos(theta)])


def _theta_hat(theta, phi):
    return np.array([math.cos(theta) * math.cos(phi), math.cos(theta) * math.sin(phi),
                     -math.sin(theta)])


def _phi_hat(phi):
    return np.array([-math.sin(phi), math.cos(phi), 0.0])


def _los_rotation(direction, carrier_frequency, elevation_angle):
    """The angle [deg] by which the field of the LOS path that arrives at the receiver is
    turned against the field of the transmitter, about the direction from the UT to the
    satellite"""
    config.precision = "double"
    v = Antenna(polarization="single", polarization_type="V", antenna_pattern="omni",
                carrier_frequency=carrier_frequency)
    vh = Antenna(polarization="dual", polarization_type="VH", antenna_pattern="omni",
                 carrier_frequency=carrier_frequency)
    ut_array, bs_array = (vh, v) if direction == "downlink" else (v, vh)
    model = DenseUrban(carrier_frequency=carrier_frequency, ut_array=ut_array,
                       bs_array=bs_array, direction=direction,
                       elevation_angle=elevation_angle, enable_pathloss=False,
                       enable_shadow_fading=False)
    topology = list(gen_single_sector_topology(batch_size=1, num_ut=1, scenario="dur",
                                               elevation_angle=elevation_angle,
                                               bs_height=SATELLITE_HEIGHT))
    # Global and local coordinates coincide
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
    top = captured["topology"]
    # [batch, tx, rx, 1, rx antennas, tx antennas, time]: the V and the H element of the
    # receiver; their ratio is the ratio of the field components, without the phase of
    # the path that both share
    h = captured["h_los"][0, 0, 0, 0, :, 0, 0].cpu().numpy()
    ratio = h[1] / h[0]
    assert abs(ratio.imag) < 1e-12
    zod, aod, zoa, aoa = (float(a[0, 0, 0]) for a in
                          (top.los_zod, top.los_aod, top.los_zoa, top.los_aoa))
    transmitted = _theta_hat(zod, aod)
    received = _theta_hat(zoa, aoa) + ratio.real * _phi_hat(aoa)
    # The direction of departure points from the transmitter to the receiver
    ut_to_satellite = _unit(zod, aod) * (-1.0 if direction == "downlink" else 1.0)
    return math.degrees(math.atan2(np.dot(np.cross(transmitted, received), ut_to_satellite),
                                   np.dot(transmitted, received)))


@pytest.mark.parametrize("elevation_angle", [30.0, 60.0])
@pytest.mark.parametrize("carrier_frequency", [2.0e9, 20.0e9])
def test_los_rotation_as_in_the_downlink(carrier_frequency, elevation_angle):
    psi = 108.0 / (carrier_frequency / 1e9) ** 2
    downlink = _los_rotation("downlink", carrier_frequency, elevation_angle)
    uplink = _los_rotation("uplink", carrier_frequency, elevation_angle)
    message = (f"{carrier_frequency / 1e9:g} GHz, {elevation_angle:g} deg: the rotation "
               f"about the direction from the UT to the satellite is {downlink:+.4f} deg "
               f"in the downlink and {uplink:+.4f} deg in the uplink, psi = {psi:.4f} deg")
    assert abs(downlink + psi) < 1e-6, message
    assert abs(uplink + psi) < 1e-6, message


@pytest.mark.parametrize("satellite_polarization", ["VH", "cross"])
@pytest.mark.parametrize("direction", ["downlink", "uplink"])
def test_nlos_power_over_the_satellite_pair(direction, satellite_polarization):
    # Urban, 2 GHz, 30 deg, forced NLOS links, a cross-polarized pair at each UT: the
    # power of every path summed over the two orthogonal elements of the satellite is
    # that of the same model with the Faraday rotation replaced by the identity, while
    # the power of a single element changes with the rotation
    config.precision = "double"
    config.seed = 7
    carrier_frequency = 2.0e9
    ut_array = Antenna(polarization="dual", polarization_type="cross",
                       antenna_pattern="38.901", carrier_frequency=carrier_frequency)
    bs_array = Antenna(polarization="dual", polarization_type=satellite_polarization,
                       antenna_pattern="38.901", carrier_frequency=carrier_frequency)
    model = Urban(carrier_frequency=carrier_frequency, ut_array=ut_array,
                  bs_array=bs_array, direction=direction, elevation_angle=30.0)
    model.set_topology(*gen_single_sector_topology(batch_size=4, num_ut=4, scenario="urb",
                                                   elevation_angle=30.0,
                                                   bs_height=SATELLITE_HEIGHT), los=False)

    def power():
        config.seed = 11
        h, _ = model(2, 30e3)
        # [batch, rx, rx antennas, tx, tx antennas, paths, time]
        return (torch.abs(h) ** 2).cpu().numpy()

    rotated = power()
    rotation = model._cir_sampler._faraday_rotation
    rotation.copy_(torch.eye(2, dtype=rotation.dtype, device=rotation.device))
    identity = power()
    # The satellite transmits in the downlink and receives in the uplink
    axis = 4 if direction == "downlink" else 2
    total_rotated, total_identity = rotated.sum(axis=axis), identity.sum(axis=axis)
    assert np.array_equal(total_rotated > 0, total_identity > 0)
    positive = total_identity > 0
    change_db = np.abs(10 * np.log10(total_rotated[positive] / total_identity[positive]))
    both = (rotated > 0) & (identity > 0)
    element_change_db = np.abs(10 * np.log10(rotated[both] / identity[both]))
    assert element_change_db.max() > 1.0
    assert change_db.max() < 1e-9, (
        f"{direction}, satellite {satellite_polarization}: the power of a path summed over "
        f"the satellite's pair changes by up to {change_db.max():.2f} dB with the rotation")
