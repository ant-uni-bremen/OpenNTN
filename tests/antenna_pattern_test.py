# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the radiation patterns of the antenna elements.
#
# The pattern "aperture" is the normalized gain pattern of a reflector antenna with a
# circular aperture of radius a, TR 38.811 V15.4.0, clause 6.4.1: 1 at the boresight and
# 4 |J_1(ka sin(theta)) / (ka sin(theta))|^2 = (2 J_1(x) / x)^2 with x = ka sin(theta)
# for 0 < |theta| <= 90 deg, theta the angle from the boresight, J_1 the Bessel function
# of the first kind and first order and ka the number of wavelengths on the
# circumference of the aperture, independent of the frequency. The radius a is given in
# wavelengths; Figure 6.4.1-1 of the clause shows a = 10. The pattern halves where
# (2 J_1(x) / x)^2 = 1/2, at x = x_3, so the half-power beamwidth is 2 arcsin(x_3 / ka).
# TR 38.821 V16.2.0, Table 6.1.1.1-1, gives the satellite apertures as diameters D (its
# NOTE 1: "This value is equivalent to the antenna diameter in Sec. 6.4.1 of [2]", TR
# 38.811) with their 3 dB beamwidths, for example 2 m at 2 GHz (LEO-600 and LEO-1200)
# with 4.4127 deg; the radius in wavelengths is a = D f / (2c). The values of J_1 come
# from mpmath, independent of the Bessel function of PyTorch that the element uses.
#
# The pattern "dlp" is the element pattern of the co-phased dual linearly polarized UE
# array of R1-1802551, which TR 38.811 V15.4.0, clause 6.4.2, uses: Table 2 of
# R1-1802551, in dB, A''(theta'', phi'') = G_E,max - min{-(A_E,V + A_E,H), A_m} with
# A_E,V = -min{12 ((theta'' - 90 deg) / theta_3dB)^2, SLA_V} and
# A_E,H = -min{12 (phi'' / phi_3dB)^2, A_m}, theta_3dB = phi_3dB = 90 deg,
# SLA_V = A_m = 25 dB and the maximum directional gain G_E,max = 5 dBi (Table 1:
# "Antenna element gain 5 dBi").
import math

import mpmath
import numpy as np
import pytest
import torch

from openntn import Antenna, AntennaArray, PanelArray
from openntn.antenna import AntennaElement

SPEED_OF_LIGHT = 299792458.0


def _gain(x):
    """(2 J_1(x) / x)^2, 1 at x = 0, in double precision from mpmath"""
    if x == 0.0:
        return 1.0
    with mpmath.workdps(30):
        return float((2 * mpmath.besselj(1, x) / x) ** 2)


def _x_3():
    """The root of (2 J_1(x) / x)^2 = 1/2 in the main lobe"""
    with mpmath.workdps(30):
        return float(mpmath.findroot(lambda x: (2 * mpmath.besselj(1, x) / x) ** 2 - 0.5,
                                     1.6))


def _pattern(element, angle):
    """The power pattern of the element at the angle [rad] off the boresight, in the
    horizontal plane (theta = 90 deg, phi = angle), where the boresight lies"""
    theta = torch.tensor([math.pi / 2], dtype=torch.float64)
    phi = torch.tensor([angle], dtype=torch.float64)
    f_theta, f_phi = element.field(theta, phi)
    return float(f_theta[0] ** 2 + f_phi[0] ** 2)


def _half_power_beamwidth(element, ka):
    """Twice the angle [rad] at which the pattern of the element halves, by bisection
    between the boresight and the first null of the pattern, sin(angle) = 3.8317 / ka"""
    low, high = 0.0, math.asin(min(1.0, 3.8317 / ka))
    for _ in range(200):
        middle = 0.5 * (low + high)
        if _pattern(element, middle) > 0.5:
            low = middle
        else:
            high = middle
    return low + high


def _element(**kwargs):
    return AntennaElement("aperture", precision="double", **kwargs)


def test_half_power_beamwidth_of_tr38821():
    # 2 m at 2 GHz, the S-band aperture of LEO-600 and LEO-1200 in TR 38.821
    a = 2.0 * 2.0e9 / (2.0 * SPEED_OF_LIGHT)
    assert abs(a - 6.6713) < 1e-4
    ka = 2.0 * math.pi * a
    beamwidth = _half_power_beamwidth(_element(aperture_radius=a), ka)
    expected = 2.0 * math.asin(_x_3() / ka)
    assert abs(beamwidth - expected) <= 1e-9 * expected, (beamwidth, expected)
    # The beamwidth of the table is 0.16 % below that of the pattern, 4.4198 deg
    assert abs(math.degrees(beamwidth) / 4.4127 - 1.0) <= 0.002, math.degrees(beamwidth)


def test_pattern_for_ka_10():
    # ka = 10, the aperture of the calibration of TR 38.811 Annex A, 10 deg off the
    # boresight: (2 J_1(x) / x)^2 with x = 10 sin(10 deg), -3.51 dB
    element = _element(aperture_radius=10.0 / (2.0 * math.pi))
    angle = math.radians(10.0)
    expected = _gain(10.0 * math.sin(angle))
    actual = _pattern(element, angle)
    assert abs(actual - expected) <= 1e-12 * expected, (actual, expected)
    assert round(10.0 * math.log10(actual), 2) == -3.51


def test_default_radius():
    # Without the parameter the radius is 10 wavelengths, as before: the half-power
    # beamwidth 2.9482 deg and -30.54 dB at 10 deg off the boresight
    default, ten = _element(), _element(aperture_radius=10)
    ka = 20.0 * math.pi
    beamwidth = _half_power_beamwidth(default, ka)
    assert abs(beamwidth - 2.0 * math.asin(_x_3() / ka)) <= 1e-9 * beamwidth
    assert round(math.degrees(beamwidth), 4) == 2.9482
    angle = math.radians(10.0)
    assert round(10.0 * math.log10(_pattern(default, angle)), 2) == -30.54
    angles = torch.linspace(0.0, math.pi / 2, 91, dtype=torch.float64)
    theta = torch.full_like(angles, math.pi / 2)
    assert all(torch.equal(f, g) for f, g in zip(default.field(theta, angles),
                                                  ten.field(theta, angles)))


ARRAYS = ("PanelArray", "Antenna", "AntennaArray")


def _array(name, aperture_radius=None):
    kwargs = {"antenna_pattern": "aperture", "carrier_frequency": 2.0e9,
              "precision": "double"}
    if aperture_radius is not None:
        kwargs["aperture_radius"] = aperture_radius
    if name == "PanelArray":
        return PanelArray(num_rows_per_panel=2, num_cols_per_panel=2, polarization="dual",
                          polarization_type="cross", **kwargs)
    if name == "Antenna":
        return Antenna(polarization="single", polarization_type="V", **kwargs)
    return AntennaArray(num_rows=1, num_cols=4, polarization="dual",
                        polarization_type="cross", **kwargs)


@pytest.mark.parametrize("name", ARRAYS)
def test_arrays_pass_the_radius(name):
    # The fields of the elements of an array built with a radius are those of an element
    # built with it, and differ from those of the default radius
    a = 2.0 * 2.0e9 / (2.0 * SPEED_OF_LIGHT)
    angles = torch.linspace(0.0, math.pi / 2, 91, dtype=torch.float64)
    theta = torch.full_like(angles, math.pi / 2)
    array, default = _array(name, a), _array(name)
    pairs = [(array.ant_pol1, default.ant_pol1)]
    if array.polarization == "dual":
        pairs.append((array.ant_pol2, default.ant_pol2))
    for element, default_element in pairs:
        slant = float(element._slant_angle)
        expected = _element(slant_angle=slant, aperture_radius=a).field(theta, angles)
        actual = element.field(theta, angles)
        assert all(torch.equal(f, g) for f, g in zip(actual, expected))
        assert not torch.equal(actual[0], default_element.field(theta, angles)[0])


@pytest.mark.parametrize("aperture_radius", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_radius(aperture_radius):
    with pytest.raises(ValueError, match="aperture_radius"):
        _element(aperture_radius=aperture_radius)
    for name in ARRAYS:
        with pytest.raises(ValueError, match="aperture_radius"):
            _array(name, aperture_radius)


@pytest.mark.parametrize("aperture_radius", [True, "10"])
def test_radius_of_wrong_type(aperture_radius):
    with pytest.raises(TypeError, match="aperture_radius"):
        _element(aperture_radius=aperture_radius)
    for name in ARRAYS:
        with pytest.raises(TypeError, match="aperture_radius"):
            _array(name, aperture_radius)


def _table_2(theta, phi):
    """The element pattern of R1-1802551, Table 2 [dBi], for theta'' and phi'' [deg]"""
    theta_3db = phi_3db = 90.0
    sla_v = a_m = 25.0
    g_e_max = 5.0
    a_e_v = -np.minimum(12.0 * ((theta - 90.0) / theta_3db) ** 2, sla_v)
    a_e_h = -np.minimum(12.0 * (phi / phi_3db) ** 2, a_m)
    return g_e_max - np.minimum(-(a_e_v + a_e_h), a_m)


def test_dlp_pattern():
    # The pattern of the element "dlp" equals Table 2 of R1-1802551 with its 5 dBi on a
    # grid of 181 zenith and 361 azimuth angles, and is 5 dBi at the boresight
    theta, phi = np.meshgrid(np.linspace(0.0, 180.0, 181), np.linspace(-180.0, 180.0, 361),
                             indexing="ij")
    element = AntennaElement("dlp", precision="double")
    f_theta, f_phi = element.field(torch.tensor(np.radians(theta)),
                                   torch.tensor(np.radians(phi)))
    pattern = 10.0 * np.log10((f_theta ** 2 + f_phi ** 2).cpu().numpy())
    difference = np.abs(pattern - _table_2(theta, phi))
    assert difference.max() <= 1e-9, f"largest difference {difference.max():.3g} dB"
    boresight = _pattern(element, 0.0)
    assert abs(10.0 * math.log10(boresight) - 5.0) <= 1e-12, boresight
