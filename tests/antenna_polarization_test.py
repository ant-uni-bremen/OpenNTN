# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the polarization of the elements of dual-polarized antenna arrays.
#
# TR 38.901 V16.1.0, clause 7.3.2, polarization model 2: an element with the slant angle
# zeta has the field pattern F_theta = sqrt(A) cos(zeta) and F_phi = sqrt(A) sin(zeta)
# (eqs. (7.3-4) and (7.3-5)), with its power pattern A; zeta = 0 is a vertically
# polarized element and zeta = 90 deg a horizontally polarized one. A dual-polarized
# array with polarization_type="VH" has a vertically and a horizontally polarized
# element in each pair, the slant angles 0 and 90 deg, as the UT configuration "X-pol
# (0/+90 degree)" of clause 7.8.2 and the "Polarization: 0°, 90°" of R1-1802551, Table 1,
# for the UE arrays that TR 38.811 V15.4.0, clause 6.4.2, uses; polarization_type="cross"
# has the slant angles -45 and +45 deg. At the boresight (theta = 90 deg, phi = 0) the
# element pattern "38.901" has its maximum directional gain G = 8 dBi (TR 38.901 V16.1.0,
# Table 7.3-1), so the two elements of a "VH" pair have the fields (sqrt(G), 0) and
# (0, sqrt(G)).
import math

import numpy as np
import pytest
import torch

from openntn import Antenna, AntennaArray, PanelArray

ARRAYS = ("PanelArray", "Antenna", "AntennaArray")
# sqrt(G) for the maximum directional gain G = 8 dBi
SQRT_G = 10.0 ** (8.0 / 20.0)


def _array(name, polarization_type):
    kwargs = dict(polarization="dual", polarization_type=polarization_type,
                  antenna_pattern="38.901", carrier_frequency=2e9, precision="double")
    if name == "PanelArray":
        return PanelArray(num_rows_per_panel=1, num_cols_per_panel=2, **kwargs)
    if name == "Antenna":
        return Antenna(**kwargs)
    return AntennaArray(num_rows=1, num_cols=4, **kwargs)


def _pair(array):
    """The slant angles [rad] of the two elements of a pair and their fields
    (F_theta, F_phi) at the boresight"""
    theta = torch.tensor([math.pi / 2], dtype=torch.float64)
    phi = torch.tensor([0.0], dtype=torch.float64)
    slants, fields = [], []
    for element in (array.ant_pol1, array.ant_pol2):
        slants.append(float(element._slant_angle))
        f_theta, f_phi = element.field(theta, phi)
        fields.append((float(f_theta[0]), float(f_phi[0])))
    return np.array(slants), np.array(fields)


def _describe(slants, fields):
    return (f"slant angles {np.degrees(slants).round(6).tolist()} deg, fields "
            f"({fields[0, 0]:.6f}, {fields[0, 1]:.6f}) and ({fields[1, 0]:.6f}, "
            f"{fields[1, 1]:.6f})")


@pytest.mark.parametrize("name", ARRAYS)
def test_vh_pair(name):
    slants, fields = _pair(_array(name, "VH"))
    message = f"{name}, VH: {_describe(slants, fields)}"
    np.testing.assert_allclose(slants, [0.0, math.pi / 2], rtol=0, atol=1e-15,
                               err_msg=message)
    np.testing.assert_allclose(fields, [[SQRT_G, 0.0], [0.0, SQRT_G]], rtol=0, atol=1e-12,
                               err_msg=message)


@pytest.mark.parametrize("name", ARRAYS)
def test_cross_pair(name):
    slants, fields = _pair(_array(name, "cross"))
    message = f"{name}, cross: {_describe(slants, fields)}"
    np.testing.assert_allclose(slants, [-math.pi / 4, math.pi / 4], rtol=0, atol=1e-15,
                               err_msg=message)
    half = SQRT_G * math.sqrt(0.5)
    np.testing.assert_allclose(fields, [[half, -half], [half, half]], rtol=0, atol=1e-12,
                               err_msg=message)
