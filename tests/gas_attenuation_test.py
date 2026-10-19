# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the gas attenuation against Annex 2 of ITU-R P.676-10 (09/2013), implemented
# in the test from the printed equations.
#
# TR 38.811 V15.4.0, clause 6.6.4: "Annex 1 of Recommendation ITU-R P.676 gives a
# complete method for calculating gaseous attenuation, while Annex 2 of the same
# Recommendation gives an approximate method for frequencies up to 350 GHz. For system
# level simulations, the baseline method is as follows: - The method of Annex 2 in
# ITU-R P.676 is considered". P.676-10 is the last revision whose Annex 2 gives the
# specific attenuations and the equivalent heights in closed form. Annex 2, for a UE at
# the surface and the elevation angle theta:
#     A = (gamma_o h_o + gamma_w h_w) / sin(theta)                       eqs. (27), (28)
# with the specific attenuation of dry air gamma_o of eqs. (22a) to (22u), that of water
# vapour gamma_w of eqs. (23a) to (23d), and the equivalent heights h_o of eqs. (25a) to
# (25e) and h_w of eqs. (26a) and (26b). Annex 2 takes rp = ptot/1013, "where ptot
# represents total air pressure", and rt = 288/(273 + t), with the temperature t in
# degrees Celsius; Annex 1 writes the total pressure as p + e, with the dry air pressure
# p and the water vapour partial pressure e = rho T / 216.7 (eq. (4)). TR 38.811 clause
# 6.6.4 calls its 1013.25 hPa "the dry air pressure", so the model takes its atmospheric
# pressure as p, and t = T - 273.15.
import math
import types

import numpy as np
import pytest
import torch
from sionna.phy import config

from openntn import Antenna, Urban
from openntn import utils
from openntn.utils import gen_single_sector_topology

# Reference atmosphere of TR 38.811 clause 6.6.4, the defaults of the models
TEMPERATURE, DRY_PRESSURE, WATER_VAPOUR_DENSITY = 288.15, 1013.25, 7.5

# Annex 2, eqs. (22g) to (22t), as printed: factor * phi(rp, rt, a, b, c, d) of eq.
# (22u), given as (factor, a, b, c, d)
PHI = {
    "xi1": (1.0, 0.0717, -1.8132, 0.0156, -1.6515),                  # (22g)
    "xi2": (1.0, 0.5146, -4.6368, -0.1921, -5.7416),                 # (22h)
    "xi3": (1.0, 0.3414, -6.5851, 0.2130, -8.5854),                  # (22i)
    "xi4": (1.0, -0.0112, 0.0092, -0.1033, -0.0009),                 # (22j)
    "xi5": (1.0, 0.2705, -2.7192, -0.3016, -4.1033),                 # (22k)
    "xi6": (1.0, 0.2445, -5.9191, 0.0422, -8.0719),                  # (22l)
    "xi7": (1.0, -0.1833, 6.5589, -0.2402, 6.131),                   # (22m)
    "gamma54": (2.192, 1.8286, -1.9487, 0.4051, -2.8509),            # (22n)
    "gamma58": (12.59, 1.0045, 3.5610, 0.1588, 1.2834),              # (22o)
    "gamma60": (15.0, 0.9003, 4.1335, 0.0427, 1.6088),               # (22p)
    "gamma62": (14.28, 0.9886, 3.4176, 0.1827, 1.3429),              # (22q)
    "gamma64": (6.819, 1.4320, 0.6258, 0.3177, -0.5914),             # (22r)
    "gamma66": (1.908, 2.0717, -4.1404, 0.4910, -4.8718),            # (22s)
    "delta": (-0.00306, 3.211, -14.94, 1.583, -16.37),               # (22t)
}

# Zenith checks of every branch: the limits of the branches of eqs. (22a) to (22f) (54,
# 60, 62, 66 and 120 GHz) and of eq. (25e) (70 GHz), a frequency within each branch, the
# limits of the range (1 and 350 GHz) and the centres of the lines of eqs. (23a) and
# (25c) (22.235, 118.75, 183.31 and 325.153 GHz)
BRANCH_FREQUENCIES = (1.0, 22.235, 54.0, 54.5, 57.0, 60.0, 61.0, 62.0, 63.0, 64.0, 66.0,
                      66.5, 70.0, 90.0, 118.75, 120.0, 120.5, 183.31, 325.153, 350.0)


def _gamma_o(f, rp, rt):
    """Specific attenuation of dry air [dB/km], eqs. (22a) to (22u)."""
    v = {name: k * rp ** a * rt ** b * math.exp(c * (1 - rp) + d * (1 - rt))
         for name, (k, a, b, c, d) in PHI.items()}                                  # (22u)
    if f <= 54:
        return (7.2 * rt ** 2.8 / (f ** 2 + 0.34 * rp ** 2 * rt ** 1.6)
                + 0.62 * v["xi3"] / ((54 - f) ** (1.16 * v["xi1"]) + 0.83 * v["xi2"])
                ) * f ** 2 * rp ** 2 * 1e-3                                          # (22a)
    if f <= 60:
        return math.exp(math.log(v["gamma54"]) / 24 * (f - 58) * (f - 60)
                        - math.log(v["gamma58"]) / 8 * (f - 54) * (f - 60)
                        + math.log(v["gamma60"]) / 12 * (f - 54) * (f - 58))        # (22b)
    if f <= 62:
        return v["gamma60"] + (v["gamma62"] - v["gamma60"]) * (f - 60) / 2         # (22c)
    if f <= 66:
        return math.exp(math.log(v["gamma62"]) / 8 * (f - 64) * (f - 66)
                        - math.log(v["gamma64"]) / 4 * (f - 62) * (f - 66)
                        + math.log(v["gamma66"]) / 8 * (f - 62) * (f - 64))        # (22d)
    if f <= 120:
        return (3.02e-4 * rt ** 3.5
                + 0.283 * rt ** 3.8 / ((f - 118.75) ** 2 + 2.91 * rp ** 2 * rt ** 1.6)
                + 0.502 * v["xi6"] * (1 - 0.0163 * v["xi7"] * (f - 66))
                / ((f - 66) ** (1.4346 * v["xi4"]) + 1.15 * v["xi5"])
                ) * f ** 2 * rp ** 2 * 1e-3                                          # (22e)
    return (3.02e-4 / (1 + 1.9e-5 * f ** 1.5)
            + 0.283 * rt ** 0.3 / ((f - 118.75) ** 2 + 2.91 * rp ** 2 * rt ** 1.6)
            ) * f ** 2 * rp ** 2 * rt ** 3.5 * 1e-3 + v["delta"]                    # (22f)


def _gamma_w(f, rp, rt, rho):
    """Specific attenuation of water vapour [dB/km], eqs. (23a) to (23d)."""
    eta1 = 0.955 * rp * rt ** 0.68 + 0.006 * rho                                    # (23b)
    eta2 = 0.735 * rp * rt ** 0.5 + 0.0353 * rt ** 4 * rho                          # (23c)

    def g(fi):
        return 1 + ((f - fi) / (f + fi)) ** 2                                       # (23d)

    lines = (
        3.98 * eta1 * math.exp(2.23 * (1 - rt))
        / ((f - 22.235) ** 2 + 9.42 * eta1 ** 2) * g(22),
        11.96 * eta1 * math.exp(0.7 * (1 - rt)) / ((f - 183.31) ** 2 + 11.14 * eta1 ** 2),
        0.081 * eta1 * math.exp(6.44 * (1 - rt)) / ((f - 321.226) ** 2 + 6.29 * eta1 ** 2),
        3.66 * eta1 * math.exp(1.6 * (1 - rt)) / ((f - 325.153) ** 2 + 9.22 * eta1 ** 2),
        25.37 * eta1 * math.exp(1.09 * (1 - rt)) / (f - 380) ** 2,
        17.4 * eta1 * math.exp(1.46 * (1 - rt)) / (f - 448) ** 2,
        844.6 * eta1 * math.exp(0.17 * (1 - rt)) / (f - 557) ** 2 * g(557),
        290 * eta1 * math.exp(0.41 * (1 - rt)) / (f - 752) ** 2 * g(752),
        8.3328e4 * eta2 * math.exp(0.99 * (1 - rt)) / (f - 1780) ** 2 * g(1780),
    )
    return sum(lines) * f ** 2 * rt ** 2.5 * rho * 1e-4                             # (23a)


def _h_o(f, rp):
    """Equivalent height of dry air [km], eqs. (25a) to (25e)."""
    t1 = 4.64 / (1 + 0.066 * rp ** -2.3) * math.exp(
        -((f - 59.7) / (2.87 + 12.4 * math.exp(-7.9 * rp))) ** 2)                   # (25b)
    t2 = (0.14 * math.exp(2.12 * rp)
          / ((f - 118.75) ** 2 + 0.031 * math.exp(2.2 * rp)))                       # (25c)
    t3 = (0.0114 / (1 + 0.14 * rp ** -2.6) * f
          * (-0.0247 + 0.0001 * f + 1.61e-6 * f ** 2)
          / (1 - 0.0169 * f + 4.1e-5 * f ** 2 + 3.2e-7 * f ** 3))                   # (25d)
    h_o = 6.1 / (1 + 0.17 * rp ** -1.1) * (1 + t1 + t2 + t3)                        # (25a)
    return min(h_o, 10.7 * rp ** 0.3) if f < 70 else h_o                            # (25e)


def _h_w(f, rp):
    """Equivalent height of water vapour [km], eqs. (26a) and (26b)."""
    sigma_w = 1.013 / (1 + math.exp(-8.6 * (rp - 0.57)))                            # (26b)
    return 1.66 * (1 + 1.39 * sigma_w / ((f - 22.235) ** 2 + 2.56 * sigma_w)
                   + 3.37 * sigma_w / ((f - 183.31) ** 2 + 4.69 * sigma_w)
                   + 1.58 * sigma_w / ((f - 325.1) ** 2 + 2.89 * sigma_w))          # (26a)


def _annex_2(f, elevation_angle, rho=WATER_VAPOUR_DENSITY, temperature=TEMPERATURE,
             dry_pressure=DRY_PRESSURE):
    """Slant path attenuation [dB] by P.676-10 Annex 2, eqs. (27) and (28), for the
    frequency f [GHz], the elevation angle [deg], the water vapour density [g/m3], the
    temperature [K] and the dry air pressure [hPa]."""
    e = rho * temperature / 216.7                                     # Annex 1, eq. (4)
    rp = (dry_pressure + e) / 1013
    rt = 288 / (273 + (temperature - 273.15))
    zenith = (_gamma_o(f, rp, rt) * _h_o(f, rp)
              + _gamma_w(f, rp, rt, rho) * _h_w(f, rp))                             # (27)
    return zenith / math.sin(math.radians(elevation_angle))                         # (28)


def _stub(f_ghz, elevation_angle=90.0, rho=WATER_VAPOUR_DENSITY):
    stub = types.SimpleNamespace(
        _temperature=TEMPERATURE, _atmospheric_pressure=DRY_PRESSURE,
        _water_vapor_density=rho, _carrier_frequency=f_ghz * 1e9,
        _elevation_angle=elevation_angle, los=torch.ones(1, 1, 1, dtype=torch.bool),
        dtype=torch.float64)
    # The loss functions store their results through the method _update_attr of the
    # scenario, which writes them into its existing tensors; the stub assigns them
    stub._update_attr = lambda name, value: setattr(stub, name, value)
    return stub


def _gas_loss(f_ghz, elevation_angle=90.0, rho=WATER_VAPOUR_DENSITY):
    stub = _stub(f_ghz, elevation_angle, rho)
    utils._compute_pathloss_gas(stub)
    return float(stub._pl_g.flatten()[0])


@pytest.mark.parametrize("carrier_frequency", [2.0e9, 2.2e9, 20e9, 30e9])
def test_annex_2(carrier_frequency):
    # The gas loss of the model with its default atmosphere (TR 38.811 clause 6.6.4)
    # equals Annex 2 of P.676-10 at every reference elevation angle.
    config.precision = "double"
    antenna = Antenna(polarization="single", polarization_type="V",
                      antenna_pattern="38.901", carrier_frequency=carrier_frequency)
    worst = 0.0
    for elevation_angle in range(10, 100, 10):
        model = Urban(carrier_frequency=carrier_frequency, ut_array=antenna,
                      bs_array=antenna, direction="downlink",
                      elevation_angle=float(elevation_angle))
        model.set_topology(*gen_single_sector_topology(
            batch_size=2, num_ut=2, scenario="urb", elevation_angle=float(elevation_angle),
            bs_height=600e3))
        expected = _annex_2(carrier_frequency / 1e9, elevation_angle)
        actual = model._scenario.gas_pathloss.cpu().numpy()
        worst = max(worst, float(np.abs(actual / expected - 1.0).max()))
    assert worst <= 1e-9, f"{carrier_frequency / 1e9:g} GHz: relative error {worst:.3g}"


def test_frequency_range():
    # Annex 2 holds from 1 GHz to 350 GHz. The zenith loss is continuous across 66 GHz,
    # and the dry air attenuates above 120 GHz.
    assert abs(_gas_loss(66.1) - _gas_loss(65.9)) < 2.0
    assert _gas_loss(121.0, rho=0.0) > 0.0
    for f in (0.9, 350.5):
        with pytest.raises(ValueError):
            _gas_loss(f)


@pytest.mark.parametrize("rho", [WATER_VAPOUR_DENSITY, 0.0])
@pytest.mark.parametrize("f", BRANCH_FREQUENCIES)
def test_branches(f, rho):
    # Every branch of Annex 2 at the zenith, with water vapour and in dry air
    expected = _annex_2(f, 90.0, rho=rho)
    actual = _gas_loss(f, rho=rho)
    assert abs(actual / expected - 1.0) <= 1e-9, (
        f"{f:g} GHz, {rho:g} g/m3: {actual:.6g} dB, Annex 2 of P.676-10 {expected:.6g} dB")
