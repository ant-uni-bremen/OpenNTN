# SPDX-FileCopyrightText: Copyright (c) 2025 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# This file tests step 3, the basic path loss of TR 38.811 Section 6.6.2, eq. (6.6-4):
# PL_b = FSPL + SF + CL, with SF ~ N(0, sigma_SF^2) and CL = 0 dB for LoS UEs.
# After subtracting FSPL, the LoS samples must have mean 0 and standard deviation
# sigma_SF(LoS), and the NLoS samples mean CL and standard deviation sigma_SF(NLoS).
# The expected values are transcribed from TR 38.811 V15.4.0, Tables 6.6.2-1 (dense
# urban), 6.6.2-2 (urban) and 6.6.2-3 (suburban and rural), at the reference elevation
# angles 10, 20, ..., 90 degrees.
# The tolerances are statistical, see statistical_checks.py; the SF samples of different
# links are independent.

from openntn import utils   # The code to test
import unittest   # The test framework
from openntn import Antenna, AntennaArray, Urban, DenseUrban, SubUrban

from statistical_checks import assert_mean, assert_std


def create_ut_ant(carrier_frequency):
    return Antenna(polarization="single",
                polarization_type="V",
                antenna_pattern="38.901",
                carrier_frequency=carrier_frequency)

def create_bs_ant(carrier_frequency):
    return AntennaArray(num_rows=1,
                        num_cols=4,
                        polarization="dual",
                        polarization_type="VH",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)


_MODELS = {"urb": Urban, "dur": DenseUrban, "sur": SubUrban}
_CARRIER = {"s_band": 2.2e9, "ka_band": 20e9}
_ELEVS = (10, 20, 30, 40, 50, 60, 70, 80, 90)
BATCH_SIZE = 100
NUM_UT = 100

# (sigma_SF LoS [dB], sigma_SF NLoS [dB], CL [dB]) per elevation angle 10..90 degrees
_EXPECTED = {
    # Table 6.6.2-1: dense urban
    ("dur", "s_band"): [(3.5, 15.5, 34.3), (3.4, 13.9, 30.9), (2.9, 12.4, 29.0),
                        (3.0, 11.7, 27.7), (3.1, 10.6, 26.8), (2.7, 10.5, 26.2),
                        (2.5, 10.1, 25.8), (2.3, 9.2, 25.5), (1.2, 9.2, 25.5)],
    ("dur", "ka_band"): [(2.9, 17.1, 44.3), (2.4, 17.1, 39.9), (2.7, 15.6, 37.5),
                         (2.4, 14.6, 35.8), (2.4, 14.2, 34.6), (2.7, 12.6, 33.8),
                         (2.6, 12.1, 33.3), (2.8, 12.3, 33.0), (0.6, 12.3, 32.9)],
    # Table 6.6.2-2: urban
    ("urb", "s_band"): [(4.0, 6.0, 34.3), (4.0, 6.0, 30.9), (4.0, 6.0, 29.0),
                        (4.0, 6.0, 27.7), (4.0, 6.0, 26.8), (4.0, 6.0, 26.2),
                        (4.0, 6.0, 25.8), (4.0, 6.0, 25.5), (4.0, 6.0, 25.5)],
    ("urb", "ka_band"): [(4.0, 6.0, 44.3), (4.0, 6.0, 39.9), (4.0, 6.0, 37.5),
                         (4.0, 6.0, 35.8), (4.0, 6.0, 34.6), (4.0, 6.0, 33.8),
                         (4.0, 6.0, 33.3), (4.0, 6.0, 33.0), (4.0, 6.0, 32.9)],
    # Table 6.6.2-3: suburban and rural
    ("sur", "s_band"): [(1.79, 8.93, 19.52), (1.14, 9.08, 18.17), (1.14, 8.78, 18.42),
                        (0.92, 10.25, 18.28), (1.42, 10.56, 18.63), (1.56, 10.74, 17.68),
                        (0.85, 10.17, 16.50), (0.72, 11.52, 16.30), (0.72, 11.52, 16.30)],
    ("sur", "ka_band"): [(1.9, 10.7, 29.5), (1.6, 10.0, 24.6), (1.9, 11.2, 21.9),
                         (2.3, 11.6, 20.0), (2.7, 11.8, 18.7), (3.1, 10.8, 17.8),
                         (3.0, 10.8, 17.2), (3.6, 10.8, 16.9), (0.4, 10.8, 16.8)],
}


def run_test(scenario, band, elevation_angle):
    carrier_frequency = _CARRIER[band]
    sf_los_sigma, sf_nlos_sigma, nlos_cl = _EXPECTED[(scenario, band)][_ELEVS.index(elevation_angle)]
    ut_array = create_ut_ant(carrier_frequency)
    bs_array = create_bs_ant(carrier_frequency)

    channel_model = _MODELS[scenario](carrier_frequency=carrier_frequency,
                                      ut_array=ut_array,
                                      bs_array=bs_array,
                                      direction="downlink",
                                      elevation_angle=float(elevation_angle),
                                      enable_pathloss=True,
                                      enable_shadow_fading=True)

    topology = utils.gen_single_sector_topology(batch_size=BATCH_SIZE, num_ut=NUM_UT,
                                                scenario=scenario,
                                                elevation_angle=float(elevation_angle),
                                                bs_height=600000.0)
    channel_model.set_topology(*topology)

    # Subtract FSPL to isolate Clutter Loss (CL) and Shadow Fading (SF)
    loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
    los = channel_model._scenario.los
    loss_los = loss_no_fspl[los]
    loss_nlos = loss_no_fspl[~los]

    where = f"{scenario}/{band}/{elevation_angle}deg"
    assert_mean(loss_los, 0.0, sf_los_sigma, f"LoS mean, {where}")
    assert_std(loss_los, sf_los_sigma, f"LoS std, {where}")
    assert_mean(loss_nlos, nlos_cl, sf_nlos_sigma, f"NLoS mean, {where}")
    assert_std(loss_nlos, sf_nlos_sigma, f"NLoS std, {where}")


def _make_case(scenario, band, elevation_angle):
    def test(self):
        run_test(scenario, band, elevation_angle)
    return test


def _populate(cls, scenario):
    for band in _CARRIER:
        for elevation_angle in _ELEVS:
            setattr(cls, f"test_{band}_{elevation_angle}_degrees_dl",
                    _make_case(scenario, band, elevation_angle))


class Test_DUR(unittest.TestCase):
    pass


class Test_URB(unittest.TestCase):
    pass


class Test_SUR(unittest.TestCase):
    pass


_populate(Test_DUR, "dur")
_populate(Test_URB, "urb")
_populate(Test_SUR, "sur")
