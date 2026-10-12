# SPDX-FileCopyrightText: Copyright (c) 2025 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# This file tests the path loss of a GEO downlink in Ka band, the link budget case SC1 of
# 3GPP TR 38.821 Table 6.1.3.3-1 (20 GHz, free space path loss 210.6 dB), in the urban
# scenario at 12.5 degrees elevation. It checks the components of TR 38.811 V15.4.0,
# Section 6.6.2:
#   - slant range, eq. (6.6-3), with the Earth radius of 6371 km of Section 6.3;
#   - free space path loss, eq. (6.6-2);
#   - total path loss PL = PL_b + PL_g + PL_s + PL_e, eq. (6.6-1);
#   - basic path loss PL_b = FSPL + SF + CL, eq. (6.6-4), with the shadow fading and
#     clutter loss of Table 6.6.2-2 and the LOS probability of Table 6.6.1-1, both at
#     the reference elevation angle nearest to 12.5 degrees, 10 degrees. The models
#     apply the shadow fading SF once, as the large scale parameter SF of TR 38.901
#     V16.1.0, clause 7.5, step 4, in step 12 (Tables 6.7.2-1a to 6.7.2-8b of TR 38.811
#     give its cross-correlations and correlation distance), so PL_b is FSPL + CL, bit
#     for bit, and the shadow fading is 10 lg(sf) of the large scale parameter sf
#     (positive means more received power, NOTE 2 of Tables 6.7.2-1a to 6.7.2-8b).
# The statistical checks use the tolerances of statistical_checks.py. The topology has
# one UT per batch example, so that the shadow fading samples of different links are
# independent (the shadow fading of UTs of one example is spatially correlated).
import math
import unittest

import torch

from openntn import utils   # The code to test
from openntn import Antenna, AntennaArray, Urban

from statistical_checks import assert_mean, assert_proportion, assert_std

CARRIER_FREQUENCY = 20e9
ELEVATION_ANGLE = 12.5
GEO_HEIGHT = 35786000.0
EARTH_RADIUS = 6371000.0
# Table 6.6.2-2 (urban, Ka band) and Table 6.6.1-1 (urban) at 10 degrees
SIGMA_SF_LOS = 4.0
SIGMA_SF_NLOS = 6.0
CLUTTER_LOSS = 44.3
LOS_PROBABILITY = 0.246


class TestCouplingLoss(unittest.TestCase):

    def setUp(self):
        ut_array = Antenna(polarization="single",
                           polarization_type="V",
                           antenna_pattern="38.901",
                           carrier_frequency=CARRIER_FREQUENCY)
        bs_array = AntennaArray(num_rows=1,
                                num_cols=4,
                                polarization="dual",
                                polarization_type="VH",
                                antenna_pattern="38.901",
                                carrier_frequency=CARRIER_FREQUENCY)
        self.channel_model = Urban(carrier_frequency=CARRIER_FREQUENCY,
                                   ut_array=ut_array,
                                   bs_array=bs_array,
                                   direction="downlink",
                                   elevation_angle=ELEVATION_ANGLE,
                                   enable_pathloss=True,
                                   enable_shadow_fading=True)
        topology = utils.gen_single_sector_topology(batch_size=4000,
                                                    num_ut=1,
                                                    scenario="urb",
                                                    elevation_angle=ELEVATION_ANGLE,
                                                    bs_height=GEO_HEIGHT)
        self.channel_model.set_topology(*topology)
        self.scenario = self.channel_model._scenario

    def test_slant_range(self):
        sin_alpha = math.sin(math.radians(ELEVATION_ANGLE))
        expected = (math.sqrt(EARTH_RADIUS ** 2 * sin_alpha ** 2 + GEO_HEIGHT ** 2
                              + 2 * GEO_HEIGHT * EARTH_RADIUS) - EARTH_RADIUS * sin_alpha)
        distance = self.scenario.distance_3d.double()
        torch.testing.assert_close(distance, torch.full_like(distance, expected),
                                   rtol=1e-6, atol=0.0)

    def test_free_space_path_loss(self):
        distance = self.scenario.distance_3d.double()
        expected = 32.45 + 20 * math.log10(CARRIER_FREQUENCY / 1e9) + 20 * torch.log10(distance)
        fspl = self.scenario.free_space_pathloss.double()
        # float32 rounding of a 210 dB quantity
        torch.testing.assert_close(fspl, expected, rtol=0.0, atol=1e-3)
        # TR 38.821 prints 210.6 dB, so the value lies within half the last digit.
        self.assertLessEqual(float((fspl - 210.6).abs().max()), 0.05)

    def test_total_path_loss(self):
        sc = self.scenario
        total = self.channel_model._lsp_sampler.sample_pathloss().double()
        components = (sc.basic_pathloss.double() + sc.gas_pathloss.double()
                      + sc.scintillation_pathloss.double() + sc.entry_pathloss.double())
        torch.testing.assert_close(total, components, rtol=0.0, atol=1e-3)

    def test_basic_path_loss_distribution(self):
        sc = self.scenario
        los = sc.los
        assert_proportion(int(los.sum()), los.numel(), LOS_PROBABILITY, "LOS fraction")
        basic, fspl = sc.basic_pathloss, sc.free_space_pathloss
        cl = torch.tensor(CLUTTER_LOSS, dtype=basic.dtype, device=basic.device)
        self.assertTrue(torch.equal(basic[los], fspl[los]), "LOS: PL_b differs from FSPL")
        self.assertTrue(torch.equal(basic[~los], fspl[~los] + cl),
                        "NLOS: PL_b differs from FSPL + CL")
        shadow_fading = 10.0 * torch.log10(self.channel_model._lsp.sf.double())
        assert_mean(shadow_fading[los], 0.0, SIGMA_SF_LOS, "LOS: shadow fading, mean")
        assert_std(shadow_fading[los], SIGMA_SF_LOS, "LOS: shadow fading, std")
        assert_mean(shadow_fading[~los], 0.0, SIGMA_SF_NLOS, "NLOS: shadow fading, mean")
        assert_std(shadow_fading[~los], SIGMA_SF_NLOS, "NLOS: shadow fading, std")


if __name__ == '__main__':
    unittest.main()
