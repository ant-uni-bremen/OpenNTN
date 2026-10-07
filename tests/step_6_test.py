# SPDX-FileCopyrightText: Copyright (c) 2025 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# This file tests the implementation of step 6, the cluster power generation. To do this, the ideal
# values for all calculations are done and the average calculation is compared to it. As step 4 already
# tests the correct creation of the LSPs Delay Spread (DS) and the Rician K Factor (K), we assume these
# to be correct here.
# Step 6 has no easily measurable output, so that a mockup 

from openntn import utils   # The code to test
import unittest   # The test framework

import pytest
from openntn import Antenna, AntennaArray, DenseUrban, SubUrban, Urban, Topology
import numpy as np
import torch
import math
from sionna.phy import config
from sionna.phy.channel.utils import deg_2_rad
import json
import os

# Every test of this file takes about a second or more.
pytestmark = pytest.mark.slow


def create_ut_ant(carrier_frequency):
    ut_ant = Antenna(polarization="single",
                    polarization_type="V",
                    antenna_pattern="38.901",
                    carrier_frequency=carrier_frequency)
    return ut_ant

def create_bs_ant(carrier_frequency):
    bs_ant = AntennaArray(num_rows=1,
                            num_cols=4,
                            polarization="dual",
                            polarization_type="VH",
                            antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency)
    return bs_ant

class TestClusterPowerGeneration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.carrier_frequency = 2.2e9  
        cls.elevation_angle = 10.0     
        cls.batch_size = 1000           
        cls.ut_array = create_ut_ant(cls.carrier_frequency)
        cls.bs_array = create_bs_ant(cls.carrier_frequency)
        cls.channel_model = Urban(
            carrier_frequency=cls.carrier_frequency,
            ut_array=cls.ut_array,
            bs_array=cls.bs_array,
            direction="downlink",
            elevation_angle=cls.elevation_angle,
            enable_pathloss=True,
            enable_shadow_fading=True
        )
    # def test_sum_of_clusters_one(self):
        
    #     scenario = "urb"
    #     topology = utils.gen_single_sector_topology(
    #         batch_size=self.batch_size, num_ut=100, scenario=scenario, 
    #         elevation_angle=self.elevation_angle, bs_height=600000.0
    #     )
    #     self.channel_model.set_topology(*topology)
    #     rays_generator = self.channel_model._ray_sampler
    #     lsp = self.channel_model._lsp
    #     delays, unscaled_delays = rays_generator._cluster_delays(lsp.ds, lsp.k_factor)
    #     powers, _ = rays_generator._cluster_powers(
    #         self.channel_model._lsp.ds, self.channel_model._lsp.k_factor, unscaled_delays
    #     )     
    #     print(powers.shape)  
    #     i = 1
    #     for power in powers: 
    #         self.assertAlmostEqual(torch.sum(power[:,:,:i]).numpy(), 1.0, places=5)
    #         print(power[:,:,1])
    #         i+=1

    def test_specular_component_los(self):
        scenario = "urb"
        topology = utils.gen_single_sector_topology(
            batch_size=self.batch_size, num_ut=100, scenario=scenario,
            elevation_angle=self.elevation_angle, bs_height=600000.0
        )
        self.channel_model.set_topology(*topology)
        rays_generator = self.channel_model._ray_sampler
        lsp = self.channel_model._lsp
        delays, unscaled_delays = rays_generator._cluster_delays(lsp.ds, lsp.k_factor)
        powers, powers_los = rays_generator._cluster_powers(lsp.ds, lsp.k_factor, unscaled_delays)

        # TR 38.901 V16.1.0, eqs. (7.5-7) and (7.5-8): for LoS links the normalized powers
        # P'_n are scaled by 1/(K_R+1), and P_1,LOS = K_R/(K_R+1) is added to the first
        # cluster. NLoS links keep the powers of eq. (7.5-6). The relation is exact, so the
        # tolerance covers float32 rounding only.
        k_factor = torch.unsqueeze(lsp.k_factor, 3)
        p_1_los = k_factor / (k_factor + 1.0)
        expected = powers / (k_factor + 1.0)
        expected = torch.cat([expected[..., :1] + p_1_los, expected[..., 1:]], dim=3)
        los = torch.unsqueeze(self.channel_model._scenario.los, 3)
        expected = torch.where(los, expected, powers)
        self.assertTrue(bool(torch.any(los)) and bool(torch.any(~los)),
                        "the topology must contain LoS and NLoS links")
        torch.testing.assert_close(powers_los, expected, rtol=1e-5, atol=1e-7)

    def test_rays_equal_power(self):
        """TR 38.901 V16.1.0, 7.5 step 6: each ray of cluster n has the power P_n/M."""
        # The ray powers are applied in step 11 (eq. 7.5-22). With omnidirectional,
        # vertically polarised single antennas in the global orientation, the field,
        # array and Doppler terms of each ray have unit modulus, so the power of each ray
        # of the NLOS channel matrix is its ray power. Faraday rotation, which mixes the
        # polarisations, is applied only for satellite heights of 600 km and more, so the
        # topology passed to step 11 uses height 0. The relation is exact; the tolerance
        # covers float32 rounding.
        fc = self.carrier_frequency
        omni = Antenna(polarization="single", polarization_type="V",
                       antenna_pattern="omni", carrier_frequency=fc)
        model = Urban(carrier_frequency=fc, ut_array=omni, bs_array=omni,
                      direction="downlink", elevation_angle=self.elevation_angle)
        model.set_topology(*utils.gen_single_sector_topology(
            batch_size=50, num_ut=10, scenario="urb",
            elevation_angle=self.elevation_angle, bs_height=600000.0))
        sc = model._scenario
        rays = model._ray_sampler(model._lsp)
        topology = Topology(velocities=sc.ut_velocities,
                            moving_end="rx",
                            los_aoa=deg_2_rad(sc.los_aoa),
                            los_aod=deg_2_rad(sc.los_aod),
                            los_zoa=deg_2_rad(sc.los_zoa),
                            los_zod=deg_2_rad(sc.los_zod),
                            los=sc.los,
                            distance_3d=sc.distance_3d,
                            tx_orientations=torch.zeros_like(sc.bs_orientations),
                            rx_orientations=torch.zeros_like(sc.ut_orientations),
                            bs_height=torch.zeros_like(sc.bs_loc[:, :, 2][0]),
                            elevation_angle=sc.elevation_angle,
                            doppler_enabled=sc.doppler_enabled)
        ccg = model._cir_sampler
        phi = ccg._step_10(rays.aoa.shape)
        sample_times = torch.arange(4, dtype=sc.dtype, device=sc.device) * 1e-4
        # [batch, num_tx, num_rx, num_clusters, num_rays, num_rx_ant, num_tx_ant, time]
        h = ccg._step_11_nlos(phi, topology, rays, sample_times, fc)
        num_rays = h.shape[4]
        self.assertEqual(num_rays, int(sc.rays_per_cluster))
        power = torch.abs(h) ** 2
        expected = (rays.powers / num_rays)[..., None, None, None, None].expand_as(power)
        torch.testing.assert_close(power, expected, rtol=1e-5, atol=1e-12)

    def test_cluster_elimination(self):
        """TR 38.901 V16.1.0, 7.5 step 6: "Remove clusters with less than -25 dB power
        compared to the maximum cluster power." """
        # The cluster powers are those of eq. (7.5-6), on LOS and NLOS links; the
        # specular component of eq. (7.5-8) is not part of the comparison. A removed
        # cluster has zero power.
        self.channel_model.set_topology(*utils.gen_single_sector_topology(
            batch_size=100, num_ut=100, scenario="urb",
            elevation_angle=self.elevation_angle, bs_height=600000.0))
        rays = self.channel_model._ray_sampler(self.channel_model._lsp)
        relative = rays.powers / rays.powers.amax(dim=3, keepdim=True)
        los = torch.unsqueeze(self.channel_model._scenario.los, 3).expand_as(relative)
        kept_weak = (relative > 0) & (relative < 10 ** (-25 / 10))
        used = self.channel_model._ray_sampler._cluster_mask == 0.0
        removed = used & (relative == 0)
        for name, state in (("LOS", los), ("NLOS", ~los)):
            self.assertTrue(bool(state.any()), f"no {name} links")
            self.assertEqual(int((kept_weak & state).sum()), 0,
                             f"{name}: clusters below -25 dB of the strongest cluster are "
                             "still present")
            self.assertGreater(int((removed & state).sum()), 0, f"{name}: no cluster removed")

if __name__ == '__main__':
    unittest.main()