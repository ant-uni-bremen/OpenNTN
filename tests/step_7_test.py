# SPDX-FileCopyrightText: Copyright (c) 2025 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# This file tests the implementation of step 7, the angles of arrival and departure. 

import unittest
import torch
import numpy as np

# Importing necessary modules from Sionna
from sionna.phy import config
from sionna.phy.utils.random import uniform
import openntn.rays as rays
import openntn.dense_urban_scenario as sys_scenario
import openntn.antenna as antenna
from openntn.utils import gen_single_sector_topology as gen_topology
from openntn import Antenna, AntennaArray, DenseUrban, SubUrban, Urban

class Test_A_D_angles(unittest.TestCase):
    
    def setUp(self):
        self.batch_size = 100
        self.num_bs = 1
        self.num_ut = 2
        # Creating a mock antenna configuration
        self.antenna = antenna.Antenna(polarization="single",
                                           polarization_type="V",
                                           antenna_pattern="38.901",
                                           carrier_frequency=30e9)
        # Setting up a dense urban scenario
        self.scenario = DenseUrban(carrier_frequency=30e9, 
                                                            ut_array=self.antenna, 
                                                            bs_array=self.antenna,
                                                            direction="uplink", 
                                                            elevation_angle=80.0, 
                                                            enable_pathloss=True, 
                                                            enable_shadow_fading=True, 
                                                            doppler_enabled=True,
                                                            precision="single")
        # Generate the topology
        topology = gen_topology(batch_size=self.batch_size, num_ut=self.num_ut , scenario="dur", bs_height = 600000.0)

        # Set the topology
        self.scenario.set_topology(*topology)
        self.raysGen = self.scenario._ray_sampler
        self.lsp = self.scenario._lsp
        delays, unscaled_delays = self.raysGen._cluster_delays(self.lsp.ds, self.lsp.k_factor)
        self.cluster_powers, _ = self.raysGen._cluster_powers(
        self.lsp.ds, self.lsp.k_factor, unscaled_delays)
        
        
    def test_azimuth_angles_ranges(self):
        """
        Check that the azimuth angles of arrival (AoA) and departure (AoD) are wrapped within (-180, 180) degrees.
        """
        bs = self.scenario._scenario.num_bs
        ut = self.scenario._scenario.num_ut
        batch_size = self.scenario._scenario.batch_size
        #Mock values for the azimuth spread angles
        asa = uniform([batch_size, bs, ut], low=5.0, high=15.0, dtype=torch.float32, device=config.device, generator=config.torch_rng(config.device))
        asd = uniform([batch_size, bs, ut], low=5.0, high=15.0, dtype=torch.float32, device=config.device, generator=config.torch_rng(config.device))
        rician_k = self.lsp.k_factor
        cluster_powers = self.cluster_powers

        aoa = self.raysGen._azimuth_angles_of_arrival(asa, rician_k, cluster_powers)
        aod = self.raysGen._azimuth_angles_of_departure(asd, rician_k, cluster_powers)

        self.assertTrue(torch.all(aoa >= -180).cpu().numpy())
        self.assertTrue(torch.all(aoa <= 180).cpu().numpy())
        self.assertTrue(torch.all(aod >= -180).cpu().numpy())
        self.assertTrue(torch.all(aod <= 180).cpu().numpy())

    def test_zenith_angles_ranges(self):
        """
        Check that the zenith angles of arrival (ZoA) and departure (ZoD) are wrapped within (0, 180) degrees.
        """
        bs = self.scenario._scenario.num_bs
        ut = self.scenario._scenario.num_ut
        batch_size = self.scenario._scenario.batch_size
        #Mock values for the zenith spread angles
        zsa = uniform([batch_size, bs, ut], low=5.0, high=15.0, dtype=torch.float32, device=config.device, generator=config.torch_rng(config.device))
        zsd = uniform([batch_size, bs, ut], low=5.0, high=15.0, dtype=torch.float32, device=config.device, generator=config.torch_rng(config.device))
        rician_k = self.lsp.k_factor
        cluster_powers = self.cluster_powers
        zoa = self.raysGen._zenith_angles_of_arrival(zsa, rician_k, cluster_powers)
        zod = self.raysGen._zenith_angles_of_departure(zsd, rician_k, cluster_powers)

        self.assertTrue(torch.all(zoa >= 0).cpu().numpy())
        self.assertTrue(torch.all(zoa <= 180).cpu().numpy())
        self.assertTrue(torch.all(zod >= 0).cpu().numpy())
        self.assertTrue(torch.all(zod <= 180).cpu().numpy())

    def test_azimuth_angles_variability(self):
        """
        Verify that a larger azimuth spread input yields increased variability in the computed azimuth angles (AoA).
        """
        bs = self.scenario._scenario.num_bs
        ut = self.scenario._scenario.num_ut
        batch_size = self.scenario._scenario.batch_size

        # Use a low spread and a high spread.
        low_spread = torch.full([batch_size, bs, ut], 0.5, device=config.device)
        high_spread = torch.full([batch_size, bs, ut], 10.0, device=config.device)
        rician_k = self.lsp.k_factor
        cluster_powers = self.cluster_powers
        aoa_low = self.raysGen._azimuth_angles_of_arrival(low_spread, rician_k, cluster_powers)
        aoa_high = self.raysGen._azimuth_angles_of_arrival(high_spread, rician_k, cluster_powers)

        # Compute variability (standard deviation over the cluster dimension).
        var_low = torch.mean(torch.std(aoa_low, axis=3))
        var_high = torch.mean(torch.std(aoa_high, axis=3))
        self.assertGreater(var_high, var_low)

    def test_zenith_angles_variability(self):
        """
        Verify that a larger zenith spread input yields increased variability in the computed zenith angles (ZoA).
        """
        bs = self.scenario._scenario.num_bs
        ut = self.scenario._scenario.num_ut
        batch_size = self.scenario._scenario.batch_size

        low_spread = torch.full([batch_size, bs, ut], 0.5, device=config.device)
        high_spread = torch.full([batch_size, bs, ut], 10.0, device=config.device)
        rician_k = self.lsp.k_factor
        cluster_powers = self.cluster_powers
        zoa_low = self.raysGen._zenith_angles_of_arrival(low_spread, rician_k, cluster_powers)
        zoa_high = self.raysGen._zenith_angles_of_arrival(high_spread, rician_k, cluster_powers)

        var_low = torch.mean(torch.std(zoa_low, axis=3))
        var_high = torch.mean(torch.std(zoa_high, axis=3))
        self.assertGreater(var_high, var_low)


        

if __name__ == '__main__':
    unittest.main()