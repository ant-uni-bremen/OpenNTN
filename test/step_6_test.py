# This file tests the implementation of step 6, the cluster power generation. To do this, the ideal
# values for all calculations are done and the average calculation is compared to it. As step 4 already
# tests the correct creation of the LSPs Delay Spread (DS) and the Rician K Factor (K), we assume these
# to be correct here.
# Step 6 has no easily measurable output, so that a mockup 

from sionna.phy.channel.tr38811 import utils   # The code to test
import unittest   # The test framework
from sionna.phy.channel.tr38811 import Antenna, AntennaArray, DenseUrban, SubUrban, Urban, CDL
import numpy as np
import tensorflow as tf
import math
from sionna.phy import config
import json
import os

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
    #         self.assertAlmostEqual(tf.reduce_sum(power[:,:,:i]).numpy(), 1.0, places=5)
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
        ric_fac = lsp.k_factor
        ric_fac = tf.expand_dims(ric_fac, axis=3)
        P1_los = ric_fac/(1+ric_fac)
        los_powers, _ = rays_generator._cluster_powers(
            self.channel_model._lsp.ds, self.channel_model._lsp.k_factor, unscaled_delays
        )
        first_cluster_power = los_powers[:,:,:, :1]  # First cluster
        #first_cluster_power = tf.reduce_mean(first_cluster_power).numpy()
        
        # print("Size :", first_cluster_power.shape)
        # print("Size_P1_los :", P1_los.shape)
        # print("First cluster power :", tf.reduce_mean(first_cluster_power).numpy())
        # print("P1_los :", tf.reduce_mean(P1_los).numpy())

        assert math.isclose(tf.reduce_mean(first_cluster_power).numpy(), tf.reduce_mean(P1_los).numpy(), abs_tol=0.3)

    def test_rays_equal_power(self):
        """" Testing if each ray has equal power"""""
        scenario = "urb"
        topology = utils.gen_single_sector_topology(
            batch_size=self.batch_size, num_ut=100, scenario=scenario,
            elevation_angle=self.elevation_angle, bs_height=600000.0
        )
        self.channel_model.set_topology(*topology)
        rays_generator = self.channel_model._ray_sampler
        lsp = self.channel_model._lsp
        delays, unscaled_delays = rays_generator._cluster_delays(lsp.ds, lsp.k_factor)
        cluster_powers, _ = rays_generator._cluster_powers(
        lsp.ds, lsp.k_factor, unscaled_delays)
        for cluster_power in cluster_powers:
            for power in cluster_power:
                self.assertTrue(tf.reduce_all(tf.equal(power, cluster_power[0])).numpy())  # All powers equal within cluster

    def test_cluster_elimination(self):
        """Check if any cluster has -25 dB power compared to the maximum cluster power"""
        scenario = "urb"
        topology = utils.gen_single_sector_topology(
            batch_size=self.batch_size, num_ut=100, scenario=scenario,
            elevation_angle=self.elevation_angle, bs_height=600000.0
        )
        self.channel_model.set_topology(*topology)
        threshold_db = -25  # dB threshold
        threshold_power = 10 ** (threshold_db / 10)  # Convert dB to linear scale
        rays_generator = self.channel_model._ray_sampler
        lsp = self.channel_model._lsp
        delays, unscaled_delays = rays_generator._cluster_delays(lsp.ds, lsp.k_factor)
        cluster_powers, _ = rays_generator._cluster_powers(lsp.ds, lsp.k_factor, unscaled_delays)

        epsilon = 1e-12  # Small value to avoid log issues
        for power in cluster_powers:
            power = tf.maximum(power, epsilon)  # Avoid log of zero
            max_power = tf.reduce_max(cluster_powers)  # Find maximum power in the cluster
            difference = max_power - power
            #print(difference.shape)
            # Ensure no clusters remain below the threshold
            self.assertTrue(tf.reduce_all(tf.reduce_mean(difference)>= threshold_power).numpy())
            
if __name__ == '__main__':
    unittest.main()