# This file tests the implementation of step 7, the angles of arrival and departure. 

import unittest
import tensorflow as tf
import numpy as np

# Importing necessary modules from Sionna
import sionna.phy.channel.tr38811.rays as rays
import sionna.phy.channel.tr38811.dense_urban_scenario as sys_scenario
import sionna.phy.channel.tr38811.antenna as antenna
from sionna.phy.channel.tr38811.utils import gen_single_sector_topology as gen_topology
from sionna.phy.channel.tr38811 import Antenna, AntennaArray, DenseUrban, SubUrban, Urban, CDL

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
        asa = tf.random.uniform([batch_size, bs, ut], minval=5.0, maxval=15.0, dtype=tf.float32)
        asd = tf.random.uniform([batch_size, bs, ut], minval=5.0, maxval=15.0, dtype=tf.float32)
        rician_k = self.lsp.k_factor
        cluster_powers = self.cluster_powers

        aoa = self.raysGen._azimuth_angles_of_arrival(asa, rician_k, cluster_powers)
        aod = self.raysGen._azimuth_angles_of_departure(asd, rician_k, cluster_powers)

        self.assertTrue(tf.reduce_all(aoa >= -180).numpy())
        self.assertTrue(tf.reduce_all(aoa <= 180).numpy())
        self.assertTrue(tf.reduce_all(aod >= -180).numpy())
        self.assertTrue(tf.reduce_all(aod <= 180).numpy())

    def test_zenith_angles_ranges(self):
        """
        Check that the zenith angles of arrival (ZoA) and departure (ZoD) are wrapped within (0, 180) degrees.
        """
        bs = self.scenario._scenario.num_bs
        ut = self.scenario._scenario.num_ut
        batch_size = self.scenario._scenario.batch_size
        #Mock values for the zenith spread angles
        zsa = tf.random.uniform([batch_size, bs, ut], minval=5.0, maxval=15.0, dtype=tf.float32)
        zsd = tf.random.uniform([batch_size, bs, ut], minval=5.0, maxval=15.0, dtype=tf.float32)
        rician_k = self.lsp.k_factor
        cluster_powers = self.cluster_powers
        zoa = self.raysGen._zenith_angles_of_arrival(zsa, rician_k, cluster_powers)
        zod = self.raysGen._zenith_angles_of_departure(zsd, rician_k, cluster_powers)

        self.assertTrue(tf.reduce_all(zoa >= 0).numpy())
        self.assertTrue(tf.reduce_all(zoa <= 180).numpy())
        self.assertTrue(tf.reduce_all(zod >= 0).numpy())
        self.assertTrue(tf.reduce_all(zod <= 180).numpy())

    def test_azimuth_angles_variability(self):
        """
        Verify that a larger azimuth spread input yields increased variability in the computed azimuth angles (AoA).
        """
        bs = self.scenario._scenario.num_bs
        ut = self.scenario._scenario.num_ut
        batch_size = self.scenario._scenario.batch_size

        # Use a low spread and a high spread.
        low_spread = tf.fill([batch_size, bs, ut], 0.5)
        high_spread = tf.fill([batch_size, bs, ut], 10.0)
        rician_k = self.lsp.k_factor
        cluster_powers = self.cluster_powers
        aoa_low = self.raysGen._azimuth_angles_of_arrival(low_spread, rician_k, cluster_powers)
        aoa_high = self.raysGen._azimuth_angles_of_arrival(high_spread, rician_k, cluster_powers)

        # Compute variability (standard deviation over the cluster dimension).
        var_low = tf.reduce_mean(tf.math.reduce_std(aoa_low, axis=3))
        var_high = tf.reduce_mean(tf.math.reduce_std(aoa_high, axis=3))
        self.assertGreater(var_high, var_low)

    def test_zenith_angles_variability(self):
        """
        Verify that a larger zenith spread input yields increased variability in the computed zenith angles (ZoA).
        """
        bs = self.scenario._scenario.num_bs
        ut = self.scenario._scenario.num_ut
        batch_size = self.scenario._scenario.batch_size

        low_spread = tf.fill([batch_size, bs, ut], 0.5)
        high_spread = tf.fill([batch_size, bs, ut], 10.0)
        rician_k = self.lsp.k_factor
        cluster_powers = self.cluster_powers
        zoa_low = self.raysGen._zenith_angles_of_arrival(low_spread, rician_k, cluster_powers)
        zoa_high = self.raysGen._zenith_angles_of_arrival(high_spread, rician_k, cluster_powers)

        var_low = tf.reduce_mean(tf.math.reduce_std(zoa_low, axis=3))
        var_high = tf.reduce_mean(tf.math.reduce_std(zoa_high, axis=3))
        self.assertGreater(var_high, var_low)


        

if __name__ == '__main__':
    unittest.main()