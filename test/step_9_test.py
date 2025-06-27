# This file tests the implementation of step 9, the Cross polarization ratio generation. 
# Step 9 test is  a mockup 
import unittest
import tensorflow as tf
import numpy as np
import sionna.phy.channel.tr38811.rays as rays
import sionna.phy.channel.tr38811.dense_urban_scenario as sys_scenario
import sionna.phy.channel.tr38811.antenna as antenna
from sionna.phy.channel.tr38811.utils import gen_single_sector_topology as gen_topology

class Step_9(unittest.TestCase):
    def setUp(self):
        # Initialize the required attributes and setup the scenario
        self.batch_size = 200
        self.num_bs = 1
        self.num_ut = 1
        self.num_clusters_max = 4
        self.rays_per_cluster = 20
        self.precision = "single"
        self.mu_xpr = 17.6
        self.sigma_xpr = 12.7

        # Create the antenna
        self.mock_antenna = antenna.Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=30e9
        )

        # Create the scenario
        self.mock_scenario = sys_scenario.DenseUrbanScenario(
            carrier_frequency=30e9,
            ut_array=self.mock_antenna,
            bs_array=self.mock_antenna,
            direction="uplink",
            elevation_angle=90.0,
            enable_pathloss=True,
            enable_shadow_fading=True,
            doppler_enabled=True,
            precision=self.precision
        )

        # Generate the topology
        topology = gen_topology(
            batch_size=self.batch_size,
            num_ut=self.num_ut,
            scenario="dur",
            elevation_angle=90,
            bs_height=600000.0
        )

        # Set the topology for the scenario
        self.mock_scenario.set_topology(*topology)

        # Create an instance of the RaysGenerator
        self.rays_generator = rays.RaysGenerator(self.mock_scenario)

    def test_cross_polarization_power_ratios(self):
        # Generate the cross-polarization power ratios
        result = self.rays_generator._cross_polarization_power_ratios()

        # Expected tensor shape
        expected_shape = (
            self.batch_size,
            self.num_bs,
            self.num_ut,
            self.num_clusters_max,
            self.rays_per_cluster
        )

        # Check if the received tensor shape is as expected
        self.assertEqual(result.shape, expected_shape)

        # Check if the values are positive
        self.assertTrue(tf.reduce_all(result > 0).numpy())

        # Check the mean and std of the distribution against hardcoded values
        # Linear to dB
        log_result = 10 * np.log10(result.numpy())

        expected_mean = self.mu_xpr
        expected_std = self.sigma_xpr

        measured_mean = np.mean(log_result)
        measured_std = np.std(log_result)

        self.assertAlmostEqual(measured_mean, expected_mean, delta=1.0)
        self.assertAlmostEqual(measured_std, expected_std, delta=1.0)

if __name__ == "__main__":
    unittest.main()
