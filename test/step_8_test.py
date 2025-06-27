# This file tests the implementation of step 8, the angles coupling and shuffling. 
# Step 8 is a mockup 
import unittest
import tensorflow as tf
import numpy as np

# Importing necessary modules from Sionna
import sionna.phy.channel.tr38811.rays as rays
import sionna.phy.channel.tr38811.dense_urban_scenario as sys_scenario
import sionna.phy.channel.tr38811.antenna as antenna
from sionna.phy.channel.tr38811.utils import gen_single_sector_topology as gen_topology

class TestShuffle_Coupling(unittest.TestCase):
    
    def setUp(self):
        # Creating a mock antenna configuration
        self.mockAntenna = antenna.Antenna(polarization="single",
                                           polarization_type="V",
                                           antenna_pattern="38.901",
                                           carrier_frequency=30e9)
        # Setting up a dense urban scenario
        self.mockScenario = sys_scenario.DenseUrbanScenario(carrier_frequency=30e9, 
                                                            ut_array=self.mockAntenna, 
                                                            bs_array=self.mockAntenna,
                                                            direction="uplink", 
                                                            elevation_angle=90.0, 
                                                            enable_pathloss=True, 
                                                            enable_shadow_fading=True, 
                                                            doppler_enabled=True,
                                                            precision="single")
        # Generate the topology
        topology = gen_topology(batch_size=2, num_ut=1, scenario="dur", elevation_angle=90, bs_height = 600000.0)

        # Set the topology
        self.mockScenario.set_topology(*topology)
        self.raysGenerator = rays.RaysGenerator(self.mockScenario)

    def test_random_coupling(self):
        # Creating test data for angles
        aoa = tf.constant(np.random.rand(2, 1, 1, 4, 20), dtype=tf.float32)
        aod = tf.constant(np.random.rand(2, 1, 1, 4, 20), dtype=tf.float32)
        zoa = tf.constant(np.random.rand(2, 1, 1, 4, 20), dtype=tf.float32)
        zod = tf.constant(np.random.rand(2, 1, 1, 4, 20), dtype=tf.float32)

        # Testing random coupling function
        shuffled_aoa, shuffled_aod, shuffled_zoa, shuffled_zod = self.raysGenerator._random_coupling(
            aoa, aod, zoa, zod
        )

        # Checking the shape of the shuffled outputs
        self.assertEqual(aoa.shape, shuffled_aoa.shape)
        self.assertEqual(aod.shape, shuffled_aod.shape)
        self.assertEqual(zoa.shape, shuffled_zoa.shape)
        self.assertEqual(zod.shape, shuffled_zod.shape)

        # Ensuring that the angles are shuffled
        self.assertFalse(tf.reduce_all(tf.equal(aoa, shuffled_aoa)).numpy())
        self.assertFalse(tf.reduce_all(tf.equal(aod, shuffled_aod)).numpy())
        self.assertFalse(tf.reduce_all(tf.equal(zoa, shuffled_zoa)).numpy())
        self.assertFalse(tf.reduce_all(tf.equal(zod, shuffled_zod)).numpy())

if __name__ == "__main__":
    unittest.main()