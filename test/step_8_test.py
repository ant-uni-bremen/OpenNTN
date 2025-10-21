# This file tests the implementation of step 8, the angles coupling and shuffling.
# Step 8 is a mockup 

import unittest
import tensorflow as tf
import numpy as np

from sionna.phy.channel.tr38811.utils import gen_single_sector_topology as gen_topology
from sionna.phy.channel.tr38811 import Antenna, RaysGenerator, DenseUrbanScenario, SubUrbanScenario, UrbanScenario, RuralScenario

class TestShuffle_Coupling(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 128
        cls.num_ut = 1
        cls.carrier_frequency = 30e9
        cls.elevation_angle = 60.0

        cls.antenna = Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=cls.carrier_frequency
        )
        cls.scenario_classes = {
            "sur": SubUrbanScenario,
            "urb": UrbanScenario,
            "dur": DenseUrbanScenario,
            "rur": RuralScenario
        }

    def setUpScenario(self, scenario_key):
        """Helper function to configure a given scenario."""
        ScenarioClass = self.scenario_classes[scenario_key]
        scenario = ScenarioClass(
            carrier_frequency=self.carrier_frequency,
            ut_array=self.antenna,
            bs_array=self.antenna,
            direction="downlink",
            elevation_angle=self.elevation_angle,
            enable_pathloss=True,
            enable_shadow_fading=True,
            doppler_enabled=True
        )

        topology = gen_topology(
            batch_size=self.batch_size,
            num_ut=self.num_ut,
            scenario=scenario_key,
            bs_height=600000.0
        )
        scenario.set_topology(*topology)

        raysGenerator = RaysGenerator(scenario)
        max_num_clusters = scenario.num_clusters_max

        return scenario, raysGenerator, max_num_clusters

    def test_random_coupling(self):
        """Verify that the random coupling correctly shuffles angles while maintaining shape."""
        for key in self.scenario_classes:
            with self.subTest(scenario=key):
                scenario, raysGenerator, max_num_clusters = self.setUpScenario(key)

                aoa = tf.constant(np.random.rand(self.batch_size, 1, 1, max_num_clusters, 20), dtype=tf.float32)
                aod = tf.constant(np.random.rand(self.batch_size, 1, 1, max_num_clusters, 20), dtype=tf.float32)
                zoa = tf.constant(np.random.rand(self.batch_size, 1, 1, max_num_clusters, 20), dtype=tf.float32)
                zod = tf.constant(np.random.rand(self.batch_size, 1, 1, max_num_clusters, 20), dtype=tf.float32)

                # Apply random coupling
                shuffled_aoa, shuffled_aod, shuffled_zoa, shuffled_zod = raysGenerator._random_coupling(
                    aoa, aod, zoa, zod
                )

                self.assertEqual(aoa.shape, shuffled_aoa.shape)
                self.assertEqual(aod.shape, shuffled_aod.shape)
                self.assertEqual(zoa.shape, shuffled_zoa.shape)
                self.assertEqual(zod.shape, shuffled_zod.shape)

                self.assertFalse(tf.reduce_all(tf.equal(aoa, shuffled_aoa)).numpy())
                self.assertFalse(tf.reduce_all(tf.equal(aod, shuffled_aod)).numpy())
                self.assertFalse(tf.reduce_all(tf.equal(zoa, shuffled_zoa)).numpy())
                self.assertFalse(tf.reduce_all(tf.equal(zod, shuffled_zod)).numpy())

if __name__ == "__main__":
    unittest.main()
