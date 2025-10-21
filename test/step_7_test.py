# This file tests the implementation of step 7, the angles of arrival and departure. 

# This file tests the implementation of step 7, the angles of arrival and departure. 

import unittest
import tensorflow as tf
import numpy as np

# Importing necessary modules from Sionna
from sionna.phy.channel.tr38811.utils import gen_single_sector_topology as gen_topology
from sionna.phy.channel.tr38811 import Antenna, AntennaArray, DenseUrban, SubUrban, Urban, Rural

class Test_A_D_angles(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.batch_size = 100
        cls.num_bs = 1
        cls.num_ut = 1
        cls.carrier_frequency = 30e9
        cls.elevation_angle = 60.0

        # Common antenna configuration
        cls.antenna = Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=cls.carrier_frequency
        )

        # Dictionary for scenarios
        cls.scenario_classes = {
            "sur": SubUrban,
            "urb": Urban,
            "dur": DenseUrban,
            "rur": Rural
        }

    def setUpScenario(self, scenario_key):
        """Helper to create and configure scenario"""
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
        raysGen = scenario._ray_sampler
        lsp = scenario._lsp
        delays, unscaled_delays = raysGen._cluster_delays(lsp.ds, lsp.k_factor)
        cluster_powers, _ = raysGen._cluster_powers(lsp.ds, lsp.k_factor, unscaled_delays)

        return scenario, raysGen, lsp, cluster_powers

    def test_azimuth_angles_ranges(self):
        """Check AoA and AoD azimuth angles stay within (-180, 180) degrees."""
        for key in self.scenario_classes:
            with self.subTest(scenario=key):
                scenario, raysGen, lsp, cluster_powers = self.setUpScenario(key)
                bs = scenario._scenario.num_bs
                ut = scenario._scenario.num_ut
                batch_size = scenario._scenario.batch_size

                # Mock azimuth spreads
                asa = tf.random.uniform([batch_size, bs, ut], minval=5.0, maxval=15.0)
                asd = tf.random.uniform([batch_size, bs, ut], minval=5.0, maxval=15.0)
                rician_k = lsp.k_factor

                aoa = raysGen._azimuth_angles_of_arrival(asa, rician_k, cluster_powers)
                aod = raysGen._azimuth_angles_of_departure(asd, rician_k, cluster_powers)

                self.assertTrue(tf.reduce_all(aoa >= -180).numpy())
                self.assertTrue(tf.reduce_all(aoa <= 180).numpy())
                self.assertTrue(tf.reduce_all(aod >= -180).numpy())
                self.assertTrue(tf.reduce_all(aod <= 180).numpy())

    def test_zenith_angles_ranges(self):
        """Check ZoA and ZoD zenith angles stay within (0, 180) degrees."""
        for key in self.scenario_classes:
            with self.subTest(scenario=key):
                scenario, raysGen, lsp, cluster_powers = self.setUpScenario(key)
                bs = scenario._scenario.num_bs
                ut = scenario._scenario.num_ut
                batch_size = scenario._scenario.batch_size

                # Mock zenith spreads
                zsa = tf.random.uniform([batch_size, bs, ut], minval=5.0, maxval=15.0)
                zsd = tf.random.uniform([batch_size, bs, ut], minval=5.0, maxval=15.0)
                rician_k = lsp.k_factor

                zoa = raysGen._zenith_angles_of_arrival(zsa, rician_k, cluster_powers)
                zod = raysGen._zenith_angles_of_departure(zsd, rician_k, cluster_powers)

                self.assertTrue(tf.reduce_all(zoa >= 0).numpy())
                self.assertTrue(tf.reduce_all(zoa <= 180).numpy())
                self.assertTrue(tf.reduce_all(zod >= 0).numpy())
                self.assertTrue(tf.reduce_all(zod <= 180).numpy())

    def test_azimuth_angles_variability(self):
        """Check that higher azimuth spread increases variability in AoA."""
        for key in self.scenario_classes:
            with self.subTest(scenario=key):
                scenario, raysGen, lsp, cluster_powers = self.setUpScenario(key)
                bs = scenario._scenario.num_bs
                ut = scenario._scenario.num_ut
                batch_size = scenario._scenario.batch_size

                low_spread = tf.fill([batch_size, bs, ut], 0.5)
                high_spread = tf.fill([batch_size, bs, ut], 10.0)
                rician_k = lsp.k_factor

                aoa_low = raysGen._azimuth_angles_of_arrival(low_spread, rician_k, cluster_powers)
                aoa_high = raysGen._azimuth_angles_of_arrival(high_spread, rician_k, cluster_powers)

                var_low = tf.reduce_mean(tf.math.reduce_std(aoa_low, axis=3))
                var_high = tf.reduce_mean(tf.math.reduce_std(aoa_high, axis=3))

                self.assertGreater(var_high, var_low)

    def test_zenith_angles_variability(self):
        """Check that higher zenith spread increases variability in ZoA."""
        for key in self.scenario_classes:
            with self.subTest(scenario=key):
                scenario, raysGen, lsp, cluster_powers = self.setUpScenario(key)
                bs = scenario._scenario.num_bs
                ut = scenario._scenario.num_ut
                batch_size = scenario._scenario.batch_size

                low_spread = tf.fill([batch_size, bs, ut], 0.5)
                high_spread = tf.fill([batch_size, bs, ut], 10.0)
                rician_k = lsp.k_factor

                zoa_low = raysGen._zenith_angles_of_arrival(low_spread, rician_k, cluster_powers)
                zoa_high = raysGen._zenith_angles_of_arrival(high_spread, rician_k, cluster_powers)

                var_low = tf.reduce_mean(tf.math.reduce_std(zoa_low, axis=3))
                var_high = tf.reduce_mean(tf.math.reduce_std(zoa_high, axis=3))

                self.assertGreater(var_high, var_low)


if __name__ == '__main__':
    unittest.main()
