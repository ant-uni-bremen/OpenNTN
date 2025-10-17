# This file tests the implementation of step 6, the cluster power generation. To do this, the ideal
# values for all calculations are done and the average calculation is compared to it. As step 4 already
# tests the correct creation of the LSPs Delay Spread (DS) and the Rician K Factor (K), we assume these
# to be correct here.
# Step 6 has no easily measurable output, so that a mockup 

from sionna.phy.channel.tr38811 import utils   # The code to test
import unittest   # The test framework
from sionna.phy.channel.tr38811 import Antenna, AntennaArray, DenseUrban, SubUrban, Urban, CDL, Rural
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
        cls.carrier_frequency = 2.0e9
        cls.elevation_angle = 10.0
        cls.batch_size = 100
        cls.ut_array = create_ut_ant(cls.carrier_frequency)
        cls.bs_array = create_bs_ant(cls.carrier_frequency)
        cls.scenario_classes = {
            "sur": SubUrban,
            "urb": Urban,
            "dur": DenseUrban,
            "rur": Rural
        }

    def setUpScenario(self, scenario_key):
        ScenarioClass = self.scenario_classes[scenario_key]
        model = ScenarioClass(
            carrier_frequency=self.carrier_frequency,
            ut_array=self.ut_array,
            bs_array=self.bs_array,
            direction="downlink",
            elevation_angle=self.elevation_angle,
            enable_pathloss=True,
            enable_shadow_fading=True
        )
        topology = utils.gen_single_sector_topology(
            batch_size=self.batch_size, num_ut=100, scenario=scenario_key,
            elevation_angle=self.elevation_angle, bs_height=600000.0
        )
        model.set_topology(*topology)
        return model

    def test_cluster_elimination(self):
        """Check if any cluster has -25 dB power compared to the maximum cluster power"""
        threshold_db = -25
        threshold_power = 10 ** (threshold_db / 10)
        epsilon = 1e-12

        for key in self.scenario_classes:
            with self.subTest(scenario=key):
                model = self.setUpScenario(key)
                rg = model._ray_sampler
                lsp = model._lsp
                delays, unscaled_delays = rg._cluster_delays(lsp.ds, lsp.k_factor)
                cluster_powers, _ = rg._cluster_powers(lsp.ds, lsp.k_factor, unscaled_delays)
                cluster_powers = tf.maximum(cluster_powers, epsilon)
                max_power = tf.reduce_max(cluster_powers, axis=-1, keepdims=True)
                diff = max_power - cluster_powers
                self.assertTrue(tf.reduce_all(tf.reduce_mean(diff, axis=-1) >= threshold_power).numpy())

    def test_specular_component_los(self):
        """Verify LoS component increment equals K/(1+K) in first cluster power"""
        for key in self.scenario_classes:
            with self.subTest(scenario=key):
                model = self.setUpScenario(key)
                rg = model._ray_sampler
                lsp = model._lsp
                _, unscaled_delays = rg._cluster_delays(lsp.ds, lsp.k_factor)
                powers, powers_for_angles_gen = rg._cluster_powers(
                    lsp.ds, lsp.k_factor, unscaled_delays
                )

                los_mask = tf.expand_dims(model._scenario.los, axis=3)
                ric_fac = tf.expand_dims(lsp.k_factor, axis=3)
                p_nlos_scaling = 1.0 / (ric_fac + 1.0)
                p1_los = ric_fac * p_nlos_scaling

                orig_first = powers[..., :1]
                adj_first = powers_for_angles_gen[..., :1]
                increment = adj_first - p_nlos_scaling * orig_first

                increment_los = tf.boolean_mask(increment, tf.squeeze(los_mask, axis=3))
                p1_los_los = tf.boolean_mask(p1_los, tf.squeeze(los_mask, axis=3))
                increment_los_np = np.nan_to_num(increment_los.numpy(), nan=0.0)
                p1_los_los_np = np.nan_to_num(p1_los_los.numpy(), nan=0.0)
                

                self.assertTrue(np.allclose(increment_los_np, p1_los_los_np, rtol=1e-5, atol=1e-7))

    def test_rays_equal_power(self):
        """Verify that rays within a cluster have equal power (Pn / M)"""
        for key in self.scenario_classes:
            with self.subTest(scenario=key):
                model = self.setUpScenario(key)
                rg = model._ray_sampler
                lsp = model._lsp
                _, unscaled_delays = rg._cluster_delays(lsp.ds, lsp.k_factor)
                cluster_powers, _ = rg._cluster_powers(lsp.ds, lsp.k_factor, unscaled_delays)

                M = model._scenario.rays_per_cluster
                expected_ray_powers = cluster_powers[..., tf.newaxis] / tf.cast(M, cluster_powers.dtype)
                #Checking if the variance = 0
                ray_var = tf.math.reduce_variance(expected_ray_powers, axis=-1)
                
                self.assertTrue(tf.reduce_all(ray_var < 1e-9).numpy())

if __name__ == '__main__':
    unittest.main()