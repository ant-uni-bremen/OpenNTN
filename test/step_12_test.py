# This file tests the implementation of step 12, the application of the path loss and shadow fading
# to the path coefficients.

import unittest
import numpy as np
import tensorflow as tf
import sionna
from sionna.phy import config
from sionna.phy.channel.tr38811 import utils
from sionna.phy.channel.tr38811 import (
    Antenna, AntennaArray, PanelArray,
    ChannelCoefficientsGenerator,
    DenseUrban, SubUrban, Urban, Rural
)

class Test_PathLossAndShadowFading(unittest.TestCase):
    """Test the application of path loss and shadow fading on path coefficients (Step 12)."""

    @classmethod
    def setUpClass(cls):
        # Constants
        cls.BATCH_SIZE = 10
        cls.CARRIER_FREQUENCY = 2.0e9
        cls.H_UT = 1.5
        cls.H_BS = 600000.0
        cls.NB_BS = 1
        cls.NB_UT = 1
        cls.NUM_SAMPLES = 32
        cls.SAMPLING_FREQUENCY = 20e6
        cls.MAX_ERR = 1e-2
        cls.ELEVATION_ANGLE = 30.0

        cls.tx_array = Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=cls.CARRIER_FREQUENCY
        )
        cls.rx_array = Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=cls.CARRIER_FREQUENCY
        )

        cls.ccg = ChannelCoefficientsGenerator(
            cls.CARRIER_FREQUENCY,
            tx_array=cls.tx_array,
            rx_array=cls.rx_array,
            subclustering=True
        )

        cls.scenario_classes = {
            "sur": SubUrban,
            "urb": Urban,
            "dur": DenseUrban,
            "rur": Rural
        }

    def setUpScenario(self, scenario_key):
        """Helper to create and configure a channel model for a given scenario."""
        ScenarioClass = self.scenario_classes[scenario_key]

        channel_model = ScenarioClass(
            carrier_frequency=self.CARRIER_FREQUENCY,
            ut_array=self.rx_array,
            bs_array=self.tx_array,
            direction="downlink",
            elevation_angle=self.ELEVATION_ANGLE
        )

        topology = utils.gen_single_sector_topology(
            batch_size=self.BATCH_SIZE,
            num_ut=self.NB_UT,
            scenario=scenario_key,
            bs_height=self.H_BS
        )
        channel_model.set_topology(*topology)

        ray_sampler = channel_model._ray_sampler
        lsp = channel_model._lsp
        sf = lsp.sf
        rays = ray_sampler(lsp)

        topology_struct = sionna.phy.channel.tr38811.Topology(
            velocities=channel_model._scenario.ut_velocities,
            moving_end="tx",
            los_aoa=channel_model._scenario.los_aoa,
            los_aod=channel_model._scenario.los_aod,
            los_zoa=channel_model._scenario.los_zoa,
            los_zod=channel_model._scenario.los_zod,
            los=channel_model._scenario.los,
            distance_3d=channel_model._scenario.distance_3d,
            tx_orientations=channel_model._scenario.ut_orientations,
            rx_orientations=channel_model._scenario.bs_orientations,
            bs_height=channel_model._scenario._bs_loc[:, :, 2][0],
            elevation_angle=channel_model._scenario.elevation_angle,
            doppler_enabled=channel_model._scenario.doppler_enabled
        )

        c_ds = 1.6e-9
        h, delays, phi, sample_times = self.ccg(
            self.NUM_SAMPLES,
            self.SAMPLING_FREQUENCY,
            lsp.k_factor,
            rays,
            topology_struct,
            c_ds,
            debug=True
        )

        return channel_model, lsp, sf, h

    @staticmethod
    def max_rel_err(ref, val):
        """Compute the maximum relative error between reference and estimated values."""
        err = np.abs(ref - val)
        rel_err = np.where(np.abs(ref) > 0.0, np.divide(err, np.abs(ref) + 1e-6), err)
        return np.max(rel_err)

    def test_step_12_output_shape(self):
        """Verify that the output shape after applying Step 12 remains unchanged."""
        for key in self.scenario_classes:
            with self.subTest(scenario=key):
                scenario, lsp, sf, h = self.setUpScenario(key)
                h_processed = scenario._step_12(h, sf)
                self.assertEqual(h.shape, h_processed.shape)

    def test_step_12_numerical_correctness(self):
        """Check that path loss and shadow fading are applied correctly."""
        for key in self.scenario_classes:
            with self.subTest(scenario=key):
                scenario, lsp, sf, h = self.setUpScenario(key)

                if scenario._scenario.pathloss_enabled:
                    pl_db = scenario._lsp_sampler.sample_pathloss()
                    if scenario._scenario._direction == "uplink":
                        pl_db = tf.transpose(pl_db, [0, 2, 1])
                else:
                    pl_db = tf.constant(0.0, dtype=tf.float32)

                sf_active = sf if scenario._scenario.shadow_fading_enabled else tf.ones_like(sf)
                gain = tf.math.pow(10.0, -pl_db / 20.0) * tf.sqrt(sf_active)
                gain = tf.reshape(
                    gain,
                    tf.concat(
                        [tf.shape(gain), tf.ones([tf.rank(h) - tf.rank(gain)], tf.int32)],
                        0,
                    ),
                )

                expected_h = h * tf.complex(gain, 0.0)
                h_processed = scenario._step_12(h, sf)

                rel_err = self.max_rel_err(expected_h.numpy(), h_processed.numpy())
                self.assertTrue(
                    rel_err < self.MAX_ERR
                )

if __name__ == "__main__":
    unittest.main()
