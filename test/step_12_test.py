# This file tests the implementation of step 12, the application of the path loss and shadow fading
# to the path coefficients. 
import tensorflow as tf
import unittest
import numpy as np
import sionna

from sionna.phy.channel.tr38811 import utils
from sionna.phy.channel.tr38811 import Antenna, AntennaArray,PanelArray,ChannelCoefficientsGenerator
from sionna.phy.channel.tr38811 import DenseUrban, SubUrban, Urban, CDL


class Step_12(unittest.TestCase):
    r"""Test the computation of channel coefficients"""

    # Batch size used to check the LSP distribution
    BATCH_SIZE = 10

    # Carrier frequency
    CARRIER_FREQUENCY = 2.0e9 # Hz

    # Maximum allowed deviation for calculation (relative error)
    MAX_ERR = 1e-2

    # # Heigh of UTs
    H_UT = 1.5

    # # Heigh of BSs
    H_BS = 600000.0

    # # Number of BS
    NB_BS =1

    # Number of UT
    NB_UT = 1

    # Number of channel time samples
    NUM_SAMPLES = 32

    # Sampling frequency
    SAMPLING_FREQUENCY = 20e6

    def setUp(self):
        batch_size = Step_12.BATCH_SIZE
        nb_ut = Step_12.NB_UT
        nb_bs = Step_12.NB_BS
        h_ut = Step_12.H_UT
        h_bs = Step_12.H_BS
        fc = Step_12.CARRIER_FREQUENCY

        
        self.tx_array = Antenna(polarization="single",
                                    polarization_type="V",
                                    antenna_pattern="38.901",
                                    carrier_frequency=fc)
        self.rx_array = Antenna(polarization='single',
                                polarization_type='V',
                                antenna_pattern='38.901',
                                carrier_frequency=fc)

        self.ccg = ChannelCoefficientsGenerator(
            fc,
            tx_array=self.tx_array,
            rx_array=self.rx_array,
            subclustering=True)

        channel_model = DenseUrban(
            carrier_frequency=fc,
            ut_array=self.rx_array,
            bs_array=self.tx_array,
            direction='downlink',
            elevation_angle=30.0)
        topology = utils.gen_single_sector_topology(
            batch_size=batch_size, num_ut=nb_ut, scenario='dur', bs_height=h_bs
        )
        channel_model.set_topology(*topology)
        self.scenario = channel_model
        

        ray_sampler = self.scenario._ray_sampler
        self.lsp = self.scenario._lsp
        self.sf = self.lsp.sf
        
        # # lsp = lsp_sampler()
        self.rays = ray_sampler(self.lsp)   
        topology = sionna.phy.channel.tr38811.Topology(velocities=channel_model._scenario.ut_velocities,
                                moving_end="tx", 
                                los_aoa=channel_model._scenario.los_aoa,
                                los_aod=channel_model._scenario.los_aod,
                                los_zoa=channel_model._scenario.los_zoa,
                                los_zod=channel_model._scenario.los_zod,
                                los=channel_model._scenario.los,
                                distance_3d=channel_model._scenario.distance_3d,
                                tx_orientations=channel_model._scenario.ut_orientations,
                                rx_orientations=channel_model._scenario.bs_orientations,
                                bs_height = channel_model._scenario._bs_loc[:,:,2][0],
                                elevation_angle = channel_model._scenario.elevation_angle,
                                doppler_enabled = channel_model._scenario.doppler_enabled
                                )
        self.topology = topology
        
        num_time_samples = Step_12.NUM_SAMPLES 
        sampling_frequency = Step_12.SAMPLING_FREQUENCY
        # c_ds = scenario.get_param("cDS")*1e-9
        c_ds = 1.6*1e-9
        h, delays, phi, sample_times = self.ccg(num_time_samples,
            sampling_frequency, self.lsp.k_factor, self.rays, topology, c_ds,
            debug=True)
        self.phi = phi.numpy()
        self.sample_times = sample_times.numpy()
        self.c_ds = c_ds
        self.h = h 

    def max_rel_err(self, r, x):
        """Compute the maximum relative error, ``r`` being the reference value,
        ``x`` an esimate of ``r``."""
        err = np.abs(r-x)
        rel_err = np.where(np.abs(r) > 0.0, np.divide(err,np.abs(r)+1e-6), err)
        return np.max(rel_err)

    def test_step_12_output_shape(self):
        """Test that the shape of h"""
        h_processed = self.scenario._step_12(self.h, self.sf)
        self.assertEqual(self.h.shape, h_processed.shape)
    
    def test_step_12_numerical_correctness(self):
        """Verify that path loss and shadow fading are applied correctly"""
        if self.scenario._scenario.pathloss_enabled:
            pl_db = self.scenario._lsp_sampler.sample_pathloss()
            if self.scenario._scenario._direction == 'uplink':
                pl_db = tf.transpose(pl_db, [0,2,1])
        else:
            pl_db = tf.constant(0.0, dtype=tf.float32)
        
        sf = self.sf if self.scenario._scenario.shadow_fading_enabled else tf.ones_like(self.sf)
        gain = tf.math.pow(10.0, -pl_db/20.0) * tf.sqrt(sf)
        gain = tf.reshape(gain, tf.concat([tf.shape(gain), tf.ones([tf.rank(self.h)-tf.rank(gain)], tf.int32)], 0))
        expected_h = self.h * tf.complex(gain, 0.0)
        
        h_processed = self.scenario._step_12(self.h, self.sf)
        rel_err = self.max_rel_err(expected_h.numpy(), h_processed.numpy())
        self.assertTrue(rel_err < Step_12.MAX_ERR) 
        


if __name__ == "__main__":
    unittest.main()
