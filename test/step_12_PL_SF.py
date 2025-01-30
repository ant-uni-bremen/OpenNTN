# This file tests the implementation of step 12, the application of the path loss and shadow fading
# to the path coefficients. 
import tensorflow as tf
import unittest
import numpy as np
import sionna
from sionna import config
from sionna.channel.tr38811 import utils
from sionna.channel.tr38811 import Antenna, AntennaArray,PanelArray,ChannelCoefficientsGenerator
from sionna.channel.tr38811 import DenseUrban, SubUrban, Urban, CDL


class Step_12(unittest.TestCase):
    r"""Test the computation of channel coefficients"""

    # Batch size used to check the LSP distribution
    BATCH_SIZE = 100

    # Carrier frequency
    CARRIER_FREQUENCY = 2.0e9 # Hz

    # Maximum allowed deviation for calculation (relative error)
    MAX_ERR = 1e-2

    # # Heigh of UTs
    H_UT = 1.5

    # # Heigh of BSs
    H_BS = 600000.0

    # # Number of BS
    NB_BS = 1

    # Number of UT
    NB_UT = 100

    # Number of channel time samples
    NUM_SAMPLES = 64

    # Sampling frequency
    SAMPLING_FREQUENCY = 20e6

    def setUp(self):
        batch_size = Step_12.BATCH_SIZE
        nb_ut = Step_12.NB_UT
        nb_bs = Step_12.NB_BS
        h_ut = Step_12.H_UT
        h_bs = Step_12.H_BS
        fc = Step_12.CARRIER_FREQUENCY
        los = tf.zeros(shape=[batch_size, nb_bs, nb_ut], dtype=tf.bool)
        distance_3d = tf.random.uniform([batch_size, nb_ut, nb_ut], 0.0, 2000.0, dtype=tf.float32)

        self.tx_array = PanelArray(num_rows_per_panel=1,
                                    num_cols_per_panel=1,
                                    polarization="single",
                                    polarization_type="V",
                                    antenna_pattern="38.901",
                                    carrier_frequency=fc)
        self.rx_array = PanelArray(num_rows_per_panel=1,
                                num_cols_per_panel=1,
                                polarization='dual',
                                polarization_type='VH',
                                antenna_pattern='38.901',
                                carrier_frequency=fc)

        self.ccg = ChannelCoefficientsGenerator(
            fc,
            tx_array=self.tx_array,
            rx_array=self.rx_array,
            subclustering=True,
            dtype=tf.complex64)

        rx_orientations = config.tf_rng.uniform([batch_size, nb_ut, 3], 0.0,
                                            2*np.pi, dtype=tf.float32)
        tx_orientations = config.tf_rng.uniform([batch_size, nb_bs, 3], 0.0,
                                            2*np.pi, dtype=tf.float32)
        ut_velocities = config.tf_rng.uniform([batch_size, nb_ut, 3], 0.0, 5.0,
                                                dtype=tf.float32)

        channel_model = SubUrban(
            carrier_frequency=fc,
            ut_array=self.rx_array,
            bs_array=self.tx_array,
            direction='downlink',
            elevation_angle=30.0)
        topology = utils.gen_single_sector_topology(
            batch_size=batch_size, num_ut=nb_ut, scenario='sur', bs_height=h_bs
        )
        channel_model.set_topology(*topology)
        self.scenario = channel_model

        ray_sampler = self.scenario._ray_sampler
        lsp = self.scenario._lsp
        self.rays = ray_sampler(lsp)
        self.los_aoa = self.rays.aoa[..., 0]  
        self.los_aod = self.rays.aod[..., 0]  
        self.los_zoa = self.rays.zoa[..., 0]  
        self.los_zod = self.rays.zod[..., 0]     
        self.lsp = lsp
        self.sf = self.lsp.sf
        los = tf.boolean_mask(los, channel_model._scenario.los)
        topology = sionna.channel.tr38811.Topology(
            velocities=ut_velocities,
            moving_end='rx',
            los_aoa= self.los_aoa,
            los_aod= self.los_aod,
            los_zoa= self.los_zoa,
            los_zod= self.los_zod,
            los= los,
            distance_3d=distance_3d,
            tx_orientations=tx_orientations,
            bs_height= h_bs ,
            elevation_angle = 30,
            doppler_enabled = False,
            rx_orientations=rx_orientations)
        self.topology = topology
        num_time_samples = Step_12.NUM_SAMPLES
        sampling_frequency = Step_12.SAMPLING_FREQUENCY
        c_ds = 1.6*1e-9
        h, delays, phi, sample_times = self.ccg(num_time_samples,
            sampling_frequency, lsp.k_factor, self.rays, topology, c_ds,
            debug=True)
        self.h = h

    def test_step_12(self):
        h_processed = self.scenario._step_12(self.h,self.sf)


if __name__ == "__main__":
    unittest.main()
