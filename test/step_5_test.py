# This file tests the implementation of step 5, the cluster delay generation. To do this, the ideal
# values for all calculations are done and the average calculation is compared to it. As step 4 already
# tests the correct creation of the LSPs Delay Spread (DS) and the Rician K Factor (K), we assume these
# to be correct here.
# Step 5 has no easily measurable output, so that a mockup 

from sionna.phy.channel.tr38811 import utils   # The code to test
import unittest   # The test framework
from sionna.phy.channel.tr38811 import Antenna, AntennaArray, DenseUrban, SubUrban, Urban, CDL
import numpy as np
import tensorflow as tf
import math
from sionna.phy.utils import log10



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



class Test_URB(unittest.TestCase):

    def setUp(self):
        self.elevation_angles = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
        self.batch_size = 100
        self.num_ut = 100
        self.bs_height = 600000.0

    def run_test(self, direction, carrier_frequency, rTau_los=2.5, rTau_nlos=2.3):

        scenario = "urb"
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        # Iterate over each elevation angle
        for elevation_angle in self.elevation_angles:
            channel_model = Urban(
                carrier_frequency=carrier_frequency,
                ut_array=ut_array,
                bs_array=bs_array,
                direction=direction,
                elevation_angle=elevation_angle,
                enable_pathloss=True,
                enable_shadow_fading=True
            )
            
            topology = utils.gen_single_sector_topology(
                batch_size=self.batch_size, num_ut=self.num_ut, scenario=scenario, 
                elevation_angle=elevation_angle, bs_height=self.bs_height
            )
            channel_model.set_topology(*topology)

            rays_generator = channel_model._ray_sampler
            lsp = channel_model._lsp
         
            reference_delays, _ = rays_generator._cluster_delays(lsp.ds, lsp.k_factor)
            cluster_mask = rays_generator._cluster_mask  # Binary mask to apply to clusters

            # Define delay scaling based on LOS/NLOS scenario
            delay_scaling_parameter = tf.where(channel_model._scenario._los, rTau_los, rTau_nlos)
            delay_spread = lsp.ds  
            x = tf.random.uniform(
                shape=[self.batch_size, channel_model._scenario.num_bs, 
                       channel_model._scenario.num_ut, channel_model._scenario.num_clusters_max],
                minval=1e-6, maxval=1.0,
                dtype=channel_model._scenario.rdtype
            )
            delay_scaling_parameter = tf.expand_dims(delay_scaling_parameter,axis=3)
            
            delay_spread = tf.expand_dims(delay_spread, axis=3)

            # Calculate unscaled delays by applying scaling based on channel conditions
            unscaled_delays = -delay_scaling_parameter* delay_spread* tf.math.log(x)
            unscaled_delays = unscaled_delays * (1. - cluster_mask) + cluster_mask  # Apply cluster mask
            unscaled_delays -= tf.reduce_min(unscaled_delays, axis=3, keepdims=True)  # Normalize by subtracting min value
            unscaled_delays = tf.sort(unscaled_delays, axis=3)  # Sort delays for ordered processing

            # Convert Rician K-factor to dB and calculate scaling factor
            rician_k_factor_db = 10.0 * log10(lsp.k_factor)
            scaling_factor = (0.7705 - 0.0433 * rician_k_factor_db +
                              0.0002 * tf.square(rician_k_factor_db) +
                              0.000017 * tf.math.pow(rician_k_factor_db, 3))
            scaling_factor = tf.expand_dims(scaling_factor, axis=3)
            
            # Apply scaling factor for LOS conditions; NLOS delays remain unchanged
            delays = tf.where(
                tf.expand_dims(channel_model._scenario.los, axis=3), 
                unscaled_delays / scaling_factor, 
                unscaled_delays
            )

            # Validate the delays by checking mean and standard deviation against reference values
            for reference, actual, delay_type in [
                (reference_delays, delays, "reference_delays")
            ]:
                mean_diff = tf.abs(tf.reduce_mean(reference) - tf.reduce_mean(actual))
                std_diff = tf.abs(tf.math.reduce_std(reference) - tf.math.reduce_std(actual))
                # Assert that both mean and std deviation differences are within tolerance
                assert mean_diff < 1e-5, f"Mean mismatch" f"Expected mean = {tf.math.reduce_std(actual).numpy()}, Calculated mean = {tf.reduce_mean(reference).numpy()}"
                assert std_diff < 1e-5, f"Std deviation mismatch" f"Expected mean = {tf.math.reduce_std(actual).numpy()}, Calculated mean = {tf.reduce_mean(reference).numpy()}"

    def test_s_band_dl(self):
        self.run_test(direction="downlink", carrier_frequency=2.2e9)

    def test_s_band_ul(self):
        self.run_test(direction="uplink", carrier_frequency=2.0e9)

    def test_ka_band_dl(self):
        self.run_test(direction="downlink", carrier_frequency=20.0e9)

    def test_ka_band_ul(self):
        self.run_test(direction="uplink", carrier_frequency=30.0e9)



class Test_SUR(unittest.TestCase):

    def setUp(self):
        self.elevation_angles = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
        self.batch_size = 100
        self.num_ut = 100
        self.bs_height = 600000.0

    def run_test(self, direction, carrier_frequency, rTau_los=2.5, rTau_nlos=2.3):
        rTau_los_s_band = [2.20, 3.36, 3.50, 2.81, 2.39, 2.73, 2.07, 2.04, 2.04]
        rTau_nlos_s_band = [2.28, 2.33, 2.43, 2.26, 2.71, 2.10, 2.19, 2.06, 2.06]  


        scenario = "sur"
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)


        for idx, elevation_angle in enumerate(self.elevation_angles):
            if carrier_frequency == 2.2e9 or carrier_frequency == 2.0e9: 
                rTau_los = rTau_los_s_band[idx]
                rTau_nlos = rTau_nlos_s_band[idx]
            elif carrier_frequency == 30.0e9 or carrier_frequency == 20.0e9:  
                rTau_los = rTau_los
                rTau_nlos = rTau_nlos



            channel_model = SubUrban(
                carrier_frequency=carrier_frequency,
                ut_array=ut_array,
                bs_array=bs_array,
                direction=direction,
                elevation_angle=elevation_angle,
                enable_pathloss=True,
                enable_shadow_fading=True
            )
            
            topology = utils.gen_single_sector_topology(
                batch_size=self.batch_size, num_ut=self.num_ut, scenario=scenario, 
                elevation_angle=elevation_angle, bs_height=self.bs_height
            )
            channel_model.set_topology(*topology)

            rays_generator = channel_model._ray_sampler
            lsp = channel_model._lsp
         
            reference_delays, _ = rays_generator._cluster_delays(lsp.ds, lsp.k_factor)
            cluster_mask = rays_generator._cluster_mask  # Binary mask to apply to clusters

            delay_scaling_parameter = tf.where(channel_model._scenario._los, rTau_los, rTau_nlos)
            delay_spread = lsp.ds  
            x = tf.random.uniform(
                shape=[self.batch_size, channel_model._scenario.num_bs, 
                       channel_model._scenario.num_ut, channel_model._scenario.num_clusters_max],
                minval=1e-6, maxval=1.0,
                dtype=channel_model._scenario.rdtype
            )

            # Calculate unscaled delays by applying scaling based on channel conditions
            unscaled_delays = -delay_scaling_parameter[..., tf.newaxis] * delay_spread[..., tf.newaxis] * tf.math.log(x)
            unscaled_delays = unscaled_delays * (1. - cluster_mask) + cluster_mask  # Apply cluster mask
            unscaled_delays -= tf.reduce_min(unscaled_delays, axis=3, keepdims=True)  # Normalize by subtracting min value
            unscaled_delays = tf.sort(unscaled_delays, axis=3)  # Sort delays for ordered processing

            # Convert Rician K-factor to dB and calculate scaling factor
            rician_k_factor_db = 10.0 * log10(lsp.k_factor)
            scaling_factor = (0.7705 - 0.0433 * rician_k_factor_db +
                              0.0002 * tf.square(rician_k_factor_db) +
                              0.000017 * tf.math.pow(rician_k_factor_db, 3))
            scaling_factor = tf.expand_dims(scaling_factor, axis=3)
            
            # Apply scaling factor for LOS conditions; NLOS delays remain unchanged
            delays = tf.where(
                tf.expand_dims(channel_model._scenario.los, axis=3), 
                unscaled_delays / scaling_factor, 
                unscaled_delays
            )

            # Validate the delays by checking mean and standard deviation against reference values
            for reference, actual  in [
                (reference_delays, delays)
            ]:
                mean_diff = tf.abs(tf.reduce_mean(reference) - tf.reduce_mean(actual))
                std_diff = tf.abs(tf.math.reduce_std(reference) - tf.math.reduce_std(actual))
                # Assert that both mean and std deviation differences are within tolerance
                assert mean_diff < 1e-5, f"Mean mismatch" f"Expected mean = {tf.math.reduce_std(actual).numpy()}, Calculated mean = {tf.reduce_mean(reference).numpy()}"
                assert std_diff < 1e-5, f"Std deviation mismatch" f"Expected mean = {tf.math.reduce_std(actual).numpy()}, Calculated mean = {tf.reduce_mean(reference).numpy()}"

    def test_s_band_dl(self):
        self.run_test(direction="downlink", carrier_frequency=2.2e9)

    def test_s_band_ul(self):
        self.run_test(direction="uplink", carrier_frequency=2.0e9)

    def test_ka_band_dl(self):
        self.run_test(direction="downlink", carrier_frequency=20.0e9)

    def test_ka_band_ul(self):
        self.run_test(direction="uplink", carrier_frequency=30.0e9)

class Test_DUR(unittest.TestCase):

    def setUp(self):
        self.elevation_angles = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
        self.batch_size = 100
        self.num_ut = 100
        self.bs_height = 600000.0

    def run_test(self, direction, carrier_frequency, rTau_los=2.5, rTau_nlos=2.3):

        scenario = "dur"
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)


        for elevation_angle in self.elevation_angles:
            channel_model = DenseUrban(
                carrier_frequency=carrier_frequency,
                ut_array=ut_array,
                bs_array=bs_array,
                direction=direction,
                elevation_angle=elevation_angle,
                enable_pathloss=True,
                enable_shadow_fading=True
            )
            
            topology = utils.gen_single_sector_topology(
                batch_size=self.batch_size, num_ut=self.num_ut, scenario=scenario, 
                elevation_angle=elevation_angle, bs_height=self.bs_height
            )
            channel_model.set_topology(*topology)

            rays_generator = channel_model._ray_sampler
            lsp = channel_model._lsp
         
            reference_delays, _ = rays_generator._cluster_delays(lsp.ds, lsp.k_factor)
            cluster_mask = rays_generator._cluster_mask  # Binary mask to apply to clusters

            delay_scaling_parameter = tf.where(channel_model._scenario._los, rTau_los, rTau_nlos)
            delay_spread = lsp.ds  
            x = tf.random.uniform(
                shape=[self.batch_size, channel_model._scenario.num_bs, 
                       channel_model._scenario.num_ut, channel_model._scenario.num_clusters_max],
                minval=1e-6, maxval=1.0,
                dtype=channel_model._scenario.rdtype
            )

            # Calculate unscaled delays by applying scaling based on channel conditions
            unscaled_delays = -delay_scaling_parameter[..., tf.newaxis] * delay_spread[..., tf.newaxis] * tf.math.log(x)
            unscaled_delays = unscaled_delays * (1. - cluster_mask) + cluster_mask  # Apply cluster mask
            unscaled_delays -= tf.reduce_min(unscaled_delays, axis=3, keepdims=True)  # Normalize by subtracting min value
            unscaled_delays = tf.sort(unscaled_delays, axis=3)  # Sort delays for ordered processing

            # Convert Rician K-factor to dB and calculate scaling factor
            rician_k_factor_db = 10.0 * log10(lsp.k_factor)
            scaling_factor = (0.7705 - 0.0433 * rician_k_factor_db +
                              0.0002 * tf.square(rician_k_factor_db) +
                              0.000017 * tf.math.pow(rician_k_factor_db, 3))
            scaling_factor = tf.expand_dims(scaling_factor, axis=3)
            
            # Apply scaling factor for LOS conditions; NLOS delays remain unchanged
            delays = tf.where(
                tf.expand_dims(channel_model._scenario.los, axis=3), 
                unscaled_delays / scaling_factor, 
                unscaled_delays
            )

            # Validate the delays by checking mean and standard deviation against reference values
            for reference, actual  in [
                (reference_delays, delays)
            ]:
                mean_diff = tf.abs(tf.reduce_mean(reference) - tf.reduce_mean(actual))
                std_diff = tf.abs(tf.math.reduce_std(reference) - tf.math.reduce_std(actual))
                # Assert that both mean and std deviation differences are within tolerance
                assert mean_diff < 1e-5, f"Mean mismatch" f"Expected mean = {tf.math.reduce_std(actual).numpy()}, Calculated mean = {tf.reduce_mean(reference).numpy()}"
                assert std_diff < 1e-5, f"Std deviation mismatch" f"Expected mean = {tf.math.reduce_std(actual).numpy()}, Calculated mean = {tf.reduce_mean(reference).numpy()} in {elevation_angle} degrees"

    def test_s_band_dl(self):
        self.run_test(direction="downlink", carrier_frequency=2.2e9)

    def test_s_band_ul(self):
        self.run_test(direction="uplink", carrier_frequency=2.0e9)

    def test_ka_band_dl(self):
        self.run_test(direction="downlink", carrier_frequency=20.0e9)

    def test_ka_band_ul(self):
        self.run_test(direction="uplink", carrier_frequency=30.0e9)        

if __name__ == '__main__':
    unittest.main()