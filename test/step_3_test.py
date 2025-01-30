# The testing of the Link Budget is covered in the file Link Budget Test and is based on 3GPP TR38.821. This
# is is an expansion, testing the proper generation in differnt settings. Thus only the Clutter Loss is tested,
# as well as the Shadow Fading, as it is generated from a distribution
"""
The basic Pathloss (PLb) is caluclated as the sum of the Free Space Pathloss (FSPL), Shadow Fading (SF) and 
Clutter Loss (CL). SF is a distribution and the other values are deterministic. As only the FSPL and PLb are
given in the interface, we calcualted SF = PLb - FSPL - CL and take the mean and variance of this distribution,
to verify both CL and SF.
"""

from sionna.channel.tr38811 import utils   # The code to test
import unittest   # The test framework
from sionna.channel.tr38811 import Antenna, AntennaArray, DenseUrban, SubUrban, Urban, CDL
import numpy as np
import tensorflow as tf
import math

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
# For more insight, see 3GPP Table 6.6.2-2: Shadow fading and clutter loss for urban scenario
    def test_s_band_10_degrees_dl(self):
        elevation_angle = 10.0

        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 34.3
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_10_degrees_ul(self):
        elevation_angle = 10.0

        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 34.3
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_20_degrees_dl(self):
        elevation_angle = 20.0

        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 30.9
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_20_degrees_ul(self):
        elevation_angle = 20.0

        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 30.9
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_30_degrees_dl(self):
        elevation_angle = 30.0

        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 29.0
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_30_degrees_ul(self):
        elevation_angle = 30.0

        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 29.0
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_40_degrees_dl(self):
        elevation_angle = 40.0

        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 27.7
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_40_degrees_ul(self):
        elevation_angle = 40.0

        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 27.7
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_50_degrees_dl(self):
        elevation_angle = 50.0

        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 26.8
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_50_degrees_ul(self):
        elevation_angle = 50.0

        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 26.8
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_60_degrees_dl(self):
        elevation_angle = 60.0

        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 26.2
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_60_degrees_ul(self):
        elevation_angle = 60.0

        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 26.2
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_70_degrees_dl(self):
        elevation_angle = 70.0

        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 25.8
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_70_degrees_ul(self):
        elevation_angle = 70.0

        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 25.8
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_80_degrees_dl(self):
        elevation_angle = 80.0

        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 25.5
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_80_degrees_ul(self):
        elevation_angle = 80.0

        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 25.5
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_90_degrees_dl(self):
        elevation_angle = 90.0

        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        #Increased number of samples, as NLOS probability is otherwise too low to get enough samples
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 25.5
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=1.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=1.5)

    def test_s_band_90_degrees_ul(self):
        elevation_angle = 90.0

        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 25.5
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=1.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=1.5)

    def test_ka_band_10_degrees_dl(self):
        elevation_angle = 10.0

        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 44.3
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_10_degrees_ul(self):
        elevation_angle = 10.0

        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 44.3
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_20_degrees_dl(self):
        elevation_angle = 20.0

        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 39.9
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_20_degrees_ul(self):
        elevation_angle = 20.0

        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 39.9
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_30_degrees_dl(self):
        elevation_angle = 30.0

        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 37.5
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_30_degrees_ul(self):
        elevation_angle = 30.0

        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 37.5
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_40_degrees_dl(self):
        elevation_angle = 40.0

        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 35.8
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_40_degrees_ul(self):
        elevation_angle = 40.0

        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 35.8
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_50_degrees_dl(self):
        elevation_angle = 50.0

        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 34.6
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_50_degrees_ul(self):
        elevation_angle = 50.0

        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 34.6
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_60_degrees_dl(self):
        elevation_angle = 60.0

        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 33.8
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_60_degrees_ul(self):
        elevation_angle = 60.0

        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 33.8
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_70_degrees_dl(self):
        elevation_angle = 70.0

        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 33.3
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_70_degrees_ul(self):
        elevation_angle = 70.0

        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 33.3
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_80_degrees_dl(self):
        elevation_angle = 80.0

        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 33.0
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_80_degrees_ul(self):
        elevation_angle = 80.0

        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 33.0
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_90_degrees_dl(self):
        elevation_angle = 90.0

        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        #Increased number of samples, as NLOS probability is otherwise too low to get enough samples
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 32.9
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=1.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=1.5)

    def test_ka_band_90_degrees_ul(self):
        elevation_angle = 90.0

        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 32.9
        sf_los_sigma = 4.0
        sf_nlos_sigma = 6.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=1.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=1.5)

class Test_DUR(unittest.TestCase):
# For more instight, see 3GPP Table 6.6.2-1: Shadow fading and clutter loss for dense urban scenario
    def test_s_band_10_degrees_dl(self):
        elevation_angle = 10.0

        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 34.3
        sf_los_sigma = 3.5
        sf_nlos_sigma = 15.5

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_10_degrees_ul(self):
        elevation_angle = 10.0

        direction = "uplink"
        scenario = "dur"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 34.3
        sf_los_sigma = 3.5
        sf_nlos_sigma = 15.5

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_20_degrees_dl(self):
        elevation_angle = 20.0

        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 30.9
        sf_los_sigma = 3.4
        sf_nlos_sigma = 13.9

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_20_degrees_ul(self):
        elevation_angle = 20.0

        direction = "uplink"
        scenario = "dur"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 30.9
        sf_los_sigma = 3.4
        sf_nlos_sigma = 13.9

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_30_degrees_dl(self):
        elevation_angle = 30.0

        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 29.0
        sf_los_sigma = 2.9
        sf_nlos_sigma = 12.4

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_30_degrees_ul(self):
        elevation_angle = 30.0

        direction = "uplink"
        scenario = "dur"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 29.0
        sf_los_sigma = 2.9
        sf_nlos_sigma = 12.4

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_40_degrees_dl(self):
        elevation_angle = 40.0

        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 27.7
        sf_los_sigma = 3.0
        sf_nlos_sigma = 11.7

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_40_degrees_ul(self):
        elevation_angle = 40.0

        direction = "uplink"
        scenario = "dur"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 27.7
        sf_los_sigma = 3.0
        sf_nlos_sigma = 11.7

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_50_degrees_dl(self):
        elevation_angle = 50.0

        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 26.8
        sf_los_sigma = 3.1
        sf_nlos_sigma = 10.6

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_50_degrees_ul(self):
        elevation_angle = 50.0

        direction = "uplink"
        scenario = "dur"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 26.8
        sf_los_sigma = 3.1
        sf_nlos_sigma = 10.6

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_60_degrees_dl(self):
        elevation_angle = 60.0

        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 26.2
        sf_los_sigma = 2.7
        sf_nlos_sigma = 10.5

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_60_degrees_ul(self):
        elevation_angle = 60.0

        direction = "uplink"
        scenario = "dur"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 26.2
        sf_los_sigma = 2.7
        sf_nlos_sigma = 10.5

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_70_degrees_dl(self):
        elevation_angle = 70.0

        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 25.8
        sf_los_sigma = 2.5
        sf_nlos_sigma = 10.1

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_70_degrees_ul(self):
        elevation_angle = 70.0

        direction = "uplink"
        scenario = "dur"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 25.8
        sf_los_sigma = 2.5
        sf_nlos_sigma = 10.1

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_80_degrees_dl(self):
        elevation_angle = 80.0

        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 25.5
        sf_los_sigma = 2.3
        sf_nlos_sigma = 9.2

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_80_degrees_ul(self):
        elevation_angle = 80.0

        direction = "uplink"
        scenario = "dur"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 25.5
        sf_los_sigma = 2.3
        sf_nlos_sigma = 9.2

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_90_degrees_dl(self):
        elevation_angle = 90.0

        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        #Increased number of samples, as NLOS probability is otherwise too low to get enough samples
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 25.5
        sf_los_sigma = 1.2
        sf_nlos_sigma = 9.2

        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=1.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=1.5)

    def test_s_band_90_degrees_ul(self):
        elevation_angle = 90.0

        direction = "uplink"
        scenario = "dur"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 25.5
        sf_los_sigma = 1.2
        sf_nlos_sigma = 9.2

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=1.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=1.5)

    def test_ka_band_10_degrees_dl(self):
        elevation_angle = 10.0

        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 44.3
        sf_los_sigma = 2.9
        sf_nlos_sigma = 17.1

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_10_degrees_ul(self):
        elevation_angle = 10.0

        direction = "uplink"
        scenario = "dur"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 44.3
        sf_los_sigma = 2.9
        sf_nlos_sigma = 17.1

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_20_degrees_dl(self):
        elevation_angle = 20.0

        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 39.9
        sf_los_sigma = 2.4
        sf_nlos_sigma = 17.1

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_20_degrees_ul(self):
        elevation_angle = 20.0

        direction = "uplink"
        scenario = "dur"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 39.9
        sf_los_sigma = 2.4
        sf_nlos_sigma = 17.1

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_30_degrees_dl(self):
        elevation_angle = 30.0

        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 37.5
        sf_los_sigma = 2.7
        sf_nlos_sigma = 15.6

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_30_degrees_ul(self):
        elevation_angle = 30.0

        direction = "uplink"
        scenario = "dur"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 37.5
        sf_los_sigma = 2.7
        sf_nlos_sigma = 15.6

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_40_degrees_dl(self):
        elevation_angle = 40.0

        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 35.8
        sf_los_sigma = 2.4
        sf_nlos_sigma = 14.6

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_40_degrees_ul(self):
        elevation_angle = 40.0

        direction = "uplink"
        scenario = "dur"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 35.8
        sf_los_sigma = 2.4
        sf_nlos_sigma = 14.6

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_50_degrees_dl(self):
        elevation_angle = 50.0

        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 34.6
        sf_los_sigma = 2.4
        sf_nlos_sigma = 14.2

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_50_degrees_ul(self):
        elevation_angle = 50.0

        direction = "uplink"
        scenario = "dur"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 34.6
        sf_los_sigma = 2.4
        sf_nlos_sigma = 14.2

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_60_degrees_dl(self):
        elevation_angle = 60.0

        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 33.8
        sf_los_sigma = 2.7
        sf_nlos_sigma = 12.6

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_60_degrees_ul(self):
        elevation_angle = 60.0

        direction = "uplink"
        scenario = "dur"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 33.8
        sf_los_sigma = 2.7
        sf_nlos_sigma = 12.6

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_70_degrees_dl(self):
        elevation_angle = 70.0

        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 33.3
        sf_los_sigma = 2.6
        sf_nlos_sigma = 12.1

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_70_degrees_ul(self):
        elevation_angle = 70.0

        direction = "uplink"
        scenario = "dur"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 33.3
        sf_los_sigma = 2.6
        sf_nlos_sigma = 12.1

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_80_degrees_dl(self):
        elevation_angle = 80.0

        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 33.0
        sf_los_sigma = 2.8
        sf_nlos_sigma = 12.3

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_80_degrees_ul(self):
        elevation_angle = 80.0

        direction = "uplink"
        scenario = "dur"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 33.0
        sf_los_sigma = 2.8
        sf_nlos_sigma = 12.3

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_90_degrees_dl(self):
        elevation_angle = 90.0

        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        #Increased number of samples, as NLOS probability is otherwise too low to get enough samples
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 32.9
        sf_los_sigma = 0.6
        sf_nlos_sigma = 12.3

        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=1.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=1.5)

    def test_ka_band_90_degrees_ul(self):
        elevation_angle = 90.0

        direction = "uplink"
        scenario = "dur"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 32.9
        sf_los_sigma = 0.6
        sf_nlos_sigma = 12.3

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=1.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=1.5)

class Test_SUR(unittest.TestCase):
# For more instight, see 3GPP Table 6.6.2-3: Shadow fading and clutter loss for suburban and rural scenarios
# Due to the high LOS probability, the number of samples is increased in all cases in this class
    def test_s_band_10_degrees_dl(self):
        elevation_angle = 10.0

        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 19.52
        sf_los_sigma = 1.79
        sf_nlos_sigma = 8.93

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_10_degrees_ul(self):
        elevation_angle = 10.0

        direction = "uplink"
        scenario = "sur"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 19.52
        sf_los_sigma = 1.79
        sf_nlos_sigma = 8.93

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_20_degrees_dl(self):
        elevation_angle = 20.0

        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 18.17
        sf_los_sigma = 1.14
        sf_nlos_sigma = 9.08

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_20_degrees_ul(self):
        elevation_angle = 20.0

        direction = "uplink"
        scenario = "sur"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 18.17
        sf_los_sigma = 1.14
        sf_nlos_sigma = 9.08

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_30_degrees_dl(self):
        elevation_angle = 30.0

        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 18.42
        sf_los_sigma = 1.14
        sf_nlos_sigma = 8.78

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_30_degrees_ul(self):
        elevation_angle = 30.0

        direction = "uplink"
        scenario = "sur"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 18.42
        sf_los_sigma = 1.14
        sf_nlos_sigma = 8.78

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_40_degrees_dl(self):
        elevation_angle = 40.0

        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 18.28
        sf_los_sigma = 0.92
        sf_nlos_sigma = 10.25

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_40_degrees_ul(self):
        elevation_angle = 40.0

        direction = "uplink"
        scenario = "sur"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 18.28
        sf_los_sigma = 0.92
        sf_nlos_sigma = 10.25

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_50_degrees_dl(self):
        elevation_angle = 50.0

        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 18.63
        sf_los_sigma = 1.42
        sf_nlos_sigma = 10.56

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_50_degrees_ul(self):
        elevation_angle = 50.0

        direction = "uplink"
        scenario = "sur"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 18.63
        sf_los_sigma = 1.42
        sf_nlos_sigma = 10.56

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_60_degrees_dl(self):
        elevation_angle = 60.0

        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 17.68
        sf_los_sigma = 1.56
        sf_nlos_sigma = 10.74

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_60_degrees_ul(self):
        elevation_angle = 60.0

        direction = "uplink"
        scenario = "sur"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 17.68
        sf_los_sigma = 1.56
        sf_nlos_sigma = 10.74

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_70_degrees_dl(self):
        elevation_angle = 70.0

        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 16.5
        sf_los_sigma = 0.85
        sf_nlos_sigma = 10.17

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_70_degrees_ul(self):
        elevation_angle = 70.0

        direction = "uplink"
        scenario = "sur"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 16.5
        sf_los_sigma = 0.85
        sf_nlos_sigma = 10.17

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_80_degrees_dl(self):
        elevation_angle = 80.0

        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 16.3
        sf_los_sigma = 0.72
        sf_nlos_sigma = 11.52

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_80_degrees_ul(self):
        elevation_angle = 80.0

        direction = "uplink"
        scenario = "sur"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 16.3
        sf_los_sigma = 0.72
        sf_nlos_sigma = 11.52

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_s_band_90_degrees_dl(self):
        elevation_angle = 90.0

        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        #Increased number of samples, as NLOS probability is otherwise too low to get enough samples
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=200, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 16.3
        sf_los_sigma = 0.72
        sf_nlos_sigma = 11.52

        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=1.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=1.5)

    def test_s_band_90_degrees_ul(self):
        elevation_angle = 90.0

        direction = "uplink"
        scenario = "sur"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 16.3
        sf_los_sigma = 0.72
        sf_nlos_sigma = 11.52

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=1.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=1.5)

    def test_ka_band_10_degrees_dl(self):
        elevation_angle = 10.0

        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 29.5
        sf_los_sigma = 1.9
        sf_nlos_sigma = 10.7

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_10_degrees_ul(self):
        elevation_angle = 10.0

        direction = "uplink"
        scenario = "sur"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 29.5
        sf_los_sigma = 1.9
        sf_nlos_sigma = 10.7

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_20_degrees_dl(self):
        elevation_angle = 20.0

        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 24.6
        sf_los_sigma = 1.6
        sf_nlos_sigma = 10.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_20_degrees_ul(self):
        elevation_angle = 20.0

        direction = "uplink"
        scenario = "sur"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 24.6
        sf_los_sigma = 1.6
        sf_nlos_sigma = 10.0

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_30_degrees_dl(self):
        elevation_angle = 30.0

        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 21.9
        sf_los_sigma = 1.9
        sf_nlos_sigma = 11.2

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_30_degrees_ul(self):
        elevation_angle = 30.0

        direction = "uplink"
        scenario = "sur"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 21.9
        sf_los_sigma = 1.9
        sf_nlos_sigma = 11.2

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_40_degrees_dl(self):
        elevation_angle = 40.0

        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 20.0
        sf_los_sigma = 2.3
        sf_nlos_sigma = 11.6

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_40_degrees_ul(self):
        elevation_angle = 40.0

        direction = "uplink"
        scenario = "sur"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 20.0
        sf_los_sigma = 2.3
        sf_nlos_sigma = 11.6

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_50_degrees_dl(self):
        elevation_angle = 50.0

        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 18.7
        sf_los_sigma = 2.7
        sf_nlos_sigma = 11.8

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_50_degrees_ul(self):
        elevation_angle = 50.0

        direction = "uplink"
        scenario = "sur"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 18.7
        sf_los_sigma = 2.7
        sf_nlos_sigma = 11.8

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_60_degrees_dl(self):
        elevation_angle = 60.0

        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 17.8
        sf_los_sigma = 3.1
        sf_nlos_sigma = 10.8

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_60_degrees_ul(self):
        elevation_angle = 60.0

        direction = "uplink"
        scenario = "sur"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        #Increased number of samples due to high number of fails otherwise
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 17.8
        sf_los_sigma = 3.1
        sf_nlos_sigma = 10.8

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_70_degrees_dl(self):
        elevation_angle = 70.0

        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 17.2
        sf_los_sigma = 3.0
        sf_nlos_sigma = 10.8

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_70_degrees_ul(self):
        elevation_angle = 70.0

        direction = "uplink"
        scenario = "sur"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 17.2
        sf_los_sigma = 3.0
        sf_nlos_sigma = 10.8

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_80_degrees_dl(self):
        elevation_angle = 80.0

        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 16.9
        sf_los_sigma = 3.6
        sf_nlos_sigma = 10.8

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_80_degrees_ul(self):
        elevation_angle = 80.0

        direction = "uplink"
        scenario = "sur"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 16.9
        sf_los_sigma = 3.6
        sf_nlos_sigma = 10.8

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=0.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=0.5)

    def test_ka_band_90_degrees_dl(self):
        elevation_angle = 90.0

        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        #Increased number of samples, as NLOS probability is otherwise too low to get enough samples
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 16.8
        sf_los_sigma = 0.4
        sf_nlos_sigma = 10.8

        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=1.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=1.5)

    def test_ka_band_90_degrees_ul(self):
        elevation_angle = 90.0

        direction = "uplink"
        scenario = "sur"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=1000, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)

        #Subtract the FSPL to isolate CL + SF
        loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss
        
        loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
        loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

        #Values from table
        los_cl = 0.0
        nlos_cl = 16.8
        sf_los_sigma = 0.4
        sf_nlos_sigma = 10.8

        #Toleance of 0.5 for 100 samples, which should catch incorrect behavior realibly enough, but tolerate variation in 100000 samples
        #with split in NLOS and los cases
        #Average Loss is CL, as SF is zero centered Gaussian
        assert math.isclose(tf.math.reduce_mean(loss_los), los_cl, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_mean(loss_nlos), nlos_cl, abs_tol=1.5)
        #Standard_deviation is CL std, as CL is a single value
        assert math.isclose(tf.math.reduce_std(loss_los), sf_los_sigma, abs_tol=0.5)
        #Very low NLOS probability -> Increased tolerance
        assert math.isclose(tf.math.reduce_std(loss_nlos), sf_nlos_sigma, abs_tol=1.5)


if __name__ == '__main__':
    unittest.main()