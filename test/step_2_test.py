# This file tests the generation of the LOS states according to Step 2 of 3GPP TR38.901 7.5
# using the parameters of 3GPP TR38.811 Table 6.6.1-1 LOS probability

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

    def test_urb_los_probabilities(self):        
        
        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 2.17e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        elevation_angle = 10.0
        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 24.6, abs_tol=2)

        elevation_angle = 20.0
        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 38.6, abs_tol=2)

        elevation_angle = 30.0
        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 49.3, abs_tol=2)

        elevation_angle = 40.0
        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 61.3, abs_tol=2)

        elevation_angle = 50.0
        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 72.6, abs_tol=2)

        elevation_angle = 60.0
        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 80.5, abs_tol=2)

        elevation_angle = 70.0
        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 91.9, abs_tol=2)

        elevation_angle = 80.0
        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 96.8, abs_tol=2)

        elevation_angle = 90.0
        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 99.2, abs_tol=2)

class Test_SUR(unittest.TestCase):

    def test_sur_los_probabilities(self):        
        
        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 2.17e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        elevation_angle = 10.0
        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 78.2, abs_tol=2)

        elevation_angle = 20.0
        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 86.9, abs_tol=2)

        elevation_angle = 30.0
        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 91.9, abs_tol=2)

        elevation_angle = 40.0
        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 92.9, abs_tol=2)

        elevation_angle = 50.0
        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 93.5, abs_tol=2)

        elevation_angle = 60.0
        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 94.0, abs_tol=2)

        elevation_angle = 70.0
        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 94.9, abs_tol=2)

        elevation_angle = 80.0
        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 95.2, abs_tol=2)

        elevation_angle = 90.0
        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 99.8, abs_tol=2)

class Test_DUR(unittest.TestCase):

    def test_dur_los_probabilities(self):        
        
        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 2.17e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        elevation_angle = 10.0
        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 28.2, abs_tol=2)

        elevation_angle = 20.0
        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 33.1, abs_tol=2)

        elevation_angle = 30.0
        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 39.8, abs_tol=2)

        elevation_angle = 40.0
        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 46.8, abs_tol=2)

        elevation_angle = 50.0
        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 53.7, abs_tol=2)

        elevation_angle = 60.0
        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 61.2, abs_tol=2)

        elevation_angle = 70.0
        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 73.8, abs_tol=2)

        elevation_angle = 80.0
        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 82.0, abs_tol=2)

        elevation_angle = 90.0
        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(batch_size=100, num_ut=100, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
        channel_model.set_topology(*topology)
        assert math.isclose(tf.cast(tf.reduce_sum(tf.cast(channel_model._scenario.los, tf.int32)),tf.float32)/100.0, 98.1, abs_tol=2)
        
       
if __name__ == '__main__':
    unittest.main()