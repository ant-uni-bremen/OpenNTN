# This file tests the generation of the LOS states according to Step 2 of 3GPP TR38.901 7.5
# using the parameters of 3GPP TR38.811 Table 6.6.1-1 LOS probability
# The LoS states of the 10000 links of each case are independent Bernoulli draws, so the
# observed fraction is checked with a statistical tolerance, see statistical_checks.py.

from openntn import utils   # The code to test
import unittest   # The test framework

import pytest
from openntn import Antenna, AntennaArray, DenseUrban, SubUrban, Urban
import numpy as np
import torch
import math

from statistical_checks import assert_proportion
  
# Every test of this file takes about a second or more.
pytestmark = pytest.mark.slow


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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.246, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.386, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.493, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.613, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.726, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.805, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.919, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.968, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.992, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.782, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.869, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.919, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.929, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.935, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.940, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.949, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.952, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.998, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.282, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.331, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.398, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.468, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.537, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.612, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.738, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.820, f"LoS probability at {elevation_angle} degrees")

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
        assert_proportion(int(channel_model._scenario.los.sum()), channel_model._scenario.los.numel(), 0.981, f"LoS probability at {elevation_angle} degrees")
        
       
if __name__ == '__main__':
    unittest.main()