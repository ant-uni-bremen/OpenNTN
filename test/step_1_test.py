# This file tests the generation of the Scenarios according to Step 1 of 3GPP TR38.901 7.5
# combined wiht the additions of 3GPP TR38.811 6.7.2
# The test mainly test for the correct detection of illegal scenario configurations based on 38.811 5.2 and 4.5

from sionna.phy.channel.tr38811 import utils   # The code to test
import unittest   # The test framework
from sionna.phy.channel.tr38811 import Antenna, AntennaArray, DenseUrban, SubUrban, Urban, CDL
import numpy as np
  
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

    def test_legal_s_band_freq(self):        
        
        elevation_angle = 12.5

        direction = "downlink"
        #Minimal legal
        carrier_frequency = 2.17e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        #Max legal
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

        #Random legal values
        for i in range(10):
            carrier_frequency = np.random.uniform(high=2.2e9, low=2.17e9)
            ut_array = create_ut_ant(carrier_frequency)
            bs_array = create_bs_ant(carrier_frequency)
            channel_model = Urban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
            
        direction = "uplink"
        #Minimal legal
        carrier_frequency = 1.98e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        #Max legal
        carrier_frequency = 2.01e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
  
        #Random legal values
        for i in range(10):
            carrier_frequency = np.random.uniform(high=1.98e9, low=2.01e9)
            ut_array = create_ut_ant(carrier_frequency)
            bs_array = create_bs_ant(carrier_frequency)
            channel_model = Urban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        
    def test_dl_s_band_too_high(self):
        
        elevation_angle = 12.5

        direction = "downlink"
        #Too high
        carrier_frequency = 5e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        with self.assertRaises(Exception) as context:
            channel_model = Urban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("Carrier frequency must be either in S Band (1.9GHz-4GHz) or Ka Band (19GHz - 40GHz)" in str(context.exception))

    def test_dl_s_band_too_low(self):
        elevation_angle = 12.5

        direction = "downlink"
        #Too low
        carrier_frequency = 1e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        with self.assertRaises(Exception) as context:
            channel_model = Urban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("Carrier frequency must be either in S Band (1.9GHz-4GHz) or Ka Band (19GHz - 40GHz)" in str(context.exception))
    
    def test_dl_s_band_in_ka(self):
        elevation_angle = 12.5

        direction = "downlink"
        #In Ka Band
        carrier_frequency = 20e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)

    def test_ul_s_band_too_high(self):
        elevation_angle = 12.5

        direction = "uplink"
        #Too high
        carrier_frequency = 5e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        with self.assertRaises(Exception) as context:
            channel_model = Urban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("Carrier frequency must be either in S Band (1.9GHz-4GHz) or Ka Band (19GHz - 40GHz)" in str(context.exception))

    def test_ul_s_band_too_low(self):
        elevation_angle = 12.5

        direction = "uplink"
        #Too low
        carrier_frequency = 1.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        with self.assertRaises(Exception) as context:
            channel_model = Urban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("Carrier frequency must be either in S Band (1.9GHz-4GHz) or Ka Band (19GHz - 40GHz)" in str(context.exception))
    
    def test_ul_s_band_in_ka(self):
        elevation_angle = 12.5

        direction = "uplink"
        #In Ka Band
        carrier_frequency = 29.8e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        channel_model = Urban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)

    def test_ut_freq_incorrect(self):
        elevation_angle = 12.5

        direction = "uplink"
        #In Ka Band
        carrier_frequency = 29.8e9
        ut_array = create_ut_ant(29.9e9)
        bs_array = create_bs_ant(carrier_frequency)
        with self.assertRaises(Exception) as context:
            channel_model = Urban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("The carrier frequencies of ut antenna, bs antenna and scenario must match" in str(context.exception))
            
    def test_bs_freq_incorrect(self):
        elevation_angle = 12.5

        direction = "uplink"
        #In Ka Band
        carrier_frequency = 29.8e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(29.9e9)
        with self.assertRaises(Exception) as context:
            channel_model = Urban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("The carrier frequencies of ut antenna, bs antenna and scenario must match" in str(context.exception))

    def test_channel_freq_incorrect(self):
        elevation_angle = 12.5

        direction = "uplink"
        #In Ka Band
        carrier_frequency = 29.8e9
        ut_array = create_ut_ant(29.9e9)
        bs_array = create_bs_ant(29.9e9)
        with self.assertRaises(Exception) as context:
            channel_model = Urban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("The carrier frequencies of ut antenna, bs antenna and scenario must match" in str(context.exception))

    def test_all_freq_incorrect(self):
        elevation_angle = 12.5

        direction = "uplink"
        carrier_frequency = 29.8e9
        ut_array = create_ut_ant(29.9e9)
        bs_array = create_bs_ant(29.98e9)
        with self.assertRaises(Exception) as context:
            channel_model = Urban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("The carrier frequencies of ut antenna, bs antenna and scenario must match" in str(context.exception))

    # Currently untested due to change in structure to allow graph mode execution
    """
    def test_sat_too_high(self):
        elevation_angle = 12.5

        direction = "uplink"
        carrier_frequency = 29.8e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        with self.assertRaises(Exception) as context:
            channel_model = Urban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
            topology = utils.gen_single_sector_topology(batch_size=1, num_ut=1, scenario="urb", elevation_angle=elevation_angle, bs_height=37000000)
            channel_model.set_topology(*topology)
        self.assertTrue("The base station height must be at least 600'000m and not more than 36'000'000m, as only satellites are currently supported" in str(context.exception))

   
    
    def test_sat_too_low(self):
        elevation_angle = 12.5

        direction = "uplink"
        carrier_frequency = 29.8e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        with self.assertRaises(Exception) as context:
            channel_model = Urban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
            topology = utils.gen_single_sector_topology(batch_size=1, num_ut=1, scenario="urb", elevation_angle=elevation_angle, bs_height=400000)
            channel_model.set_topology(*topology)
        self.assertTrue("The base station height must be at least 500'000m and not more than 36'000'000m, as only satellites are currently supported" in str(context.exception))
    """
    def test_sat_at_legal_height(self):
        elevation_angle = 12.5

        direction = "uplink"
        carrier_frequency = 29.8e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        for i in range(10):
            bs_height = np.random.uniform(high=36000000, low=600000)
            channel_model = Urban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
            topology = utils.gen_single_sector_topology(batch_size=1, num_ut=1, scenario="urb", elevation_angle=elevation_angle, bs_height=bs_height)
            channel_model.set_topology(*topology)
            

class Test_DUR(unittest.TestCase):
    def test_legal_s_band_freq(self):        
        
        elevation_angle = 12.5

        direction = "downlink"
        #Minimal legal
        carrier_frequency = 2.17e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        #Max legal
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

        #Random legal values
        for i in range(10):
            carrier_frequency = np.random.uniform(high=2.2e9, low=2.17e9)
            ut_array = create_ut_ant(carrier_frequency)
            bs_array = create_bs_ant(carrier_frequency)
            channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
            
        direction = "uplink"
        #Minimal legal
        carrier_frequency = 1.98e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        #Max legal
        carrier_frequency = 2.01e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
  
        #Random legal values
        for i in range(10):
            carrier_frequency = np.random.uniform(high=1.98e9, low=2.01e9)
            ut_array = create_ut_ant(carrier_frequency)
            bs_array = create_bs_ant(carrier_frequency)
            channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        
    def test_dl_s_band_too_high(self):
        elevation_angle = 12.5

        direction = "downlink"
        #Too high
        carrier_frequency = 5e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        with self.assertRaises(Exception) as context:
            channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("Carrier frequency must be either in S Band (1.9GHz-4GHz) or Ka Band (19GHz - 40GHz)" in str(context.exception))

    def test_dl_s_band_too_low(self):
        elevation_angle = 12.5

        direction = "downlink"
        #Too low
        carrier_frequency = 1e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        with self.assertRaises(Exception) as context:
            channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("Carrier frequency must be either in S Band (1.9GHz-4GHz) or Ka Band (19GHz - 40GHz)" in str(context.exception))
    
    def test_dl_s_band_in_ka(self):
        elevation_angle = 12.5

        direction = "downlink"
        #In Ka Band
        carrier_frequency = 20e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)

    def test_ul_s_band_too_high(self):
        elevation_angle = 12.5

        direction = "uplink"
        #Too high
        carrier_frequency = 45e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        with self.assertRaises(Exception) as context:
            channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("Carrier frequency must be either in S Band (1.9GHz-4GHz) or Ka Band (19GHz - 40GHz)" in str(context.exception))

    def test_ul_s_band_too_low(self):
        elevation_angle = 12.5

        direction = "uplink"
        #Too low
        carrier_frequency = 1.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        with self.assertRaises(Exception) as context:
            channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("Carrier frequency must be either in S Band (1.9GHz-4GHz) or Ka Band (19GHz - 40GHz)" in str(context.exception))
    
    def test_ul_s_band_in_ka(self):
        elevation_angle = 12.5

        direction = "uplink"
        #In Ka Band
        carrier_frequency = 29.8e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
    def test_ut_freq_incorrect(self):
        elevation_angle = 12.5

        direction = "uplink"
        #In Ka Band
        carrier_frequency = 29.8e9
        ut_array = create_ut_ant(29.9e9)
        bs_array = create_bs_ant(carrier_frequency)
        with self.assertRaises(Exception) as context:
            channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("The carrier frequencies of ut antenna, bs antenna and scenario must match" in str(context.exception))
            

    def test_bs_freq_incorrect(self):
        elevation_angle = 12.5

        direction = "uplink"
        #In Ka Band
        carrier_frequency = 29.8e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(29.9e9)
        with self.assertRaises(Exception) as context:
            channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("The carrier frequencies of ut antenna, bs antenna and scenario must match" in str(context.exception))

    def test_channel_freq_incorrect(self):
        elevation_angle = 12.5

        direction = "uplink"
        #In Ka Band
        carrier_frequency = 29.8e9
        ut_array = create_ut_ant(29.9e9)
        bs_array = create_bs_ant(29.9e9)
        with self.assertRaises(Exception) as context:
            channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("The carrier frequencies of ut antenna, bs antenna and scenario must match" in str(context.exception))

    def test_all_freq_incorrect(self):
        elevation_angle = 12.5

        direction = "uplink"
        carrier_frequency = 29.8e9
        ut_array = create_ut_ant(29.9e9)
        bs_array = create_bs_ant(29.98e9)
        with self.assertRaises(Exception) as context:
            channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("The carrier frequencies of ut antenna, bs antenna and scenario must match" in str(context.exception))

class Test_SUR(unittest.TestCase):
    def test_legal_s_band_freq(self):        
        elevation_angle = 12.5

        direction = "downlink"
        #Minimal legal
        carrier_frequency = 2.17e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        #Max legal
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

        #Random legal values
        for i in range(10):
            carrier_frequency = np.random.uniform(high=2.2e9, low=2.17e9)
            ut_array = create_ut_ant(carrier_frequency)
            bs_array = create_bs_ant(carrier_frequency)
            channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
            
        direction = "uplink"
        #Minimal legal
        carrier_frequency = 1.98e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
        
        #Max legal
        carrier_frequency = 2.01e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)
  
        #Random legal values
        for i in range(10):
            carrier_frequency = np.random.uniform(high=1.98e9, low=2.01e9)
            ut_array = create_ut_ant(carrier_frequency)
            bs_array = create_bs_ant(carrier_frequency)
            channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        

    def test_dl_s_band_too_high(self):
        elevation_angle = 12.5

        direction = "downlink"
        #Too high
        carrier_frequency = 45e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        with self.assertRaises(Exception) as context:
            channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("Carrier frequency must be either in S Band (1.9GHz-4GHz) or Ka Band (19GHz - 40GHz)" in str(context.exception))

    def test_dl_s_band_too_low(self):
        elevation_angle = 12.5

        direction = "downlink"
        #Too low
        carrier_frequency = 1e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        with self.assertRaises(Exception) as context:
            channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("Carrier frequency must be either in S Band (1.9GHz-4GHz) or Ka Band (19GHz - 40GHz)" in str(context.exception))
    
    def test_dl_s_band_in_ka(self):
        elevation_angle = 12.5

        direction = "downlink"
        #In Ka Band
        carrier_frequency = 20e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)

    def test_ul_s_band_too_high(self):
        elevation_angle = 12.5

        direction = "uplink"
        #Too high
        carrier_frequency = 5e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        with self.assertRaises(Exception) as context:
            channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("Carrier frequency must be either in S Band (1.9GHz-4GHz) or Ka Band (19GHz - 40GHz)" in str(context.exception))

    def test_ul_s_band_too_low(self):
        elevation_angle = 12.5

        direction = "uplink"
        #Too low
        carrier_frequency = 1.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        with self.assertRaises(Exception) as context:
            channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("Carrier frequency must be either in S Band (1.9GHz-4GHz) or Ka Band (19GHz - 40GHz)" in str(context.exception))
    
    def test_ul_s_band_in_ka(self):
        elevation_angle = 12.5

        direction = "uplink"
        #In Ka Band
        carrier_frequency = 29.8e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                            ut_array=ut_array,
                                            bs_array=bs_array,
                                            direction=direction,
                                            elevation_angle=elevation_angle,
                                            enable_pathloss=True,
                                            enable_shadow_fading=True)

    def test_ut_freq_incorrect(self):
        elevation_angle = 12.5

        direction = "uplink"
        #In Ka Band
        carrier_frequency = 29.8e9
        ut_array = create_ut_ant(29.9e9)
        bs_array = create_bs_ant(carrier_frequency)
        with self.assertRaises(Exception) as context:
            channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("The carrier frequencies of ut antenna, bs antenna and scenario must match" in str(context.exception))
            

    def test_bs_freq_incorrect(self):
        elevation_angle = 12.5

        direction = "uplink"
        #In Ka Band
        carrier_frequency = 29.8e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(29.9e9)
        with self.assertRaises(Exception) as context:
            channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("The carrier frequencies of ut antenna, bs antenna and scenario must match" in str(context.exception))

    def test_channel_freq_incorrect(self):
        elevation_angle = 12.5

        direction = "uplink"
        #In Ka Band
        carrier_frequency = 29.8e9
        ut_array = create_ut_ant(29.9e9)
        bs_array = create_bs_ant(29.9e9)
        with self.assertRaises(Exception) as context:
            channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("The carrier frequencies of ut antenna, bs antenna and scenario must match" in str(context.exception))

    def test_all_freq_incorrect(self):
        elevation_angle = 12.5

        direction = "uplink"
        carrier_frequency = 29.8e9
        ut_array = create_ut_ant(29.9e9)
        bs_array = create_bs_ant(29.98e9)
        with self.assertRaises(Exception) as context:
            channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
        self.assertTrue("The carrier frequencies of ut antenna, bs antenna and scenario must match" in str(context.exception))

if __name__ == '__main__':
    unittest.main()