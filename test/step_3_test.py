from sionna.phy.channel.tr38811 import utils   # The code to test
import unittest   # The test framework
from sionna.phy.channel.tr38811 import Antenna, AntennaArray, Urban, DenseUrban, SubUrban
import tensorflow as tf
import math

def create_ut_ant(carrier_frequency):
    return Antenna(polarization="single",
                polarization_type="V",
                antenna_pattern="38.901",
                carrier_frequency=carrier_frequency)

def create_bs_ant(carrier_frequency):
    return AntennaArray(num_rows=1,
                        num_cols=4,
                        polarization="dual",
                        polarization_type="VH",
                        antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)

# Helper function to test different configurations with dynamic inputs
def run_test(channel_model_class, direction, elevation_angle, carrier_frequency, nlos_cl, sf_los_sigma, sf_nlos_sigma, scenario, batch_size, num_ut ):
    ut_array = create_ut_ant(carrier_frequency)
    bs_array = create_bs_ant(carrier_frequency)

    # Initialize the channel model dynamically based on the class passed
    channel_model = channel_model_class(carrier_frequency=carrier_frequency,
                                        ut_array=ut_array,
                                        bs_array=bs_array,
                                        direction=direction,
                                        elevation_angle=elevation_angle,
                                        enable_pathloss=True,
                                        enable_shadow_fading=True)

    # Generate topology and set it for the model, with dynamic scenario
    topology = utils.gen_single_sector_topology(batch_size=batch_size, num_ut=num_ut,
                                                scenario=scenario,
                                                elevation_angle=elevation_angle,
                                                bs_height=600000.0)
    channel_model.set_topology(*topology)

    # Subtract FSPL to isolate Clutter Loss (CL) and Shadow Fading (SF)
    loss_no_fspl = channel_model._scenario.basic_pathloss - channel_model._scenario.free_space_pathloss

    # Separate Line-of-Sight (LoS) and Non-Line-of-Sight (NLoS) losses
    loss_los = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los)
    loss_nlos = tf.boolean_mask(loss_no_fspl, channel_model._scenario.los == False)

    # Get mean and standard deviation for LoS and NLoS
    los_mean = tf.math.reduce_mean(loss_los)
    los_std = tf.math.reduce_std(loss_los)
    nlos_mean = tf.math.reduce_mean(loss_nlos)
    nlos_std = tf.math.reduce_std(loss_nlos)

    # Assertions for Line-of-Sight (LoS) cases
    try:
        assert math.isclose(los_mean, 0.0, abs_tol=0.5), f"LoS Mean mismatch: got {los_mean}, expected 0.0"
        assert math.isclose(los_std, sf_los_sigma, abs_tol=0.5), f"LoS Std deviation mismatch: got {los_std}, expected {sf_los_sigma}"
    except AssertionError as e:
        print(f"Test failed for LOS in {scenario} with elevation {elevation_angle}° , f {carrier_frequency} and direction {direction}: {str(e)}")

    # Assertions for Non-Line-of-Sight (NLoS) cases
    try:
        assert math.isclose(nlos_mean, nlos_cl, abs_tol=0.5), f"NLoS Mean mismatch: got {nlos_mean}, expected {nlos_cl}"
        assert math.isclose(nlos_std, sf_nlos_sigma, abs_tol=0.5), f"NLoS Std deviation mismatch: got {nlos_std}, expected {sf_nlos_sigma}"
    except AssertionError as e:
        print(f"Test failed for NLoS in {scenario} with elevation {elevation_angle}° , f {carrier_frequency} and direction {direction}: {str(e)}")
    print("\n")


class Test_Urb(unittest.TestCase):

    def setUp(self):
        self.batch_size = 100
        self.num_ut = 100

    # Test for Urban downlink with 10 degrees elevation
    def test_s_band_10_degrees_dl(self):
        run_test(channel_model_class=Urban,
                 direction="downlink",
                 elevation_angle=10.0,
                 carrier_frequency=2.2e9,
                 nlos_cl=34.3,
                 sf_los_sigma=4.0,
                 sf_nlos_sigma=6.0,
                 scenario='urb',
                 batch_size = self.batch_size,
                 num_ut= self.num_ut)

    # Test for Urban downlink with 20 degrees elevation
    def test_s_band_20_degrees_dl(self):
        run_test(channel_model_class=DenseUrban,
                 direction="downlink",
                 elevation_angle=20.0,
                 carrier_frequency=2.2e9,
                 nlos_cl=30.9,
                 sf_los_sigma=4.0,
                 sf_nlos_sigma=6.0,
                 scenario='urb',
                 batch_size = self.batch_size,
                 num_ut= self.num_ut)

    # Test for SubUrban downlink with 20 degrees elevation
    def test_s_band_20_degrees_dl(self):
        run_test(channel_model_class=SubUrban,
                 direction="downlink",
                 elevation_angle=20.0,
                 carrier_frequency=2.2e9,
                 nlos_cl=18.17,
                 sf_los_sigma=1.14,
                 sf_nlos_sigma=9.08,
                 scenario='urb',
                 batch_size = self.batch_size,
                 num_ut= self.num_ut)

class Test_SubUrban(unittest.TestCase):
    def setUp(self):
        self.batch_size = 1000
        self.num_ut = 100
    # Test for Urban downlink with 10 degrees elevation
    def test_s_band_10_degrees_dl(self):
        run_test(channel_model_class=SubUrban,
                 direction="downlink",
                 elevation_angle=10.0,
                 carrier_frequency=2.2e9,
                 nlos_cl=19.52,
                 sf_los_sigma=1.79,
                 sf_nlos_sigma=8.93,
                 scenario='sur',
                 batch_size = self.batch_size,
                 num_ut= self.num_ut)

    # Test for SubUrban downlink with 20 degrees elevation
    def test_s_band_20_degrees_dl(self):
        run_test(channel_model_class=SubUrban,
                 direction="downlink",
                 elevation_angle=20.0,
                 carrier_frequency=2.2e9,
                 nlos_cl=18.17,
                 sf_los_sigma=1.14,
                 sf_nlos_sigma=9.08,
                 scenario='sur',
                 batch_size = self.batch_size,
                 num_ut= self.num_ut)

    # Test for SubUrban downlink with 30 degrees elevation
    def test_s_band_30_degrees_dl(self):
        run_test(channel_model_class=SubUrban,
                 direction="downlink",
                 elevation_angle=30.0,
                 carrier_frequency=2.2e9,
                 nlos_cl=18.42,
                 sf_los_sigma=1.14,
                 sf_nlos_sigma=8.78,
                 scenario='sur',
                 batch_size = self.batch_size,
                 num_ut= self.num_ut)

    # Test for SubUrban downlink with 40 degrees elevation
    def test_s_band_40_degrees_dl(self):
        run_test(channel_model_class=SubUrban,
                 direction="downlink",
                 elevation_angle=40.0,
                 carrier_frequency=2.2e9,
                 nlos_cl=18.28,
                 sf_los_sigma=0.92,
                 sf_nlos_sigma=10.25,
                 scenario='sur',
                 batch_size = self.batch_size,
                 num_ut= self.num_ut)

    # Test for SubUrban downlink with 50 degrees elevation
    def test_s_band_50_degrees_dl(self):
        run_test(channel_model_class=SubUrban,
                 direction="downlink",
                 elevation_angle=50.0,
                 carrier_frequency=2.2e9,
                 nlos_cl=18.63,
                 sf_los_sigma=1.42,
                 sf_nlos_sigma=10.56,
                 scenario='sur',
                 batch_size = self.batch_size,
                 num_ut= self.num_ut)

    # Test for SubUrban downlink with 60 degrees elevation
    def test_s_band_60_degrees_dl(self):
        run_test(channel_model_class=SubUrban,
                 direction="downlink",
                 elevation_angle=60.0,
                 carrier_frequency=2.2e9,
                 nlos_cl=17.68,
                 sf_los_sigma=1.56,
                 sf_nlos_sigma=10.74,
                 scenario='sur',
                 batch_size = self.batch_size,
                 num_ut= self.num_ut)
    # Test for SubUrban downlink with 70 degrees elevation    
    def test_s_band_70_degrees_dl(self):
        run_test(channel_model_class=SubUrban,
                 direction="downlink",
                 elevation_angle=70.0,
                 carrier_frequency=2.2e9,
                 nlos_cl=16.50,
                 sf_los_sigma=0.85,
                 sf_nlos_sigma=10.17,
                 scenario='sur',
                 batch_size = self.batch_size,
                 num_ut= self.num_ut)

    # Test for SubUrban downlink with 80 degrees elevation
    def test_s_band_80_degrees_dl(self):
        run_test(channel_model_class=SubUrban,
                 direction="downlink",
                 elevation_angle=80.0,
                 carrier_frequency=2.2e9,
                 nlos_cl=16.30,
                 sf_los_sigma=0.72,
                 sf_nlos_sigma=11.52,
                 scenario='sur',
                 batch_size = self.batch_size,
                 num_ut= self.num_ut)

    # Test for SubUrban downlink with 90 degrees elevation
    def test_s_band_90_degrees_dl(self):
        run_test(channel_model_class=SubUrban,
                 direction="downlink",
                 elevation_angle=90.0,
                 carrier_frequency=2.2e9,
                 nlos_cl=16.30,
                 sf_los_sigma=0.72,
                 sf_nlos_sigma=11.52,
                 scenario='sur',
                 batch_size = 1000,
                 num_ut= 200)
        
    # Test for Ka-band downlink with 10 degrees elevation
    def test_ka_band_10_degrees_dl(self):
        run_test(channel_model_class=SubUrban,
                 direction="downlink",
                 elevation_angle=10.0,
                 carrier_frequency=20e9,  
                 nlos_cl=29.5,
                 sf_los_sigma=1.9,
                 sf_nlos_sigma=10.7,
                 scenario='sur',
                 batch_size = self.batch_size,
                 num_ut= self.num_ut)

    # Test for Ka-band downlink with 20 degrees elevation
    def test_ka_band_20_degrees_dl(self):
        run_test(channel_model_class=SubUrban,
                 direction="downlink",
                 elevation_angle=20.0,
                 carrier_frequency=20e9,  
                 nlos_cl=24.6,
                 sf_los_sigma=1.6,
                 sf_nlos_sigma=10.0,
                 scenario='sur',
                 batch_size = self.batch_size,
                 num_ut= self.num_ut)

    # Test for Ka-band downlink with 30 degrees elevation
    def test_ka_band_30_degrees_dl(self):
        run_test(channel_model_class=SubUrban,
                 direction="downlink",
                 elevation_angle=30.0,
                 carrier_frequency=20e9,  
                 nlos_cl=21.9,
                 sf_los_sigma=1.9,
                 sf_nlos_sigma=11.2,
                 scenario='sur',
                 batch_size = self.batch_size,
                 num_ut= self.num_ut)

    # Test for Ka-band downlink with 40 degrees elevation
    def test_ka_band_40_degrees_dl(self):
        run_test(channel_model_class=SubUrban,
                 direction="downlink",
                 elevation_angle=40.0,
                 carrier_frequency=20e9,  
                 nlos_cl=20.0,
                 sf_los_sigma=2.3,
                 sf_nlos_sigma=11.6,
                 scenario='sur',
                 batch_size = self.batch_size,
                 num_ut= self.num_ut)

    # Test for Ka-band downlink with 50 degrees elevation
    def test_ka_band_50_degrees_dl(self):
        run_test(channel_model_class=SubUrban,
                 direction="downlink",
                 elevation_angle=50.0,
                 carrier_frequency=20e9,  
                 nlos_cl=18.7,
                 sf_los_sigma=2.7,
                 sf_nlos_sigma=11.8,
                 scenario='sur',
                 batch_size = self.batch_size,
                 num_ut= self.num_ut)

    # Test for Ka-band downlink with 60 degrees elevation
    def test_ka_band_60_degrees_dl(self):
        run_test(channel_model_class=SubUrban,
                 direction="downlink",
                 elevation_angle=60.0,
                 carrier_frequency=20e9,  
                 nlos_cl=17.8,
                 sf_los_sigma=3.1,
                 sf_nlos_sigma=10.8,
                 scenario='sur',
                 batch_size = self.batch_size,
                 num_ut= self.num_ut)
        
    # Test for Ka-band downlink with 70 degrees elevation
    def test_ka_band_70_degrees_dl(self):
        run_test(channel_model_class=SubUrban,
                 direction="downlink",
                 elevation_angle=70.0,
                 carrier_frequency=20e9,  
                 nlos_cl=17.2,
                 sf_los_sigma=3.0,
                 sf_nlos_sigma=10.8,
                 scenario='sur',
                 batch_size = self.batch_size,
                 num_ut= self.num_ut)

    # Test for Ka-band downlink with 80 degrees elevation
    def test_ka_band_80_degrees_dl(self):
        run_test(channel_model_class=SubUrban,
                 direction="downlink",
                 elevation_angle=80.0,
                 carrier_frequency=20e9,  
                 nlos_cl=16.9,
                 sf_los_sigma=3.6,
                 sf_nlos_sigma=10.8,
                 scenario='sur',
                 batch_size = self.batch_size,
                 num_ut= self.num_ut)

    # Test for Ka-band downlink with 90 degrees elevation
    def test_ka_band_90_degrees_dl(self):
        run_test(channel_model_class=SubUrban,
                 direction="downlink",
                 elevation_angle=90.0,
                 carrier_frequency=20e9,  
                 nlos_cl=16.8,
                 sf_los_sigma=0.4,
                 sf_nlos_sigma=10.8,
                 scenario='sur',
                 batch_size = self.batch_size,
                 num_ut= self.num_ut)


if __name__ == '__main__':
    unittest.main()
