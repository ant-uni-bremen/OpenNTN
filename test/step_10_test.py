# This file tests the implementation of step 10, the initial random phase generation. 
# Step 10 test is  a mockup 
import unittest
import tensorflow as tf
import numpy as np
import math
from sionna.phy.constants import PI
from sionna.phy.channel.tr38811 import utils
from sionna.phy.channel.tr38811 import Antenna, AntennaArray,PanelArray,ChannelCoefficientsGenerator


class Test_Step10(unittest.TestCase):
    def setUp(self):
        # Initialize the required attributes
        self.shape = tf.constant([2, 3], dtype=tf.int32)
    
        self.mock_antenna = Antenna(
            polarization="single",
            polarization_type="V",
            antenna_pattern="38.901",
            carrier_frequency=30e9
        )
        
        # Create an instance of ChannelCoefficientsGenerator
        self.channel_generator = ChannelCoefficientsGenerator(
            carrier_frequency=30e9,
            tx_array=self.mock_antenna, 
            rx_array=self.mock_antenna,  
            subclustering=False,
            precision="single"
        )

    def test_step_10(self):
        # Call the _step_10 method
        phi = self.channel_generator._step_10(self.shape)

        # Compare the shapes
        expected_shape = (2, 3, 4) 
        self.assertEqual(phi.shape, expected_shape)

        phi = phi.numpy()
        self.assertTrue(np.all(phi > -PI))
        self.assertTrue(np.all(phi < PI))

        # Check that the distribution mean is close to zero
        mean_val = np.mean(phi)
        self.assertTrue(math.isclose(mean_val, 0.0, abs_tol=1))

if __name__ == "__main__":
    unittest.main()

