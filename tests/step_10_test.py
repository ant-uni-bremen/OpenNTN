# SPDX-FileCopyrightText: Copyright (c) 2025 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# This file tests the implementation of step 10, the initial random phase generation. 
# Step 10 test is  a mockup 
import unittest
import torch
import numpy as np
import math
from sionna.phy.constants import PI
from openntn import utils
from openntn import Antenna, AntennaArray,PanelArray,ChannelCoefficientsGenerator

from statistical_checks import assert_uniform_mean


class Test_Step10(unittest.TestCase):
    def setUp(self):
        # Initialize the required attributes
        self.shape = torch.tensor([2, 3], dtype=torch.int32)
    
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

        phi = phi.cpu().numpy()
        self.assertTrue(np.all(phi >= -PI))
        self.assertTrue(np.all(phi < PI))

    def test_step_10_distribution(self):
        # TR 38.901 V16.1.0, step 10: the initial phases are uniform within (-pi, pi).
        # A large sample makes the statistical check on the mean meaningful; see
        # statistical_checks.py.
        phi = self.channel_generator._step_10(torch.tensor([100, 100], dtype=torch.int32))
        assert_uniform_mean(phi, -PI, PI, "mean of the initial phases")

if __name__ == "__main__":
    unittest.main()

