# This file tests the implementation of step 8, the angles coupling and shuffling. 
# Step 8 is a mockup 
import unittest
import torch
from sionna.phy import config
import numpy as np

# Importing necessary modules from Sionna
import openntn.rays as rays
import openntn.dense_urban_scenario as sys_scenario
import openntn.antenna as antenna
from openntn.utils import gen_single_sector_topology as gen_topology

class TestShuffle_Coupling(unittest.TestCase):
    
    def setUp(self):
        # Creating a mock antenna configuration
        self.mockAntenna = antenna.Antenna(polarization="single",
                                           polarization_type="V",
                                           antenna_pattern="38.901",
                                           carrier_frequency=30e9)
        # Setting up a dense urban scenario
        self.mockScenario = sys_scenario.DenseUrbanScenario(carrier_frequency=30e9, 
                                                            ut_array=self.mockAntenna, 
                                                            bs_array=self.mockAntenna,
                                                            direction="uplink", 
                                                            elevation_angle=90.0, 
                                                            enable_pathloss=True, 
                                                            enable_shadow_fading=True, 
                                                            doppler_enabled=True,
                                                            precision="single")
        # Generate the topology
        topology = gen_topology(batch_size=2, num_ut=1, scenario="dur", elevation_angle=90, bs_height = 600000.0)

        # Set the topology
        self.mockScenario.set_topology(*topology)
        self.raysGenerator = rays.RaysGenerator(self.mockScenario)

    def test_random_coupling(self):
        # Creating test data for angles
        aoa = torch.tensor(np.random.rand(2, 1, 1, 4, 20), dtype=torch.float32, device=config.device)
        aod = torch.tensor(np.random.rand(2, 1, 1, 4, 20), dtype=torch.float32, device=config.device)
        zoa = torch.tensor(np.random.rand(2, 1, 1, 4, 20), dtype=torch.float32, device=config.device)
        zod = torch.tensor(np.random.rand(2, 1, 1, 4, 20), dtype=torch.float32, device=config.device)

        # Testing random coupling function
        shuffled_aoa, shuffled_aod, shuffled_zoa, shuffled_zod = self.raysGenerator._random_coupling(
            aoa, aod, zoa, zod
        )

        # Checking the shape of the shuffled outputs
        self.assertEqual(aoa.shape, shuffled_aoa.shape)
        self.assertEqual(aod.shape, shuffled_aod.shape)
        self.assertEqual(zoa.shape, shuffled_zoa.shape)
        self.assertEqual(zod.shape, shuffled_zod.shape)

        # Ensuring that the angles are shuffled
        self.assertFalse(torch.all(torch.eq(aoa, shuffled_aoa)).cpu().numpy())
        self.assertFalse(torch.all(torch.eq(aod, shuffled_aod)).cpu().numpy())
        self.assertFalse(torch.all(torch.eq(zoa, shuffled_zoa)).cpu().numpy())
        self.assertFalse(torch.all(torch.eq(zod, shuffled_zod)).cpu().numpy())

if __name__ == "__main__":
    unittest.main()