#This file simuates the coupling losses and does the calibration test.
from sionna.phy.channel.tr38811 import utils   # The code to test
import unittest   # The test framework
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sionna.phy.channel.tr38811 import Antenna, AntennaArray, Urban


class TestLinkBudget(unittest.TestCase):

    def create_antenna_arrays(self, carrier_frequency):  # Helper function for Antenna Initialization
        ut_array = Antenna(polarization="single",
                           polarization_type="V",
                           antenna_pattern="38.901",
                           carrier_frequency=carrier_frequency)

        bs_array = AntennaArray(num_rows=1,
                                num_cols=4,
                                polarization="dual",
                                polarization_type="VH",
                                antenna_pattern="38.901",
                                carrier_frequency=carrier_frequency)
        return ut_array, bs_array

    def test_sc1_dl(self):  
        carrier_frequency = 20e9
        elevation_angle = 12.5
        direction = "downlink"
        scenario = "urb"

        ut_array, bs_array = self.create_antenna_arrays(carrier_frequency)
        channel_model = Urban(carrier_frequency=carrier_frequency,
                              ut_array=ut_array,
                              bs_array=bs_array,
                              direction=direction,
                              elevation_angle=elevation_angle,
                              enable_pathloss=True,
                              enable_shadow_fading=True)
        
        topology = utils.gen_single_sector_topology(
            batch_size=100,
            num_ut=2,
            scenario=scenario,
            elevation_angle=elevation_angle,
            bs_height=35786000.0
        )
        channel_model.set_topology(*topology)

        # Extracting coupling loss values
        lsp_generator = channel_model._lsp_sampler
        sample_path_loss =lsp_generator.sample_pathloss()
        sample_path_loss_los = tf.boolean_mask(sample_path_loss, channel_model._scenario.los)
        sample_path_loss_nlos = tf.boolean_mask(sample_path_loss, channel_model._scenario.los == False)
        gas_path_loss = channel_model._scenario.gas_pathloss[:, 0, 0]
        scintillation_path_loss = channel_model._scenario.scintillation_pathloss[:, 0, 0]
        free_space_path_loss = channel_model._scenario.free_space_pathloss[:, 0, 0]

        coupling_loss = free_space_path_loss + gas_path_loss + scintillation_path_loss

        pl_basic_db = channel_model._scenario.basic_pathloss[0, 0, 0]
        pl_entry_db = channel_model._scenario.entry_pathloss[0, 0, 0]

        # print("Gas Path Loss:", GPL)
        # print("Scintillation Path Loss:", SPL)
        # print("Free Space Path Loss:", FSPL)
        # print(CNR)
        print("basic pathloss :",pl_basic_db)
        print("\nEntry pathloss :", pl_entry_db)
        print("\nFree Space Path loss", free_space_path_loss)
        print("\nGas Path loss", gas_path_loss)
        print("\nScintilation Path loss", scintillation_path_loss)
        print("\nSample Path loss", sample_path_loss)

        # # Calculate the CDF
        # sorted_cl = np.sort(coupling_loss)
        # cdf = np.arange(1, len(sorted_cl) + 1) / len(sorted_cl)

        # # Plot the CDF
        # plt.figure(figsize=(8, 6))
        # plt.plot(sorted_cl, cdf, label="Coupling Loss CDF")
        # plt.title("CDF of Coupling Loss")
        # plt.xlabel("Coupling Loss (dB)")
        # plt.ylabel("CDF")
        # plt.grid()
        # plt.legend()
        # plt.show()

        # # Print statistics
        # print("Coupling Loss Statistics:")
        # print(f"Mean: {np.mean(coupling_loss):.2f} dB")
        # print(f"Median: {np.median(coupling_loss):.2f} dB")
        # print(f"Min: {np.min(coupling_loss):.2f} dB")
        # print(f"Max: {np.max(coupling_loss):.2f} dB")
        # Flatten the sample path loss for all users
        sample_path_loss_np = sample_path_loss_los.numpy()
        sample_path_loss_flattened = sample_path_loss_np.flatten()

        # Calculate the CDF
        sorted_sample_pl = np.sort(sample_path_loss_flattened)
        cdf_sample_pl = np.arange(1, len(sorted_sample_pl) + 1) / len(sorted_sample_pl)

        # Plot the CDF
        #plt.figure(figsize=(8, 6))
        #plt.plot(sorted_sample_pl, cdf_sample_pl, label="Sample Path Loss CDF", color="orange")
        #plt.title("CDF of Sample Path Loss")
        #plt.xlabel("Sample Path Loss (dB)")
        #plt.ylabel("CDF")
        #plt.grid()
        #plt.legend()
        #plt.show()

        # Print statistics
        print("Sample Path Loss Statistics:")
        print(f"Mean: {np.mean(sample_path_loss_flattened):.2f} dB")
        print(f"Median: {np.median(sample_path_loss_flattened):.2f} dB")
        print(f"Min: {np.min(sample_path_loss_flattened):.2f} dB")
        print(f"Max: {np.max(sample_path_loss_flattened):.2f} dB")


if __name__ == '__main__':
    unittest.main()

