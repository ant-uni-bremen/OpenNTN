# This file is very huge, as it effectively tests a significant portion of the values in the parametrization tables for all scenarios.
# The order of Large Scale parameters (LSPs) differs from the standard and is: DS ASD ASA SF K ZSA ZSD. This is to keep
# in line with the rest of the Sionna implementation. At the moment the proper distribution of all sampled parameters is evaluated.
# This file only tests the correlation between the LSP, whereas the correct distribution is tested in step_4_lsp_generation_test.py. The standard
# 3GPP TR38.811 uses the value N/A for a few correlations, which is interpreted as 0.0 in this implementation. 
# The spatial correlation is calculated as exp(d*c) for correlated parameters, in which d is the 2d distance and c is the correlation factor, which
# is equal to -1/the value in the corresponding table. 

from sionna.phy.channel.tr38811 import utils   # The code to test
import unittest   # The test framework
from sionna.phy.channel.tr38811 import Antenna, AntennaArray, DenseUrban, SubUrban, Urban, CDL
import numpy as np
import tensorflow as tf
import math
#from sionna.utils import matrix_sqrt


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

def square_matrix(mat):
    return tf.linalg.matmul(mat,tf.linalg.adjoint(mat))



class Test_URB(unittest.TestCase):
# Values taken from Table 6.7.2-4a: Channel model parameters for Urban Scenario (NLOS) at S band and 
# Table 6.7.2-3a: Channel model parameters for Urban Scenario (LOS) at S band
    def test_s_band_dl(self):
        
        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        num_ut = 100

        DS_los = 30.0
        ASD_los = 18.0
        ASA_los = 15.0
        SF_los = 37.0
        K_los = 12.0
        ZSA_los = 15.0
        ZSD_los = 15.0

        DS_nlos = 40.0
        ASD_nlos = 50.0
        ASA_nlos = 50.0
        SF_nlos = 50.0
        K_nlos = 1.0
        ZSA_nlos = 50.0
        ZSD_nlos = 50.0

        for elevation_angle in [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0]:

            channel_model = Urban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
            
            topology = utils.gen_single_sector_topology(batch_size=100, num_ut=num_ut, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
            channel_model.set_topology(*topology)
            indoor = tf.tile(tf.expand_dims(channel_model._scenario.indoor, axis=1),
                         [1, channel_model._scenario.num_bs, 1])
            # LoS
            los_ut = channel_model._scenario.los
            los_pair_bool = tf.logical_and(tf.expand_dims(los_ut, axis=3),
                                        tf.expand_dims(los_ut, axis=2))
            # NLoS
            nlos_ut = tf.logical_and(tf.logical_not(channel_model._scenario.los),
                                    tf.logical_not(indoor))
            nlos_pair_bool = tf.logical_and(tf.expand_dims(nlos_ut, axis=3),
                                            tf.expand_dims(nlos_ut, axis=2))
            # O2I
            o2i_pair_bool = tf.logical_and(tf.expand_dims(indoor, axis=3),
                                        tf.expand_dims(indoor, axis=2))

            # Stacking the correlation matrix
            # One correlation matrix per LSP
            filtering_matrices = []
            distance_scaling_matrices = []
            for parameter_name in ('corrDistDS', 'corrDistASD', 'corrDistASA',
                'corrDistSF', 'corrDistK', 'corrDistZSA', 'corrDistZSD'):
                
                if parameter_name == 'corrDistDS':
                    parameter_value_los = DS_los
                    parameter_value_nlos = DS_nlos
                elif parameter_name == 'corrDistASD':
                    parameter_value_los = ASD_los
                    parameter_value_nlos = ASD_nlos
                elif parameter_name == 'corrDistASA':
                    parameter_value_los = ASA_los
                    parameter_value_nlos = ASA_nlos
                elif parameter_name == 'corrDistSF':
                    parameter_value_los = SF_los
                    parameter_value_nlos = SF_nlos
                elif parameter_name == 'corrDistK':
                    parameter_value_los = K_los
                    parameter_value_nlos = K_nlos
                elif parameter_name == 'corrDistZSA':
                    parameter_value_los = ZSA_los
                    parameter_value_nlos = ZSA_nlos
                elif parameter_name == 'corrDistZSD':
                    parameter_value_los = ZSD_los
                    parameter_value_nlos = ZSD_nlos

                filtering_matrix = tf.eye(channel_model._scenario.num_ut,
                    channel_model._scenario.num_ut, batch_shape=[channel_model._scenario.batch_size,
                    channel_model._scenario.num_bs], dtype=channel_model._scenario.rdtype)
                
                distance_scaling_matrix = tf.where(channel_model._scenario.los, parameter_value_los,parameter_value_nlos)
                #distance_scaling_matrix = channel_model._scenario.get_param(parameter_name)
                distance_scaling_matrix = tf.tile(tf.expand_dims(
                    distance_scaling_matrix, axis=3),
                    [1, 1, 1, channel_model._scenario.num_ut])
    
                epsilon = 1e-12
                distance_scaling_matrix = -1. / (distance_scaling_matrix + epsilon)
                # LoS
                filtering_matrix = tf.where(los_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # NLoS
                filtering_matrix = tf.where(nlos_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # indoor
                filtering_matrix = tf.where(o2i_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # Stacking
                filtering_matrices.append(filtering_matrix)
                distance_scaling_matrices.append(distance_scaling_matrix)
            filtering_matrices = tf.stack(filtering_matrices, axis=2)
            distance_scaling_matrices = tf.stack(distance_scaling_matrices, axis=2)
            ut_dist_2d = channel_model._scenario.matrix_ut_distance_2d
            # Adding a dimension for broadcasting with BS
            ut_dist_2d = tf.expand_dims(tf.expand_dims(ut_dist_2d, axis=1), axis=2)
            
            spatial_lsp_correlation = (tf.math.exp(ut_dist_2d*distance_scaling_matrices)*filtering_matrices)  
            spatial_lsp_correlation = tf.linalg.cholesky(spatial_lsp_correlation)

            difference = channel_model._lsp_sampler._spatial_lsp_correlation_matrix_sqrt - spatial_lsp_correlation

            difference = tf.abs(difference) < 1e-6

            assert tf.reduce_all(difference)



    def test_s_band_ul(self):

        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        num_ut = 100

        DS_los = 30.0
        ASD_los = 18.0
        ASA_los = 15.0
        SF_los = 37.0
        K_los = 12.0
        ZSA_los = 15.0
        ZSD_los = 15.0

        DS_nlos = 40.0
        ASD_nlos = 50.0
        ASA_nlos = 50.0
        SF_nlos = 50.0
        K_nlos = 1.0
        ZSA_nlos = 50.0
        ZSD_nlos = 50.0

        for elevation_angle in [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0]:

            channel_model = Urban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
            
            topology = utils.gen_single_sector_topology(batch_size=100, num_ut=num_ut, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
            channel_model.set_topology(*topology)
            indoor = tf.tile(tf.expand_dims(channel_model._scenario.indoor, axis=1),
                         [1, channel_model._scenario.num_bs, 1])
            # LoS
            los_ut = channel_model._scenario.los
            los_pair_bool = tf.logical_and(tf.expand_dims(los_ut, axis=3),
                                        tf.expand_dims(los_ut, axis=2))
            # NLoS
            nlos_ut = tf.logical_and(tf.logical_not(channel_model._scenario.los),
                                    tf.logical_not(indoor))
            nlos_pair_bool = tf.logical_and(tf.expand_dims(nlos_ut, axis=3),
                                            tf.expand_dims(nlos_ut, axis=2))
            # O2I
            o2i_pair_bool = tf.logical_and(tf.expand_dims(indoor, axis=3),
                                        tf.expand_dims(indoor, axis=2))

            # Stacking the correlation matrix
            # One correlation matrix per LSP
            filtering_matrices = []
            distance_scaling_matrices = []
            for parameter_name in ('corrDistDS', 'corrDistASD', 'corrDistASA',
                'corrDistSF', 'corrDistK', 'corrDistZSA', 'corrDistZSD'):
                
                if parameter_name == 'corrDistDS':
                    parameter_value_los = DS_los
                    parameter_value_nlos = DS_nlos
                elif parameter_name == 'corrDistASD':
                    parameter_value_los = ASD_los
                    parameter_value_nlos = ASD_nlos
                elif parameter_name == 'corrDistASA':
                    parameter_value_los = ASA_los
                    parameter_value_nlos = ASA_nlos
                elif parameter_name == 'corrDistSF':
                    parameter_value_los = SF_los
                    parameter_value_nlos = SF_nlos
                elif parameter_name == 'corrDistK':
                    parameter_value_los = K_los
                    parameter_value_nlos = K_nlos
                elif parameter_name == 'corrDistZSA':
                    parameter_value_los = ZSA_los
                    parameter_value_nlos = ZSA_nlos
                elif parameter_name == 'corrDistZSD':
                    parameter_value_los = ZSD_los
                    parameter_value_nlos = ZSD_nlos

                filtering_matrix = tf.eye(channel_model._scenario.num_ut,
                    channel_model._scenario.num_ut, batch_shape=[channel_model._scenario.batch_size,
                    channel_model._scenario.num_bs], dtype=channel_model._scenario.rdtype)
                
                distance_scaling_matrix = tf.where(channel_model._scenario.los, parameter_value_los,parameter_value_nlos)
                #distance_scaling_matrix = channel_model._scenario.get_param(parameter_name)
                distance_scaling_matrix = tf.tile(tf.expand_dims(
                    distance_scaling_matrix, axis=3),
                    [1, 1, 1, channel_model._scenario.num_ut])
    
                epsilon = 1e-12
                distance_scaling_matrix = -1. / (distance_scaling_matrix + epsilon)
                # LoS
                filtering_matrix = tf.where(los_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # NLoS
                filtering_matrix = tf.where(nlos_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # indoor
                filtering_matrix = tf.where(o2i_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # Stacking
                filtering_matrices.append(filtering_matrix)
                distance_scaling_matrices.append(distance_scaling_matrix)
            filtering_matrices = tf.stack(filtering_matrices, axis=2)
            distance_scaling_matrices = tf.stack(distance_scaling_matrices, axis=2)
            ut_dist_2d = channel_model._scenario.matrix_ut_distance_2d
            # Adding a dimension for broadcasting with BS
            ut_dist_2d = tf.expand_dims(tf.expand_dims(ut_dist_2d, axis=1), axis=2)
            
            spatial_lsp_correlation = (tf.math.exp(ut_dist_2d*distance_scaling_matrices)*filtering_matrices)  
            spatial_lsp_correlation = tf.linalg.cholesky(spatial_lsp_correlation)

            difference = channel_model._lsp_sampler._spatial_lsp_correlation_matrix_sqrt - spatial_lsp_correlation
            difference = tf.abs(difference) < 1e-6

            assert tf.reduce_all(difference)



    # Values from tables Table 6.7.2-2b: Channel model parameters for Dense Urban Scenario (NLOS) in Ka band
    # and Table 6.7.2-5b: Channel model parameters for Suburban Scenario (LOS) in Ka band

    def test_ka_band_dl(self):
        direction = "downlink"
        scenario = "urb"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        num_ut = 100

        DS_los = 30.0
        ASD_los = 18.0
        ASA_los = 15.0
        SF_los = 37.0
        K_los = 12.0
        ZSA_los = 15.0
        ZSD_los = 15.0

        DS_nlos = 40.0
        ASD_nlos = 50.0
        ASA_nlos = 50.0
        SF_nlos = 50.0
        K_nlos = 1.0
        ZSA_nlos = 50.0
        ZSD_nlos = 50.0

        for elevation_angle in [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0]:

            channel_model = Urban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
            
            topology = utils.gen_single_sector_topology(batch_size=100, num_ut=num_ut, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
            channel_model.set_topology(*topology)
            indoor = tf.tile(tf.expand_dims(channel_model._scenario.indoor, axis=1),
                         [1, channel_model._scenario.num_bs, 1])
            # LoS
            los_ut = channel_model._scenario.los
            los_pair_bool = tf.logical_and(tf.expand_dims(los_ut, axis=3),
                                        tf.expand_dims(los_ut, axis=2))
            # NLoS
            nlos_ut = tf.logical_and(tf.logical_not(channel_model._scenario.los),
                                    tf.logical_not(indoor))
            nlos_pair_bool = tf.logical_and(tf.expand_dims(nlos_ut, axis=3),
                                            tf.expand_dims(nlos_ut, axis=2))
            # O2I
            o2i_pair_bool = tf.logical_and(tf.expand_dims(indoor, axis=3),
                                        tf.expand_dims(indoor, axis=2))

            # Stacking the correlation matrix
            # One correlation matrix per LSP
            filtering_matrices = []
            distance_scaling_matrices = []
            for parameter_name in ('corrDistDS', 'corrDistASD', 'corrDistASA',
                'corrDistSF', 'corrDistK', 'corrDistZSA', 'corrDistZSD'):
                
                if parameter_name == 'corrDistDS':
                    parameter_value_los = DS_los
                    parameter_value_nlos = DS_nlos
                elif parameter_name == 'corrDistASD':
                    parameter_value_los = ASD_los
                    parameter_value_nlos = ASD_nlos
                elif parameter_name == 'corrDistASA':
                    parameter_value_los = ASA_los
                    parameter_value_nlos = ASA_nlos
                elif parameter_name == 'corrDistSF':
                    parameter_value_los = SF_los
                    parameter_value_nlos = SF_nlos
                elif parameter_name == 'corrDistK':
                    parameter_value_los = K_los
                    parameter_value_nlos = K_nlos
                elif parameter_name == 'corrDistZSA':
                    parameter_value_los = ZSA_los
                    parameter_value_nlos = ZSA_nlos
                elif parameter_name == 'corrDistZSD':
                    parameter_value_los = ZSD_los
                    parameter_value_nlos = ZSD_nlos

                filtering_matrix = tf.eye(channel_model._scenario.num_ut,
                    channel_model._scenario.num_ut, batch_shape=[channel_model._scenario.batch_size,
                    channel_model._scenario.num_bs], dtype=channel_model._scenario.rdtype)
                
                distance_scaling_matrix = tf.where(channel_model._scenario.los, parameter_value_los,parameter_value_nlos)
                #distance_scaling_matrix = channel_model._scenario.get_param(parameter_name)
                distance_scaling_matrix = tf.tile(tf.expand_dims(
                    distance_scaling_matrix, axis=3),
                    [1, 1, 1, channel_model._scenario.num_ut])
    
                epsilon = 1e-12
                distance_scaling_matrix = -1. / (distance_scaling_matrix + epsilon)
                # LoS
                filtering_matrix = tf.where(los_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # NLoS
                filtering_matrix = tf.where(nlos_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # indoor
                filtering_matrix = tf.where(o2i_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # Stacking
                filtering_matrices.append(filtering_matrix)
                distance_scaling_matrices.append(distance_scaling_matrix)
            filtering_matrices = tf.stack(filtering_matrices, axis=2)
            distance_scaling_matrices = tf.stack(distance_scaling_matrices, axis=2)
            ut_dist_2d = channel_model._scenario.matrix_ut_distance_2d
            # Adding a dimension for broadcasting with BS
            ut_dist_2d = tf.expand_dims(tf.expand_dims(ut_dist_2d, axis=1), axis=2)
            
            spatial_lsp_correlation = (tf.math.exp(ut_dist_2d*distance_scaling_matrices)*filtering_matrices)  
            spatial_lsp_correlation = tf.linalg.cholesky(spatial_lsp_correlation)

            difference = channel_model._lsp_sampler._spatial_lsp_correlation_matrix_sqrt - spatial_lsp_correlation

            difference = tf.abs(difference) < 1e-6

            assert tf.reduce_all(difference)




    def test_ka_band_ul(self):
        direction = "uplink"
        scenario = "urb"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        num_ut = 100

        DS_los = 30.0
        ASD_los = 18.0
        ASA_los = 15.0
        SF_los = 37.0
        K_los = 12.0
        ZSA_los = 15.0
        ZSD_los = 15.0

        DS_nlos = 40.0
        ASD_nlos = 50.0
        ASA_nlos = 50.0
        SF_nlos = 50.0
        K_nlos = 1.0
        ZSA_nlos = 50.0
        ZSD_nlos = 50.0

        for elevation_angle in [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0]:

            channel_model = Urban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
            
            topology = utils.gen_single_sector_topology(batch_size=100, num_ut=num_ut, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
            channel_model.set_topology(*topology)
            indoor = tf.tile(tf.expand_dims(channel_model._scenario.indoor, axis=1),
                         [1, channel_model._scenario.num_bs, 1])
            # LoS
            los_ut = channel_model._scenario.los
            los_pair_bool = tf.logical_and(tf.expand_dims(los_ut, axis=3),
                                        tf.expand_dims(los_ut, axis=2))
            # NLoS
            nlos_ut = tf.logical_and(tf.logical_not(channel_model._scenario.los),
                                    tf.logical_not(indoor))
            nlos_pair_bool = tf.logical_and(tf.expand_dims(nlos_ut, axis=3),
                                            tf.expand_dims(nlos_ut, axis=2))
            # O2I
            o2i_pair_bool = tf.logical_and(tf.expand_dims(indoor, axis=3),
                                        tf.expand_dims(indoor, axis=2))

            # Stacking the correlation matrix
            # One correlation matrix per LSP
            filtering_matrices = []
            distance_scaling_matrices = []
            for parameter_name in ('corrDistDS', 'corrDistASD', 'corrDistASA',
                'corrDistSF', 'corrDistK', 'corrDistZSA', 'corrDistZSD'):
                
                if parameter_name == 'corrDistDS':
                    parameter_value_los = DS_los
                    parameter_value_nlos = DS_nlos
                elif parameter_name == 'corrDistASD':
                    parameter_value_los = ASD_los
                    parameter_value_nlos = ASD_nlos
                elif parameter_name == 'corrDistASA':
                    parameter_value_los = ASA_los
                    parameter_value_nlos = ASA_nlos
                elif parameter_name == 'corrDistSF':
                    parameter_value_los = SF_los
                    parameter_value_nlos = SF_nlos
                elif parameter_name == 'corrDistK':
                    parameter_value_los = K_los
                    parameter_value_nlos = K_nlos
                elif parameter_name == 'corrDistZSA':
                    parameter_value_los = ZSA_los
                    parameter_value_nlos = ZSA_nlos
                elif parameter_name == 'corrDistZSD':
                    parameter_value_los = ZSD_los
                    parameter_value_nlos = ZSD_nlos

                filtering_matrix = tf.eye(channel_model._scenario.num_ut,
                    channel_model._scenario.num_ut, batch_shape=[channel_model._scenario.batch_size,
                    channel_model._scenario.num_bs], dtype=channel_model._scenario.rdtype)
                
                distance_scaling_matrix = tf.where(channel_model._scenario.los, parameter_value_los,parameter_value_nlos)
                #distance_scaling_matrix = channel_model._scenario.get_param(parameter_name)
                distance_scaling_matrix = tf.tile(tf.expand_dims(
                    distance_scaling_matrix, axis=3),
                    [1, 1, 1, channel_model._scenario.num_ut])
    
                epsilon = 1e-12
                distance_scaling_matrix = -1. / (distance_scaling_matrix + epsilon)
                # LoS
                filtering_matrix = tf.where(los_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # NLoS
                filtering_matrix = tf.where(nlos_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # indoor
                filtering_matrix = tf.where(o2i_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # Stacking
                filtering_matrices.append(filtering_matrix)
                distance_scaling_matrices.append(distance_scaling_matrix)
            filtering_matrices = tf.stack(filtering_matrices, axis=2)
            distance_scaling_matrices = tf.stack(distance_scaling_matrices, axis=2)
            ut_dist_2d = channel_model._scenario.matrix_ut_distance_2d
            # Adding a dimension for broadcasting with BS
            ut_dist_2d = tf.expand_dims(tf.expand_dims(ut_dist_2d, axis=1), axis=2)
            
            spatial_lsp_correlation = (tf.math.exp(ut_dist_2d*distance_scaling_matrices)*filtering_matrices)  
            spatial_lsp_correlation = tf.linalg.cholesky(spatial_lsp_correlation)

            difference = channel_model._lsp_sampler._spatial_lsp_correlation_matrix_sqrt - spatial_lsp_correlation
            difference = tf.abs(difference) < 1e-6

            assert tf.reduce_all(difference)

class Test_DUR(unittest.TestCase):
# Values taken from Table 6.7.2-2a: Channel model parameters for Urban Scenario (NLOS) at S band and 
# Table 6.7.2-1a: Channel model parameters for Dense Urban Scenario (LOS) at S band
    def test_s_band_dl(self):
        
        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        num_ut = 100

        DS_los = 30.0
        ASD_los = 18.0
        ASA_los = 15.0
        SF_los = 37.0
        K_los = 12.0
        ZSA_los = 15.0
        ZSD_los = 15.0

        DS_nlos = 40.0
        ASD_nlos = 50.0
        ASA_nlos = 50.0
        SF_nlos = 50.0
        K_nlos = 1.0
        ZSA_nlos = 50.0
        ZSD_nlos = 50.0

        for elevation_angle in [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0]:

            channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
            
            topology = utils.gen_single_sector_topology(batch_size=100, num_ut=num_ut, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
            channel_model.set_topology(*topology)
            indoor = tf.tile(tf.expand_dims(channel_model._scenario.indoor, axis=1),
                         [1, channel_model._scenario.num_bs, 1])
            # LoS
            los_ut = channel_model._scenario.los
            los_pair_bool = tf.logical_and(tf.expand_dims(los_ut, axis=3),
                                        tf.expand_dims(los_ut, axis=2))
            # NLoS
            nlos_ut = tf.logical_and(tf.logical_not(channel_model._scenario.los),
                                    tf.logical_not(indoor))
            nlos_pair_bool = tf.logical_and(tf.expand_dims(nlos_ut, axis=3),
                                            tf.expand_dims(nlos_ut, axis=2))
            # O2I
            o2i_pair_bool = tf.logical_and(tf.expand_dims(indoor, axis=3),
                                        tf.expand_dims(indoor, axis=2))

            # Stacking the correlation matrix
            # One correlation matrix per LSP
            filtering_matrices = []
            distance_scaling_matrices = []
            for parameter_name in ('corrDistDS', 'corrDistASD', 'corrDistASA',
                'corrDistSF', 'corrDistK', 'corrDistZSA', 'corrDistZSD'):
                
                if parameter_name == 'corrDistDS':
                    parameter_value_los = DS_los
                    parameter_value_nlos = DS_nlos
                elif parameter_name == 'corrDistASD':
                    parameter_value_los = ASD_los
                    parameter_value_nlos = ASD_nlos
                elif parameter_name == 'corrDistASA':
                    parameter_value_los = ASA_los
                    parameter_value_nlos = ASA_nlos
                elif parameter_name == 'corrDistSF':
                    parameter_value_los = SF_los
                    parameter_value_nlos = SF_nlos
                elif parameter_name == 'corrDistK':
                    parameter_value_los = K_los
                    parameter_value_nlos = K_nlos
                elif parameter_name == 'corrDistZSA':
                    parameter_value_los = ZSA_los
                    parameter_value_nlos = ZSA_nlos
                elif parameter_name == 'corrDistZSD':
                    parameter_value_los = ZSD_los
                    parameter_value_nlos = ZSD_nlos

                filtering_matrix = tf.eye(channel_model._scenario.num_ut,
                    channel_model._scenario.num_ut, batch_shape=[channel_model._scenario.batch_size,
                    channel_model._scenario.num_bs], dtype=channel_model._scenario.rdtype)
                
                distance_scaling_matrix = tf.where(channel_model._scenario.los, parameter_value_los,parameter_value_nlos)
                #distance_scaling_matrix = channel_model._scenario.get_param(parameter_name)
                distance_scaling_matrix = tf.tile(tf.expand_dims(
                    distance_scaling_matrix, axis=3),
                    [1, 1, 1, channel_model._scenario.num_ut])
    
                epsilon = 1e-12
                distance_scaling_matrix = -1. / (distance_scaling_matrix + epsilon)
                # LoS
                filtering_matrix = tf.where(los_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # NLoS
                filtering_matrix = tf.where(nlos_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # indoor
                filtering_matrix = tf.where(o2i_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # Stacking
                filtering_matrices.append(filtering_matrix)
                distance_scaling_matrices.append(distance_scaling_matrix)
            filtering_matrices = tf.stack(filtering_matrices, axis=2)
            distance_scaling_matrices = tf.stack(distance_scaling_matrices, axis=2)
            ut_dist_2d = channel_model._scenario.matrix_ut_distance_2d
            # Adding a dimension for broadcasting with BS
            ut_dist_2d = tf.expand_dims(tf.expand_dims(ut_dist_2d, axis=1), axis=2)
            
            spatial_lsp_correlation = (tf.math.exp(ut_dist_2d*distance_scaling_matrices)*filtering_matrices)  
            spatial_lsp_correlation = tf.linalg.cholesky(spatial_lsp_correlation)

            difference = channel_model._lsp_sampler._spatial_lsp_correlation_matrix_sqrt - spatial_lsp_correlation

            difference = tf.abs(difference) < 1e-6

            assert tf.reduce_all(difference)


            


    def test_s_band_ul(self):

        direction = "uplink"
        scenario = "dur"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        num_ut = 100

        DS_los = 30.0
        ASD_los = 18.0
        ASA_los = 15.0
        SF_los = 37.0
        K_los = 12.0
        ZSA_los = 15.0
        ZSD_los = 15.0

        DS_nlos = 40.0
        ASD_nlos = 50.0
        ASA_nlos = 50.0
        SF_nlos = 50.0
        K_nlos = 1.0
        ZSA_nlos = 50.0
        ZSD_nlos = 50.0

        for elevation_angle in [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0]:

            channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
            
            topology = utils.gen_single_sector_topology(batch_size=100, num_ut=num_ut, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
            channel_model.set_topology(*topology)
            indoor = tf.tile(tf.expand_dims(channel_model._scenario.indoor, axis=1),
                         [1, channel_model._scenario.num_bs, 1])
            # LoS
            los_ut = channel_model._scenario.los
            los_pair_bool = tf.logical_and(tf.expand_dims(los_ut, axis=3),
                                        tf.expand_dims(los_ut, axis=2))
            # NLoS
            nlos_ut = tf.logical_and(tf.logical_not(channel_model._scenario.los),
                                    tf.logical_not(indoor))
            nlos_pair_bool = tf.logical_and(tf.expand_dims(nlos_ut, axis=3),
                                            tf.expand_dims(nlos_ut, axis=2))
            # O2I
            o2i_pair_bool = tf.logical_and(tf.expand_dims(indoor, axis=3),
                                        tf.expand_dims(indoor, axis=2))

            # Stacking the correlation matrix
            # One correlation matrix per LSP
            filtering_matrices = []
            distance_scaling_matrices = []
            for parameter_name in ('corrDistDS', 'corrDistASD', 'corrDistASA',
                'corrDistSF', 'corrDistK', 'corrDistZSA', 'corrDistZSD'):
                
                if parameter_name == 'corrDistDS':
                    parameter_value_los = DS_los
                    parameter_value_nlos = DS_nlos
                elif parameter_name == 'corrDistASD':
                    parameter_value_los = ASD_los
                    parameter_value_nlos = ASD_nlos
                elif parameter_name == 'corrDistASA':
                    parameter_value_los = ASA_los
                    parameter_value_nlos = ASA_nlos
                elif parameter_name == 'corrDistSF':
                    parameter_value_los = SF_los
                    parameter_value_nlos = SF_nlos
                elif parameter_name == 'corrDistK':
                    parameter_value_los = K_los
                    parameter_value_nlos = K_nlos
                elif parameter_name == 'corrDistZSA':
                    parameter_value_los = ZSA_los
                    parameter_value_nlos = ZSA_nlos
                elif parameter_name == 'corrDistZSD':
                    parameter_value_los = ZSD_los
                    parameter_value_nlos = ZSD_nlos

                filtering_matrix = tf.eye(channel_model._scenario.num_ut,
                    channel_model._scenario.num_ut, batch_shape=[channel_model._scenario.batch_size,
                    channel_model._scenario.num_bs], dtype=channel_model._scenario.rdtype)
                
                distance_scaling_matrix = tf.where(channel_model._scenario.los, parameter_value_los,parameter_value_nlos)
                #distance_scaling_matrix = channel_model._scenario.get_param(parameter_name)
                distance_scaling_matrix = tf.tile(tf.expand_dims(
                    distance_scaling_matrix, axis=3),
                    [1, 1, 1, channel_model._scenario.num_ut])
    
                epsilon = 1e-12
                distance_scaling_matrix = -1. / (distance_scaling_matrix + epsilon)
                # LoS
                filtering_matrix = tf.where(los_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # NLoS
                filtering_matrix = tf.where(nlos_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # indoor
                filtering_matrix = tf.where(o2i_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # Stacking
                filtering_matrices.append(filtering_matrix)
                distance_scaling_matrices.append(distance_scaling_matrix)
            filtering_matrices = tf.stack(filtering_matrices, axis=2)
            distance_scaling_matrices = tf.stack(distance_scaling_matrices, axis=2)
            ut_dist_2d = channel_model._scenario.matrix_ut_distance_2d
            # Adding a dimension for broadcasting with BS
            ut_dist_2d = tf.expand_dims(tf.expand_dims(ut_dist_2d, axis=1), axis=2)
            
            spatial_lsp_correlation = (tf.math.exp(ut_dist_2d*distance_scaling_matrices)*filtering_matrices)  
            spatial_lsp_correlation = tf.linalg.cholesky(spatial_lsp_correlation)

            difference = channel_model._lsp_sampler._spatial_lsp_correlation_matrix_sqrt - spatial_lsp_correlation
            difference = tf.abs(difference) < 1e-6

            assert tf.reduce_all(difference)



    # Values from tables Table 6.7.2-2b: Channel model parameters for Dense Urban Scenario (NLOS) in Ka band
    # and Table 6.7.2-5b: Channel model parameters for Suburban Scenario (LOS) in Ka band

    def test_ka_band_dl(self):
        direction = "downlink"
        scenario = "dur"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        num_ut = 100

        DS_los = 30.0
        ASD_los = 18.0
        ASA_los = 15.0
        SF_los = 37.0
        K_los = 12.0
        ZSA_los = 15.0
        ZSD_los = 15.0

        DS_nlos = 40.0
        ASD_nlos = 50.0
        ASA_nlos = 50.0
        SF_nlos = 50.0
        K_nlos = 1.0
        ZSA_nlos = 50.0
        ZSD_nlos = 50.0

        for elevation_angle in [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0]:

            channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
            
            topology = utils.gen_single_sector_topology(batch_size=100, num_ut=num_ut, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
            channel_model.set_topology(*topology)
            indoor = tf.tile(tf.expand_dims(channel_model._scenario.indoor, axis=1),
                         [1, channel_model._scenario.num_bs, 1])
            # LoS
            los_ut = channel_model._scenario.los
            los_pair_bool = tf.logical_and(tf.expand_dims(los_ut, axis=3),
                                        tf.expand_dims(los_ut, axis=2))
            # NLoS
            nlos_ut = tf.logical_and(tf.logical_not(channel_model._scenario.los),
                                    tf.logical_not(indoor))
            nlos_pair_bool = tf.logical_and(tf.expand_dims(nlos_ut, axis=3),
                                            tf.expand_dims(nlos_ut, axis=2))
            # O2I
            o2i_pair_bool = tf.logical_and(tf.expand_dims(indoor, axis=3),
                                        tf.expand_dims(indoor, axis=2))

            # Stacking the correlation matrix
            # One correlation matrix per LSP
            filtering_matrices = []
            distance_scaling_matrices = []
            for parameter_name in ('corrDistDS', 'corrDistASD', 'corrDistASA',
                'corrDistSF', 'corrDistK', 'corrDistZSA', 'corrDistZSD'):
                
                if parameter_name == 'corrDistDS':
                    parameter_value_los = DS_los
                    parameter_value_nlos = DS_nlos
                elif parameter_name == 'corrDistASD':
                    parameter_value_los = ASD_los
                    parameter_value_nlos = ASD_nlos
                elif parameter_name == 'corrDistASA':
                    parameter_value_los = ASA_los
                    parameter_value_nlos = ASA_nlos
                elif parameter_name == 'corrDistSF':
                    parameter_value_los = SF_los
                    parameter_value_nlos = SF_nlos
                elif parameter_name == 'corrDistK':
                    parameter_value_los = K_los
                    parameter_value_nlos = K_nlos
                elif parameter_name == 'corrDistZSA':
                    parameter_value_los = ZSA_los
                    parameter_value_nlos = ZSA_nlos
                elif parameter_name == 'corrDistZSD':
                    parameter_value_los = ZSD_los
                    parameter_value_nlos = ZSD_nlos

                filtering_matrix = tf.eye(channel_model._scenario.num_ut,
                    channel_model._scenario.num_ut, batch_shape=[channel_model._scenario.batch_size,
                    channel_model._scenario.num_bs], dtype=channel_model._scenario.rdtype)
                
                distance_scaling_matrix = tf.where(channel_model._scenario.los, parameter_value_los,parameter_value_nlos)
                #distance_scaling_matrix = channel_model._scenario.get_param(parameter_name)
                distance_scaling_matrix = tf.tile(tf.expand_dims(
                    distance_scaling_matrix, axis=3),
                    [1, 1, 1, channel_model._scenario.num_ut])
    
                epsilon = 1e-12
                distance_scaling_matrix = -1. / (distance_scaling_matrix + epsilon)
                # LoS
                filtering_matrix = tf.where(los_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # NLoS
                filtering_matrix = tf.where(nlos_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # indoor
                filtering_matrix = tf.where(o2i_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # Stacking
                filtering_matrices.append(filtering_matrix)
                distance_scaling_matrices.append(distance_scaling_matrix)
            filtering_matrices = tf.stack(filtering_matrices, axis=2)
            distance_scaling_matrices = tf.stack(distance_scaling_matrices, axis=2)
            ut_dist_2d = channel_model._scenario.matrix_ut_distance_2d
            # Adding a dimension for broadcasting with BS
            ut_dist_2d = tf.expand_dims(tf.expand_dims(ut_dist_2d, axis=1), axis=2)
            
            spatial_lsp_correlation = (tf.math.exp(ut_dist_2d*distance_scaling_matrices)*filtering_matrices)  
            spatial_lsp_correlation = tf.linalg.cholesky(spatial_lsp_correlation)

            difference = channel_model._lsp_sampler._spatial_lsp_correlation_matrix_sqrt - spatial_lsp_correlation

            difference = tf.abs(difference) < 1e-6

            assert tf.reduce_all(difference)



    def test_ka_band_ul(self):
        direction = "uplink"
        scenario = "dur"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        num_ut = 100

        DS_los = 30.0
        ASD_los = 18.0
        ASA_los = 15.0
        SF_los = 37.0
        K_los = 12.0
        ZSA_los = 15.0
        ZSD_los = 15.0

        DS_nlos = 40.0
        ASD_nlos = 50.0
        ASA_nlos = 50.0
        SF_nlos = 50.0
        K_nlos = 1.0
        ZSA_nlos = 50.0
        ZSD_nlos = 50.0

        for elevation_angle in [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0]:

            channel_model = DenseUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
            
            topology = utils.gen_single_sector_topology(batch_size=100, num_ut=num_ut, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
            channel_model.set_topology(*topology)
            indoor = tf.tile(tf.expand_dims(channel_model._scenario.indoor, axis=1),
                         [1, channel_model._scenario.num_bs, 1])
            # LoS
            los_ut = channel_model._scenario.los
            los_pair_bool = tf.logical_and(tf.expand_dims(los_ut, axis=3),
                                        tf.expand_dims(los_ut, axis=2))
            # NLoS
            nlos_ut = tf.logical_and(tf.logical_not(channel_model._scenario.los),
                                    tf.logical_not(indoor))
            nlos_pair_bool = tf.logical_and(tf.expand_dims(nlos_ut, axis=3),
                                            tf.expand_dims(nlos_ut, axis=2))
            # O2I
            o2i_pair_bool = tf.logical_and(tf.expand_dims(indoor, axis=3),
                                        tf.expand_dims(indoor, axis=2))

            # Stacking the correlation matrix
            # One correlation matrix per LSP
            filtering_matrices = []
            distance_scaling_matrices = []
            for parameter_name in ('corrDistDS', 'corrDistASD', 'corrDistASA',
                'corrDistSF', 'corrDistK', 'corrDistZSA', 'corrDistZSD'):
                
                if parameter_name == 'corrDistDS':
                    parameter_value_los = DS_los
                    parameter_value_nlos = DS_nlos
                elif parameter_name == 'corrDistASD':
                    parameter_value_los = ASD_los
                    parameter_value_nlos = ASD_nlos
                elif parameter_name == 'corrDistASA':
                    parameter_value_los = ASA_los
                    parameter_value_nlos = ASA_nlos
                elif parameter_name == 'corrDistSF':
                    parameter_value_los = SF_los
                    parameter_value_nlos = SF_nlos
                elif parameter_name == 'corrDistK':
                    parameter_value_los = K_los
                    parameter_value_nlos = K_nlos
                elif parameter_name == 'corrDistZSA':
                    parameter_value_los = ZSA_los
                    parameter_value_nlos = ZSA_nlos
                elif parameter_name == 'corrDistZSD':
                    parameter_value_los = ZSD_los
                    parameter_value_nlos = ZSD_nlos

                filtering_matrix = tf.eye(channel_model._scenario.num_ut,
                    channel_model._scenario.num_ut, batch_shape=[channel_model._scenario.batch_size,
                    channel_model._scenario.num_bs], dtype=channel_model._scenario.rdtype)
                
                distance_scaling_matrix = tf.where(channel_model._scenario.los, parameter_value_los,parameter_value_nlos)
                #distance_scaling_matrix = channel_model._scenario.get_param(parameter_name)
                distance_scaling_matrix = tf.tile(tf.expand_dims(
                    distance_scaling_matrix, axis=3),
                    [1, 1, 1, channel_model._scenario.num_ut])
    
                epsilon = 1e-12
                distance_scaling_matrix = -1. / (distance_scaling_matrix + epsilon)
                # LoS
                filtering_matrix = tf.where(los_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # NLoS
                filtering_matrix = tf.where(nlos_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # indoor
                filtering_matrix = tf.where(o2i_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # Stacking
                filtering_matrices.append(filtering_matrix)
                distance_scaling_matrices.append(distance_scaling_matrix)
            filtering_matrices = tf.stack(filtering_matrices, axis=2)
            distance_scaling_matrices = tf.stack(distance_scaling_matrices, axis=2)
            ut_dist_2d = channel_model._scenario.matrix_ut_distance_2d
            # Adding a dimension for broadcasting with BS
            ut_dist_2d = tf.expand_dims(tf.expand_dims(ut_dist_2d, axis=1), axis=2)
            
            spatial_lsp_correlation = (tf.math.exp(ut_dist_2d*distance_scaling_matrices)*filtering_matrices)  
            spatial_lsp_correlation = tf.linalg.cholesky(spatial_lsp_correlation)

            difference = channel_model._lsp_sampler._spatial_lsp_correlation_matrix_sqrt - spatial_lsp_correlation
            difference = tf.abs(difference) < 1e-6

            assert tf.reduce_all(difference)



class Test_SUR(unittest.TestCase):
# Values taken from Table 6.7.2-4a: Channel model parameters for Urban Scenario (NLOS) at S band and 
# Table 6.7.2-3a: Channel model parameters for Urban Scenario (LOS) at S band
    def test_s_band_dl(self):
        
        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 2.2e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        num_ut = 100

        DS_los = 30.0
        ASD_los = 18.0
        ASA_los = 15.0
        SF_los = 37.0
        K_los = 12.0
        ZSA_los = 15.0
        ZSD_los = 15.0

        DS_nlos = 40.0
        ASD_nlos = 50.0
        ASA_nlos = 50.0
        SF_nlos = 50.0
        K_nlos = 1.0
        ZSA_nlos = 50.0
        ZSD_nlos = 50.0

        for elevation_angle in [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0]:

            channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
            
            topology = utils.gen_single_sector_topology(batch_size=100, num_ut=num_ut, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
            channel_model.set_topology(*topology)
            indoor = tf.tile(tf.expand_dims(channel_model._scenario.indoor, axis=1),
                         [1, channel_model._scenario.num_bs, 1])
            # LoS
            los_ut = channel_model._scenario.los
            los_pair_bool = tf.logical_and(tf.expand_dims(los_ut, axis=3),
                                        tf.expand_dims(los_ut, axis=2))
            # NLoS
            nlos_ut = tf.logical_and(tf.logical_not(channel_model._scenario.los),
                                    tf.logical_not(indoor))
            nlos_pair_bool = tf.logical_and(tf.expand_dims(nlos_ut, axis=3),
                                            tf.expand_dims(nlos_ut, axis=2))
            # O2I
            o2i_pair_bool = tf.logical_and(tf.expand_dims(indoor, axis=3),
                                        tf.expand_dims(indoor, axis=2))

            # Stacking the correlation matrix
            # One correlation matrix per LSP
            filtering_matrices = []
            distance_scaling_matrices = []
            for parameter_name in ('corrDistDS', 'corrDistASD', 'corrDistASA',
                'corrDistSF', 'corrDistK', 'corrDistZSA', 'corrDistZSD'):
                
                if parameter_name == 'corrDistDS':
                    parameter_value_los = DS_los
                    parameter_value_nlos = DS_nlos
                elif parameter_name == 'corrDistASD':
                    parameter_value_los = ASD_los
                    parameter_value_nlos = ASD_nlos
                elif parameter_name == 'corrDistASA':
                    parameter_value_los = ASA_los
                    parameter_value_nlos = ASA_nlos
                elif parameter_name == 'corrDistSF':
                    parameter_value_los = SF_los
                    parameter_value_nlos = SF_nlos
                elif parameter_name == 'corrDistK':
                    parameter_value_los = K_los
                    parameter_value_nlos = K_nlos
                elif parameter_name == 'corrDistZSA':
                    parameter_value_los = ZSA_los
                    parameter_value_nlos = ZSA_nlos
                elif parameter_name == 'corrDistZSD':
                    parameter_value_los = ZSD_los
                    parameter_value_nlos = ZSD_nlos

                filtering_matrix = tf.eye(channel_model._scenario.num_ut,
                    channel_model._scenario.num_ut, batch_shape=[channel_model._scenario.batch_size,
                    channel_model._scenario.num_bs], dtype=channel_model._scenario.rdtype)
                
                distance_scaling_matrix = tf.where(channel_model._scenario.los, parameter_value_los,parameter_value_nlos)
                #distance_scaling_matrix = channel_model._scenario.get_param(parameter_name)
                distance_scaling_matrix = tf.tile(tf.expand_dims(
                    distance_scaling_matrix, axis=3),
                    [1, 1, 1, channel_model._scenario.num_ut])
    
                epsilon = 1e-12
                distance_scaling_matrix = -1. / (distance_scaling_matrix + epsilon)
                # LoS
                filtering_matrix = tf.where(los_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # NLoS
                filtering_matrix = tf.where(nlos_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # indoor
                filtering_matrix = tf.where(o2i_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # Stacking
                filtering_matrices.append(filtering_matrix)
                distance_scaling_matrices.append(distance_scaling_matrix)
            filtering_matrices = tf.stack(filtering_matrices, axis=2)
            distance_scaling_matrices = tf.stack(distance_scaling_matrices, axis=2)
            ut_dist_2d = channel_model._scenario.matrix_ut_distance_2d
            # Adding a dimension for broadcasting with BS
            ut_dist_2d = tf.expand_dims(tf.expand_dims(ut_dist_2d, axis=1), axis=2)
            
            spatial_lsp_correlation = (tf.math.exp(ut_dist_2d*distance_scaling_matrices)*filtering_matrices)  
            spatial_lsp_correlation = tf.linalg.cholesky(spatial_lsp_correlation)

            difference = channel_model._lsp_sampler._spatial_lsp_correlation_matrix_sqrt - spatial_lsp_correlation

            difference = tf.abs(difference) < 1e-6

            assert tf.reduce_all(difference)


            


    def test_s_band_ul(self):

        direction = "uplink"
        scenario = "sur"
        carrier_frequency = 2.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        num_ut = 100

        DS_los = 30.0
        ASD_los = 18.0
        ASA_los = 15.0
        SF_los = 37.0
        K_los = 12.0
        ZSA_los = 15.0
        ZSD_los = 15.0

        DS_nlos = 40.0
        ASD_nlos = 50.0
        ASA_nlos = 50.0
        SF_nlos = 50.0
        K_nlos = 1.0
        ZSA_nlos = 50.0
        ZSD_nlos = 50.0

        for elevation_angle in [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0]:

            channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
            
            topology = utils.gen_single_sector_topology(batch_size=100, num_ut=num_ut, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
            channel_model.set_topology(*topology)
            indoor = tf.tile(tf.expand_dims(channel_model._scenario.indoor, axis=1),
                         [1, channel_model._scenario.num_bs, 1])
            # LoS
            los_ut = channel_model._scenario.los
            los_pair_bool = tf.logical_and(tf.expand_dims(los_ut, axis=3),
                                        tf.expand_dims(los_ut, axis=2))
            # NLoS
            nlos_ut = tf.logical_and(tf.logical_not(channel_model._scenario.los),
                                    tf.logical_not(indoor))
            nlos_pair_bool = tf.logical_and(tf.expand_dims(nlos_ut, axis=3),
                                            tf.expand_dims(nlos_ut, axis=2))
            # O2I
            o2i_pair_bool = tf.logical_and(tf.expand_dims(indoor, axis=3),
                                        tf.expand_dims(indoor, axis=2))

            # Stacking the correlation matrix
            # One correlation matrix per LSP
            filtering_matrices = []
            distance_scaling_matrices = []
            for parameter_name in ('corrDistDS', 'corrDistASD', 'corrDistASA',
                'corrDistSF', 'corrDistK', 'corrDistZSA', 'corrDistZSD'):
                
                if parameter_name == 'corrDistDS':
                    parameter_value_los = DS_los
                    parameter_value_nlos = DS_nlos
                elif parameter_name == 'corrDistASD':
                    parameter_value_los = ASD_los
                    parameter_value_nlos = ASD_nlos
                elif parameter_name == 'corrDistASA':
                    parameter_value_los = ASA_los
                    parameter_value_nlos = ASA_nlos
                elif parameter_name == 'corrDistSF':
                    parameter_value_los = SF_los
                    parameter_value_nlos = SF_nlos
                elif parameter_name == 'corrDistK':
                    parameter_value_los = K_los
                    parameter_value_nlos = K_nlos
                elif parameter_name == 'corrDistZSA':
                    parameter_value_los = ZSA_los
                    parameter_value_nlos = ZSA_nlos
                elif parameter_name == 'corrDistZSD':
                    parameter_value_los = ZSD_los
                    parameter_value_nlos = ZSD_nlos

                filtering_matrix = tf.eye(channel_model._scenario.num_ut,
                    channel_model._scenario.num_ut, batch_shape=[channel_model._scenario.batch_size,
                    channel_model._scenario.num_bs], dtype=channel_model._scenario.rdtype)
                
                distance_scaling_matrix = tf.where(channel_model._scenario.los, parameter_value_los,parameter_value_nlos)
                #distance_scaling_matrix = channel_model._scenario.get_param(parameter_name)
                distance_scaling_matrix = tf.tile(tf.expand_dims(
                    distance_scaling_matrix, axis=3),
                    [1, 1, 1, channel_model._scenario.num_ut])
    
                epsilon = 1e-12
                distance_scaling_matrix = -1. / (distance_scaling_matrix + epsilon)
                # LoS
                filtering_matrix = tf.where(los_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # NLoS
                filtering_matrix = tf.where(nlos_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # indoor
                filtering_matrix = tf.where(o2i_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # Stacking
                filtering_matrices.append(filtering_matrix)
                distance_scaling_matrices.append(distance_scaling_matrix)
            filtering_matrices = tf.stack(filtering_matrices, axis=2)
            distance_scaling_matrices = tf.stack(distance_scaling_matrices, axis=2)
            ut_dist_2d = channel_model._scenario.matrix_ut_distance_2d
            # Adding a dimension for broadcasting with BS
            ut_dist_2d = tf.expand_dims(tf.expand_dims(ut_dist_2d, axis=1), axis=2)
            
            spatial_lsp_correlation = (tf.math.exp(ut_dist_2d*distance_scaling_matrices)*filtering_matrices)  
            spatial_lsp_correlation = tf.linalg.cholesky(spatial_lsp_correlation)

            difference = channel_model._lsp_sampler._spatial_lsp_correlation_matrix_sqrt - spatial_lsp_correlation
            difference = tf.abs(difference) < 1e-6

            assert tf.reduce_all(difference)



    # Values from tables Table 6.7.2-2b: Channel model parameters for Dense Urban Scenario (NLOS) in Ka band
    # and Table 6.7.2-5b: Channel model parameters for Suburban Scenario (LOS) in Ka band

    def test_ka_band_dl(self):
        direction = "downlink"
        scenario = "sur"
        carrier_frequency = 20.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)
        num_ut = 100

        DS_los = 30.0
        ASD_los = 18.0
        ASA_los = 15.0
        SF_los = 37.0
        K_los = 12.0
        ZSA_los = 15.0
        ZSD_los = 15.0

        DS_nlos = 40.0
        ASD_nlos = 50.0
        ASA_nlos = 50.0
        SF_nlos = 50.0
        K_nlos = 1.0
        ZSA_nlos = 50.0
        ZSD_nlos = 50.0

        for elevation_angle in [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0]:

            channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
            
            topology = utils.gen_single_sector_topology(batch_size=100, num_ut=num_ut, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
            channel_model.set_topology(*topology)
            indoor = tf.tile(tf.expand_dims(channel_model._scenario.indoor, axis=1),
                         [1, channel_model._scenario.num_bs, 1])
            # LoS
            los_ut = channel_model._scenario.los
            los_pair_bool = tf.logical_and(tf.expand_dims(los_ut, axis=3),
                                        tf.expand_dims(los_ut, axis=2))
            # NLoS
            nlos_ut = tf.logical_and(tf.logical_not(channel_model._scenario.los),
                                    tf.logical_not(indoor))
            nlos_pair_bool = tf.logical_and(tf.expand_dims(nlos_ut, axis=3),
                                            tf.expand_dims(nlos_ut, axis=2))
            # O2I
            o2i_pair_bool = tf.logical_and(tf.expand_dims(indoor, axis=3),
                                        tf.expand_dims(indoor, axis=2))

            # Stacking the correlation matrix
            # One correlation matrix per LSP
            filtering_matrices = []
            distance_scaling_matrices = []
            for parameter_name in ('corrDistDS', 'corrDistASD', 'corrDistASA',
                'corrDistSF', 'corrDistK', 'corrDistZSA', 'corrDistZSD'):
                
                if parameter_name == 'corrDistDS':
                    parameter_value_los = DS_los
                    parameter_value_nlos = DS_nlos
                elif parameter_name == 'corrDistASD':
                    parameter_value_los = ASD_los
                    parameter_value_nlos = ASD_nlos
                elif parameter_name == 'corrDistASA':
                    parameter_value_los = ASA_los
                    parameter_value_nlos = ASA_nlos
                elif parameter_name == 'corrDistSF':
                    parameter_value_los = SF_los
                    parameter_value_nlos = SF_nlos
                elif parameter_name == 'corrDistK':
                    parameter_value_los = K_los
                    parameter_value_nlos = K_nlos
                elif parameter_name == 'corrDistZSA':
                    parameter_value_los = ZSA_los
                    parameter_value_nlos = ZSA_nlos
                elif parameter_name == 'corrDistZSD':
                    parameter_value_los = ZSD_los
                    parameter_value_nlos = ZSD_nlos

                filtering_matrix = tf.eye(channel_model._scenario.num_ut,
                    channel_model._scenario.num_ut, batch_shape=[channel_model._scenario.batch_size,
                    channel_model._scenario.num_bs], dtype=channel_model._scenario.rdtype)
                
                distance_scaling_matrix = tf.where(channel_model._scenario.los, parameter_value_los,parameter_value_nlos)
                #distance_scaling_matrix = channel_model._scenario.get_param(parameter_name)
                distance_scaling_matrix = tf.tile(tf.expand_dims(
                    distance_scaling_matrix, axis=3),
                    [1, 1, 1, channel_model._scenario.num_ut])
    
                epsilon = 1e-12
                distance_scaling_matrix = -1. / (distance_scaling_matrix + epsilon)
                # LoS
                filtering_matrix = tf.where(los_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # NLoS
                filtering_matrix = tf.where(nlos_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # indoor
                filtering_matrix = tf.where(o2i_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # Stacking
                filtering_matrices.append(filtering_matrix)
                distance_scaling_matrices.append(distance_scaling_matrix)
            filtering_matrices = tf.stack(filtering_matrices, axis=2)
            distance_scaling_matrices = tf.stack(distance_scaling_matrices, axis=2)
            ut_dist_2d = channel_model._scenario.matrix_ut_distance_2d
            # Adding a dimension for broadcasting with BS
            ut_dist_2d = tf.expand_dims(tf.expand_dims(ut_dist_2d, axis=1), axis=2)
            
            spatial_lsp_correlation = (tf.math.exp(ut_dist_2d*distance_scaling_matrices)*filtering_matrices)  
            spatial_lsp_correlation = tf.linalg.cholesky(spatial_lsp_correlation)

            difference = channel_model._lsp_sampler._spatial_lsp_correlation_matrix_sqrt - spatial_lsp_correlation

            difference = tf.abs(difference) < 1e-6

            assert tf.reduce_all(difference)




    def test_ka_band_ul(self):
        direction = "uplink"
        scenario = "sur"
        carrier_frequency = 30.0e9
        ut_array = create_ut_ant(carrier_frequency)
        bs_array = create_bs_ant(carrier_frequency)

        num_ut = 100

        DS_los = 30.0
        ASD_los = 18.0
        ASA_los = 15.0
        SF_los = 37.0
        K_los = 12.0
        ZSA_los = 15.0
        ZSD_los = 15.0

        DS_nlos = 40.0
        ASD_nlos = 50.0
        ASA_nlos = 50.0
        SF_nlos = 50.0
        K_nlos = 1.0
        ZSA_nlos = 50.0
        ZSD_nlos = 50.0

        for elevation_angle in [10.0,20.0,30.0,40.0,50.0,60.0,70.0,80.0,90.0]:

            channel_model = SubUrban(carrier_frequency=carrier_frequency,
                                                ut_array=ut_array,
                                                bs_array=bs_array,
                                                direction=direction,
                                                elevation_angle=elevation_angle,
                                                enable_pathloss=True,
                                                enable_shadow_fading=True)
            
            topology = utils.gen_single_sector_topology(batch_size=100, num_ut=num_ut, scenario=scenario, elevation_angle=elevation_angle, bs_height=600000.0)
            channel_model.set_topology(*topology)
            indoor = tf.tile(tf.expand_dims(channel_model._scenario.indoor, axis=1),
                         [1, channel_model._scenario.num_bs, 1])
            # LoS
            los_ut = channel_model._scenario.los
            los_pair_bool = tf.logical_and(tf.expand_dims(los_ut, axis=3),
                                        tf.expand_dims(los_ut, axis=2))
            # NLoS
            nlos_ut = tf.logical_and(tf.logical_not(channel_model._scenario.los),
                                    tf.logical_not(indoor))
            nlos_pair_bool = tf.logical_and(tf.expand_dims(nlos_ut, axis=3),
                                            tf.expand_dims(nlos_ut, axis=2))
            # O2I
            o2i_pair_bool = tf.logical_and(tf.expand_dims(indoor, axis=3),
                                        tf.expand_dims(indoor, axis=2))

            # Stacking the correlation matrix
            # One correlation matrix per LSP
            filtering_matrices = []
            distance_scaling_matrices = []
            for parameter_name in ('corrDistDS', 'corrDistASD', 'corrDistASA',
                'corrDistSF', 'corrDistK', 'corrDistZSA', 'corrDistZSD'):
                
                if parameter_name == 'corrDistDS':
                    parameter_value_los = DS_los
                    parameter_value_nlos = DS_nlos
                elif parameter_name == 'corrDistASD':
                    parameter_value_los = ASD_los
                    parameter_value_nlos = ASD_nlos
                elif parameter_name == 'corrDistASA':
                    parameter_value_los = ASA_los
                    parameter_value_nlos = ASA_nlos
                elif parameter_name == 'corrDistSF':
                    parameter_value_los = SF_los
                    parameter_value_nlos = SF_nlos
                elif parameter_name == 'corrDistK':
                    parameter_value_los = K_los
                    parameter_value_nlos = K_nlos
                elif parameter_name == 'corrDistZSA':
                    parameter_value_los = ZSA_los
                    parameter_value_nlos = ZSA_nlos
                elif parameter_name == 'corrDistZSD':
                    parameter_value_los = ZSD_los
                    parameter_value_nlos = ZSD_nlos

                filtering_matrix = tf.eye(channel_model._scenario.num_ut,
                    channel_model._scenario.num_ut, batch_shape=[channel_model._scenario.batch_size,
                    channel_model._scenario.num_bs], dtype=channel_model._scenario.rdtype)
                
                distance_scaling_matrix = tf.where(channel_model._scenario.los, parameter_value_los,parameter_value_nlos)
                #distance_scaling_matrix = channel_model._scenario.get_param(parameter_name)
                distance_scaling_matrix = tf.tile(tf.expand_dims(
                    distance_scaling_matrix, axis=3),
                    [1, 1, 1, channel_model._scenario.num_ut])
    
                epsilon = 1e-12
                distance_scaling_matrix = -1. / (distance_scaling_matrix + epsilon)
                # LoS
                filtering_matrix = tf.where(los_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # NLoS
                filtering_matrix = tf.where(nlos_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # indoor
                filtering_matrix = tf.where(o2i_pair_bool,
                    tf.constant(1.0, channel_model._scenario.rdtype),
                        filtering_matrix)
                # Stacking
                filtering_matrices.append(filtering_matrix)
                distance_scaling_matrices.append(distance_scaling_matrix)
            filtering_matrices = tf.stack(filtering_matrices, axis=2)
            distance_scaling_matrices = tf.stack(distance_scaling_matrices, axis=2)
            ut_dist_2d = channel_model._scenario.matrix_ut_distance_2d
            # Adding a dimension for broadcasting with BS
            ut_dist_2d = tf.expand_dims(tf.expand_dims(ut_dist_2d, axis=1), axis=2)
            
            spatial_lsp_correlation = (tf.math.exp(ut_dist_2d*distance_scaling_matrices)*filtering_matrices)  
            spatial_lsp_correlation = tf.linalg.cholesky(spatial_lsp_correlation)

            difference = channel_model._lsp_sampler._spatial_lsp_correlation_matrix_sqrt - spatial_lsp_correlation
            difference = tf.abs(difference) < 1e-6

            assert tf.reduce_all(difference)



if __name__ == '__main__':
    unittest.main()