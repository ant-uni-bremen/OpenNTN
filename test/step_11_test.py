# This file tests the implementation of step 11, the channel coefficients generation. 
import tensorflow as tf
import unittest
import numpy as np
import sionna
from sionna import config
from sionna.phy.channel.tr38811 import utils
from sionna.phy.channel.tr38811 import Antenna, AntennaArray,PanelArray,ChannelCoefficientsGenerator
from sionna.phy.channel.tr38811 import DenseUrban, SubUrban, Urban, CDL
from sionna.channel.utils import deg_2_rad
from sionna.phy.channel.tr38811.utils import gen_single_sector_topology as gen_ntn_topology


class Step_11(unittest.TestCase):
    r"""Test the computation of channel coefficients"""

    # Batch size used to check the LSP distribution
    BATCH_SIZE = 10

    # Carrier frequency
    CARRIER_FREQUENCY = 2.0e9 # Hz

    # Maximum allowed deviation for calculation (relative error)
    MAX_ERR = 1e-2

    # # Heigh of UTs
    H_UT = 1.5

    # # Heigh of BSs
    H_BS = 600000.0

    # # Number of BS
    NB_BS =1

    # Number of UT
    NB_UT = 1

    # Number of channel time samples
    NUM_SAMPLES = 32

    # Sampling frequency
    SAMPLING_FREQUENCY = 20e6

    def setUp(self):
        batch_size = Step_11.BATCH_SIZE
        nb_ut = Step_11.NB_UT
        nb_bs = Step_11.NB_BS
        h_ut = Step_11.H_UT
        h_bs = Step_11.H_BS
        fc = Step_11.CARRIER_FREQUENCY
        # los_aoa = tf.random.uniform(
        # shape=[batch_size, nb_bs, nb_ut],
        # minval=-np.pi,
        # maxval=np.pi,
        # dtype=tf.float32)

        # los_aod = tf.random.uniform(
        #     shape=[batch_size, nb_bs, nb_ut],
        #     minval=-np.pi,
        #     maxval=np.pi,
        #     dtype=tf.float32
        # )

        # los_zoa = tf.random.uniform(
        #     shape=[batch_size, nb_bs, nb_ut],
        #     minval=-np.pi,
        #     maxval=np.pi,
        #     dtype=tf.float32
        # )

        # los_zod = tf.random.uniform(
        #     shape=[batch_size, nb_bs, nb_ut],
        #     minval=-np.pi,
        #     maxval=np.pi,
        #     dtype=tf.float32
        # )

        los = tf.ones(shape=[batch_size, nb_bs, nb_ut], dtype=tf.bool)
        distance_3d = tf.random.uniform([batch_size, nb_ut, nb_ut], 0.0, 2000.0, dtype=tf.float32)

        
        self.tx_array = Antenna(polarization="single",
                                    polarization_type="V",
                                    antenna_pattern="38.901",
                                    carrier_frequency=fc)
        self.rx_array = Antenna(polarization='single',
                                polarization_type='V',
                                antenna_pattern='38.901',
                                carrier_frequency=fc)

        self.ccg = ChannelCoefficientsGenerator(
            fc,
            tx_array=self.tx_array,
            rx_array=self.rx_array,
            subclustering=True)

        # rx_orientations = config.tf_rng.uniform([batch_size, nb_ut, 3], 0.0,
        #                                     2*np.pi, dtype=tf.float32)
        # tx_orientations = config.tf_rng.uniform([batch_size, nb_bs, 3], 0.0,
        #                                     2*np.pi, dtype=tf.float32)
        # ut_velocities = config.tf_rng.uniform([batch_size, nb_ut, 3], 0.0, 5.0,
        #                                         dtype=tf.float32)

        channel_model = DenseUrban(
            carrier_frequency=fc,
            ut_array=self.rx_array,
            bs_array=self.tx_array,
            direction='downlink',
            elevation_angle=30.0)
        topology = utils.gen_single_sector_topology(
            batch_size=batch_size, num_ut=nb_ut, scenario='dur', bs_height=h_bs
        )
        channel_model.set_topology(*topology)
        self.scenario = channel_model
        

        ray_sampler = self.scenario._ray_sampler
        self.lsp = self.scenario._lsp
        
        # # lsp = lsp_sampler()
        self.rays = ray_sampler(self.lsp)   
        topology = sionna.phy.channel.tr38811.Topology(velocities=channel_model._scenario.ut_velocities,
                                moving_end="tx", 
                                los_aoa=channel_model._scenario.los_aoa,
                                los_aod=channel_model._scenario.los_aod,
                                los_zoa=channel_model._scenario.los_zoa,
                                los_zod=channel_model._scenario.los_zod,
                                los=channel_model._scenario.los,
                                distance_3d=channel_model._scenario.distance_3d,
                                tx_orientations=channel_model._scenario.ut_orientations,
                                rx_orientations=channel_model._scenario.bs_orientations,
                                bs_height = channel_model._scenario._bs_loc[:,:,2][0],
                                elevation_angle = channel_model._scenario.elevation_angle,
                                doppler_enabled = channel_model._scenario.doppler_enabled
                                )
        self.topology = topology
        
        num_time_samples = Step_11.NUM_SAMPLES 
        sampling_frequency = Step_11.SAMPLING_FREQUENCY
        # c_ds = scenario.get_param("cDS")*1e-9
        c_ds = 1.6*1e-9
        _, _, phi, sample_times = self.ccg(num_time_samples,
            sampling_frequency, self.lsp.k_factor, self.rays, topology, c_ds,
            debug=True)
        self.phi = phi.numpy()
        self.sample_times = sample_times.numpy()
        self.c_ds = c_ds


    def max_rel_err(self, r, x):
        """Compute the maximum relative error, ``r`` being the reference value,
        ``x`` an esimate of ``r``."""
        err = np.abs(r-x)
        rel_err = np.where(np.abs(r) > 0.0, np.divide(err,np.abs(r)+1e-6), err)
        return np.max(rel_err)

    def unit_sphere_vector_ref(self, theta, phi):
        """Reference implementation: Unit to sphere vector"""
        uvec = np.stack([np.sin(theta)*np.cos(phi),
                            np.sin(theta)*np.sin(phi), np.cos(theta)],
                            axis=-1)
        uvec = np.expand_dims(uvec, axis=-1)
        return uvec

    def test_unit_sphere_vector(self):
        """Test 3GPP channel coefficient calculation: Unit sphere vector"""
        #
        batch_size = Step_11.BATCH_SIZE
        theta = config.tf_rng.normal(shape=[batch_size]).numpy()
        phi = config.tf_rng.normal(shape=[batch_size]).numpy()
        uvec_ref = self.unit_sphere_vector_ref(theta, phi)
        uvec = self.ccg._unit_sphere_vector(theta, phi).numpy()
        max_err = self.max_rel_err(uvec_ref, uvec)
        err_tol = Step_11.MAX_ERR
        self.assertLessEqual(max_err, err_tol)
    def forward_rotation_matrix_ref(self, orientations):
        """Reference implementation: Forward rotation matrix"""
        a, b, c = orientations[...,0], orientations[...,1], orientations[...,2]
        #
        R = np.zeros(list(a.shape) + [3,3])
        #
        R[...,0,0] = np.cos(a)*np.cos(b)
        R[...,1,0] = np.sin(a)*np.cos(b)
        R[...,2,0] = -np.sin(b)
        #
        R[...,0,1] = np.cos(a)*np.sin(b)*np.sin(c) - np.sin(a)*np.cos(c)
        R[...,1,1] = np.sin(a)*np.sin(b)*np.sin(c) + np.cos(a)*np.cos(c)
        R[...,2,1] = np.cos(b)*np.sin(c)
        #
        R[...,0,2] = np.cos(a)*np.sin(b)*np.cos(c) + np.sin(a)*np.sin(c)
        R[...,1,2] = np.sin(a)*np.sin(b)*np.cos(c) - np.cos(a)*np.sin(c)
        R[...,2,2] = np.cos(b)*np.cos(c)
        #
        return R

    def test_forward_rotation_matrix(self):
        """Test 3GPP channel coefficient calculation: Forward rotation matrix"""
        batch_size = Step_11.BATCH_SIZE
        orientation = config.tf_rng.normal(shape=[batch_size,3]).numpy()
        R_ref = self.forward_rotation_matrix_ref(orientation)
        R = self.ccg._forward_rotation_matrix(orientation).numpy()
        max_err = self.max_rel_err(R_ref, R)
        err_tol = Step_11.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def reverse_rotation_matrix_ref(self, orientations):
        """Reference implementation: Reverse rotation matrix"""
        R = self.forward_rotation_matrix_ref(orientations)
        dim_ind = np.arange(len(R.shape))
        dim_ind = np.concatenate([dim_ind[:-2], [dim_ind[-1]], [dim_ind[-2]]],
                                    axis=0)
        R_inv = np.transpose(R, dim_ind)
        return R_inv

    def test_reverse_rotation_matrix(self):
        """Test 3GPP channel coefficient calculation: Reverse rotation matrix"""
        batch_size = Step_11.BATCH_SIZE
        orientation = config.tf_rng.normal(shape=[batch_size,3]).numpy()
        R_ref = self.reverse_rotation_matrix_ref(orientation)
        R = self.ccg._reverse_rotation_matrix(orientation).numpy()
        max_err = self.max_rel_err(R_ref, R)
        err_tol = Step_11.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def gcs_to_lcs_ref(self, orientations, theta, phi):
        """Reference implementation: GCS to LCS angles"""
        rho = self.unit_sphere_vector_ref(theta, phi)
        Rinv = self.reverse_rotation_matrix_ref(orientations)
        rho_prime = Rinv@rho

        x = np.array([1,0,0])
        x = np.expand_dims(x, axis=-1)
        x = np.broadcast_to(x, rho_prime.shape)

        y = np.array([0,1,0])
        y = np.expand_dims(y, axis=-1)
        y = np.broadcast_to(y, rho_prime.shape)

        z = np.array([0,0,1])
        z = np.expand_dims(z, axis=-1)
        z = np.broadcast_to(z, rho_prime.shape)

        theta_prime = np.sum(rho_prime*z, axis=-2)
        theta_prime = np.clip(theta_prime, -1., 1.)
        theta_prime = np.arccos(theta_prime)
        phi_prime = np.angle(np.sum(rho_prime*x, axis=-2)\
            + 1j*np.sum(rho_prime*y, axis=-2))

        theta_prime = np.squeeze(theta_prime, axis=-1)
        phi_prime = np.squeeze(phi_prime, axis=-1)

        return (theta_prime, phi_prime)

    def test_gcs_to_lcs(self):
        """Test 3GPP channel coefficient calculation: GCS to LCS"""
        batch_size = Step_11.BATCH_SIZE
        orientation = config.tf_rng.normal(shape=[batch_size,3]).numpy()
        theta = config.tf_rng.normal(shape=[batch_size]).numpy()
        phi = config.tf_rng.normal(shape=[batch_size]).numpy()

        theta_prime_ref, phi_prime_ref = self.gcs_to_lcs_ref(orientation, theta,
                                                            phi)
        theta_prime, phi_prime = self.ccg._gcs_to_lcs(
            tf.cast(orientation, tf.float32),
            tf.cast(theta, tf.float32),
            tf.cast(phi, tf.float32))
        theta_prime = theta_prime.numpy()
        phi_prime = phi_prime.numpy()

        err_tol = Step_11.MAX_ERR
        max_err = self.max_rel_err(theta_prime_ref, theta_prime)
        self.assertLessEqual(max_err, err_tol)
        max_err = self.max_rel_err(phi_prime_ref, phi_prime)
        self.assertLessEqual(max_err, err_tol)

    def compute_psi_ref(self, orientations, theta, phi):
        """Reference implementation: Compute psi angle"""
        a = orientations[...,0]
        b = orientations[...,1]
        c = orientations[...,2]

        real = np.sin(c)*np.cos(theta)*np.sin(phi-a)\
            + np.cos(c)*(np.cos(b)*np.sin(theta)\
                -np.sin(b)*np.cos(theta)*np.cos(phi-a))
        imag = np.sin(c)*np.cos(phi-a) + np.sin(b)*np.cos(c)*np.sin(phi-a)
        return np.angle(real+1j*imag)

    def l2g_response_ref(self, F_prime, orientations, theta, phi):
        """Reference implementation: L2G response"""

        psi = self.compute_psi_ref(orientations, theta, phi)

        mat = np.zeros(list(np.shape(psi)) + [2,2])
        mat[...,0,0] = np.cos(psi)
        mat[...,0,1] = -np.sin(psi)
        mat[...,1,0] = np.sin(psi)
        mat[...,1,1] = np.cos(psi)

        F = mat@np.expand_dims(F_prime, axis=-1)
        return F

    def test_l2g_response(self):
        """Test 3GPP channel coefficient calculation: L2G antenna response"""
        batch_size = Step_11.BATCH_SIZE
        orientation = config.tf_rng.normal(shape=[batch_size,3]).numpy()
        theta = config.tf_rng.normal(shape=[batch_size]).numpy()
        phi = config.tf_rng.normal(shape=[batch_size]).numpy()
        F_prime = config.tf_rng.normal(shape=[batch_size,2]).numpy()

        F_ref = self.l2g_response_ref(F_prime, orientation, theta, phi)
        F = self.ccg._l2g_response( tf.cast(F_prime, tf.float32),
                                    tf.cast(orientation,tf.float32),
                                    tf.cast(theta, tf.float32),
                                    tf.cast(phi, tf.float32)).numpy()

        max_err = self.max_rel_err(F_ref, F)
        err_tol = Step_11.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def rot_pos_ref(self, orientations, positions):
        R = self.forward_rotation_matrix_ref(orientations)
        pos_r = R@positions
        return pos_r

    def rot_pos(self, orientations, positions):
        """Reference implementation: Rotate according to an orientation"""
        R = self.forward_rotation_matrix_ref(orientations)
        pos_r = R@positions
        return pos_r

    def test_rot_pos(self):
        """Test 3GPP channel coefficient calculation: Rotate position according
        to orientation"""
        batch_size = Step_11.BATCH_SIZE
        orientations = config.tf_rng.normal(shape=[batch_size,3]).numpy()
        positions = config.tf_rng.normal(shape=[batch_size,3, 1]).numpy()

        pos_r_ref = self.rot_pos_ref(orientations, positions)
        pos_r = self.ccg._rot_pos(  tf.cast(orientations, tf.float32),
                                    tf.cast(positions, tf.float32)).numpy()
        max_err = self.max_rel_err(pos_r_ref, pos_r)
        err_tol = Step_11.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def step_11_get_tx_antenna_positions_ref(self, topology):
        """Reference implementation: Positions of the TX array elements"""

        tx_orientations = topology.tx_orientations.numpy()

        # Antenna locations in LCS and reshape for broadcasting
        ant_loc_lcs = self.tx_array.ant_pos.numpy()
        ant_loc_lcs = np.expand_dims(np.expand_dims(
            np.expand_dims(ant_loc_lcs, axis=0), axis=1), axis=-1)

        # Antenna loc in GCS relative to BS location
        tx_orientations = np.expand_dims(tx_orientations, axis=2)
        ant_loc_gcs = np.squeeze(self.rot_pos_ref(tx_orientations, ant_loc_lcs),
                                 axis=-1)

        return ant_loc_gcs

    def test_step_11_get_tx_antenna_positions(self):
        """Test 3GPP channel coefficient calculation: Positions of the TX array
        elements"""
        tx_ant_pos_ref= self.step_11_get_tx_antenna_positions_ref(self.topology)
        tx_ant_pos = self.ccg._step_11_get_tx_antenna_positions(self.topology)
        tx_ant_pos = tx_ant_pos.numpy()
        max_err = self.max_rel_err(tx_ant_pos_ref, tx_ant_pos)
        err_tol = Step_11.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def step_11_get_rx_antenna_positions_ref(self, topology):
        """Reference implementation: Positions of the RX array elements"""

        rx_orientations = topology.rx_orientations.numpy()

        # Antenna locations in LCS and reshape for broadcasting
        ant_loc_lcs = self.rx_array.ant_pos.numpy()
        ant_loc_lcs = np.expand_dims(np.expand_dims(
            np.expand_dims(ant_loc_lcs, axis=0), axis=1), axis=-1)

        # Antenna loc in GCS relative to UT location
        rx_orientations = np.expand_dims(rx_orientations, axis=2)
        ant_loc_gcs = np.squeeze(self.rot_pos_ref(rx_orientations, ant_loc_lcs),
                                    axis=-1)

        return ant_loc_gcs

    def test_step_11_get_rx_antenna_positions(self):
        """Test 3GPP channel coefficient calculation: Positions of the RX array
        elements"""
        rx_ant_pos_ref= self.step_11_get_rx_antenna_positions_ref(self.topology)
        rx_ant_pos = self.ccg._step_11_get_rx_antenna_positions(self.topology)
        rx_ant_pos = rx_ant_pos.numpy()
        max_err = self.max_rel_err(rx_ant_pos_ref, rx_ant_pos)
        err_tol = Step_11.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def step_11_phase_matrix_ref(self, Phi, kappa):
        """Reference implementation: Phase matrix"""
        xpr_scaling = np.sqrt(1./kappa)
        H_phase = np.zeros(list(Phi.shape[:-1]) + [2,2])\
            +1j*np.zeros(list(Phi.shape[:-1]) + [2,2])
        H_phase[...,0,0] = np.exp(1j*Phi[...,0])
        H_phase[...,0,1] = xpr_scaling*np.exp(1j*Phi[...,1])
        H_phase[...,1,0] = xpr_scaling*np.exp(1j*Phi[...,2])
        H_phase[...,1,1] = np.exp(1j*Phi[...,3])
        return H_phase

    def test_step_11_phase_matrix(self):
        """Test 3GPP channel coefficient calculation:
        Phase matrix calculation"""
        H_phase_ref = self.step_11_phase_matrix_ref(self.phi, self.rays.xpr)
        H_phase = self.ccg._step_11_phase_matrix(self.phi, self.rays).numpy()
        max_err = self.max_rel_err(H_phase_ref, H_phase)
        err_tol = Step_11.MAX_ERR
        self.assertLessEqual(max_err, err_tol)
    def step_11_field_matrix_ref(self, topology, aoa, aod, zoa, zod, H_phase):
        """Reference implementation: Field matrix"""

        tx_orientations = topology.tx_orientations.numpy()
        rx_orientations = topology.rx_orientations.numpy()

        # Convert departure angles to LCS
        tx_orientations = np.expand_dims(np.expand_dims(
            np.expand_dims(tx_orientations, axis=2), axis=2), axis=2)
        zod_prime, aod_prime = self.gcs_to_lcs_ref(tx_orientations, zod, aod)

        # Convert arrival angles to LCS
        rx_orientations = np.expand_dims(np.expand_dims(
            np.expand_dims(rx_orientations, axis=1), axis=3), axis=3)
        zoa_prime, aoa_prime = self.gcs_to_lcs_ref(rx_orientations, zoa, aoa)

        # Compute the TX antenna reponse in LCS and map it to GCS
        F_tx_prime_pol1_1, F_tx_prime_pol1_2 = self.tx_array.ant_pol1.field(
           tf.constant(zod_prime,tf.float32), tf.constant(aod_prime,tf.float32))
        F_tx_prime_pol1_1 = F_tx_prime_pol1_1.numpy()
        F_tx_prime_pol1_2 = F_tx_prime_pol1_2.numpy()
        F_tx_prime_pol1 = np.stack([F_tx_prime_pol1_1, F_tx_prime_pol1_2],
            axis=-1)
        F_tx_pol1 = self.l2g_response_ref(F_tx_prime_pol1, tx_orientations,
                                            zod, aod)

        # Dual polarization case for TX
        if (self.tx_array.polarization == 'dual'):
            F_tx_prime_pol2_1, F_tx_prime_pol2_2 = self.tx_array.ant_pol2.field(
                tf.constant(zod_prime, tf.float32),
                tf.constant(aod_prime, tf.float32))
            F_tx_prime_pol2_1 = F_tx_prime_pol2_1.numpy()
            F_tx_prime_pol2_2 = F_tx_prime_pol2_2.numpy()
            F_tx_prime_pol2 = np.stack([F_tx_prime_pol2_1, F_tx_prime_pol2_2],
                axis=-1)
            F_tx_pol2 = self.l2g_response_ref(F_tx_prime_pol2, tx_orientations,
                zod, aod)

        # Compute the RX antenna reponse in LCS and map it to GCS
        F_rx_prime_pol1_1, F_rx_prime_pol1_2 = self.rx_array.ant_pol1.field(
            tf.constant(zoa_prime, tf.float32),
            tf.constant(aoa_prime, tf.float32))
        F_rx_prime_pol1_1 = F_rx_prime_pol1_1.numpy()
        F_rx_prime_pol1_2 = F_rx_prime_pol1_2.numpy()
        F_rx_prime_pol1 = np.stack([F_rx_prime_pol1_1, F_rx_prime_pol1_2],
            axis=-1)
        F_rx_pol1 = self.l2g_response_ref(F_rx_prime_pol1, rx_orientations,
            zoa, aoa)

        # Dual polarization case for RX
        if (self.rx_array.polarization == 'dual'):
            F_rx_prime_pol2_1, F_rx_prime_pol2_2 = self.rx_array.ant_pol2.field(
                tf.constant(zoa_prime, tf.float32),
                tf.constant(aoa_prime, tf.float32))
            F_rx_prime_pol2_1 = F_rx_prime_pol2_1.numpy()
            F_rx_prime_pol2_2 = F_rx_prime_pol2_2.numpy()
            F_rx_prime_pol2 = np.stack([F_rx_prime_pol2_1, F_rx_prime_pol2_2],
                axis=-1)
            F_rx_pol2 = self.l2g_response_ref(F_rx_prime_pol2, rx_orientations,
                zoa, aoa)

        # Compute prtoduct between the phase matrix and the TX antenna field.
        F_tx_pol1 = H_phase@F_tx_pol1
        if (self.tx_array.polarization == 'dual'):
            F_tx_pol2 = H_phase@F_tx_pol2

        # TX: Scatteing the antenna response
        # Single polarization case is easy, as one only needs to repeat the same
        # antenna response for all elements
        F_tx_pol1 = np.expand_dims(np.squeeze(F_tx_pol1, axis=-1), axis=-2)
        if (self.tx_array.polarization == 'single'):
            F_tx = np.tile(F_tx_pol1, [1,1,1,1,1, self.tx_array.num_ant,1])
        # Dual-polarization requires scatterting the responses at the right
        # place
        else:
            F_tx_pol2 = np.expand_dims(np.squeeze(F_tx_pol2, axis=-1), axis=-2)
            F_tx = np.zeros(F_tx_pol1.shape) + 1j*np.zeros(F_tx_pol1.shape)
            F_tx = np.tile(F_tx, [1,1,1,1,1, self.tx_array.num_ant,1])
            F_tx[:,:,:,:,:,self.tx_array.ant_ind_pol1,:] = F_tx_pol1
            F_tx[:,:,:,:,:,self.tx_array.ant_ind_pol2,:] = F_tx_pol2

        # RX: Scatteing the antenna response
        # Single polarization case is easy, as one only needs to repeat the same
        # antenna response for all elements
        F_rx_pol1 = np.expand_dims(np.squeeze(F_rx_pol1, axis=-1), axis=-2)
        if (self.rx_array.polarization == 'single'):
            F_rx = np.tile(F_rx_pol1, [1,1,1,1,1,self.rx_array.num_ant,1])
        # Dual-polarization requires scatterting the responses at the right
        # place
        else:
            F_rx_pol2 = np.expand_dims(np.squeeze(F_rx_pol2, axis=-1), axis=-2)
            F_rx = np.zeros(F_rx_pol1.shape) + 1j*np.zeros(F_rx_pol1.shape)
            F_rx = np.tile(F_rx, [1,1,1,1,1,self.rx_array.num_ant,1])
            F_rx[:,:,:,:,:,self.rx_array.ant_ind_pol1,:] = F_rx_pol1
            F_rx[:,:,:,:,:,self.rx_array.ant_ind_pol2,:] = F_rx_pol2

        # Computing H_field
        F_tx = np.expand_dims(F_tx, axis=-3)
        F_rx = np.expand_dims(F_rx, axis=-2)
        H_field = np.sum(F_tx*F_rx, axis=-1)
        return H_field

    def test_step_11_field_matrix(self):
        """Test 3GPP channel coefficient calculation:
        Field matrix calculation"""
        H_phase = self.step_11_phase_matrix_ref(self.phi, self.rays.xpr)
        H_field_ref = self.step_11_field_matrix_ref(self.topology,
                                                    self.rays.aoa,
                                                    self.rays.aod,
                                                    self.rays.zoa,
                                                    self.rays.zod,
                                                    H_phase)

        H_field = self.ccg._step_11_field_matrix(self.topology,
                                    tf.constant(self.rays.aoa, tf.float32),
                                    tf.constant(self.rays.aod, tf.float32),
                                    tf.constant(self.rays.zoa, tf.float32),
                                    tf.constant(self.rays.zod, tf.float32),
                                    tf.constant(H_phase, tf.complex64)).numpy()
        max_err = self.max_rel_err(H_field_ref, H_field)
        err_tol = Step_11.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def step_11_array_offsets_ref(self, aoa, aod, zoa, zod, topology):
        """Reference implementation: Array offset matrix"""

        # Arrival spherical unit vector
        r_hat_rx = np.squeeze(self.unit_sphere_vector_ref(zoa, aoa), axis=-1)
        r_hat_rx = np.expand_dims(r_hat_rx, axis=-2)

        # Departure spherical unit vector
        r_hat_tx =  np.squeeze(self.unit_sphere_vector_ref(zod, aod), axis=-1)
        r_hat_tx = np.expand_dims(r_hat_tx, axis=-2)

        # TX location vector
        d_bar_tx = self.step_11_get_tx_antenna_positions_ref(topology)
        d_bar_tx = np.expand_dims(np.expand_dims(
            np.expand_dims(d_bar_tx, axis=2), axis=3), axis=4)

        # RX location vector
        d_bar_rx = self.step_11_get_rx_antenna_positions_ref(topology)
        d_bar_rx = np.expand_dims(np.expand_dims(
                np.expand_dims(d_bar_rx, axis=1), axis=3), axis=4)

        lambda_0 = self.scenario._scenario.lambda_0.numpy()

        # TX offset matrix

        tx_offset = np.sum(r_hat_tx*d_bar_tx, axis=-1)
        rx_offset = np.sum(r_hat_rx*d_bar_rx, axis=-1)

        tx_offset = np.expand_dims(tx_offset, -2)
        rx_offset = np.expand_dims(rx_offset, -1)
        antenna_offset = np.exp(1j*2*np.pi*(tx_offset+rx_offset)/lambda_0)

        return antenna_offset

    def test_step_11_array_offsets(self):
        """Test 3GPP channel coefficient calculation: Array offset matrix"""
        H_array_ref = self.step_11_array_offsets_ref(self.rays.aoa,
                                                     self.rays.aod,
                                                     self.rays.zoa,
                                                     self.rays.zod,
                                                     self.topology)

        H_array = self.ccg._step_11_array_offsets(self.topology,
                                tf.constant(self.rays.aoa, tf.float32),
                                tf.constant(self.rays.aod, tf.float32),
                                tf.constant(self.rays.zoa, tf.float32),
                                tf.constant(self.rays.zod, tf.float32)).numpy()

        max_err = self.max_rel_err(H_array_ref, H_array)
        err_tol = Step_11.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def step_11_doppler_matrix_ref(self, topology, aoa, zoa, aod, zod, t):
        """Reference implementation: Doppler matrix calculation"""

        lambda_0 = self.scenario._scenario.lambda_0.numpy()
        velocities = topology.velocities.numpy()

        # Determine which end of the channel is moving (TX or RX)
        if topology.moving_end == "rx":
            velocities = np.expand_dims(velocities, axis=1)
            r_hat_ut = np.squeeze(self.unit_sphere_vector_ref(zoa, aoa), axis=-1)
        elif topology.moving_end == "tx":
            velocities = np.expand_dims(velocities, axis=2)
            r_hat_ut = np.squeeze(self.unit_sphere_vector_ref(zod, aod), axis=-1)

        # Expand dimensions to make broadcastable
        velocities = np.expand_dims(np.expand_dims(np.expand_dims(velocities, axis=3), axis=4), axis=-1)
        r_hat_ut = np.expand_dims(r_hat_ut, axis=-1)
        # Calculate the phase shift due to Doppler effect
        exponent = 2 * np.pi / lambda_0 * np.sum(r_hat_ut * velocities, axis=-2) * t

        # Handle satellite-specific Doppler effects (if applicable)
        if topology.bs_height >= 600000.0 and topology.doppler_enabled:
            max_sat_speed_for_elevation_angle = topology.sat_speed.numpy()
            max_rotation_per_time = (2.0 * np.pi / lambda_0) * max_sat_speed_for_elevation_angle
            rotation_for_time = np.outer(max_rotation_per_time, t)  # Shape: [batch size, num time steps]

            # Broadcast to match the shape of `exponent`
            rotation_for_time = np.expand_dims(rotation_for_time, axis=(1, 2, 3, 4))
            rotation_for_time = np.broadcast_to(rotation_for_time, exponent.shape)

            # Add satellite Doppler rotation to the exponent
            exponent += rotation_for_time

        # Compute the Doppler matrix
        H_doppler = np.exp(1j * exponent)

        return H_doppler

    def test_step_11_doppler_matrix(self):
        """Test 3GPP channel coefficient calculation: Doppler matrix"""
        H_doppler_ref = self.step_11_doppler_matrix_ref(self.topology,
                                                        self.rays.aoa,
                                                        self.rays.zoa,
                                                        self.rays.aod,
                                                        self.rays.zod,
                                                        self.sample_times)

        H_doppler = self.ccg._step_11_doppler_matrix(self.topology,
                            tf.constant(self.rays.aoa, tf.float32),
                            tf.constant(self.rays.zoa, tf.float32),
                            tf.constant(self.rays.aod, tf.float32),
                            tf.constant(self.rays.zod, tf.float32),
                            tf.constant(self.sample_times, tf.float32)).numpy()

        max_err = self.max_rel_err(H_doppler_ref, H_doppler)
        err_tol = Step_11.MAX_ERR
        self.assertLessEqual(max_err, err_tol)

    def step_11_nlos_ref(self, phi, aoa, aod, zoa, zod, kappa, powers, t,
        topology):
        """Reference implemenrtation: Compute the channel matrix of the NLoS
        component"""

                # Compute the phase matrix
        H_phase = self.step_11_phase_matrix_ref(phi, kappa)

        # Add Faraday phase rotation for high BS height (e.g., satellites)
        if topology.bs_height >= 600000.0:
            rays_shape = aoa.shape
            faraday_phase_rotation = self.ccg._step_11_faraday_rotation(
                carrier_frequency=self.CARRIER_FREQUENCY, aod_shape=rays_shape
            )
            H_phase = np.matmul(H_phase, faraday_phase_rotation)

        # Compute other components
        H_field = self.step_11_field_matrix_ref(topology, aoa, aod, zoa, zod, H_phase)
        H_array = self.step_11_array_offsets_ref(aoa, aod, zoa, zod, topology)
        H_doppler = self.step_11_doppler_matrix_ref(topology, aoa, zoa, aod, zod, t)

        # Expand dimensions for consistent broadcasting
        H_field = np.expand_dims(H_field, axis=-1)
        H_array = np.expand_dims(H_array, axis=-1)
        H_doppler = np.expand_dims(np.expand_dims(H_doppler, axis=-2), axis=-3)

        # Combine components
        H_full = H_field * H_array * H_doppler

        # Apply power scaling
        power_scaling = np.sqrt(powers / aoa.shape[4])
        power_scaling = np.expand_dims(np.expand_dims(np.expand_dims(
            np.expand_dims(power_scaling, axis=4), axis=5), axis=6), axis=7)
        H_full = H_full * power_scaling

        return H_full

    def test_step_11_nlos_ref(self):
        """Test 3GPP channel coefficient calculation: Doppler matrix"""
        H_full_ref = self.step_11_nlos_ref( self.phi,
                                            self.rays.aoa,
                                            self.rays.aod,
                                            self.rays.zoa,
                                            self.rays.zod,
                                            self.rays.xpr,
                                            self.rays.powers,
                                            self.sample_times,
                                            self.topology)

        H_full = self.ccg._step_11_nlos(tf.constant(self.phi, tf.float32),
                            self.topology,
                            self.rays,
                            tf.constant(self.sample_times, tf.float32),
                            Step_11.CARRIER_FREQUENCY).numpy()
        max_err = self.max_rel_err(H_full_ref, H_full)
        err_tol = Step_11.MAX_ERR
        self.assertLessEqual(max_err, err_tol)   

    def step_11_reduce_nlos_ref(self, H_full, powers, delays, c_DS):
        """Reference implementation: Compute the channel matrix of the NLoS
        component 2"""

        # Step 1: Sort clusters in descending order of power
        strongest_clusters = tf.argsort(powers, axis=-1, direction="DESCENDING")

        # Step 2: Sort delays and H_full according to cluster order
        delays_sorted = tf.gather(delays, strongest_clusters, batch_dims=3, axis=3)
        H_full_sorted = tf.gather(H_full, strongest_clusters, batch_dims=3, axis=3)

        # Step 3: Split delays into strong and weak clusters
        delays_strong = delays_sorted[..., :2]
        delays_weak = delays_sorted[..., 2:]

        # Step 4: Compute sub-cluster delays for strong clusters
        offsets = tf.reshape([0.0, 1.28, 2.56], [1, 1, 1, 1, -1])
        c_DS_expanded = tf.expand_dims(tf.expand_dims(c_DS, axis=-1), axis=-1)
        delays_sub_cl = (tf.expand_dims(delays_strong, -2) + offsets * c_DS_expanded)
        delays_sub_cl = tf.reshape(delays_sub_cl, tf.concat([tf.shape(delays_sub_cl)[:-2], [-1]], axis=0))

        # Step 5: Split H_full into strong and weak clusters
        H_strong = tf.gather(H_full_sorted, range(2), batch_dims=3, axis=3)
        H_weak = tf.gather(H_full_sorted, range(2, tf.shape(H_full_sorted)[3]), batch_dims=3, axis=3)

        # Step 6: Aggregate rays for sub-clusters of strong clusters
        H_sub_cl_1 = tf.reduce_sum(tf.gather(H_strong, [0, 1, 2, 3, 4, 5, 6, 7, 18, 19], axis=4), axis=4)
        H_sub_cl_2 = tf.reduce_sum(tf.gather(H_strong, [8, 9, 10, 11, 16, 17], axis=4), axis=4)
        H_sub_cl_3 = tf.reduce_sum(tf.gather(H_strong, [12, 13, 14, 15], axis=4), axis=4)

        # Step 7: Aggregate rays for weak clusters
        H_weak = tf.reduce_sum(H_weak, axis=4)

        # Step 8: Concatenate sub-cluster coefficients and weak clusters
        H_nlos = tf.concat([H_sub_cl_1, H_sub_cl_2, H_sub_cl_3, H_weak], axis=3)
        delays_nlos = tf.concat([delays_sub_cl, delays_weak], axis=3)

        # Step 9: Sort delays in ascending order and reorder H_nlos accordingly
        delays_sorted_indices = tf.argsort(delays_nlos, axis=-1, direction="ASCENDING")
        delays_nlos = tf.gather(delays_nlos, delays_sorted_indices, batch_dims=3, axis=3)
        H_nlos = tf.gather(H_nlos, delays_sorted_indices, batch_dims=3, axis=3)

        return (H_nlos, delays_nlos)


    def test_step_11_reduce_nlos(self):
        """Test 3GPP channel coefficient calculation: NLoS channel matrix
        computation"""

        H_full_ref = self.step_11_nlos_ref( self.phi,
                                            self.rays.aoa,
                                            self.rays.aod,
                                            self.rays.zoa,
                                            self.rays.zod,
                                            self.rays.xpr,
                                            self.rays.powers,
                                            self.sample_times,
                                            self.topology)

        H_nlos_ref, delays_nlos_ref = self.step_11_reduce_nlos_ref(
                                                    H_full_ref,
                                                    self.rays.powers.numpy(),
                                                    self.rays.delays.numpy(),
                                                    self.c_ds)

        H_nlos, delays_nlos = self.ccg._step_11_reduce_nlos(
            tf.constant(H_full_ref, tf.complex128), self.rays, self.c_ds)
        H_nlos = H_nlos.numpy()
        delays_nlos = delays_nlos.numpy()

        err_tol = Step_11.MAX_ERR
        max_err = self.max_rel_err(H_nlos_ref, H_nlos)
        self.assertLessEqual(max_err, err_tol)

        max_err = self.max_rel_err(delays_nlos_ref, delays_nlos)
        self.assertLessEqual(max_err, err_tol)
if __name__ == "__main__":
    unittest.main()