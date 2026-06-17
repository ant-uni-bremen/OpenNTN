#
# This file has been created by the Dept. of Communications Engineering of the University of Bremen.
# The code is based on implementations provided by the NVIDIA CORPORATION & AFFILIATES
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""
Class for sampling channel impulse responses following 3GPP TR38.811
specifications and giving LSPs and rays. The process is defined mainly
in Section 6 and especially 6.5 of 3GPP TR38.811. The process is
based on 3GPP TR38.901, with TR38.811 serving mainly as an extension.
"""

import math

import torch

from sionna.phy import PI, SPEED_OF_LIGHT, config
from sionna.phy import Object
from sionna.phy.utils import rand
from .utils import compute_satellite_speed

__all__ = ["Topology", "ChannelCoefficientsGenerator"]


class Topology:
    # pylint: disable=line-too-long
    r"""
    Class for conveniently storing the network topology information required
    for sampling channel impulse responses

    Parameters
    -----------

    velocities : [batch size, number of UTs], `torch.float`
        UT velocities

    moving_end : str
        Indicated which end of the channel (TX or RX) is moving. Either "tx" or
        "rx".

    los_aoa : [batch size, number of BSs, number of UTs], `torch.float`
        Azimuth angle of arrival of LoS path [radian]

    los_aod : [batch size, number of BSs, number of UTs], `torch.float`
        Azimuth angle of departure of LoS path [radian]

    los_zoa : [batch size, number of BSs, number of UTs], `torch.float`
        Zenith angle of arrival for of path [radian]

    los_zod : [batch size, number of BSs, number of UTs], `torch.float`
        Zenith angle of departure for of path [radian]

    los : [batch size, number of BSs, number of UTs], `torch.bool`
        Indicate for each BS-UT link if it is in LoS

    distance_3d : [batch size, number of UTs, number of UTs], `torch.float`
        Distance between the UTs in X-Y-Z space (not only X-Y plan).

    tx_orientations : [batch size, number of TXs, 3], `torch.float`
        Orientations of the transmitters, which are either BSs or UTs depending
        on the link direction [radian].

    rx_orientations : [batch size, number of RXs, 3], `torch.float`
        Orientations of the receivers, which are either BSs or UTs depending on
        the link direction [radian].

    doppler_enabled : bool
        Indicates if Doppler shift induced phase rotation should be simulated.
    """

    def __init__(self,  velocities,
                        moving_end,
                        los_aoa,
                        los_aod,
                        los_zoa,
                        los_zod,
                        los,
                        distance_3d,
                        tx_orientations,
                        rx_orientations,
                        bs_height,
                        elevation_angle,
                        doppler_enabled):
        self.velocities = velocities
        self.moving_end = moving_end
        self.los_aoa = los_aoa
        self.los_aod = los_aod
        self.los_zoa = los_zoa
        self.los_zod = los_zod
        self.los = los
        self.tx_orientations = tx_orientations
        self.rx_orientations = rx_orientations
        self.distance_3d = distance_3d
        self.bs_height = bs_height
        self.elevation_angle = elevation_angle
        self.doppler_enabled = doppler_enabled
        # TODO In the best case we would verify the height, however, this ran into issues with eager execution at the moment
        # for now, we assume the satellite to always be used, despite the other checks for it already in place
        # In a future version we will test the sat height and set sat_speed to none if the bs is too low to be a satellite
        # self.sat_speed = None
        # if torch.any(torch.greater_equal(bs_height, 600000.0)):
        sat_speed = compute_satellite_speed(bs_height).to(los_aoa.device)
        elevation_angle_rad = torch.as_tensor(elevation_angle, dtype=los_aoa.dtype,
            device=los_aoa.device) * (PI/180.0)
        max_sat_speed_for_elevation_angle = torch.cos(elevation_angle_rad) * sat_speed
        batch_size = los_aoa.shape[0]
        random_direction_per_batch = rand([batch_size], dtype=los_aoa.dtype,
            device=los_aoa.device,
            generator=config.torch_rng(str(los_aoa.device))) * 2 * PI
        self.sat_speed = torch.cos(random_direction_per_batch) * max_sat_speed_for_elevation_angle


class ChannelCoefficientsGenerator(Object):
    # pylint: disable=line-too-long
    r"""
    Sample channel impulse responses according to LSPs rays.

    This class implements steps 10 and 11 from the TR 38.901 specifications,
    (section 7.5).

    Parameters
    ----------
    carrier_frequency : float
        Carrier frequency [Hz]

    tx_array : PanelArray
        Panel array used by the transmitters.
        All transmitters share the same antenna array configuration.

    rx_array : PanalArray
        Panel array used by the receivers.
        All transmitters share the same antenna array configuration.

    subclustering : bool
        Use subclustering if set to `True` (see step 11 for section 7.5 in
        TR 38.901). CDL does not use subclustering. System level models (dur, urb,
        and sur) do.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    device : `None` (default) | str
        Device for computation (e.g., ``"cpu"``, ``"cuda:0"``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    Input
    -----
    num_time_samples : int
        Number of samples

    sampling_frequency : float
        Sampling frequency [Hz]

    k_factor : [batch_size, number of TX, number of RX]
        K-factor

    rays : Rays
        Rays from which to compute thr CIR

    topology : Topology
        Topology of the network

    c_ds : [batch size, number of TX, number of RX]
        Cluster DS [ns]. Only needed when subclustering is used
        (``subclustering`` set to `True`), i.e., with system level models.
        Otherwise can be set to None.
        Defaults to None.

    debug : bool
        If set to `True`, additional information is returned in addition to
        paths coefficients and delays: The random phase shifts (see step 10 of
        section 7.5 in TR38.901 specification), and the time steps at which the
        channel is sampled.

    Output
    ------
    h : [batch size, num TX, num RX, num paths, num RX antenna, num TX antenna, num samples], `torch.complex`
        Paths coefficients

    delays : [batch size, num TX, num RX, num paths], `torch.real`
        Paths delays [s]

    phi : [batch size, number of BSs, number of UTs, 4], `torch.real`
        Initial phases (see step 10 of section 7.5 in TR 38.901 specification).
        Last dimension corresponds to the four polarization combinations.

    sample_times : [number of time steps], `torch.float`
        Sampling time steps
    """

    def __init__(self,  carrier_frequency,
                        tx_array, rx_array,
                        subclustering,
                        precision=None,
                        device=None):
        super().__init__(precision=precision, device=device)

        self._carrier_frequency = carrier_frequency
        # Wavelength (m)
        self._lambda_0 = torch.as_tensor(SPEED_OF_LIGHT/carrier_frequency,
            dtype=self.dtype, device=self.device)
        self._tx_array = tx_array
        self._rx_array = rx_array
        self._subclustering = subclustering

        # Sub-cluster information for intra cluster delay spread clusters
        # This is hardcoded from Table 7.5-5
        # Register as buffers for device/CUDAGraph compatibility
        self.register_buffer("_sub_cl_1_ind", torch.tensor(
            [0, 1, 2, 3, 4, 5, 6, 7, 18, 19], dtype=torch.int64, device=self.device))
        self.register_buffer("_sub_cl_2_ind", torch.tensor(
            [8, 9, 10, 11, 16, 17], dtype=torch.int64, device=self.device))
        self.register_buffer("_sub_cl_3_ind", torch.tensor(
            [12, 13, 14, 15], dtype=torch.int64, device=self.device))
        self.register_buffer("_sub_cl_delay_offsets", torch.tensor(
            [0, 1.28, 2.56], dtype=self.dtype, device=self.device))

        # Pre-computed complex constants (see tr38901 twin) for torch.compile
        # compatibility; created here rather than inside compiled functions.
        self.register_buffer("_v1_const", torch.tensor(
            [0, 0, 1], dtype=self.dtype, device=self.device))
        self.register_buffer("_v2_const", torch.tensor(
            [1 + 0j, 1j, 0], dtype=self.cdtype, device=self.device))
        self.register_buffer("_h_phase_los_const", torch.tensor(
            [[1.0, 0.0], [0.0, -1.0]], dtype=self.cdtype, device=self.device))

    def __call__(self, num_time_samples, sampling_frequency, k_factor, rays,
                 topology, c_ds=None, debug=False):
        # Sample times
        sample_times = (torch.arange(num_time_samples,
                dtype=self.dtype, device=self.device)/sampling_frequency)

        # Step 10
        phi = self._step_10(rays.aoa.shape)

        # Step 11
        h, delays = self._step_11(phi, topology, k_factor, rays, sample_times,
                                                        c_ds, self._carrier_frequency)

        # Return additional information if requested
        if debug:
            return h, delays, phi, sample_times

        return h, delays

    ###########################################
    # Utility functions
    ###########################################

    def _unit_sphere_vector(self, theta, phi):
        r"""
        Generate vector on unit sphere (7.1-6)

        Input
        -------
        theta : Arbitrary shape, `torch.float`
            Zenith [radian]

        phi : Same shape as ``theta``, `torch.float`
            Azimuth [radian]

        Output
        --------
        rho_hat : ``phi.shape`` + [3, 1]
            Vector on unit sphere

        """
        rho_hat = torch.stack([torch.sin(theta)*torch.cos(phi),
                            torch.sin(theta)*torch.sin(phi),
                            torch.cos(theta)], dim=-1)
        return rho_hat.unsqueeze(-1)

    def _forward_rotation_matrix(self, orientations):
        r"""
        Forward composite rotation matrix (7.1-4)

        Input
        ------
            orientations : [...,3], `torch.float`
                Orientation to which to rotate [radian]

        Output
        -------
        R : [...,3,3], `torch.float`
            Rotation matrix
        """
        a, b, c = orientations[...,0], orientations[...,1], orientations[...,2]

        row_1 = torch.stack([torch.cos(a)*torch.cos(b),
            torch.cos(a)*torch.sin(b)*torch.sin(c)-torch.sin(a)*torch.cos(c),
            torch.cos(a)*torch.sin(b)*torch.cos(c)+torch.sin(a)*torch.sin(c)], dim=-1)

        row_2 = torch.stack([torch.sin(a)*torch.cos(b),
            torch.sin(a)*torch.sin(b)*torch.sin(c)+torch.cos(a)*torch.cos(c),
            torch.sin(a)*torch.sin(b)*torch.cos(c)-torch.cos(a)*torch.sin(c)], dim=-1)

        row_3 = torch.stack([-torch.sin(b),
            torch.cos(b)*torch.sin(c),
            torch.cos(b)*torch.cos(c)], dim=-1)

        rot_mat = torch.stack([row_1, row_2, row_3], dim=-2)
        return rot_mat

    def _rot_pos(self, orientations, positions):
        r"""
        Rotate the ``positions`` according to the ``orientations``

        Input
        ------
        orientations : [...,3], `torch.float`
            Orientation to which to rotate [radian]

        positions : [...,3,1], `torch.float`
            Positions to rotate

        Output
        -------
        : [...,3,1], `torch.float`
            Rotated positions
        """
        rot_mat = self._forward_rotation_matrix(orientations)
        return torch.matmul(rot_mat, positions)

    def _reverse_rotation_matrix(self, orientations):
        r"""
        Reverse composite rotation matrix (7.1-4)

        Input
        ------
        orientations : [...,3], `torch.float`
            Orientations to rotate to  [radian]

        Output
        -------
        R_inv : [...,3,3], `torch.float`
            Inverse of the rotation matrix corresponding to ``orientations``
        """
        rot_mat = self._forward_rotation_matrix(orientations)
        rot_mat_inv = rot_mat.mT
        return rot_mat_inv

    def _gcs_to_lcs(self, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Compute the angles ``theta``, ``phi`` in LCS rotated according to
        ``orientations`` (7.1-7/8)

        Input
        ------
        orientations : [...,3] of rank K, `torch.float`
            Orientations to which to rotate to [radian]

        theta : Broadcastable to the first K-1 dimensions of ``orientations``, `torch.float`
            Zenith to rotate [radian]

        phi : Same dimension as ``theta``, `torch.float`
            Azimuth to rotate [radian]

        Output
        -------
        theta_prime : Same dimension as ``theta``, `torch.float`
            Rotated zenith

        phi_prime : Same dimensions as ``theta`` and ``phi``, `torch.float`
            Rotated azimuth
        """

        rho_hat = self._unit_sphere_vector(theta, phi)
        rot_inv = self._reverse_rotation_matrix(orientations)
        rot_rho = torch.matmul(rot_inv, rho_hat)
        v1 = self._v1_const.reshape([1]*(rot_rho.dim()-1)+[3])
        v2 = self._v2_const.reshape([1]*(rot_rho.dim()-1)+[3])
        z = torch.matmul(v1, rot_rho)
        z = z.clamp(-1.0, 1.0)
        theta_prime = torch.acos(z)
        phi_prime = torch.angle((torch.matmul(v2, rot_rho.to(self.cdtype))))
        theta_prime = theta_prime.squeeze(phi.dim()).squeeze(phi.dim())
        phi_prime = phi_prime.squeeze(phi.dim()).squeeze(phi.dim())

        return (theta_prime, phi_prime)

    def _compute_psi(self, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Compute displacement angle :math:`Psi` for the transformation of LCS-GCS
        field components in (7.1-15) of TR38.901 specification

        Input
        ------
        orientations : [...,3], `torch.float`
            Orientations to which to rotate to [radian]

        theta :  Broadcastable to the first K-1 dimensions of ``orientations``, `torch.float`
            Spherical position zenith [radian]

        phi : Same dimensions as ``theta``, `torch.float`
            Spherical position azimuth [radian]

        Output
        -------
            Psi : Same shape as ``theta`` and ``phi``, `torch.float`
                Displacement angle :math:`Psi`
        """
        a = orientations[...,0]
        b = orientations[...,1]
        c = orientations[...,2]
        real = torch.sin(c)*torch.cos(theta)*torch.sin(phi-a)
        real = real + torch.cos(c)*(torch.cos(b)*torch.sin(theta)-torch.sin(b)*torch.cos(theta)*torch.cos(phi-a))
        imag = torch.sin(c)*torch.cos(phi-a) + torch.sin(b)*torch.cos(c)*torch.sin(phi-a)
        psi = torch.angle(torch.complex(real, imag))
        return psi

    def _l2g_response(self, f_prime, orientations, theta, phi):
        # pylint: disable=line-too-long
        r"""
        Transform field components from LCS to GCS (7.1-11)

        Input
        ------
        f_prime : K-Dim Tensor of shape [...,2], `torch.float`
            Field components

        orientations : K-Dim Tensor of shape [...,3], `torch.float`
            Orientations of LCS-GCS [radian]

        theta : K-1-Dim Tensor with matching dimensions to ``f_prime`` and ``phi``, `torch.float`
            Spherical position zenith [radian]

        phi : Same dimensions as ``theta``, `torch.float`
            Spherical position azimuth [radian]

        Output
        ------
            F : K+1-Dim Tensor with shape [...,2,1], `torch.float`
                The first K dimensions are identical to those of ``f_prime``
        """
        psi = self._compute_psi(orientations, theta, phi)
        row1 = torch.stack([torch.cos(psi), -torch.sin(psi)], dim=-1)
        row2 = torch.stack([torch.sin(psi), torch.cos(psi)], dim=-1)
        mat = torch.stack([row1, row2], dim=-2)
        f = torch.matmul(mat, f_prime.unsqueeze(-1))
        return f

    def _step_11_get_tx_antenna_positions(self, topology):
        r"""Compute d_bar_tx in (7.5-22), i.e., the positions in GCS of elements
        forming the transmit panel

        Input
        -----
        topology : Topology
            Topology of the network

        Output
        -------
        d_bar_tx : [batch_size, num TXs, num TX antenna, 3]
            Positions of the antenna elements in the GCS
        """
        # Get BS orientations got broadcasting
        tx_orientations = topology.tx_orientations
        tx_orientations = tx_orientations.unsqueeze(2)

        # Get antenna element positions in LCS and reshape for broadcasting
        tx_ant_pos_lcs = self._tx_array.ant_pos
        tx_ant_pos_lcs = tx_ant_pos_lcs.reshape(1, 1, -1, 3, 1)

        # Compute antenna element positions in GCS
        tx_ant_pos_gcs = self._rot_pos(tx_orientations, tx_ant_pos_lcs)
        tx_ant_pos_gcs = tx_ant_pos_gcs.squeeze(-1)

        d_bar_tx = tx_ant_pos_gcs

        return d_bar_tx

    def _step_11_get_rx_antenna_positions(self, topology):
        r"""Compute d_bar_rx in (7.5-22), i.e., the positions in GCS of elements
        forming the receive antenna panel

        Input
        -----
        topology : Topology
            Topology of the network

        Output
        -------
        d_bar_rx : [batch_size, num RXs, num RX antenna, 3]
            Positions of the antenna elements in the GCS
        """
        # Get UT orientations got broadcasting
        rx_orientations = topology.rx_orientations
        rx_orientations = rx_orientations.unsqueeze(2)

        # Get antenna element positions in LCS and reshape for broadcasting
        rx_ant_pos_lcs = self._rx_array.ant_pos
        rx_ant_pos_lcs = rx_ant_pos_lcs.reshape(1, 1, -1, 3, 1)

        # Compute antenna element positions in GCS
        rx_ant_pos_gcs = self._rot_pos(rx_orientations, rx_ant_pos_lcs)
        rx_ant_pos_gcs = rx_ant_pos_gcs.squeeze(-1)

        d_bar_rx = rx_ant_pos_gcs

        return d_bar_rx

    def _step_10(self, shape):
        r"""
        Generate random and uniformly distributed phases for all rays and
        polarization combinations

        Input
        -----
        shape : Shape tensor
            Shape of the leading dimensions for the tensor of phases to generate

        Output
        ------
        phi : [shape] + [4], `torch.float`
            Phases for all polarization combinations
        """
        phi = rand((*tuple(shape), 4), dtype=self.dtype, device=self.device,
            generator=self.torch_rng) * 2 * PI - PI

        return phi

    def _step_11_phase_matrix(self, phi, rays):
        # pylint: disable=line-too-long
        r"""
        Compute matrix with random phases in (7.5-22)

        Input
        -----
        phi : [batch size, num TXs, num RXs, num clusters, num rays, 4], `torch.float`
            Initial phases for all combinations of polarization

        rays : Rays
            Rays

        Output
        ------
        h_phase : [batch size, num TXs, num RXs, num clusters, num rays, 2, 2], `torch.complex`
            Matrix with random phases in (7.5-22)
        """
        xpr = rays.xpr

        xpr_scaling = torch.complex(torch.sqrt(1/xpr),
            torch.zeros_like(xpr))
        e0 = torch.exp(torch.complex(torch.zeros_like(phi[...,0]),
            phi[...,0]))
        e3 = torch.exp(torch.complex(torch.zeros_like(phi[...,3]),
            phi[...,3]))
        e1 = xpr_scaling*torch.exp(torch.complex(torch.zeros_like(phi[...,1]),
            phi[...,1]))
        e2 = xpr_scaling*torch.exp(torch.complex(torch.zeros_like(phi[...,2]),
            phi[...,2]))
        h_phase = torch.stack([e0, e1, e2, e3], dim=-1).reshape(*e0.shape, 2, 2)

        return h_phase

    def _step_11_doppler_matrix(self, topology, aoa, zoa, aod, zod, t):
        # pylint: disable=line-too-long
        r"""
        Compute matrix with phase shifts due to mobility in (7.5-22)

        Input
        -----
        topology : Topology
            Topology of the network

        aoa : [batch size, num TXs, num RXs, num clusters, num rays], `torch.float`
            Azimuth angles of arrivals [radian]

        zoa : [batch size, num TXs, num RXs, num clusters, num rays], `torch.float`
            Zenith angles of arrivals [radian]

        t : [number of time steps]
            Time steps at which the channel is sampled

        Output
        ------
        h_doppler : [batch size, num_tx, num rx, num clusters, num rays, num time steps], `torch.complex`
            Matrix with phase shifts due to mobility in (7.5-22)
        """

        lambda_0 = self._lambda_0
        ut_velocities = topology.velocities

        # Add an extra dimension to make v_bar broadcastable with the time
        # dimension
        # v_bar [batch size, num tx or num rx, 3, 1]
        v_uts_bar = ut_velocities
        v_uts_bar = v_uts_bar.unsqueeze(-1)

        # Depending on which end of the channel is moving, tx or rx, we add an
        # extra dimension to make this tensor broadcastable with the other end
        # TODO with the new structure rx and tx are not adeqaute, as both ends are moving and
        # the question is which is the UT instead of which is moving. Thus in the future this should
        # be adapted. However, as it is only a naming issue and limited to this class, we will leave it
        # like this for now
        if topology.moving_end == 'rx':
            # v_bar [batch size, 1, num rx, num tx, 1]
            v_uts_bar = v_uts_bar.unsqueeze(1)
            r_hat_ut = self._unit_sphere_vector(zoa, aoa)

        # Old way to say uplink
        elif topology.moving_end == 'tx':
            # v_bar [batch size, num tx, 1, num tx, 1]
            v_uts_bar = v_uts_bar.unsqueeze(2)
            r_hat_ut = self._unit_sphere_vector(zod, aod)


        # v_bar [batch size, 1, num rx, 1, 1, 3, 1]
        # or    [batch size, num tx, 1, 1, 1, 3, 1]
        v_uts_bar = v_uts_bar.unsqueeze(-3).unsqueeze(-3)

        # v_bar [batch size, num_tx, num rx, num clusters, num rays, 3, 1]

        # Compute phase shift due to doppler
        # [batch size, num_tx, num rx, num clusters, num rays, num time steps]
        exponent = 2*PI/lambda_0*(r_hat_ut*v_uts_bar).sum(dim=-2)*t

        # Check if a satellite is used (height of satellite is at least 600km). This is done to potentially incorporate HAPS later
        bs_height_threshold = torch.less_equal(
            torch.as_tensor(600000.0, device=topology.bs_height.device),
            topology.bs_height)
        doppler_enabled = torch.as_tensor(topology.doppler_enabled,
            dtype=torch.bool, device=topology.bs_height.device)
        if torch.logical_and(bs_height_threshold, doppler_enabled):
            # We get the maximum speed of the satellite based on its orbit and the users elevation angle. This value is multiplied with a random constant between [cos(0),cos(2pi))
            # to simulate the random direction of the orbit in relation to the users.
            max_sat_speed_for_elevation_angle = topology.sat_speed
            # The Doppler shift induced phase shift between each time step
            max_rotation_per_time = (2.0*PI/lambda_0)*max_sat_speed_for_elevation_angle

            # Multiply the time with the shift per time to create the shift on each time symbol for each batch
            # [batch size, num time steps]
            rotation_for_time = torch.matmul(max_rotation_per_time.unsqueeze(1), t.unsqueeze(0))

            # As the time shift is the same within each batch, we expand the calculations per batch in the desired shape
            # to get shape [batch size, num_tx, num rx, num clusters, num rays, num time steps]
            rotation_for_time = rotation_for_time.unsqueeze(1)
            rotation_for_time = rotation_for_time.unsqueeze(1)
            rotation_for_time = rotation_for_time.unsqueeze(1)
            rotation_for_time = rotation_for_time.unsqueeze(1)

            rotation_for_time = rotation_for_time.broadcast_to(exponent.shape)
            # Add the Doppler exponent to the standard exponent based on 3GPP TR38.811 section 6.8.1
            exponent = torch.add(exponent, rotation_for_time)

        h_doppler = torch.exp(torch.complex(torch.zeros_like(exponent),
                                    exponent))

        return h_doppler

    def _step_11_array_offsets(self, topology, aoa, aod, zoa, zod):
        # pylint: disable=line-too-long
        r"""
        Compute matrix accounting for phases offsets between antenna elements

        Input
        -----
        topology : Topology
            Topology of the network

        aoa : [batch size, num TXs, num RXs, num clusters, num rays], `torch.float`
            Azimuth angles of arrivals [radian]

        aod : [batch size, num TXs, num RXs, num clusters, num rays], `torch.float`
            Azimuth angles of departure [radian]

        zoa : [batch size, num TXs, num RXs, num clusters, num rays], `torch.float`
            Zenith angles of arrivals [radian]

        zod : [batch size, num TXs, num RXs, num clusters, num rays], `torch.float`
            Zenith angles of departure [radian]
        Output
        ------
        h_array : [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas], `torch.complex`
            Matrix accounting for phases offsets between antenna elements
        """

        lambda_0 = self._lambda_0

        r_hat_rx = self._unit_sphere_vector(zoa, aoa)
        r_hat_rx = r_hat_rx.squeeze(-1)
        r_hat_tx = self._unit_sphere_vector(zod, aod)
        r_hat_tx = r_hat_tx.squeeze(-1)
        d_bar_rx = self._step_11_get_rx_antenna_positions(topology)
        d_bar_tx = self._step_11_get_tx_antenna_positions(topology)

        # Reshape tensors for broadcasting
        # r_hat_rx/tx have
        # shape [batch_size, num_tx, num_rx, num_clusters, num_rays,    3]
        # and will be reshaped to
        # [batch_size, num_tx, num_rx, num_clusters, num_rays, 1, 3]
        r_hat_tx = r_hat_tx.unsqueeze(-2)
        r_hat_rx = r_hat_rx.unsqueeze(-2)

        # d_bar_tx has shape [batch_size, num_tx,          num_tx_antennas, 3]
        # and will be reshaped to
        # [batch_size, num_tx, 1, 1, 1, num_tx_antennas, 3]
        d_bar_tx = d_bar_tx.unsqueeze(2).unsqueeze(3).unsqueeze(4)

        # d_bar_rx has shape [batch_size,    num_rx,       num_rx_antennas, 3]
        # and will be reshaped to
        # [batch_size, 1, num_rx, 1, 1, num_rx_antennas, 3]
        d_bar_rx = d_bar_rx.unsqueeze(1).unsqueeze(3).unsqueeze(4)

        # Compute all tensor elements

        # As broadcasting of such high-rank tensors is not fully supported
        # in all cases, we need to do a hack here by explicitly
        # broadcasting one dimension:
        d_bar_rx = d_bar_rx.expand(-1, r_hat_rx.shape[1], -1, -1, -1, -1, -1)
        exp_rx = 2*PI/lambda_0*(r_hat_rx*d_bar_rx).sum(dim=-1, keepdim=True)
        exp_rx = torch.exp(torch.complex(torch.zeros_like(exp_rx),
                                    exp_rx))

        # The hack is for some reason not needed for this term
        # exp_tx = 2*PI/lambda_0*(r_hat_tx*d_bar_tx).sum(
        #     dim=-1, keepdim=True)
        exp_tx = 2*PI/lambda_0*(r_hat_tx*d_bar_tx).sum(dim=-1)
        exp_tx = torch.exp(torch.complex(torch.zeros_like(exp_tx),
                                    exp_tx))
        exp_tx = exp_tx.unsqueeze(-2)

        h_array = exp_rx*exp_tx

        return h_array

    def _step_11_field_matrix(self, topology, aoa, aod, zoa, zod, h_phase):
        # pylint: disable=line-too-long
        r"""
        Compute matrix accounting for the element responses, random phases
        and xpr

        Input
        -----
        topology : Topology
            Topology of the network

        aoa : [batch size, num TXs, num RXs, num clusters, num rays], `torch.float`
            Azimuth angles of arrivals [radian]

        aod : [batch size, num TXs, num RXs, num clusters, num rays], `torch.float`
            Azimuth angles of departure [radian]

        zoa : [batch size, num TXs, num RXs, num clusters, num rays], `torch.float`
            Zenith angles of arrivals [radian]

        zod : [batch size, num TXs, num RXs, num clusters, num rays], `torch.float`
            Zenith angles of departure [radian]

        h_phase : [batch size, num_tx, num rx, num clusters, num rays, num time steps], `torch.complex`
            Matrix with phase shifts due to mobility in (7.5-22)

        Output
        ------
        h_field : [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas], `torch.complex`
            Matrix accounting for element responses, random phases and xpr
        """

        tx_orientations = topology.tx_orientations
        rx_orientations = topology.rx_orientations

        # Transform departure angles to the LCS
        tx_orientations = tx_orientations.reshape(
            tx_orientations.shape[0], tx_orientations.shape[1],
            1, 1, 1, tx_orientations.shape[-1])
        zod_prime, aod_prime = self._gcs_to_lcs(tx_orientations, zod, aod)

        # Transform arrival angles to the LCS
        rx_orientations = rx_orientations.reshape(
            rx_orientations.shape[0], 1, rx_orientations.shape[1],
            1, 1, rx_orientations.shape[-1])
        zoa_prime, aoa_prime = self._gcs_to_lcs(rx_orientations, zoa, aoa)

        # Compute transmitted and received field strength for all antennas
        # in the LCS  and convert to GCS
        f_tx_pol1_prime = torch.stack(self._tx_array.ant_pol1.field(zod_prime,
                                                            aod_prime), dim=-1)
        f_rx_pol1_prime = torch.stack(self._rx_array.ant_pol1.field(zoa_prime,
                                                            aoa_prime), dim=-1)

        f_tx_pol1 = self._l2g_response(f_tx_pol1_prime, tx_orientations,
            zod, aod)

        f_rx_pol1 = self._l2g_response(f_rx_pol1_prime, rx_orientations,
            zoa, aoa)

        if self._tx_array.polarization == 'dual':
            f_tx_pol2_prime = torch.stack(self._tx_array.ant_pol2.field(
                zod_prime, aod_prime), dim=-1)
            f_tx_pol2 = self._l2g_response(f_tx_pol2_prime, tx_orientations,
                zod, aod)

        if self._rx_array.polarization == 'dual':
            f_rx_pol2_prime = torch.stack(self._rx_array.ant_pol2.field(
                zoa_prime, aoa_prime), dim=-1)
            f_rx_pol2 = self._l2g_response(f_rx_pol2_prime, rx_orientations,
                zoa, aoa)

        # Fill the full channel matrix with field responses
        pol1_tx = torch.matmul(h_phase, torch.complex(f_tx_pol1,
            torch.zeros_like(f_tx_pol1)))
        if self._tx_array.polarization == 'dual':
            pol2_tx = torch.matmul(h_phase, torch.complex(f_tx_pol2,
                torch.zeros_like(f_tx_pol2)))

        num_ant_tx = self._tx_array.num_ant
        if self._tx_array.polarization == 'single':
            # Each BS antenna gets the polarization 1 response
            f_tx_array = pol1_tx.unsqueeze(0).expand(num_ant_tx, *pol1_tx.shape)
        else:
            # Assign polarization reponse according to polarization to each
            # antenna
            pol_tx = torch.stack([pol1_tx, pol2_tx], 0)
            ant_ind_pol2 = self._tx_array.ant_ind_pol2
            num_ant_pol2 = ant_ind_pol2.shape[0]
            # O = Pol 1, 1 = Pol 2, we only scatter the indices for Pol 1,
            # the other elements are already 0
            gather_ind = torch.zeros([num_ant_tx], dtype=torch.int64,
                device=self.device)
            gather_ind[ant_ind_pol2.reshape(-1)] = torch.ones(
                [num_ant_pol2], dtype=torch.int64, device=self.device)
            f_tx_array = pol_tx[gather_ind]

        num_ant_rx = self._rx_array.num_ant
        if self._rx_array.polarization == 'single':
            # Each UT antenna gets the polarization 1 response
            f_rx_array = f_rx_pol1.unsqueeze(0).expand(num_ant_rx, *f_rx_pol1.shape)
            f_rx_array = torch.complex(f_rx_array,
                                    torch.zeros_like(f_rx_array))
        else:
            # Assign polarization response according to polarization to each
            # antenna
            pol_rx = torch.stack([f_rx_pol1, f_rx_pol2], 0)
            ant_ind_pol2 = self._rx_array.ant_ind_pol2
            num_ant_pol2 = ant_ind_pol2.shape[0]
            # O = Pol 1, 1 = Pol 2, we only scatter the indices for Pol 1,
            # the other elements are already 0
            gather_ind = torch.zeros([num_ant_rx], dtype=torch.int64,
                device=self.device)
            gather_ind[ant_ind_pol2.reshape(-1)] = torch.ones(
                [num_ant_pol2], dtype=torch.int64, device=self.device)
            f_rx_array = torch.complex(pol_rx[gather_ind],
                            torch.zeros_like(pol_rx[gather_ind]))

        # Compute the scalar product between the field vectors through
        # reduce_sum and transpose to put antenna dimensions last
        h_field = (f_rx_array.unsqueeze(1)*f_tx_array.unsqueeze(0)).sum(dim=(-2, -1))
        h_field = h_field.permute(*range(2, h_field.dim()), 0, 1)

        return h_field

    def _step_11_nlos(self, phi, topology, rays, t, carrier_frequency):
        # pylint: disable=line-too-long
        r"""
        Compute the full NLOS channel matrix (7.5-28)

        Input
        -----
        phi: [batch size, num TXs, num RXs, num clusters, num rays, 4], `torch.float`
            Random initial phases [radian]

        topology : Topology
            Topology of the network

        rays : Rays
            Rays

        t : [num time samples], `torch.float`
            Time samples

        Output
        ------
        h_full : [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas, num time steps], `torch.complex`
            NLoS channel matrix
        """

        h_phase = self._step_11_phase_matrix(phi, rays)

        # Add Faraday phase rotation in NTN case
        if torch.greater_equal(topology.bs_height, torch.as_tensor(600000.0, device=topology.bs_height.device)):
            rays_shape = rays.aoa.shape
            faraday_phase_rotation = self._step_11_faraday_rotation(carrier_frequency=carrier_frequency, aod_shape=rays_shape)
            h_phase = torch.matmul(h_phase, faraday_phase_rotation)

        h_field = self._step_11_field_matrix(topology, rays.aoa, rays.aod,
                                                    rays.zoa, rays.zod, h_phase)
        h_array = self._step_11_array_offsets(topology, rays.aoa, rays.aod,
                                                            rays.zoa, rays.zod)
        h_doppler = self._step_11_doppler_matrix(topology, rays.aoa, rays.zoa, rays.aod, rays.zod,
                                                                            t)
        h_full = (h_field*h_array).unsqueeze(-1) * h_doppler.unsqueeze(-2).unsqueeze(-2)

        power_scaling = torch.complex(torch.sqrt(rays.powers/
            torch.as_tensor(h_full.shape[4], dtype=self.dtype, device=self.device)),
                            torch.zeros_like(rays.powers))
        for _ in range(h_full.dim() - power_scaling.dim()):
            power_scaling = power_scaling.unsqueeze(-1)
        h_full = h_full * power_scaling

        return h_full

    def _step_11_reduce_nlos(self, h_full, rays, c_ds):
        # pylint: disable=line-too-long
        r"""
        Compute the final NLOS matrix in (7.5-27)

        Input
        ------
        h_full : [batch size, num_tx, num rx, num clusters, num rays, num rx antennas, num tx antennas, num time steps], `torch.complex`
            NLoS channel matrix

        rays : Rays
            Rays

        c_ds : [batch size, num TX, num RX], `torch.float`
            Cluster delay spread

        Output
        -------
        h_nlos : [batch size, num_tx, num rx, num clusters, num rx antennas, num tx antennas, num time steps], `torch.complex`
            Paths NLoS coefficients

        delays_nlos : [batch size, num_tx, num rx, num clusters], `torch.float`
            Paths NLoS delays
        """

        if self._subclustering:

            powers = rays.powers
            delays = rays.delays

            # Sort all clusters along their power
            strongest_clusters = torch.argsort(powers, dim=-1,
                descending=True)

            # Sort delays according to the same ordering
            delays_sorted = torch.gather(delays, dim=3, index=strongest_clusters)

            # Split into delays for strong and weak clusters
            delays_strong = delays_sorted[...,:2]
            delays_weak = delays_sorted[...,2:]

            # Compute delays for sub-clusters
            offsets = self._sub_cl_delay_offsets.reshape(
                (delays_strong.dim()-1)*[1]+[-1]+[1])
            c_ds_t = c_ds if torch.is_tensor(c_ds) else torch.as_tensor(
                c_ds, dtype=delays_strong.dtype, device=delays_strong.device)
            delays_sub_cl = (delays_strong.unsqueeze(-2) +
                offsets*c_ds_t.unsqueeze(-1).unsqueeze(-1))
            delays_sub_cl = delays_sub_cl.reshape(*delays_sub_cl.shape[:-2], -1)

            # Select the strongest two clusters for sub-cluster splitting
            strongest_2 = strongest_clusters[...,:2]
            idx = strongest_2.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            idx = idx.expand(-1, -1, -1, -1, h_full.shape[4], h_full.shape[5],
                h_full.shape[6], h_full.shape[7])
            h_strong = torch.gather(h_full, dim=3, index=idx)

            # The other clusters are the weak clusters
            strongest_rest = strongest_clusters[...,2:]
            idx = strongest_rest.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            idx = idx.expand(-1, -1, -1, -1, h_full.shape[4], h_full.shape[5],
                h_full.shape[6], h_full.shape[7])
            h_weak = torch.gather(h_full, dim=3, index=idx)

            # Sum specific rays for each sub-cluster
            h_sub_cl_1 = h_strong[:, :, :, :, self._sub_cl_1_ind, ...].sum(dim=4)
            h_sub_cl_2 = h_strong[:, :, :, :, self._sub_cl_2_ind, ...].sum(dim=4)
            h_sub_cl_3 = h_strong[:, :, :, :, self._sub_cl_3_ind, ...].sum(dim=4)

            # Sum all rays for the weak clusters
            h_weak = h_weak.sum(dim=4)

            # Concatenate the channel and delay tensors
            h_nlos = torch.cat([h_sub_cl_1, h_sub_cl_2, h_sub_cl_3, h_weak],
                dim=3)
            delays_nlos = torch.cat([delays_sub_cl, delays_weak], dim=3)
        else:
            # Sum over rays
            h_nlos = h_full.sum(dim=4)
            delays_nlos = rays.delays

        # Order the delays in ascending orders
        delays_ind = torch.argsort(delays_nlos, dim=-1)
        delays_nlos = torch.gather(delays_nlos, dim=3, index=delays_ind)

        # # Order the channel clusters according to the delay, too
        idx = delays_ind.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        idx = idx.expand(-1, -1, -1, -1, h_nlos.shape[4], h_nlos.shape[5],
            h_nlos.shape[6])
        h_nlos = torch.gather(h_nlos, dim=3, index=idx)

        return h_nlos, delays_nlos

    def _step_11_los(self, topology, t, carrier_frequency):
        # pylint: disable=line-too-long
        r"""Compute the LOS channels from (7.5-29)

        Intput
        ------
        topology : Topology
            Network topology

        t : [num time samples], `torch.float`
            Number of time samples

        Output
        ------
        h_los : [batch size, num_tx, num rx, 1, num rx antennas, num tx antennas, num time steps], `torch.complex`
            Paths LoS coefficients
        """

        aoa = topology.los_aoa
        aod = topology.los_aod
        zoa = topology.los_zoa
        zod = topology.los_zod

         # LoS departure and arrival angles
        aoa = aoa.unsqueeze(3).unsqueeze(4)
        zoa = zoa.unsqueeze(3).unsqueeze(4)
        aod = aod.unsqueeze(3).unsqueeze(4)
        zod = zod.unsqueeze(3).unsqueeze(4)

        # Field matrix
        h_phase = self._h_phase_los_const.reshape(1, 1, 1, 1, 1, 2, 2)

        # Add Faraday phase rotation in NTN case
        if torch.greater_equal(topology.bs_height, torch.as_tensor(600000.0, device=topology.bs_height.device)):

            rays_shape = aoa.shape
            faraday_phase_rotation = self._step_11_faraday_rotation(carrier_frequency=carrier_frequency, aod_shape=rays_shape)
            h_phase = torch.matmul(h_phase, faraday_phase_rotation)

        h_field = self._step_11_field_matrix(topology, aoa, aod, zoa, zod,
                                                                    h_phase)
        # Array offset matrix
        h_array = self._step_11_array_offsets(topology, aoa, aod, zoa, zod)

        # Doppler matrix
        h_doppler = self._step_11_doppler_matrix(topology, aoa, zoa, aod, zod, t)

        # Phase shift due to propagation delay
        d3d = topology.distance_3d
        lambda_0 = self._lambda_0
        h_delay = torch.exp(torch.complex(torch.zeros_like(d3d),
                        2*PI*d3d/lambda_0))

        # Combining all to compute channel coefficient
        h_field = h_field.squeeze(4).unsqueeze(-1)
        h_array = h_array.squeeze(4).unsqueeze(-1)
        h_doppler = h_doppler.unsqueeze(4)
        h_delay = h_delay.unsqueeze(3).unsqueeze(4).unsqueeze(5).unsqueeze(6)

        h_los = h_field*h_array*h_doppler*h_delay
        return h_los

    def _step_11(self, phi, topology, k_factor, rays, t, c_ds, carrier_frequency):
        # pylint: disable=line-too-long
        r"""
        Combine LOS and LOS components to compute (7.5-30)

        Input
        -----
        phi: [batch size, num TXs, num RXs, num clusters, num rays, 4], `torch.float`
            Random initial phases

        topology : Topology
            Network topology

        k_factor : [batch size, num TX, num RX], `torch.float`
            Rician K-factor

        rays : Rays
            Rays

        t : [num time samples], `torch.float`
            Number of time samples

        c_ds : [batch size, num TX, num RX], `torch.float`
            Cluster delay spread
        """

        h_full = self._step_11_nlos(phi, topology, rays, t, carrier_frequency)
        h_nlos, delays_nlos = self._step_11_reduce_nlos(h_full, rays, c_ds)

        ####  LoS scenario

        h_los_los_comp = self._step_11_los(topology, t, carrier_frequency)
        k_factor_expanded = k_factor
        for _ in range(h_los_los_comp.dim() - k_factor.dim()):
            k_factor_expanded = k_factor_expanded.unsqueeze(-1)
        k_factor_complex = torch.complex(k_factor_expanded,
            torch.zeros_like(k_factor_expanded))

        # Scale NLOS and LOS components according to K-factor
        h_los_los_comp = h_los_los_comp*torch.sqrt(k_factor_complex/(k_factor_complex+1))
        h_los_nlos_comp = h_nlos*torch.sqrt(1/(k_factor_complex+1))

        # Add the LOS component to the zero-delay NLOS cluster
        h_los_cl = h_los_los_comp + h_los_nlos_comp[:,:,:,0:1,...]

        # Combine all clusters into a single tensor
        h_los = torch.cat([h_los_cl, h_los_nlos_comp[:,:,:,1:,...]], dim=3)

        #### LoS or NLoS CIR according to link configuration
        los_indicator = topology.los.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        h = torch.where(los_indicator, h_los, h_nlos)

        return h, delays_nlos

    def _step_11_faraday_rotation(self, carrier_frequency, aod_shape):
        carrier_frequency_in_GHz = carrier_frequency/(1e9)
        psi_in_deg = 108.0/(carrier_frequency_in_GHz*carrier_frequency_in_GHz)
        # TODO there is a bug report here,  appearently passing a Tensor to a NumPy call, which is not supported. Assumably psi in the functions is the issue,
        # but the bug could not yet be replicated. This comment serves as a reminder in case of future resurgence of the issue
        psi_in_rad = psi_in_deg * (PI/180.0)
        cos_val = complex(math.cos(psi_in_rad), 0.0)
        sin_val = complex(math.sin(psi_in_rad), 0.0)
        faraday_cos = torch.full(tuple(aod_shape), cos_val,
            dtype=self.cdtype, device=self.device)
        faraday_sin = torch.full(tuple(aod_shape), sin_val,
            dtype=self.cdtype, device=self.device)
        faraday_minus_sin = torch.full(tuple(aod_shape), -sin_val,
            dtype=self.cdtype, device=self.device)
        faraday_phase_rot = torch.stack(
            [faraday_cos, faraday_minus_sin, faraday_sin, faraday_cos],
            dim=-1).reshape(*tuple(aod_shape), 2, 2)
        return faraday_phase_rot
