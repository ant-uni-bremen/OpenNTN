#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by the Dept. of Communications Engineering of the University of Bremen.
# Based on the implementation of scenario RMa in Sionna TR38.901.
#
"""3GPP TR38.811 suburban channel scenario"""

import torch

from sionna.phy.constants import SPEED_OF_LIGHT, PI
from . import SystemLevelScenario
from . import utils

class SubUrbanScenario(SystemLevelScenario):
    r"""
    3GPP TR 38.811 suburban channel model scenario.

    Parameters
    -----------

    carrier_frequency : float
        Carrier frequency [Hz]

    ut_array : PanelArray
        Panel array used by the UTs. All UTs share the same antenna array
        configuration.

    bs_array : PanelArray
        Panel array used by the BS, the satellite or aerial platform.

    direction : str
        Link direction. Either "uplink" or "downlink".

    elevation_angle : float
        elevation angle of the LOS path of the satellite/HAPS vs. ground horizon in degrees

    enable_pathloss : bool
        If `True`, apply pathloss. Otherwise doesn't. Defaults to `True`.

    enable_shadow_fading : bool
        If `True`, apply shadow fading. Otherwise doesn't.
        Defaults to `True`.

    doppler_enabled : bool
        If `True`, the channel contains the Doppler shift due to the motion of
        the satellite (sections 6.7.1 and 6.8.1 of 38.811 specification); if
        `False`, it does not. The Doppler shift due to the motion of the UTs is
        applied in either case. Defaults to `True`. Must be `True` or `False`;
        `TypeError` otherwise.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    device : `None` (default) | str
        Device for computation (e.g., ``"cpu"``, ``"cuda:0"``).
        If `None`, :attr:`~sionna.phy.config.Config.device` at the
        construction is used. ``ut_array`` and ``bs_array`` must be on this
        device; `ValueError` otherwise.
    """

    def __init__(self, carrier_frequency, ut_array, bs_array,
        direction, elevation_angle, enable_pathloss=True, enable_shadow_fading=True,
        doppler_enabled=True, precision=None, device=None):

        super().__init__(carrier_frequency, ut_array, bs_array,
            direction, elevation_angle, enable_pathloss, enable_shadow_fading, doppler_enabled, precision,
            device)

    #########################################
    # Public methods and properties
    #########################################

    def clip_carrier_frequency_lsp(self, fc):
        r"""Clip the carrier frequency ``fc`` in GHz for LSP calculation

        Input
        -----
        fc : float
            Carrier frequency [GHz]

        Output
        -------
        : float
            Clipped carrier frequency, that should be used for LSp computation
        """
        return fc

    @property
    def min_2d_in(self):
        r"""Minimum indoor 2D distance for indoor UTs [m]"""
        return torch.tensor(0.0, dtype=self.dtype, device=self.device)

    @property
    def max_2d_in(self):
        r"""Maximum indoor 2D distance for indoor UTs [m]"""
        return torch.tensor(10.0, dtype=self.dtype, device=self.device)

    @property
    def los_probability(self):
        r"""Probability of each UT to be LoS. Used to generate LoS
        status of outdoor UTs.

        Taken from table 6.6.1-1 in TR38.811

        [batch size, num_ut]"""

        angle_str = str(utils._reference_elevation(self._elevation_angle))
        los_p = self._params_los["LoS_p" + '_' + angle_str]
        #self._distance_2d_out already has desired shape, this it is used here to keep the code shorter
        #los_probability = torch.zeros_like(self._distance_2d_out) + los_p
        los_probability = torch.zeros_like(self._distance_2d_out) + los_p
        return los_probability

    @property
    def rays_per_cluster(self):
        r"""Number of rays per cluster"""
        return 20

    @property
    def los_parameter_filepath(self):
        r""" Path of the configuration file for LoS scenario"""
        assert (self.carrier_frequency >= 19e9 and self.carrier_frequency <= 40e9 or self.carrier_frequency >= 1.9e9 and self.carrier_frequency <= 4e9)
        if self.direction == "uplink":
            if self.carrier_frequency >= 1.9e9 and self.carrier_frequency <= 4e9:
                return 'Sub_Urban_LOS_S_band_UL.json'
            else:
                return 'Sub_Urban_LOS_Ka_band_UL.json'
        else:
            if self.carrier_frequency >= 1.9e9 and self.carrier_frequency <= 4e9:
                return 'Sub_Urban_LOS_S_band_DL.json'
            else:
                return 'Sub_Urban_LOS_Ka_band_DL.json'
            

    @property
    def nlos_parameter_filepath(self):
        r""" Path of the configuration file for NLoS scenario"""
        assert (self.carrier_frequency >= 19e9 and self.carrier_frequency <= 40e9 or self.carrier_frequency >= 1.9e9 and self.carrier_frequency <= 4e9)
        if self.direction == "uplink":
            if self.carrier_frequency >= 1.9e9 and self.carrier_frequency <= 4e9:
                return 'Sub_Urban_NLOS_S_band_UL.json'
            else:
                return 'Sub_Urban_NLOS_Ka_band_UL.json'
        else:
            if self.carrier_frequency >= 1.9e9 and self.carrier_frequency <= 4e9:
                return 'Sub_Urban_NLOS_S_band_DL.json'
            else:
                return 'Sub_Urban_NLOS_Ka_band_DL.json'

    #########################
    # Utility methods
    #########################
    def _compute_lsp_log_mean_std(self):
        r"""Computes the mean and standard deviations of LSPs in log-domain"""

        batch_size = self.batch_size
        num_bs = self.num_bs
        num_ut = self.num_ut
        distance_2d = self.distance_2d
        h_bs = self.h_bs
        h_bs = h_bs.unsqueeze(2) # For broadcasting
        h_ut = self.h_ut
        h_ut = h_ut.unsqueeze(1) # For broadcasting

        ## Mean
        # DS
        log_mean_ds = self.get_param("muDS")
        # ASD
        log_mean_asd = self.get_param("muASD")
        # ASA
        log_mean_asa = self.get_param("muASA")
        # SF.  Has zero-mean.
        log_mean_sf = torch.zeros([batch_size, num_bs, num_ut],
                                dtype=self.dtype, device=self.device)
        # K.  Given in dB in the 3GPP tables, hence the division by 10
        log_mean_k = self.get_param("muK")/10.0
        # ZSA
        log_mean_zsa = self.get_param("muZSA")
        # ZSD mean is of the form max(-1, A*d2D/1000 - 0.01*(hUT-1.5) + B)
        log_mean_zsd = (self.get_param("muZSD"))
        # The ZSD might be -inf which the system cannot work with. Thus it is replaced by a large negative number
        # TODO -100 is chose here just to be a large negative number. However, it is not yet tested if replacing it with larger
        # numbers further improves the results.
        log_mean_max = torch.maximum(
            torch.tensor(-100.0, dtype=self.dtype, device=self.device),
            log_mean_zsd)
        log_mean_zsd = torch.where(log_mean_zsd != float("-inf"), log_mean_max, log_mean_zsd)

        lsp_log_mean = torch.stack([log_mean_ds,
                                log_mean_asd,
                                log_mean_asa,
                                log_mean_sf,
                                log_mean_k,
                                log_mean_zsa,
                                log_mean_zsd], dim=3)

        ## STD
        # DS
        log_std_ds = self.get_param("sigmaDS")
        # ASD
        log_std_asd = self.get_param("sigmaASD")
        # ASA
        log_std_asa = self.get_param("sigmaASA")
        # SF. Given in dB in the 3GPP tables, hence the division by 10
        # O2I and NLoS cases just require the use of a predefined value
        log_std_sf_o2i_nlos = self.get_param("sigmaSF")/10.0
        # For LoS, two possible scenarion depending on the 2D location of the
        # user
        distance_breakpoint = (2.*PI*h_bs*h_ut*self.carrier_frequency/
            SPEED_OF_LIGHT)
        log_std_sf_los=torch.where(distance_2d < distance_breakpoint,
            self.get_param("sigmaSF")/10.0, self.get_param("sigmaSF")/10.0)
        # Use the correct SF STD according to the user scenario: NLoS/O2I, or
        # LoS
        log_std_sf = torch.where(self.los, log_std_sf_los, log_std_sf_o2i_nlos)
        # K. Given in dB in the 3GPP tables, hence the division by 10.
        log_std_k = self.get_param("sigmaK")/10.0
        # ZSA
        log_std_zsa = self.get_param("sigmaZSA")
        # ZSD
        log_std_zsd = self.get_param("sigmaZSD")

        lsp_log_std = torch.stack([log_std_ds,
                               log_std_asd,
                               log_std_asa,
                               log_std_sf,
                               log_std_k,
                               log_std_zsa,
                               log_std_zsd], dim=3)

        self._update_attr("_lsp_log_mean", lsp_log_mean)
        self._update_attr("_lsp_log_std", lsp_log_std)

        # ZOD offset mu_offset,ZOD of TR 38.901 V16.1.0, eq. (7.5-19). TR 38.811
        # V15.4.0, clause 6.7.2, replaces the parameter tables of TR 38.901 with its
        # own tables, which have no offset row, so the offset is 0 for every link.
        self._update_attr("_zod_offset", torch.zeros_like(distance_2d))

    def _compute_pathloss_gas(self):
        utils._compute_pathloss_gas(self)

    def _compute_pathloss_scintillation(self):
        utils._compute_pathloss_scintillation(self)

    def _compute_pathloss_entry(self):
        utils._compute_pathloss_entry(self)

    def _compute_pathloss_basic(self):
        utils._compute_pathloss_basic(self)