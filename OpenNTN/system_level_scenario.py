#
# This file has been created by the Dept. of Communications Engineering of the University of Bremen.
# The code is based on implementations provided by the NVIDIA CORPORATION & AFFILIATES
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Class used to define a system level 3GPP channel simulation scenario"""

import json
from importlib_resources import files
import tensorflow as tf
from abc import abstractmethod
from sionna.phy.block import Object
from sionna.phy.constants import SPEED_OF_LIGHT, PI
from sionna.phy.utils import log10
from sionna.phy.channel.tr38811.utils import sample_bernoulli
from sionna.phy.channel.utils import rad_2_deg, wrap_angle_0_360
from .antenna import PanelArray
import math

from . import models # pylint: disable=relative-beyond-top-level


class SystemLevelScenario(Object):
    r"""
    This class is used to set up the scenario for system level 3GPP channel
    simulation.

    Scenarios for system level channel simulation, such as DenseUrban, Urban, or SubUrban,
    are defined by implementing this base class.

    Input
    ------
    carrier_frequency : float
        Carrier frequency [Hz]

    ut_array : PanelArray
        Panel array configuration used by UTs

    bs_array : PanelArray
        Panel array configuration used by BSs

    direction : str
        Link direction. Either "uplink" or "downlink"

    elevation_angle : float
        elevation angle of the LOS path of the satellite/HAPS vs. ground horizon in degrees

    enable_pathloss : bool
        If set to `True`, apply pathloss. Otherwise, does not. Defaults to True.

    enable_shadow_fading : bool
        If set to `True`, apply shadow fading. Otherwise, does not.
        Defaults to True.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.
    """

    def __init__(self, carrier_frequency, ut_array, bs_array,
        direction, elevation_angle, enable_pathloss=True, enable_shadow_fading=True, doppler_enabled=True,
        precision=None):
        super().__init__(precision=precision)
        # Direction
        assert direction in ("uplink", "downlink"), \
            "'direction' must be 'uplink' or 'downlink'"
        self._direction = direction

        #As 1.98GHz is specifically mentioned in the standard 3GPP TR38.811, we lowered the S-Band lower end from 2GHz to 1.9GHz
        #As 19.7GHz is specifically mentioned in the standard 3GPP TR38.811, we lowered the Ka-Band lower end from 26GHz to 19GHz
        assert carrier_frequency >= 19e9 and carrier_frequency <= 40e9 or carrier_frequency >= 1.9e9 and carrier_frequency <= 4e9, \
                "Carrier frequency must be either in S Band (1.9GHz-4GHz) or Ka Band (19GHz - 40GHz)"
        #if direction == "downlink":
        #    assert carrier_frequency >= 19e9 and carrier_frequency <= 40e9 or carrier_frequency >= 1.9e9 and carrier_frequency <= 4e9, \
        #        "Carrier frequency in downlink must be either in S Band (1.9GHz-4GHz) or Ka Band (19GHz - 40GHz)"
        #else:
        #    assert carrier_frequency >= 29.5e9 and carrier_frequency <= 30.0e9 or carrier_frequency >= 1.98e9 and carrier_frequency <= 2.01e9, \
        #        "Carrier frequency in downlink must be either in S Band (1.9GHz-4GHz) or Ka Band (19GHz - 40GHz)"
        
        # Carrier frequency (Hz)
        self._carrier_frequency = tf.constant(carrier_frequency,
            self.rdtype)

        # Wavelength (m)
        self._lambda_0 = tf.constant(SPEED_OF_LIGHT/carrier_frequency,
            self.rdtype)

        # UTs and BSs arrays
        assert isinstance(ut_array, PanelArray), \
            "'ut_array' must be an instance of PanelArray"
        assert isinstance(bs_array, PanelArray), \
            "'bs_array' must be an instance of PanelArray"
        self._ut_array = ut_array
        self._bs_array = bs_array

        #assert math.isclose(ut_array._lambda_0, bs_array._lambda_0, abs_tol=0.01) and math.isclose(ut_array._lambda_0, self._lambda_0, abs_tol=0.01) and math.isclose(ut_array._lambda_0, bs_array._lambda_0, abs_tol=0.01), \
        # Assert that the carrier frequencies of all elemnts are the same. As the antennas only save the wavelength, this is compared representatively
        assert ut_array._lambda_0 == bs_array._lambda_0 == self._lambda_0, \
            "The carrier frequencies of ut antenna, bs antenna and scenario must match"
        # data type

        

        assert elevation_angle >= 10.0 and elevation_angle <= 90.0, "elevation angle must be in range [10,90]"
        self._elevation_angle = elevation_angle

        # Pathloss and shadow fading
        self._enable_pathloss = enable_pathloss
        self._enable_shadow_fading = enable_shadow_fading
        self._doppler_enabled = doppler_enabled


        # Scenario
        self._ut_loc = None
        self._bs_loc = None
        self._ut_orientations = None
        self._bs_orientations = None
        self._ut_velocities = None
        self._in_state = None
        self._requested_los = None

        # Load parameters for this scenario
        self._load_params()

    @property
    def carrier_frequency(self):
        r"""Carrier frequency [Hz]"""
        return self._carrier_frequency

    @property
    def direction(self):
        r"""Direction of communication. Either "uplink" or "downlink"."""
        return self._direction

    @property
    def pathloss_enabled(self):
        r"""`True` is pathloss is enabled. `False` otherwise."""
        return self._enable_pathloss

    @property
    def shadow_fading_enabled(self):
        r"""`True` is shadow fading is enabled. `False` otherwise."""
        return self._enable_shadow_fading

    @property
    def lambda_0(self):
        r"""Wavelength [m]"""
        return self._lambda_0

    @property
    def elevation_angle(self):
        r"""Elevation angle of the LOS path of the satellite/HAPS vs. ground horizon in degrees"""
        return self._elevation_angle

    @property
    def batch_size(self):
        """Batch size"""
        return tf.shape(self._ut_loc)[0]

    @property
    def num_ut(self):
        """Number of UTs."""
        return tf.shape(self._ut_loc)[1]

    @property
    def num_bs(self):
        """
        Number of BSs.
        """
        return tf.shape(self._bs_loc)[1]

    @property
    def h_ut(self):
        r"""Heigh of UTs [m]. [batch size, number of UTs]"""
        return self._ut_loc[:,:,2]

    @property
    def h_bs(self):
        r"""Heigh of BSs [m].[batch size, number of BSs]"""
        return self._bs_loc[:,:,2]

    @property
    def ut_loc(self):
        r"""Locations of UTs [m]. [batch size, number of UTs, 3]"""
        return self._ut_loc

    @property
    def bs_loc(self):
        r"""Locations of BSs [m]. [batch size, number of BSs, 3]"""
        return self._bs_loc

    @property
    def ut_orientations(self):
        r"""Orientations of UTs [radian]. [batch size, number of UTs, 3]"""
        return self._ut_orientations

    @property
    def bs_orientations(self):
        r"""Orientations of BSs [radian]. [batch size, number of BSs, 3]"""
        return self._bs_orientations

    @property
    def ut_velocities(self):
        r"""UTs velocities [m/s]. [batch size, number of UTs, 3]"""
        return self._ut_velocities

    @property
    def ut_array(self):
        r"""PanelArray used by UTs."""
        return self._ut_array

    @property
    def bs_array(self):
        r"""PanelArray used by BSs."""
        return self._bs_array

    @property
    def indoor(self):
        r"""
        Indoor state of UTs. `True` is indoor, `False` otherwise.
        [batch size, number of UTs]"""
        return self._in_state

    @property
    def los(self):
        r"""LoS state of BS-UT links. `True` if LoS, `False` otherwise.
        [batch size, number of BSs, number of UTs]"""
        return self._los

    @property
    def distance_2d(self):
        r"""
        Distance between each UT and each BS in the X-Y plan [m].
        [batch size, number of BSs, number of UTs]"""
        return self._distance_2d

    @property
    def distance_2d_in(self):
        r"""Indoor distance between each UT and BS in the X-Y plan [m], i.e.,
        part of the total distance that corresponds to indoor propagation in the
        X-Y plan.
        Set to 0 for UTs located ourdoor.
        [batch size, number of BSs, number of UTs]"""
        return self._distance_2d_in

    @property
    def distance_2d_out(self):
        r"""Outdoor distance between each UT and BS in the X-Y plan [m], i.e.,
        part of the total distance that corresponds to outdoor propagation in
        the X-Y plan.
        Equals to ``distance_2d`` for UTs located outdoor.
        [batch size, number of BSs, number of UTs]"""
        return self._distance_2d_out

    @property
    def distance_3d(self):
        r"""
        Distance between each UT and each BS [m].
        [batch size, number of BSs, number of UTs]"""
        return self._distance_3d

    @property
    def distance_3d_in(self):
        r"""Indoor distance between each UT and BS [m], i.e.,
        part of the total distance that corresponds to indoor propagation.
        Set to 0 for UTs located ourdoor.
        [batch size, number of BSs, number of UTs]"""
        return self._distance_3d_in

    @property
    def distance_3d_out(self):
        r"""Outdoor distance between each UT and BS [m], i.e.,
        part of the total distance that corresponds to outdoor propagation.
        Equals to ``distance_3d`` for UTs located outdoor.
        [batch size, number of BSs, number of UTs]"""
        return self._distance_3d_out

    @property
    def matrix_ut_distance_2d(self):
        r"""Distance between all pairs for UTs in the X-Y plan [m].
        [batch size, number of UTs, number of UTs]"""
        return self._matrix_ut_distance_2d

    # TODO For all angles, we need to make sure that the topology and individual angles are correct.
    # Old implementations sometimes only checked either azimuth or zenith, thus they might run into issues here.
    # The angles between the user and the satellite specifically needs to be calcualated precisely. This seems
    # to work as intended at the moment, however, an additional thourough check is still planned. This is to be
    # done for aod, aoa, zod, and zoa
    @property
    def los_aod(self):
        r"""LoS azimuth angle of departure of each BS-UT link [deg].
        [batch size, number of BSs, number of UTs]"""       
        return self._los_aod


    @property
    def los_aoa(self):
        r"""LoS azimuth angle of arrival of each BS-UT link [deg].
        [batch size, number of BSs, number of UTs]"""
        return self._los_aoa


    @property
    def los_zod(self):
        r"""LoS zenith angle of departure of each BS-UT link [deg].
        [batch size, number of BSs, number of UTs]"""
        return self._los_zod

    @property
    def los_zoa(self):
        r"""considered 90 degrees"""
        return self._los_zoa

    @property
    @abstractmethod
    def los_probability(self):
        r"""Probability of each UT to be LoS. Used to randomly generate LoS
        status of outdoor UTs. [batch size, number of UTs]"""
        pass

    @property
    @abstractmethod
    def min_2d_in(self):
        r"""Minimum indoor 2D distance for indoor UTs [m]"""
        pass

    @property
    @abstractmethod
    def max_2d_in(self):
        r"""Maximum indoor 2D distance for indoor UTs [m]"""
        pass

    @property
    def lsp_log_mean(self):
        r"""
        Mean of LSPs in the log domain.
        [batch size, number of BSs, number of UTs, 7].
        The last dimension corresponds to the LSPs, in the following order:
        DS - ASD - ASA - SF - K - ZSA - ZSD - XPR"""
        return self._lsp_log_mean

    @property
    def lsp_log_std(self):
        r"""
        STD of LSPs in the log domain.
        [batch size, number of BSs, number of UTs, 7].
        The last dimension corresponds to the LSPs, in the following order:
        DS - ASD - ASA - SF - K - ZSA - ZSD - XPR"""
        return self._lsp_log_std

    @property
    @abstractmethod
    def rays_per_cluster(self):
        r"""Number of rays per cluster"""
        pass

    @property
    def zod_offset(self):
        r"""Zenith angle of departure offset"""
        return self._zod_offset

    @property
    def num_clusters_los(self):
        r"""Number of clusters for LoS scenario"""
        angle_str = str(round(self._elevation_angle/10.0)*10)
        return self._params_los["numClusters_" + angle_str]

    @property
    def num_clusters_nlos(self):
        r"""Number of clusters for NLoS scenario"""
        angle_str = str(round(self._elevation_angle/10.0)*10)
        return self._params_nlos["numClusters_" + angle_str]

    @property
    def num_clusters_max(self):
        r"""Maximum number of clusters over LoS and NLoS scenarios"""
        # Different models have different number of clusters
        angle_str = str(round(self._elevation_angle/10.0)*10)
        num_clusters_los = self._params_los["numClusters_" + angle_str]
        num_clusters_nlos = self._params_nlos["numClusters_" + angle_str]
        num_clusters_max = tf.reduce_max([num_clusters_los, num_clusters_nlos])
        return num_clusters_max

    @property
    def basic_pathloss(self):
        r"""Basic pathloss component [dB].
        See section 7.4.1 of 38.901 specification.
        [batch size, num BS, num UT]"""
        return self._pl_b
    
    @property
    def gas_pathloss(self):
        r"""Basic pathloss component [dB].
        See section 7.4.1 of 38.901 specification.
        [batch size, num BS, num UT]"""
        return self._pl_g
    
    @property
    def scintillation_pathloss(self):
        r"""Basic pathloss component [dB].
        See section 7.4.1 of 38.901 specification.
        [batch size, num BS, num UT]"""
        return self._pl_s
    
    @property
    def entry_pathloss(self):
        r"""Basic pathloss component [dB].
        See section 7.4.1 of 38.901 specification.
        [batch size, num BS, num UT]"""
        return self._pl_e
    
    @property
    def additional_pathloss(self):
        r"""Basic pathloss component [dB].
        See section 7.4.1 of 38.901 specification.
        [batch size, num BS, num UT]"""
        return self._pl_a
    
    @property
    def free_space_pathloss(self):
        r"""Free Space Pathloss, which is a part of the pathloss component [dB].
        See section 7.4.1 of 38.901 specification. This variable is only used for model callibrations and testing.
        [batch size, num BS, num UT]"""
        return self._fspl

    @property
    def latitude(self):
        r"""Latitude of each UT, used for additional pathlosses
        See section 6.6.6 of 38.811 specification.
        """
        return self._latitude
    
    @property
    def lwc(self):
        r"""Liquid water content in kg/m^2, used for cloud attenuation
        See section 6.6.5 of 38.811 specification.
        """
        return self._lwc
    
    @property
    def rain_rate(self):
        r"""Rain rate in mm/h, used for rain attenuation
        See section 6.6.5 of 38.811 specification.
        """
        return self._rain_rate
    
    @property
    def atmospheric_pressure(self):
        r"""Atmospheric pressure in hPa, used for gas pathloss
        See section 6.6.4 of 38.811 specification.
        """
        return self._atmospheric_pressure
    
    @property
    def temperature(self):
        r"""Temperature in K, used for scintillation, gas, and cloud/rain losses
        See section 6.6.6 of 38.811 specification.
        """
        return self._temperature
    
    @property
    def water_vapor_density(self):
        r"""Water vapor density in g/m^3, used for gas loss
        See section 6.6.4 of 38.811 specification.
        """
        return self._water_vapor_density
    
    @property
    def relative_humidity(self):
        r"""Relative humidity in percent, used for gas loss
        See section 6.6.4 of 38.811 specification.
        """
        return self._relative_humidity
    
    @property
    def diameter_earth_antenna(self):
        r"""The diamater of the Earth-stationed antenna, used for scintillation loss
        See section 6.6.6 of 38.811 specification.
        """
        return self._diameter_earth_antenna
    
    @property
    def antenna_efficiency(self):
        r"""Efficiency of the Earth-stationed antenna, used for scintillation loss, 0.5 is conservative estimate
        See section 6.6.6 of 38.811 specification.
        """
        return self._antenna_efficiency

    @property
    def doppler_enabled(self):
        r"""Efficiency of the Earth-stationed antenna, used for scintillation loss, 0.5 is conservative estimate
        See section 6.6.6 of 38.811 specification.
        """
        return self._doppler_enabled



    def set_topology(self, ut_loc=None, bs_loc=None, ut_orientations=None,
        bs_orientations=None, ut_velocities=None, in_state=None, los=None, latitude=None,
        lwc=None, rain_rate=None, atmospheric_pressure=None, temperature=None, 
        water_vapor_density=None, relative_humidity=None, diameter_earth_antenna=None, 
        antenna_efficiency=None, doppler_enabled=None):
        r"""
        Set the network topology.

        It is possible to set up a different network topology for each batch
        example.

        When calling this function, not specifying a parameter leads to the
        reuse of the previously given value. Not specifying a value that was not
        set at a former call rises an error.

        Input
        ------
            ut_loc : [batch size, number of UTs, 3], tf.float
                Locations of the UTs [m]

            bs_loc : [batch size, number of BSs, 3], tf.float
                Locations of BSs [m]

            ut_orientations : [batch size, number of UTs, 3], tf.float
                Orientations of the UTs arrays [radian]

            bs_orientations : [batch size, number of BSs, 3], tf.float
                Orientations of the BSs arrays [radian]

            ut_velocities : [batch size, number of UTs, 3], tf.float
                Velocity vectors of UTs [m/s]

            in_state : [batch size, number of UTs], tf.bool
                Indoor/outdoor state of UTs. `True` means indoor and `False`
                means outdoor.

            los : tf.bool or `None`
                If not `None` (default value), all UTs located outdoor are
                forced to be in LoS if ``los`` is set to `True`, or in NLoS
                if it is set to `False`. If set to `None`, the LoS/NLoS states
                of UTs is set following 3GPP specification
                (Section 7.4.2 of TR 38.901).
        """

        assert (ut_loc is not None) or (self._ut_loc is not None),\
            "`ut_loc` is None and was not previously set"

        assert (bs_loc is not None) or (self._bs_loc is not None),\
            "`bs_loc` is None and was not previously set"

        assert (in_state is not None) or (self._in_state is not None),\
            "`in_state` is None and was not previously set"

        assert (ut_orientations is not None)\
            or (self._ut_orientations is not None),\
            "`ut_orientations` is None and was not previously set"

        assert (bs_orientations is not None)\
            or (self._bs_orientations is not None),\
            "`bs_orientations` is None and was not previously set"

        assert (ut_velocities is not None)\
            or (self._ut_velocities is not None),\
            "`ut_velocities` is None and was not previously set"

        # Boolean used to keep track of whether or not we need to (re-)compute
        # the distances between users, correlation matrices...
        # This is required if the UT locations, BS locations, indoor/outdoor
        # state of UTs, or LoS/NLoS states of outdoor UTs are updated.
        need_for_update = False

        #Setting values to standard values if not further specified
        if latitude is not None:
            self._latitude = latitude
        else:
            self._latitude = 47

        if lwc is not None:
            self._lwc = lwc
        else:
            self._lwc = 0.41

        if rain_rate is not None:
            self._rain_rate = rain_rate
        else:
            self._rain_rate = 40

        if atmospheric_pressure is not None:
            self._atmospheric_pressure = atmospheric_pressure
        else:
            self._atmospheric_pressure = 1020

        if temperature is not None:
            self._temperature = temperature
        else:
            self._temperature = 273

        if water_vapor_density is not None:
            self._water_vapor_density = water_vapor_density
        else:
            self._water_vapor_density = 7.5

        if relative_humidity is not None:
            self._relative_humidity = relative_humidity
        else:
            self._relative_humidity = 50

        if diameter_earth_antenna is not None:
            self._diameter_earth_antenna = diameter_earth_antenna
        else:
            self._diameter_earth_antenna = 3.6

        if antenna_efficiency is not None:
            self._antenna_efficiency = antenna_efficiency
        else:
            self._antenna_efficiency = 0.5

        if ut_loc is not None:
            self._ut_loc = ut_loc
            need_for_update = True

        if bs_loc is not None:
            self._bs_loc = bs_loc
            need_for_update = True

        if bs_orientations is not None:
            self._bs_orientations = bs_orientations

        if ut_orientations is not None:
            self._ut_orientations = ut_orientations

        if ut_velocities is not None:
            self._ut_velocities = ut_velocities

        if in_state is not None:
            self._in_state = in_state
            need_for_update = True

        if los is not None:
            self._requested_los = los
            need_for_update = True
        
        if doppler_enabled is not None:
            self._doppler_enabled = doppler_enabled

        if need_for_update:
            # Update topology-related quantities
            self._compute_distance_2d_3d_and_angles()
            self._sample_indoor_distance()
            self._sample_los()
            # Compute the LSPs means and stds
            self._compute_lsp_log_mean_std()
            # Compute the basic path-loss
            self._compute_pathloss_basic()
            # Compute the basic path-loss
            self._compute_pathloss_gas()
            # Compute the basic path-loss
            self._compute_pathloss_entry()
            # Compute the basic path-loss
            self._compute_pathloss_scintillation()
            #The 3GPP model only considers the losses given above. However, there also are
            #models for additional losses due to clouds and rain, which are considered in this function
            #self._compute_pathloss_additional()
        return need_for_update

    def spatial_correlation_matrix(self, correlation_distance):
        r"""Computes and returns a 2D spatial exponential correlation matrix
        :math:`C` over the UTs, such that :math:`C`has shape
        (number of UTs)x(number of UTs), and

        .. math::
            C_{n,m} = \exp{-\frac{d_{n,m}}{D}}

        where :math:`d_{n,m}` is the distance between UT :math:`n` and UT
        :math:`m` in the X-Y plan, and :math:`D` the correlation distance.

        Input
        ------
        correlation_distance : float
            Correlation distance, i.e., distance such that the correlation
            is :math:`e^{-1} \approx 0.37`

        Output
        --------
        : [batch size, number of UTs, number of UTs], float
            Spatial correlation :math:`C`
        """
        spatial_correlation_matrix = tf.math.exp(-self.matrix_ut_distance_2d/
                                                 correlation_distance)
        return spatial_correlation_matrix


    @property
    @abstractmethod
    def los_parameter_filepath(self):
        r""" Path of the configuration file for LoS scenario"""
        pass

    @property
    @abstractmethod
    def nlos_parameter_filepath(self):
        r""" Path of the configuration file for NLoS scenario"""
        pass

    # TODO remove getter from old dtype structure
    #@property
    #def dtype(self):
    #    r"""Complex datatype used for internal calculation and tensors"""
    #    return self._dtype

    @abstractmethod
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
        pass

    def get_param(self, parameter_name):
        r"""
        Given a ``parameter_name`` used in the configuration file, returns a
        tensor with shape [batch size, number of BSs, number of UTs] of the
        parameter value according to each BS-UT link state (LoS, NLoS, indoor).

        Input
        ------
        parameter_name : str
            Name of the parameter used in the configuration file

        Output
        -------
        : [batch size, number of BSs, number of UTs], tf.float
            Parameter value for each BS-UT link
        """

        fc = self._carrier_frequency/1e9
        fc = self.clip_carrier_frequency_lsp(fc)

        parameter_tensor = tf.zeros(shape=[self.batch_size,
                                            self.num_bs,
                                            self.num_ut],
                                            dtype=self.rdtype)

        # Parameter value, rounds elevation angle to nearest table entry 
        if parameter_name not in ("CPhiNLoS", "CThetaNLoS"):
            angle_str = str(round(self._elevation_angle/10.0)*10)
            parameter_value_los = self._params_los[parameter_name + '_' + angle_str]
            parameter_value_nlos = self._params_nlos[parameter_name + '_' + angle_str]
        else:
            parameter_value_los = self._params_los[parameter_name]
            parameter_value_nlos = self._params_nlos[parameter_name]
        # Expand to allow broadcasting with the BS dimension
        indoor = tf.expand_dims(self.indoor, axis=1)
        # LoS
        parameter_value_los = tf.cast(parameter_value_los,
                                        self.rdtype)

        # NLoS
        parameter_value_nlos = tf.cast(parameter_value_nlos,
                                        self.rdtype)
        
        parameter_tensor = tf.where(self.los, parameter_value_los,
            parameter_value_nlos)

        return parameter_tensor

    #####################################################
    # Internal utility methods
    #####################################################

    def _compute_distance_2d_3d_and_angles(self):
        r"""
        Computes the following internal values:
        * 2D distances for all BS-UT pairs in the X-Y plane
        * 3D distances for all BS-UT pairs
        * 2D distances for all pairs of UTs in the X-Y plane
        * LoS AoA, AoD, ZoA, ZoD for all BS-UT pairs

        This function is called at every update of the topology.
        """
        ut_loc = self._ut_loc
        ut_loc = tf.expand_dims(ut_loc, axis=1)

        bs_loc = self._bs_loc
        bs_loc = tf.expand_dims(bs_loc, axis=2)

        delta_loc_xy = ut_loc[:,:,:,:2] - bs_loc[:,:,:,:2]
        delta_loc = ut_loc - bs_loc

        # 2D distances for all BS-UT pairs in the (x-y) plane
        distance_2d = tf.sqrt(tf.reduce_sum(tf.square(delta_loc_xy), axis=3))
        self._distance_2d = distance_2d

        # 3D distances for all BS-UT pairs
        # The 3D distance needs to take the curvature of Earth into account
        elevation_angle = self._elevation_angle
        #convert to radians for tf.math.sin later
        elevation_angle = math.radians(elevation_angle)
        #Radius of the Earth
        R_E = 6371000
 
        #satellite height 
        height_val = self._bs_loc[:,:,2][0]
        #Calculating the 3d distance based on 3GPP TR38.811 6.6-3 to consider the Earth's curveture
        distance_3d = (tf.sqrt(R_E**2 * tf.math.sin(elevation_angle)**2 + height_val**2 + 2*height_val*R_E) - R_E*tf.math.sin(elevation_angle))
        #Expanding to correct shape
        # TODO there is a sometimes occuring bug at this point. The squeeze assumes there to be more than one user, which can run into issues. However,
        # this bug is not yet fully understood and seems to be limited to eager executions. For the future this needs to be fixed. This bug is also assumed
        # to cause the infinite GPU memory allocation in eager execution during the use of the LMMSEEqualizer. 
        # Additionally, tf.squeeze is used here, as distance_3d needs to be a tf.constant, but casting that directly lead to bugs. This workaround works well,
        # but should be replaced by a more straightforward solution in the future.
        distance_3d = tf.fill([self.batch_size, self.num_bs, self.num_ut], tf.squeeze(distance_3d))
        self._distance_3d = distance_3d

        # LoS AoA, AoD, ZoA, ZoD
        los_aod = tf.atan2(delta_loc[:,:,:,1], delta_loc[:,:,:,0])
        los_aoa = los_aod + PI
        los_zod = tf.atan2(distance_2d, delta_loc[:,:,:,2])
        los_zoa = los_zod - PI
        # Angles are converted to degrees and wrapped to (0,360)
        self._los_aod = wrap_angle_0_360(rad_2_deg(los_aod))
        self._los_aoa = wrap_angle_0_360(rad_2_deg(los_aoa))
        self._los_zod = wrap_angle_0_360(rad_2_deg(los_zod))
        self._los_zoa = wrap_angle_0_360(rad_2_deg(los_zoa))

        # 2D distances for all pairs of UTs in the (x-y) plane
        ut_loc_xy = self._ut_loc[:,:,:2]

        ut_loc_xy_expanded_1 = tf.expand_dims(ut_loc_xy, axis=1)
        ut_loc_xy_expanded_2 = tf.expand_dims(ut_loc_xy, axis=2)

        delta_loc_xy = ut_loc_xy_expanded_1 - ut_loc_xy_expanded_2

        matrix_ut_distance_2d = tf.sqrt(tf.reduce_sum(tf.square(delta_loc_xy),
                                                       axis=3))
        self._matrix_ut_distance_2d = matrix_ut_distance_2d

    def _sample_los(self):
        r"""Set the LoS state of each UT randomly, following the procedure
        described in section 7.4.2 of TR 38.901.
        LoS state of each UT is randomly assigned according to a Bernoulli
        distribution, which probability depends on the channel model.
        """
        if self._requested_los is None:
            los_probability = self.los_probability
            los = sample_bernoulli([self.batch_size, self.num_bs,
                                        self.num_ut], los_probability,
                                        precision=self.precision)
        else:
            los = tf.fill([self.batch_size, self.num_bs, self.num_ut],
                            self._requested_los)

        self._los = tf.logical_and(los,
            tf.logical_not(tf.expand_dims(self._in_state, axis=1)))

    def _sample_indoor_distance(self):
        r"""Sample 2D indoor distances for indoor devices, according to section
        7.4.3.1 of TR 38.901.
        """

        indoor = self.indoor
        indoor = tf.expand_dims(indoor, axis=1) # For broadcasting with BS dim
        indoor_mask = tf.where(indoor, tf.constant(1.0, self.rdtype),
            tf.constant(0.0, self.rdtype))

        # Sample the indoor 2D distances for each BS-UT link
        self._distance_2d_in = tf.random.uniform(shape=[self.batch_size,
            self.num_bs, self.num_ut], minval=self.min_2d_in,
            maxval=self.max_2d_in, dtype=self.rdtype)*indoor_mask
        # Compute the outdoor 2D distances
        self._distance_2d_out = self.distance_2d - self._distance_2d_in
        # Compute the indoor 3D distances
        self._distance_3d_in = ((self._distance_2d_in/self.distance_2d)
            *self.distance_3d)
        # Compute the outdoor 3D distances
        self._distance_3d_out = self.distance_3d - self._distance_3d_in

    def _load_params(self):
        r"""Load the configuration files corresponding to the 2 possible states
        of UTs: LoS and NLoS"""
        
        source = files(models).joinpath(self.los_parameter_filepath)
        # pylint: disable=unspecified-encoding
        with open(source) as f:
            self._params_los = json.load(f)
        for param_name in self._params_los :
            v = self._params_los[param_name]
            if isinstance(v, float):
                self._params_los[param_name] = tf.constant(v,
                                                    self.rdtype)
            elif isinstance(v, int):
                self._params_los[param_name] = tf.constant(v, 
                                                    tf.int32)
            elif isinstance(v, str):
                self._params_los[param_name] = tf.constant(float(v), self.rdtype)

        source = files(models).joinpath(self.nlos_parameter_filepath)
        # pylint: disable=unspecified-encoding
        with open(source) as f:
            self._params_nlos = json.load(f)

        for param_name in self._params_nlos :
            v = self._params_nlos[param_name]
            if isinstance(v, float):
                self._params_nlos[param_name] = tf.constant(v,
                                                        self.rdtype)
            elif isinstance(v, int):
                self._params_nlos[param_name] = tf.constant(v,
                                                        tf.int32)
            elif isinstance(v, str):
                self._params_nlos[param_name] = tf.constant(float(v), self.rdtype)

    @abstractmethod
    def _compute_lsp_log_mean_std(self):
        r"""Computes the mean and standard deviations of LSPs in log-domain"""
        pass

    @abstractmethod
    def _compute_pathloss_scintillation(self):
        r"""Computes the scintillation component of the pathloss [dB]"""
        pass

    @abstractmethod
    def _compute_pathloss_basic(self):
        r"""Computes the basic component of the pathloss [dB]"""
        pass

    @abstractmethod
    def _compute_pathloss_gas(self):
        r"""Computes the atmospheric gasses component of the pathloss [dB]"""
        pass

    @abstractmethod
    def _compute_pathloss_entry(self):
        r"""Computes the building entry component of the pathloss [dB]"""
        pass


