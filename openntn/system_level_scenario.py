#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by the Dept. of Communications Engineering of the University of Bremen.
# The code is based on implementations provided by the NVIDIA CORPORATION & AFFILIATES
# in Sionna.
#
"""Class used to define a system level 3GPP channel simulation scenario"""

import json
import math
from abc import abstractmethod
from importlib.resources import files
from typing import Optional

import torch

from sionna.phy import Object
from sionna.phy.constants import SPEED_OF_LIGHT, PI
from sionna.phy.channel.utils import rad_2_deg, wrap_angle_0_360
from sionna.phy.utils import rand
from .utils import (sample_bernoulli, _validate_platform, _is_satellite,
                    _reference_elevation)
from .antenna import PanelArray

from . import models # pylint: disable=relative-beyond-top-level

# Scaling factors C_phi^NLOS and C_theta^NLOS per number of clusters, TR 38.811 V15.4.0
# Tables 6.7.2-1aa and 6.7.2-1ab. They depend on the number of clusters, which changes
# with the elevation angle and the link state, so they are not part of the parameter
# files.
_C_PHI_NLOS = {2: 0.501, 3: 0.680, 4: 0.779, 5: 0.860, 8: 1.018, 10: 1.090, 11: 1.123,
               12: 1.146, 14: 1.190, 15: 1.211, 16: 1.226, 19: 1.273, 20: 1.289}
_C_THETA_NLOS = {2: 0.430, 3: 0.594, 4: 0.697, 8: 0.889, 10: 0.957, 11: 1.031,
                 12: 1.104, 15: 1.1088, 19: 1.184, 20: 1.178}
_SCALING_FACTORS = {"CPhiNLoS": _C_PHI_NLOS, "CThetaNLoS": _C_THETA_NLOS}

# TR 38.811 V15.4.0, NOTE 8 of Tables 6.7.2-1a to 6.7.2-8b: "For satellite (e.g.GEO/LEO),
# the departure angle spreads are zeros, i.e. mu_lgASD and mu_lgZSD are -inf, and
# corresponding standard deviations are zeros." The parameter files hold the tables as
# printed; for a satellite get_param returns these values instead, and zero
# cross-correlations of ASD and ZSD with the other large scale parameters. The satellite
# is the departure end of the rays as the generator draws them in both directions, so the
# note applies to the uplink as well. With the note, the cluster ASD c_ASD at the
# satellite is 0 as well, as clause 6.9.1 states for the link-level models derived from
# this model ("the azimuth and elevation angular spreads of departure, ASD and ZSD, can
# be considered zero, and all cluster AOD and ZOD angles the same"); the urban tables
# print 0.08 to 6.63 degrees. Aerial platforms keep the values of the tables.
_NOTE_8 = {"muASD": -math.inf, "muZSD": -math.inf, "sigmaASD": 0.0, "sigmaZSD": 0.0,
           "cASD": 0.0}
_NOTE_8.update({name: 0.0 for name in (
    "corrASDvsDS", "corrASDvsSF", "corrASDvsASA", "corrASDvsK", "corrZSDvsSF", "corrZSDvsK",
    "corrZSDvsDS", "corrZSDvsASD", "corrZSAvsASD", "corrZSDvsASA", "corrZSDvsZSA")})


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

    doppler_enabled : bool
        If `True`, the channel contains the Doppler shift due to the motion of the
        satellite (sections 6.7.1 and 6.8.1 of 38.811 specification); if `False`, it
        does not. The Doppler shift due to the motion of the UTs is applied in either
        case. Defaults to `True`. Must be `True` or `False`; `TypeError` otherwise.

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
        direction, elevation_angle, enable_pathloss=True, enable_shadow_fading=True, doppler_enabled=True,
        precision=None, device=None):
        # A flag, checked so that a call that passes an average street width by position
        # after enable_shadow_fading raises instead of taking it as the flag
        if not isinstance(doppler_enabled, bool):
            raise TypeError(f"doppler_enabled must be True or False, not "
                            f"{doppler_enabled!r}")
        super().__init__(precision=precision, device=device)
        # Direction
        if not (direction in ("uplink", "downlink")):
            raise ValueError("'direction' must be 'uplink' or 'downlink'")
        self._direction = direction

        #As 1.98GHz is specifically mentioned in the standard 3GPP TR38.811, we lowered the S-Band lower end from 2GHz to 1.9GHz
        #As 19.7GHz is specifically mentioned in the standard 3GPP TR38.811, we lowered the Ka-Band lower end from 26GHz to 19GHz
        if not (carrier_frequency >= 19e9 and carrier_frequency <= 40e9 or carrier_frequency >= 1.9e9 and carrier_frequency <= 4e9):
            raise ValueError(
                "Carrier frequency must be either in S Band (1.9GHz-4GHz) or Ka Band (19GHz - 40GHz)")
        #if direction == "downlink":
        #    assert carrier_frequency >= 19e9 and carrier_frequency <= 40e9 or carrier_frequency >= 1.9e9 and carrier_frequency <= 4e9, \
        #        "Carrier frequency in downlink must be either in S Band (1.9GHz-4GHz) or Ka Band (19GHz - 40GHz)"
        #else:
        #    assert carrier_frequency >= 29.5e9 and carrier_frequency <= 30.0e9 or carrier_frequency >= 1.98e9 and carrier_frequency <= 2.01e9, \
        #        "Carrier frequency in downlink must be either in S Band (1.9GHz-4GHz) or Ka Band (19GHz - 40GHz)"

        # Carrier frequency (Hz) as a Python float, for the LOS phase: in single
        # precision the buffer below does not hold every frequency exactly (30 GHz)
        self._carrier_frequency_float = float(carrier_frequency)

        # Carrier frequency (Hz)
        # Register as buffers for CUDAGraph compatibility
        self.register_buffer("_carrier_frequency",
            torch.tensor(carrier_frequency, dtype=self.dtype, device=self.device))

        # Wavelength (m)
        self.register_buffer("_lambda_0",
            torch.tensor(SPEED_OF_LIGHT/carrier_frequency, dtype=self.dtype,
                device=self.device))

        # UTs and BSs arrays
        if not (isinstance(ut_array, PanelArray)):
            raise TypeError("'ut_array' must be an instance of PanelArray")
        if not (isinstance(bs_array, PanelArray)):
            raise TypeError("'bs_array' must be an instance of PanelArray")
        # An array creates its tensors on its own device, so an array on another device
        # would make the computations with the scenario's tensors fail. Sionna 2.2.0
        # rejects it when it is assigned; this check, before the assignment, gives the
        # same error with Sionna 2.0.1
        for name, array in (("ut_array", ut_array), ("bs_array", bs_array)):
            if array.device != self.device:
                raise ValueError(
                    f"{type(self).__name__} is on device {self.device!r}, but its "
                    f"{name} ({type(array).__name__}) is on device {array.device!r}. "
                    f"Build both on the same device, e.g. "
                    f"{type(array).__name__}(..., device={self.device!r}).")
        # An array also creates its tensors in its own precision, and a call with arrays
        # in another precision than the scenario fails. Checked before the wavelengths,
        # which each object rounds in its own precision, so that the error names the
        # precisions and not the carrier frequencies
        for name, array in (("ut_array", ut_array), ("bs_array", bs_array)):
            if array.precision != self.precision:
                raise ValueError(
                    f"{type(self).__name__} is in {self.precision} precision, but its "
                    f"{name} ({type(array).__name__}) is in {array.precision} "
                    f"precision. Build the arrays in the precision of the model, e.g. "
                    f"{type(array).__name__}(..., precision={self.precision!r}).")
        self._ut_array = ut_array
        self._bs_array = bs_array

        # Check that the carrier frequencies of all elements are the same. As the antennas only save the wavelength, this is compared representatively
        if not (ut_array._lambda_0 == bs_array._lambda_0 == self._lambda_0):
            raise ValueError(
                "The carrier frequencies of ut antenna, bs antenna and scenario must match")
        # data type



        if not (elevation_angle >= 10.0 and elevation_angle <= 90.0):
            raise ValueError("elevation angle must be in range [10,90]")
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

        # Atmospheric parameters. Set once here; set_topology changes only the
        # parameters it is passed. Pressure, temperature and water vapour density
        # default to the mean annual global reference atmosphere of TR 38.811
        # V15.4.0, clause 6.6.4: T = 288.15 K, p = 1013.25 hPa, rho = 7.5 g/m^3.
        self._latitude = 47
        self._atmospheric_pressure = 1013.25
        self._temperature = 288.15
        self._water_vapor_density = 7.5

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
        return self._ut_loc.shape[0]

    @property
    def num_ut(self):
        """Number of UTs."""
        return self._ut_loc.shape[1]

    @property
    def num_bs(self):
        """
        Number of BSs.
        """
        return self._bs_loc.shape[1]

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
        Indoor state of UTs. `True` is indoor, `False` otherwise. Always `False`: the
        models accept outdoor UTs only.
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
    def los_phase(self):
        r"""Propagation phase -2 pi d3D f_c / c of the LoS path of each BS-UT link,
        reduced to [0, 2 pi) in double precision [rad].
        [batch size, number of BSs, number of UTs]"""
        return self._los_phase

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
        r"""Zenith angle of departure offset [deg], 0 for every link: the parameter
        tables of TR 38.811 have no offset"""
        return self._zod_offset

    @property
    def num_clusters_los(self):
        r"""Number of clusters for LoS scenario"""
        angle_str = str(_reference_elevation(self._elevation_angle))
        return self._params_los["numClusters_" + angle_str]

    @property
    def num_clusters_nlos(self):
        r"""Number of clusters for NLoS scenario"""
        angle_str = str(_reference_elevation(self._elevation_angle))
        return self._params_nlos["numClusters_" + angle_str]

    @property
    def num_clusters_max(self):
        r"""Maximum number of clusters over LoS and NLoS scenarios"""
        # Different models have different number of clusters
        angle_str = str(_reference_elevation(self._elevation_angle))
        num_clusters_los = self._params_los["numClusters_" + angle_str]
        num_clusters_nlos = self._params_nlos["numClusters_" + angle_str]
        num_clusters_max = max(num_clusters_los, num_clusters_nlos)
        return num_clusters_max

    @property
    def basic_pathloss(self):
        r"""Basic path loss PL_b [dB] without its shadow fading, section 6.6.2 of 38.811
        specification: the free space path loss of eq. (6.6-2) plus, for NLOS links,
        the clutter loss of Tables 6.6.2-1 to 6.6.2-3. Eq. (6.6-4) writes the shadow
        fading SF into PL_b; the models apply it once, as the large scale parameter SF
        of step 4 of section 7.5 of 38.901 specification, in step 12, because Tables
        6.7.2-1a to 6.7.2-8b of 38.811 specification give its cross-correlations and
        correlation distance.
        [batch size, num BS, num UT]"""
        return self._pl_b

    @property
    def gas_pathloss(self):
        r"""Attenuation by atmospheric gases PL_g [dB], section 6.6.4 of 38.811
        specification, eq. (6.6-8): the zenith attenuation of ITU-R P.676-10, Annex 2,
        divided by the sine of the elevation angle.
        [batch size, num BS, num UT]"""
        return self._pl_g

    @property
    def scintillation_pathloss(self):
        r"""Scintillation loss PL_s [dB], section 6.6.6 of 38.811 specification: the
        ionospheric scintillation of section 6.6.6.1.4 below 6 GHz, the tropospheric
        scintillation of section 6.6.6.2.1 from 6 GHz on. Below 6 GHz links to an aerial
        platform get 0: Table 6.10.1-1 gives the scintillation as negligible for UAS
        including HAPS below 6 GHz.
        [batch size, num BS, num UT]"""
        return self._pl_s

    @property
    def entry_pathloss(self):
        r"""Building entry loss PL_e [dB], section 6.6.3 of 38.811 specification. It is 0
        for every link, because the models accept outdoor UTs only.
        [batch size, num BS, num UT]"""
        return self._pl_e

    @property
    def free_space_pathloss(self):
        r"""Free space path loss FSPL [dB], eq. (6.6-2) of 38.811 specification, which
        is a part of the basic path loss. This variable is only used for model
        calibrations and testing.
        [batch size, num BS, num UT]"""
        return self._fspl

    @property
    def latitude(self):
        r"""Latitude of the UTs relative to the magnetic equator in degrees. Below
        6 GHz it decides the ionospheric scintillation loss of links to a satellite,
        section 6.6.6.1.4 of 38.811 specification: A_IS within 20 deg of the magnetic
        equator, 0 at higher latitudes. Links to an aerial platform get 0 at every
        latitude (Table 6.10.1-1 of 38.811 specification).
        """
        return self._latitude

    @property
    def atmospheric_pressure(self):
        r"""Dry air pressure in hPa, used for the gas loss. Defaults to 1013.25, the
        reference atmosphere of section 6.6.4 of 38.811 specification, which calls it
        the dry air pressure; the total pressure is this plus the water vapour partial
        pressure.
        """
        return self._atmospheric_pressure

    @property
    def temperature(self):
        r"""Temperature in K, used for the gas loss (section 6.6.4 of 38.811
        specification). Defaults to 288.15, the reference atmosphere of section 6.6.4.
        """
        return self._temperature

    @property
    def water_vapor_density(self):
        r"""Water vapor density in g/m^3, used for gas loss
        See section 6.6.4 of 38.811 specification.
        """
        return self._water_vapor_density

    @property
    def doppler_enabled(self):
        r"""If `True`, the channel contains the Doppler shift due to the motion of the
        satellite (sections 6.7.1 and 6.8.1 of 38.811 specification); if `False`, it
        does not. The Doppler shift due to the motion of the UTs is applied in either
        case.
        """
        return self._doppler_enabled



    def _register_buffer_safe(self, name, tensor):
        # register a buffer, replacing any same-named non-buffer attr (topology attrs are None-init'd)
        if hasattr(self, name) and name not in self._buffers:
            delattr(self, name)
        self.register_buffer(name, tensor)

    def _update_attr(self, name, value):
        # store topology tensors as buffers so .to()/.cuda() moves them; copy_ when shape is stable, re-register otherwise
        # The tensors derived from the topology (the LSP means and standard deviations,
        # the ZOD offset, the path loss terms) are stored the same way, so that their
        # addresses stay the same across topologies of the same shapes, as CUDA graphs
        # need: a graph records the addresses of its inputs.
        existing = getattr(self, name, None)
        # A value or a buffer that requires gradients is registered anew: copy_ into a
        # buffer that carries the graph of an earlier call would chain the graphs, and
        # the backward pass of the later call would fail. Without gradients copy_ keeps
        # the buffer, as torch.compile and CUDA graphs expect.
        if (existing is not None and name in self._buffers and existing.shape == value.shape
                and not value.requires_grad and not existing.requires_grad):
            existing.copy_(value)
        else:
            # Register a copy: the caller's tensor must not become the buffer, or the
            # copy_ of a later call would overwrite it. clone() rather than detach()
            # keeps gradients with respect to the topology.
            self._register_buffer_safe(name, value.clone())

    def _check_device_and_precision(self, model: Optional[Object] = None) -> None:
        r"""Checks that the tensors are still on the device and in the precision of the
        construction

        A model stays on the device and in the precision of its construction, as
        Sionna's models do: ``.to()`` moves or casts its registered tensors, but neither
        the device and the precision of the model and its parts nor the tensors held
        outside the buffers, so a computation would fail inside with an error from
        PyTorch. The device and the dtype of the carrier frequency, a buffer registered
        at the construction, are compared with those of the model; no tensor value is
        read.

        :param model: The model that the error names with its device and precision.
            Defaults to `None`, the scenario itself.

        :output: `None`. Raises `ValueError` if the carrier frequency is on another
            device or of another dtype than the model.
        """
        model = self if model is None else model
        reference = self._carrier_frequency
        if (reference.device != torch.device(model.device)
                or reference.dtype != model.dtype):
            raise ValueError(
                f"{type(model).__name__} was built on device {model.device!r} in "
                f"{model.precision} precision, but its tensors are {reference.dtype} on "
                f"{reference.device}; build the model with the device and precision it "
                f"is to use instead of moving or casting it")

    def set_topology(self, ut_loc=None, bs_loc=None, ut_orientations=None,
        bs_orientations=None, ut_velocities=None, in_state=None, los=None, latitude=None,
        atmospheric_pressure=None, temperature=None, water_vapor_density=None,
        doppler_enabled=None):
        r"""
        Set the network topology.

        It is possible to set up a different network topology for each batch
        example.

        When calling this function, not specifying a parameter leads to the
        reuse of the previously given value. Not specifying a value that was not
        set at a former call rises an error.

        The topology tensors may require gradients. The scenario stores copies through
        which the gradients flow, so its quantities (for example the 2D distances and
        the LoS angles) stay differentiable with respect to the tensors passed, in every
        call. The path coefficients and delays of the channel models are detached, so
        gradients exist only for the quantities of the scenario.

        The topology tensors and the tensors derived from them (for example the 2D
        distances, the LoS angles, the path loss terms and the means of the large
        scale parameters) are written into the existing tensors when their shapes stay
        the same and no gradients are involved, so that their addresses stay the same,
        as CUDA graphs need. A tensor taken from the scenario after ``set_topology`` is
        therefore overwritten by the next ``set_topology`` of the same shapes; clone it
        to keep its values.

        Input
        ------
            ut_loc : [batch size, number of UTs, 3], torch.float
                Locations of the UTs [m]

            bs_loc : [batch size, number of BSs, 3], torch.float
                Locations of BSs [m]. One satellite or aerial platform per call
                (number of BSs 1), at the same height in every batch example: a
                satellite from 160 km to 35 786 km or an aerial platform from 8 km
                to 50 km; `ValueError` otherwise. TR 38.811, clause 3.1, places
                aerial platforms at 8 km to 50 km and LEO satellites from 500 km on;
                the models accept satellites from 160 km, very low orbits included.
                Table 4.5-1 gives typical altitudes (LEO 600 km to 1500 km, MEO
                7000 km to 20 000 km, GEO 35 786 km), Table 5.1-1 the reference
                deployments (GEO, non-GEO down to 600 km, UAS 8 km to 50 km). The
                Faraday rotation of clause 6.8.2, given "In case of propagation for
                spaceborne BS above the ionosphere", applies from 600 km, the lowest
                satellite altitude of TR 38.811; a satellite below 600 km flies
                within the F region of the ionosphere, for which TR 38.811 gives no
                Faraday rotation.

            ut_orientations : [batch size, number of UTs, 3], torch.float
                Orientations of the UTs arrays [radian]

            bs_orientations : [batch size, number of BSs, 3], torch.float
                Orientations of the BSs arrays [radian]

            ut_velocities : [batch size, number of UTs, 3], torch.float
                Velocity vectors of UTs [m/s]

            in_state : [batch size, number of UTs], torch.bool
                Indoor/outdoor state of UTs. `True` means indoor and `False`
                means outdoor. The models accept outdoor UTs only, for satellites
                and aerial platforms, `ValueError` otherwise. TR 38.811 considers
                only outdoor UTs for satellites (clause 6.1.2; Table 5.1-1, "100%
                Outdoors" for the satellite deployments) and indoor UTs as well
                for aerial platforms (clause 6.1.2; Table 5.1-1, "Indoor and
                Outdoor" for UAS including HAPS), but the models have neither the
                building entry loss of clause 6.6.3 nor O2I clusters.

            los : torch.bool or `None`
                If not `None` (default value), all UTs located outdoor are
                forced to be in LoS if ``los`` is set to `True`, or in NLoS
                if it is set to `False`. If set to `None`, the LoS/NLoS states
                of UTs is set following 3GPP specification
                (Section 7.4.2 of TR 38.901).

            atmospheric_pressure : float
                Dry air pressure p [hPa], used for the gas loss (ITU-R P.676-10,
                Annex 2, with the total pressure p + e and the water vapour
                partial pressure e = rho T / 216.7). Defaults to 1013.25, which
                TR 38.811, clause 6.6.4, gives as the dry air pressure of the
                mean annual global reference atmosphere.

            temperature : float
                Temperature [K], used for the gas loss. Defaults to 288.15, the
                mean annual global reference atmosphere of TR 38.811, clause
                6.6.4.

            water_vapor_density : float
                Water vapour density [g/m^3], used for the gas loss. Defaults
                to 7.5.

            latitude : float
                Latitude of the UTs relative to the magnetic equator [deg], from
                -90 to 90 (`ValueError` otherwise). Below 6 GHz the scintillation
                loss of links to a satellite is the ionospheric scintillation of
                TR 38.811, clause 6.6.6.1.4, for latitudes within 20 deg of the
                magnetic equator, and 0 at higher latitudes; links to an aerial
                platform get 0 at every latitude, as Table 6.10.1-1 gives the
                scintillation as negligible for UAS including HAPS below 6 GHz.
                Defaults to 47, where the loss is 0,
                as in the calibration of TR 38.821, Table 6.1.1.1-5 (UEs between
                20 and 60 deg of latitude).

            doppler_enabled : bool or `None`
                If not `None`, replaces the flag ``doppler_enabled`` of the
                construction, which switches the Doppler shift due to the motion
                of the satellite. Must be `True`, `False` or `None`; `TypeError`
                otherwise. Defaults to `None`.

            An atmospheric parameter that is passed without new topology
            tensors updates the gas and scintillation losses of the current
            topology; nothing is drawn again.

        Output
        -------
            need_for_update : bool
                `True` if the topology changed, so that the large scale
                parameters must be drawn again
        """

        # A scenario moved to another device or cast to another precision raises
        # ValueError before it computes anything
        self._check_device_and_precision()

        if not ((ut_loc is not None) or (self._ut_loc is not None)):
            raise ValueError("`ut_loc` is None and was not previously set")

        if not ((bs_loc is not None) or (self._bs_loc is not None)):
            raise ValueError("`bs_loc` is None and was not previously set")

        if not ((in_state is not None) or (self._in_state is not None)):
            raise ValueError("`in_state` is None and was not previously set")

        if not ((ut_orientations is not None)
                or (self._ut_orientations is not None)):
            raise ValueError("`ut_orientations` is None and was not previously set")

        if not ((bs_orientations is not None)
                or (self._bs_orientations is not None)):
            raise ValueError("`bs_orientations` is None and was not previously set")

        if not ((ut_velocities is not None)
                or (self._ut_velocities is not None)):
            raise ValueError("`ut_velocities` is None and was not previously set")

        # One satellite or aerial platform at one height per call, the latitude, the
        # Doppler flag and the outdoor state of the UTs, checked before anything is
        # stored
        if bs_loc is not None:
            bs_loc = self._convert(bs_loc)
            _validate_platform(bs_loc)
        if latitude is not None and not abs(latitude) <= 90.0:
            raise ValueError(f"latitude {latitude} deg outside -90 deg to 90 deg")
        if doppler_enabled is not None and not isinstance(doppler_enabled, bool):
            raise TypeError(f"doppler_enabled must be True, False or None, not "
                            f"{doppler_enabled!r}")
        # Outdoor UTs only, for satellites and aerial platforms: TR 38.811 V15.4.0
        # considers only outdoor UTs for satellites and indoor UTs as well for aerial
        # platforms (clause 6.1.2, Table 5.1-1), but the models have neither the
        # building entry loss of clause 6.6.3 nor O2I clusters
        if in_state is not None and bool(torch.as_tensor(in_state).any()):
            raise ValueError("indoor UTs are not supported: the models accept outdoor "
                             "UTs only (in_state False for every UT)")

        # Boolean used to keep track of whether or not we need to (re-)compute
        # the distances between users, correlation matrices...
        # This is required if the UT locations, BS locations, indoor/outdoor
        # state of UTs, or LoS/NLoS states of outdoor UTs are updated.
        need_for_update = False

        # Atmospheric parameters: a parameter that is not passed keeps its previous
        # value. The gas loss depends on the first three, the scintillation loss
        # below 6 GHz on the latitude.
        atmosphere_updated = False
        for name, value in (("atmospheric_pressure", atmospheric_pressure),
                            ("temperature", temperature),
                            ("water_vapor_density", water_vapor_density),
                            ("latitude", latitude)):
            if value is not None:
                setattr(self, "_" + name, value)
                atmosphere_updated = True

        if ut_loc is not None:
            self._update_attr("_ut_loc", self._convert(ut_loc))
            need_for_update = True

        if bs_loc is not None:
            self._update_attr("_bs_loc", bs_loc)
            need_for_update = True

        if bs_orientations is not None:
            self._update_attr("_bs_orientations", self._convert(bs_orientations))

        if ut_orientations is not None:
            self._update_attr("_ut_orientations", self._convert(ut_orientations))

        if ut_velocities is not None:
            self._update_attr("_ut_velocities", self._convert(ut_velocities))

        if in_state is not None:
            if isinstance(in_state, torch.Tensor):
                self._update_attr("_in_state", in_state.to(device=self.device))
            else:
                self._update_attr("_in_state", torch.as_tensor(in_state, device=self.device))
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
        elif atmosphere_updated:
            # Only the losses that the atmospheric parameters can change are
            # recomputed; they are deterministic, so nothing is drawn again.
            self._compute_pathloss_gas()
            self._compute_pathloss_scintillation()
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
        spatial_correlation_matrix = torch.exp(-self.matrix_ut_distance_2d/
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
        parameter value according to each BS-UT link state (LoS or NLoS; the
        models accept outdoor UTs only).

        Input
        ------
        parameter_name : str
            Name of the parameter used in the configuration file, or
            "CPhiNLoS" or "CThetaNLoS" for the scaling factors of TR 38.811
            Tables 6.7.2-1aa and 6.7.2-1ab, selected by the number of
            clusters of each link state

        For a satellite (160 km to 35 786 km), the departure spreads at the
        satellite are zero, as NOTE 8 of TR 38.811 Tables 6.7.2-1a to 6.7.2-8b
        states: "muASD" and "muZSD" are -inf, "sigmaASD" and "sigmaZSD" are 0, the
        cross-correlations of ASD and ZSD with the other large scale parameters are
        0, and so is the cluster ASD "cASD" (clause 6.9.1: "all cluster AOD and ZOD
        angles the same"), in both link directions. Aerial platforms get the values
        of the configuration file, which are those of the tables.

        Output
        -------
        : [batch size, number of BSs, number of UTs], torch.float
            Parameter value for each BS-UT link
        """

        fc = self._carrier_frequency/1e9
        fc = self.clip_carrier_frequency_lsp(fc)

        parameter_tensor = torch.zeros(self.batch_size,
                                       self.num_bs,
                                       self.num_ut,
                                       dtype=self.dtype,
                                       device=self.device)

        # Parameter value, rounds elevation angle to nearest table entry
        if parameter_name not in _SCALING_FACTORS:
            angle_str = str(_reference_elevation(self._elevation_angle))
            parameter_value_los = self._params_los[parameter_name + '_' + angle_str]
            parameter_value_nlos = self._params_nlos[parameter_name + '_' + angle_str]
        else:
            # The factor of the number of clusters of each link state at this
            # elevation angle (TR 38.901 V16.1.0, 7.5 step 7)
            factors = _SCALING_FACTORS[parameter_name]
            parameter_value_los = factors[self.num_clusters_los]
            parameter_value_nlos = factors[self.num_clusters_nlos]
        # Expand to allow broadcasting with the BS dimension
        indoor = self.indoor.unsqueeze(1)
        # LoS
        if isinstance(parameter_value_los, torch.Tensor):
            parameter_value_los = parameter_value_los.to(dtype=self.dtype,
                device=self.device)
        else:
            parameter_value_los = torch.tensor(parameter_value_los,
                dtype=self.dtype, device=self.device)

        # NLoS
        if isinstance(parameter_value_nlos, torch.Tensor):
            parameter_value_nlos = parameter_value_nlos.to(dtype=self.dtype,
                device=self.device)
        else:
            parameter_value_nlos = torch.tensor(parameter_value_nlos,
                dtype=self.dtype, device=self.device)

        parameter_tensor = torch.where(self.los, parameter_value_los,
            parameter_value_nlos)

        # NOTE 8 for satellites (see _NOTE_8)
        if parameter_name in _NOTE_8:
            satellite = _is_satellite(self.h_bs).unsqueeze(2)
            parameter_tensor = torch.where(satellite,
                torch.full_like(parameter_tensor, _NOTE_8[parameter_name]),
                parameter_tensor)

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
        ut_loc = ut_loc.unsqueeze(1)

        bs_loc = self._bs_loc
        bs_loc = bs_loc.unsqueeze(2)

        delta_loc_xy = ut_loc[:,:,:,:2] - bs_loc[:,:,:,:2]
        delta_loc = ut_loc - bs_loc

        # 2D distances for all BS-UT pairs in the (x-y) plane
        distance_2d = torch.sqrt(torch.sum(torch.square(delta_loc_xy), dim=3))
        self._update_attr("_distance_2d", distance_2d)

        # 3D distances for all BS-UT pairs
        # The 3D distance needs to take the curvature of Earth into account
        elevation_angle = self._elevation_angle
        #convert to radians for torch.sin later
        elevation_angle = math.radians(elevation_angle)
        #Radius of the Earth
        R_E = 6371000.0

        #satellite height [num_bs], the same in every batch example (set_topology)
        height_val = self._bs_loc[:,:,2][0].to(torch.float64)
        #Calculating the 3d distance based on TR 38.811 V15.4.0, eq. (6.6-3) to consider the Earth's
        #curveture.
        #The LoS phase 2 pi d3D f_c / c needs d3D to a small fraction of a wavelength, so d3D is
        #computed in double precision and in a form without cancellation,
        #d3D = (h^2 + 2 h R_E) / (sqrt(R_E^2 sin^2(a) + h^2 + 2 h R_E) + R_E sin(a)):
        #sqrt(...) - R_E sin(a) loses up to 0.77 m at 600 km in single precision.
        sin_el = math.sin(elevation_angle)
        h_term = height_val*height_val + 2.0*height_val*R_E
        distance_3d = h_term / (torch.sqrt(R_E**2 * sin_el**2 + h_term) + R_E*sin_el)
        #Expanding to correct shape. The 3D distance only depends on the elevation angle and the
        #satellite height, so it is identical across the batch and UT dimensions. It is broadcast
        #to [batch size, number of BSs, number of UTs].
        distance_3d = distance_3d.reshape(1, self.num_bs, 1).expand(
            self.batch_size, self.num_bs, self.num_ut).contiguous()
        self._update_attr("_distance_3d", distance_3d.to(self.dtype))

        # LoS propagation phase -2 pi d3D / lambda_0 of TR 38.901 V16.1.0, eq. (7.5-29),
        # reduced modulo 2 pi in double precision before the cast: at 600 km the
        # unreduced phase is 2.5e7 to 1.2e9 rad from 2 to 30 GHz, which single precision
        # resolves only to 2 to 128 rad. The carrier frequency enters as a Python float.
        los_phase = torch.remainder(
            distance_3d * (-2.0*math.pi*self._carrier_frequency_float/SPEED_OF_LIGHT),
            2.0*math.pi)
        self._update_attr("_los_phase", los_phase.to(self.dtype))

        # LoS AoA, AoD, ZoA, ZoD
        los_aod = torch.atan2(delta_loc[:,:,:,1], delta_loc[:,:,:,0])
        los_aoa = los_aod + PI
        los_zod = torch.atan2(distance_2d, delta_loc[:,:,:,2])
        # The LOS arrival direction is the reversed departure direction (TR 38.901
        # V16.1.0, clause 7.5, step 1c): ZOA = 180 deg - ZOD and AOA = AOD + 180 deg
        los_zoa = PI - los_zod
        # Angles are converted to degrees and wrapped to (0,360)
        self._update_attr("_los_aod", wrap_angle_0_360(rad_2_deg(los_aod)))
        self._update_attr("_los_aoa", wrap_angle_0_360(rad_2_deg(los_aoa)))
        self._update_attr("_los_zod", wrap_angle_0_360(rad_2_deg(los_zod)))
        self._update_attr("_los_zoa", wrap_angle_0_360(rad_2_deg(los_zoa)))

        # 2D distances for all pairs of UTs in the (x-y) plane
        ut_loc_xy = self._ut_loc[:,:,:2]

        ut_loc_xy_expanded_1 = ut_loc_xy.unsqueeze(1)
        ut_loc_xy_expanded_2 = ut_loc_xy.unsqueeze(2)

        delta_loc_xy = ut_loc_xy_expanded_1 - ut_loc_xy_expanded_2

        # The distance of a UT to itself is 0 for every UT location, so its derivative
        # is 0. The square root has an infinite derivative at 0, which the backward pass
        # would multiply by the zero offset: inf times 0 is NaN, in the gradients of the
        # distances and of the large scale parameters with respect to the UT locations.
        # The square root is therefore taken wherever the squared distance is not 0, of
        # 1 where it is 0, and the result replaced by 0 there; for every UT location the
        # values are those of the square root, bit for bit, a NaN location giving NaN.
        squared_distance = torch.sum(torch.square(delta_loc_xy), dim=3)
        nonzero = squared_distance != 0
        matrix_ut_distance_2d = torch.where(
            nonzero,
            torch.sqrt(torch.where(nonzero, squared_distance,
                                   torch.ones_like(squared_distance))),
            torch.zeros_like(squared_distance))
        self._update_attr("_matrix_ut_distance_2d", matrix_ut_distance_2d)

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
                                        precision=self.precision,
                                        device=self.device)
        else:
            los = torch.full((self.batch_size, self.num_bs, self.num_ut),
                            self._requested_los, dtype=torch.bool,
                            device=self.device)

        self._update_attr("_los", torch.logical_and(los,
            torch.logical_not(self._in_state.unsqueeze(1))))

    def _sample_indoor_distance(self):
        r"""Sample 2D indoor distances for indoor devices, according to section
        7.4.3.1 of TR 38.901.
        """

        indoor = self.indoor
        indoor = indoor.unsqueeze(1) # For broadcasting with BS dim
        indoor_mask = torch.where(indoor,
            torch.tensor(1.0, dtype=self.dtype, device=self.device),
            torch.tensor(0.0, dtype=self.dtype, device=self.device))

        # Sample the indoor 2D distances for each BS-UT link
        self._update_attr("_distance_2d_in", (rand([self.batch_size, self.num_bs,
            self.num_ut], dtype=self.dtype, device=self.device,
            generator=self.torch_rng) * (self.max_2d_in - self.min_2d_in)
            + self.min_2d_in) * indoor_mask)
        # Compute the outdoor 2D distances
        self._update_attr("_distance_2d_out", self.distance_2d - self._distance_2d_in)
        # Compute the indoor 3D distances
        self._update_attr("_distance_3d_in", (self._distance_2d_in/self.distance_2d)
            *self.distance_3d)
        # Compute the outdoor 3D distances
        self._update_attr("_distance_3d_out", self.distance_3d - self._distance_3d_in)

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
                tensor = torch.tensor(v, dtype=self.dtype, device=self.device)
                self.register_buffer(f"_params_los_{param_name}", tensor)
                self._params_los[param_name] = tensor
            elif isinstance(v, int):
                # Keep integers as Python int for num_clusters, etc.
                pass
            elif isinstance(v, str):
                tensor = torch.tensor(float(v), dtype=self.dtype,
                    device=self.device)
                self.register_buffer(f"_params_los_{param_name}", tensor)
                self._params_los[param_name] = tensor

        source = files(models).joinpath(self.nlos_parameter_filepath)
        # pylint: disable=unspecified-encoding
        with open(source) as f:
            self._params_nlos = json.load(f)

        for param_name in self._params_nlos :
            v = self._params_nlos[param_name]
            if isinstance(v, float):
                tensor = torch.tensor(v, dtype=self.dtype, device=self.device)
                self.register_buffer(f"_params_nlos_{param_name}", tensor)
                self._params_nlos[param_name] = tensor
            elif isinstance(v, int):
                # Keep integers as Python int for num_clusters, etc.
                pass
            elif isinstance(v, str):
                tensor = torch.tensor(float(v), dtype=self.dtype,
                    device=self.device)
                self.register_buffer(f"_params_nlos_{param_name}", tensor)
                self._params_nlos[param_name] = tensor

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
