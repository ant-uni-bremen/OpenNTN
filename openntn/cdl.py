#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by the Dept. of Communications Engineering of the University of Bremen.
# The code is based on implementations provided by the NVIDIA CORPORATION & AFFILIATES
# in Sionna.
#
"""Clustered delay line models NTN-CDL-A to NTN-CDL-D of 3GPP TR 38.811"""

import json
import math
from importlib.resources import files
from typing import Callable, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from sionna.phy import PI, SPEED_OF_LIGHT
from sionna.phy.channel import ChannelModel
from sionna.phy.channel.utils import deg_2_rad
from sionna.phy.utils import normal, rand

from . import models
from .antenna import PanelArray
from .channel_coefficients import (ChannelCoefficientsGenerator, Topology,
                                   _satellite_doppler_speed)
from .rays import Rays
from .tdl import (_MODELS, _check_device_and_precision, _k_factor_profile,
                  _real_number, _satellite_shift_factor)
from .utils import _AERIAL_HEIGHTS, _SATELLITE_HEIGHTS

__all__ = ["CDL"]

# The reference elevation angle of Tables 6.9.1-1 to 6.9.1-4 of TR 38.811 V15.4.0 [deg]
_ALPHA_MODEL = 50.0
# The ray offset angles of TR 38.901 V16.1.0, Table 7.5-3, for an RMS angular spread of
# 1, in the order of Sionna's TR 38.901 CDL
_RAY_OFFSETS = (0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715,
                0.5129, -0.5129, 0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481,
                1.5195, -1.5195, 2.1551, -2.1551)


def _annex_a(angles: np.ndarray, powers: np.ndarray) -> Tuple[float, float]:
    r"""Mean angle and RMS angular spread of TR 38.901 V16.1.0, Annex A [deg]

    The mean angle is :math:`\arg \sum_{n,m} \exp(j \phi_{n,m}) P_{n,m}` (eq. (A-2)) and
    the spread :math:`\sqrt{-2 \ln |\sum_{n,m} \exp(j \phi_{n,m}) P_{n,m} / \sum_{n,m}
    P_{n,m}|}` (eq. (A-1)), both computed in radians.

    :param angles: Angles of the rays [deg]
    :param powers: Powers of the rays, in the shape of ``angles``

    :output mean: `float`. Mean angle [deg], in [-180, 180]

    :output spread: `float`. RMS angular spread [deg]
    """
    resultant = np.sum(powers * np.exp(1j * np.radians(angles))) / np.sum(powers)
    return (math.degrees(math.atan2(resultant.imag, resultant.real)),
            math.degrees(math.sqrt(-2.0 * math.log(abs(resultant)))))


def _wrap_180(angles: np.ndarray) -> np.ndarray:
    r"""Azimuth angles wrapped to [-180, 180) degrees

    :param angles: Azimuth angles [deg]

    :output: Wrapped angles [deg], in the shape of ``angles``
    """
    return np.mod(angles + 180.0, 360.0) - 180.0


def _reflect_zenith(angles: np.ndarray) -> np.ndarray:
    r"""Zenith angles brought into [0, 180] degrees by the rule of TR 38.901 V16.1.0,
    clause 7.5, step 7: wrapped to [0, 360) degrees, an angle of 180 degrees or more is
    set to 360 degrees minus the angle

    :param angles: Zenith angles [deg]

    :output: Zenith angles in [0, 180] degrees [deg], in the shape of ``angles``
    """
    wrapped = np.mod(angles, 360.0)
    return np.where(wrapped >= 180.0, 360.0 - wrapped, wrapped)


def _ray_angles(params: dict, elevation_angle: float,
                angular_spreads: Optional[Tuple[float, float]]
                ) -> Dict[str, np.ndarray]:
    r"""Angles of the rays of a model at the elevation angle of the link [deg]

    In the frame of the tables, with the departure at the satellite and the arrival at
    the UT, in double precision. The rays of each cluster are the cluster angle plus the
    cluster angular spread times the offsets of TR 38.901 Table 7.5-3 (eq. (7.7-0a)).
    The zenith angles of departure are 90 degrees plus the elevation angle (TR 38.811
    eq. (6.9-1)); the azimuth angles of departure are those of the table. The zenith
    angles of arrival follow eq. (6.9-2) with the mean ZOA of the model as the desired
    mean, the ZSA of the model or the desired one, and the difference of the elevation
    angle to 50 degrees, and are then reflected into [0, 180] degrees (TR 38.901
    clause 7.5, step 7). With ``angular_spreads``, the azimuth angles of arrival are
    scaled by eq. (7.7-5) with the mean AOA of the model as the desired mean, the
    differences to the mean wrapped to [-180, 180) degrees. The spreads and the means of
    the model are those of Annex A of TR 38.901 over the rays of the table, each ray of
    cluster n with the power P_n / 20, and the specular path of a LOS model as one more
    ray, with the powers of the table normalized to one.

    :param params: The parameters of the model, as in its parameter file
    :param elevation_angle: Elevation angle of the link [deg]
    :param angular_spreads: Desired ASA and ZSA [deg], or `None` for those of the model

    :output angles: ``"aoa"``, ``"aod"``, ``"zoa"`` and ``"zod"``, the rays of the
        Laplacian clusters [deg], of shape [num_clusters, 20]; for a LOS model also
        ``"los_aoa"``, ``"los_aod"``, ``"los_zoa"`` and ``"los_zod"``, the angles of the
        specular path [deg]
    """
    offsets = np.asarray(_RAY_OFFSETS, dtype=np.float64)
    los = bool(params["los"])
    # A LOS model holds the specular path as its first entry
    first = 1 if los else 0

    def entries(name: str) -> np.ndarray:
        return np.asarray(params[name], dtype=np.float64)

    def rays(name: str, spread: str) -> np.ndarray:
        return entries(name)[first:, None] + float(params[spread]) * offsets

    powers = np.power(10.0, entries("powers") / 10.0)
    powers = powers / np.sum(powers)
    ray_powers = np.repeat(powers[first:, None] / offsets.size, offsets.size, axis=1)

    def statistics(name: str, spread: str) -> Tuple[float, float]:
        angles, weights = rays(name, spread).ravel(), ray_powers.ravel()
        if los:
            angles = np.concatenate([entries(name)[:1], angles])
            weights = np.concatenate([powers[:1], weights])
        return _annex_a(angles, weights)

    delta_alpha = elevation_angle - _ALPHA_MODEL
    zenith: Callable[[np.ndarray], np.ndarray]
    azimuth: Callable[[np.ndarray], np.ndarray]
    if angular_spreads is None:
        # Eq. (6.9-2) with the ZSA and the mean ZOA of the model is a shift by
        # -delta_alpha; the azimuth angles of arrival keep the values of the table
        def zenith(angles):
            return angles - delta_alpha

        def azimuth(angles):
            return angles
    else:
        desired_asa, desired_zsa = angular_spreads
        mean_aoa, model_asa = statistics("aoa", "cASA")
        mean_zoa, model_zsa = statistics("zoa", "cZSA")

        def zenith(angles):
            return ((desired_zsa / model_zsa) * (angles - mean_zoa) + mean_zoa
                    - delta_alpha)

        def azimuth(angles):
            return (desired_asa / model_asa) * _wrap_180(angles - mean_aoa) + mean_aoa

    # Eq. (6.9-1) for every cluster, then eq. (7.7-0a) with the cluster ZSD of the table
    zod = 90.0 + elevation_angle
    num_clusters = entries("zod").size - first
    angles = {"aod": rays("aod", "cASD"),
              "aoa": azimuth(rays("aoa", "cASA")),
              "zod": np.full((num_clusters, 1), zod) + float(params["cZSD"]) * offsets,
              "zoa": _reflect_zenith(zenith(rays("zoa", "cZSA")))}
    if los:
        angles["los_aod"] = float(entries("aod")[0])
        angles["los_aoa"] = float(azimuth(entries("aoa")[0]))
        angles["los_zod"] = zod
        angles["los_zoa"] = float(_reflect_zenith(zenith(entries("zoa")[0])))
    return angles


def _vectors(name: str, value: object, unit: str) -> np.ndarray:
    r"""Returns a vector of shape [3], or vectors of shape [batch size, 3], as float64

    :param name: Name of the argument, for the error
    :param value: Value of the argument, a tensor, an array or a sequence of numbers
    :param unit: Unit of the argument, for the error

    :output: [3] or [batch size, 3], `numpy.float64`. The value; `TypeError` for values
        that are not real numbers, `ValueError` for another shape or values that are not
        finite
    """
    if isinstance(value, torch.Tensor):
        if value.dtype == torch.bool or value.is_complex():
            raise TypeError(f"{name} must hold real numbers [{unit}], not {value!r}")
        array = value.detach().to("cpu", torch.float64).numpy()
    else:
        try:
            array = np.asarray(value)
        except (TypeError, ValueError) as error:
            raise TypeError(f"{name} must hold real numbers [{unit}], not "
                            f"{value!r}") from error
        if array.dtype.kind not in "iuf":
            raise TypeError(f"{name} must hold real numbers [{unit}], not {value!r}")
        array = array.astype(np.float64)
    if array.ndim not in (1, 2) or array.shape[-1] != 3 or array.shape[0] == 0:
        raise ValueError(f"{name} must have the shape [3] or [batch size, 3], not "
                         f"{list(array.shape)}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be finite [{unit}], not {value!r}")
    return array


class CDL(ChannelModel):
    # pylint: disable=line-too-long
    r"""Clustered delay line models NTN-CDL-A to NTN-CDL-D of :cite:p:`TR38811`

    The models of TR 38.811 V15.4.0, clause 6.9.1: NTN-CDL-A and NTN-CDL-B represent two
    channel profiles for NLOS links, NTN-CDL-C and NTN-CDL-D two for LOS links, with the
    normalized delays, the powers, the angles of the clusters, the angular spreads of
    the clusters and the cross polarization power ratio (XPR) of Tables 6.9.1-1 to
    6.9.1-4, which are given at the reference elevation angle of 50 degrees. The angular
    spreads of departure at the satellite are zero, and all clusters leave the satellite
    at the azimuth angle 0. The first cluster of NTN-CDL-C and NTN-CDL-D adds a specular
    LOS path to its Laplacian cluster. The power delay profile is normalized to a total
    power of one, the specular path included.

    The model applies the procedures of TR 38.901 V16.1.0 :cite:p:`TR38901`, clause 7.7,
    to which clause 6.9 refers, the steps of clause 6.9.1 for the elevation angle of the
    link, and the Doppler shift and the Faraday rotation of the satellite:

    - The rays of each cluster have the offset angles of Table 7.5-3 of TR 38.901 times
      the angular spreads of the clusters of the table (eq. (7.7-0a)), and they are
      coupled at random within each cluster in every batch example (clause 7.7.1,
      step 2).
    - The zenith angle of departure of every ray and of the specular path is 90 degrees
      plus the elevation angle :math:`\alpha` of the link (eq. (6.9-1)); the azimuth
      angles of departure are those of the tables, 0. At 50 degrees eq. (6.9-1) gives
      140 degrees, as the tables print it; the sentence of clause 6.9.1 that names
      190 degrees for the tables is read as 140 degrees.
    - The zenith angles of arrival of every ray and of the specular path follow
      eq. (6.9-2),

      .. math::
          \theta_{n,m,\text{ZOA}} =
          \frac{\text{ZSA}_\text{desired}}{\text{ZSA}_\text{model}}
          \left(\theta_{n,m,\text{ZOA,model}} - \mu_\text{ZOA,model}\right)
          + \mu_\text{ZOA,desired} - \Delta\alpha

      with :math:`\Delta\alpha = \alpha - 50` degrees and the desired mean ZOA equal to
      the mean ZOA of the model, so that with the angular spreads of the model the
      specular path arrives at :math:`90 - \alpha` degrees, from the direction of the
      satellite. The desired ZSA is that of the model unless ``angular_spreads`` gives
      it. The zenith angles are then brought into [0, 180] degrees by the rule of
      TR 38.901 clause 7.5, step 7, which clause 6.9.1 names: wrapped to [0, 360)
      degrees, an angle of 180 degrees or more is set to 360 degrees minus the angle, at
      the same azimuth. The rule applies at every elevation angle, also at 50 degrees,
      where four rays of the last cluster of NTN-CDL-B lie above 180 degrees in the
      table.
    - With ``angular_spreads`` the azimuth angles of arrival of every ray and of the
      specular path are scaled to the desired ASA by eq. (7.7-5) of clause 7.7.5.1, with
      the desired mean equal to the mean AOA of the model and the differences to the
      mean wrapped to [-180, 180] degrees before the scaling, as the azimuth angles of
      arrival of the tables lie near 180 degrees. Desired angular spreads thus scale the
      zenith and the azimuth angles of arrival of the specular path with those of the
      rays, so that it arrives near, not at, the direction of the satellite: NTN-CDL-C
      at 70 degrees with the desired ASA 30 degrees and ZSA 10 degrees puts it at the
      zenith angle 19.68 degrees and the azimuth angle -180.84 degrees, against 20
      degrees and -180 degrees with the angular spreads of the model.
    - The ASA and the ZSA of the model and its mean AOA and ZOA are those of eqs. (A-1)
      and (A-2) of TR 38.901, Annex A, over the rays of the table with their offsets,
      each ray of cluster :math:`n` with the power :math:`P_n/20`, and for NTN-CDL-C and
      NTN-CDL-D over the specular path as one more ray, with the powers of the table.
      The procedure scales the angles of the rays, so that the spreads of a scaled model
      may differ from the desired ones; the reflection of the zenith angles changes them
      further.
    - The normalized delays are scaled to the RMS delay spread ``delay_spread``
      (clause 7.7.3).
    - A desired K-factor ``k_factor`` of NTN-CDL-C or NTN-CDL-D changes the powers of
      the Laplacian clusters by eq. (7.7.6-1) and divides the normalized delays by the
      RMS delay spread of the scaled profile, the specular path included, so that it is
      one again before the delays are scaled (clause 7.7.6). It does not change the
      angles.
    - The XPR of the table is the same for all rays (clause 7.7.1, step 3), and the path
      coefficients follow steps 10 and 11 of clause 7.5 of TR 38.901 without
      sub-clusters (clause 7.7.1, step 4). The velocity of the UT turns the phase of
      every ray by the Doppler of its direction at the UT (clause 6.8.1). For a
      satellite from 600 km, above the ionosphere, the polarization of every ray turns
      by the Faraday rotation of clause 6.8.2 with :math:`\psi = 108/f^2` degrees,
      :math:`f` the carrier frequency in GHz; a satellite below 600 km and an aerial
      platform have none.
    - The Doppler shift due to the motion of the satellite,

      .. math::
          f_{d,\text{shift}} = \frac{v_\text{sat}}{c} \frac{R}{R+h} \cos(\alpha) f_c

      with the speed :math:`v_\text{sat}` of a circular orbit at the height :math:`h`,
      the radius :math:`R` of the Earth and the elevation angle :math:`\alpha` of the
      link (clause 6.9.2), is applied to all clusters, rays and antennas: the path
      coefficients at the sample :math:`n` of a call are multiplied by
      :math:`\exp(j 2 \pi f_{d,\text{shift}} n / f_s)`. All rays leave the satellite in
      one direction, so that the term of the satellite of clause 6.8.1 is the same for
      all of them: one shift. It is the largest shift, of an approaching satellite seen
      from a UT in its orbital plane, and it is 0 for a geostationary satellite, which
      is fixed relative to the Earth (clause 5.3.2.3), and for an aerial platform.
      ``doppler_shift`` replaces it: 0 for a link whose terminal pre-compensates the
      shift, as 3GPP TS 38.101-5 sets the Doppler shift of the satellite motion to zero
      for the performance requirements of its Table 8.2.1.2.2.1.1-3 (clause A.4.4), a
      negative value for a receding satellite, or the shift of a moving aerial platform.

    The angles are given in the coordinate system of the tables, in which the clusters
    leave the satellite at the azimuth angle 0 and, with the angular spreads of the
    model, the specular path arrives at the UT from the azimuth angle 180 degrees. By
    default both antennas point along the line of sight: the satellite with the
    orientation :math:`[0, \alpha, 0]`, the bearing 0 and the downtilt :math:`\alpha`,
    along the zenith angle of departure :math:`90 + \alpha` degrees, and the UT with
    :math:`[\pi, -\alpha, 0]`, the bearing :math:`\pi` and the uptilt :math:`\alpha`,
    toward the satellite.

    In the uplink the UT transmits: the departure and the arrival angles of the tables
    are swapped, the velocity of the UT turns the rays by their angles of departure,
    which are those at the UT, and the Faraday rotation acts between the scatterers and
    the satellite, as in the system-level models of the package.

    The shift is contained in the path coefficients at every sample. A channel in the
    frequency domain that is sampled once per OFDM symbol carries its phase from symbol
    to symbol, but not the inter-carrier interference that it causes within a symbol,
    which only the channel in the time domain at the sampling rate of the signal shows.
    The shift can exceed the subcarrier spacing: at 600 km, 50 degrees and 2 GHz it is
    29.63 kHz, 1.98 subcarrier spacings of 15 kHz.

    The speeds of the UT and of the satellite and the elevation angle are constant
    during a call, which suits simulations of a few TTIs, as clause 6.9.1 allows. The
    carrier frequency lies in the S band, 1.9 GHz to 4 GHz, or in the Ka band, 19 GHz to
    40 GHz, the bands for which clause 6.9.1 defines the models and those of the
    system-level models of the package. Suitable values of ``delay_spread``,
    ``angular_spreads`` and ``k_factor`` for a band, an environment and an elevation
    angle follow from the parameters of clause 6.7.2 (clause 6.9.1):
    :func:`~openntn.utils.link_level_parameters` returns the median delay spread, the
    median ASA and ZSA and the mean K-factor of its tables at the reference elevation
    angle nearest to that of the link.

    The arguments after ``satellite_height`` are taken by keyword only. A wrong value
    raises `ValueError` and a wrong type `TypeError`. The antenna arrays must be on the
    device and in the precision of the model and built for its carrier frequency. The
    model stays on the device and in the precision of its construction: moved with
    ``.to()`` or cast, it raises `ValueError` in its next call.

    :param model: The model, ``"A"``, ``"B"``, ``"C"`` or ``"D"``, for NTN-CDL-A to
        NTN-CDL-D; `ValueError` otherwise
    :param delay_spread: RMS delay spread [s], finite and greater than 0
    :param carrier_frequency: Carrier frequency [Hz], from 1.9 GHz to 4 GHz or from
        19 GHz to 40 GHz
    :param elevation_angle: Elevation angle of the link [deg], from 10 to 90
    :param satellite_height: Height of the satellite, from 160 km to 35 786 km, or of an
        aerial platform, from 8 km to 50 km [m]
    :param ut_array: Antenna array of the UT, a :class:`~openntn.PanelArray`, such as an
        :class:`~openntn.Antenna` or an :class:`~openntn.AntennaArray`
    :param bs_array: Antenna array of the satellite or the aerial platform, a
        :class:`~openntn.PanelArray`
    :param direction: Direction of the link, ``"downlink"`` or ``"uplink"``. Defaults to
        ``"downlink"``.
    :param doppler_shift: Doppler shift due to the motion of the satellite [Hz], a
        finite number. Defaults to `None`, the shift of clause 6.9.2 at the elevation
        angle of the link, 0 for a geostationary satellite and an aerial platform.
    :param k_factor: Desired K-factor of NTN-CDL-C or NTN-CDL-D [dB], a finite number,
        applied by clause 7.7.6 of TR 38.901; `ValueError` for NTN-CDL-A and NTN-CDL-B.
        Defaults to `None`, the K-factor of the table.
    :param angular_spreads: Desired ASA and ZSA [deg], a pair of finite numbers greater
        than 0. Defaults to `None`, the angular spreads of the model.
    :param ut_orientation: Orientation of the UT [rad], finite, of shape [3] or
        [batch size, 3]. Defaults to `None`, :math:`[\pi, -\alpha, 0]`.
    :param bs_orientation: Orientation of the satellite or the aerial platform [rad],
        finite, of shape [3] or [batch size, 3]. Defaults to `None`,
        :math:`[0, \alpha, 0]`.
    :param ut_velocity: Velocity of the UT [m/s], finite, of shape [3] or
        [batch size, 3]. Defaults to `None`: for each batch example a speed drawn
        uniformly from [``min_speed``, ``max_speed``] in the direction of an azimuth
        angle drawn uniformly from [0, :math:`2\pi`) and a zenith angle drawn uniformly
        from [0, :math:`\pi`].
    :param min_speed: Minimum speed of the UT [m/s], finite and at least 0; ignored if
        ``ut_velocity`` is given. Defaults to 0.
    :param max_speed: Maximum speed of the UT [m/s], finite and at least ``min_speed``;
        ignored if ``ut_velocity`` is given. Defaults to `None`, the value of
        ``min_speed``.
    :param normalize_delays: If `True`, the normalized delays are scaled to the RMS
        delay spread ``delay_spread``; if `False`, the delays are the normalized delays
        of the table, with an RMS delay spread of 1. Defaults to `True`.
    :param precision: Precision used for internal calculations and outputs.
        If set to `None`, :attr:`~sionna.phy.config.Config.precision` is used.
    :param device: Device for computation (e.g., ``"cpu"``, ``"cuda:0"``).
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    :input batch_size: `int`.
        Batch size.

    :input num_time_steps: `int`.
        Number of time steps.

    :input sampling_frequency: `float`.
        Sampling frequency [Hz].

    :output a: [batch size, num_rx=1, num_rx_ant, num_tx=1, num_tx_ant, num_clusters, num_time_steps], `torch.complex`.
        Path coefficients. The receiver is the UT in the downlink and the satellite in
        the uplink.

    :output tau: [batch size, num_rx=1, num_tx=1, num_clusters], `torch.float`.
        Path delays [s], relative to the first path.

    .. rubric:: Examples

    .. code-block:: python

        from openntn import CDL, Antenna, AntennaArray

        # NTN-CDL-C at 2 GHz, a satellite at 600 km seen at 50 degrees, a UT at up to
        # 3 m/s with one dual-polarized antenna, 2 x 2 dual-polarized elements at the
        # satellite
        ut_array = Antenna("dual", "cross", "omni", 2e9)
        bs_array = AntennaArray(2, 2, "dual", "VH", "38.901", 2e9)
        cdl = CDL("C", delay_spread=100e-9, carrier_frequency=2e9,
                  elevation_angle=50.0, satellite_height=600e3, ut_array=ut_array,
                  bs_array=bs_array, max_speed=3.0)
        a, tau = cdl(batch_size=16, num_time_steps=14, sampling_frequency=30e3)
        print(a.shape, tau.shape)
        # torch.Size([16, 1, 2, 1, 8, 2, 14]) torch.Size([16, 1, 1, 2])
        print(round(cdl.doppler_shift, 3))
        # 29634.973
    """

    def __init__(
        self,
        model: str,
        delay_spread: float,
        carrier_frequency: float,
        elevation_angle: float,
        satellite_height: float,
        *,
        ut_array: PanelArray,
        bs_array: PanelArray,
        direction: str = "downlink",
        doppler_shift: Optional[float] = None,
        k_factor: Optional[float] = None,
        angular_spreads: Optional[Sequence[float]] = None,
        ut_orientation: Optional[torch.Tensor] = None,
        bs_orientation: Optional[torch.Tensor] = None,
        ut_velocity: Optional[torch.Tensor] = None,
        min_speed: float = 0.0,
        max_speed: Optional[float] = None,
        normalize_delays: bool = True,
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        if not isinstance(model, str):
            raise TypeError(f"model must be one of 'A', 'B', 'C' and 'D', not "
                            f"{model!r}")
        if model not in _MODELS:
            raise ValueError(f"model must be one of 'A', 'B', 'C' and 'D', not "
                             f"{model!r}")
        delay_spread = _real_number("delay_spread", delay_spread)
        if not (math.isfinite(delay_spread) and delay_spread > 0.0):
            raise ValueError(f"delay_spread must be finite and greater than 0 s, not "
                             f"{delay_spread!r}")
        carrier_frequency = _real_number("carrier_frequency", carrier_frequency)
        if not (1.9e9 <= carrier_frequency <= 4e9 or 19e9 <= carrier_frequency <= 40e9):
            raise ValueError(f"carrier_frequency must be in the S band (1.9 GHz to "
                             f"4 GHz) or in the Ka band (19 GHz to 40 GHz), not "
                             f"{carrier_frequency!r} Hz")
        elevation_angle = _real_number("elevation_angle", elevation_angle)
        if not 10.0 <= elevation_angle <= 90.0:
            raise ValueError(f"elevation_angle must be from 10 to 90 degrees, not "
                             f"{elevation_angle!r}")
        satellite_height = _real_number("satellite_height", satellite_height)
        if not (_SATELLITE_HEIGHTS[0] <= satellite_height <= _SATELLITE_HEIGHTS[1]
                or _AERIAL_HEIGHTS[0] <= satellite_height <= _AERIAL_HEIGHTS[1]):
            raise ValueError(f"satellite_height {satellite_height!r} m: a satellite is "
                             f"at 160 km to 35 786 km, an aerial platform at 8 km to "
                             f"50 km")
        if not isinstance(ut_array, PanelArray):
            raise TypeError(f"ut_array must be a PanelArray, such as an Antenna or an "
                            f"AntennaArray, not {type(ut_array).__name__}")
        if not isinstance(bs_array, PanelArray):
            raise TypeError(f"bs_array must be a PanelArray, such as an Antenna or an "
                            f"AntennaArray, not {type(bs_array).__name__}")
        if not isinstance(direction, str):
            raise TypeError(f"direction must be 'uplink' or 'downlink', not "
                            f"{direction!r}")
        if direction not in ("uplink", "downlink"):
            raise ValueError(f"direction must be 'uplink' or 'downlink', not "
                             f"{direction!r}")
        if doppler_shift is not None:
            doppler_shift = _real_number("doppler_shift", doppler_shift)
            if not math.isfinite(doppler_shift):
                raise ValueError(f"doppler_shift must be a finite number of Hz, not "
                                 f"{doppler_shift!r}")
        if k_factor is not None:
            k_factor = _real_number("k_factor", k_factor)
            if not math.isfinite(k_factor):
                raise ValueError(f"k_factor must be a finite number of dB, not "
                                 f"{k_factor!r}")
            if model not in ("C", "D"):
                raise ValueError(f"k_factor applies to the LOS models 'C' and 'D', not "
                                 f"to {model!r}")
        if angular_spreads is not None:
            if not isinstance(angular_spreads, (tuple, list)):
                raise TypeError(f"angular_spreads must be a pair (ASA, ZSA) of numbers "
                                f"of degrees, not {angular_spreads!r}")
            if len(angular_spreads) != 2:
                raise ValueError(f"angular_spreads must be a pair (ASA, ZSA), not "
                                 f"{angular_spreads!r}")
            angular_spreads = (_real_number("angular_spreads", angular_spreads[0]),
                               _real_number("angular_spreads", angular_spreads[1]))
            if not all(math.isfinite(v) and v > 0.0 for v in angular_spreads):
                raise ValueError(f"angular_spreads must be finite and greater than 0 "
                                 f"degrees, not {angular_spreads!r}")
        if ut_orientation is not None:
            ut_orientation = _vectors("ut_orientation", ut_orientation, "rad")
        if bs_orientation is not None:
            bs_orientation = _vectors("bs_orientation", bs_orientation, "rad")
        if ut_velocity is not None:
            ut_velocity = _vectors("ut_velocity", ut_velocity, "m/s")
        min_speed = _real_number("min_speed", min_speed)
        if not (math.isfinite(min_speed) and min_speed >= 0.0):
            raise ValueError(f"min_speed must be finite and at least 0 m/s, not "
                             f"{min_speed!r}")
        if max_speed is None:
            max_speed = min_speed
        else:
            max_speed = _real_number("max_speed", max_speed)
            if not (math.isfinite(max_speed) and max_speed >= min_speed):
                raise ValueError(f"max_speed must be finite and at least min_speed "
                                 f"({min_speed!r} m/s), not {max_speed!r}")
        if not isinstance(normalize_delays, bool):
            raise TypeError(f"normalize_delays must be True or False, not "
                            f"{normalize_delays!r}")
        super().__init__(precision=precision, device=device)

        # The arrays create their tensors on their own device and in their own
        # precision, and the generator computes with them, so they must have the device,
        # the precision and the carrier frequency of the model
        # Wavelength (m)
        lambda_0 = torch.tensor(SPEED_OF_LIGHT / carrier_frequency, dtype=self.dtype,
                                device=self.device)
        for name, array in (("ut_array", ut_array), ("bs_array", bs_array)):
            if array.device != self.device:
                raise ValueError(
                    f"CDL is on device {self.device!r}, but its {name} "
                    f"({type(array).__name__}) is on device {array.device!r}. Build "
                    f"both on the same device, e.g. {type(array).__name__}(..., "
                    f"device={self.device!r}).")
            if array.precision != self.precision:
                raise ValueError(
                    f"CDL is in {self.precision} precision, but its {name} "
                    f"({type(array).__name__}) is in {array.precision} precision. "
                    f"Build the arrays in the precision of the model, e.g. "
                    f"{type(array).__name__}(..., precision={self.precision!r}).")
            if not bool(array._lambda_0 == lambda_0):
                raise ValueError(
                    f"the {name} ({type(array).__name__}) of CDL is built for another "
                    f"carrier frequency: its wavelength is "
                    f"{float(array._lambda_0)!r} m, that of {carrier_frequency!r} Hz "
                    f"{float(lambda_0)!r} m. Build it with "
                    f"carrier_frequency={carrier_frequency!r}.")

        self._model = model
        self._direction = direction
        self._elevation_angle = elevation_angle
        self._min_speed = min_speed
        self._max_speed = max_speed
        self._normalize_delays = normalize_delays
        # Register as buffers for CUDAGraph compatibility
        self.register_buffer("_carrier_frequency", torch.tensor(
            carrier_frequency, dtype=self.dtype, device=self.device))
        self.register_buffer("_delay_spread", torch.tensor(
            delay_spread, dtype=self.dtype, device=self.device))
        self.register_buffer("_bs_height", torch.tensor(
            [satellite_height], dtype=self.dtype, device=self.device))

        # Set TX and RX arrays based on direction
        if direction == "downlink":
            self._tx_array = bs_array
            self._rx_array = ut_array
        else:  # uplink
            self._tx_array = ut_array
            self._rx_array = bs_array

        # Orientations: by default along the line of sight, the satellite toward the
        # zenith angle of departure 90 + alpha and the UT toward the satellite
        alpha = math.radians(elevation_angle)
        if ut_orientation is None:
            ut_orientation = np.array([math.pi, -alpha, 0.0])
        if bs_orientation is None:
            bs_orientation = np.array([0.0, alpha, 0.0])
        self.register_buffer("_ut_orientation", torch.tensor(
            ut_orientation, dtype=self.dtype, device=self.device))
        self.register_buffer("_bs_orientation", torch.tensor(
            bs_orientation, dtype=self.dtype, device=self.device))

        # Velocity
        if ut_velocity is not None:
            self.register_buffer("_ut_velocity", torch.tensor(
                ut_velocity, dtype=self.dtype, device=self.device))
        else:
            self.register_buffer("_ut_velocity", None)

        # Load CDL model parameters
        self._load_parameters(model, elevation_angle, k_factor, angular_spreads,
                              delay_spread)

        # Create the channel coefficients generator
        self._cir_gen = ChannelCoefficientsGenerator(
            carrier_frequency=carrier_frequency,
            tx_array=self._tx_array,
            rx_array=self._rx_array,
            subclustering=False,
            precision=self.precision,
            device=self.device,
        )

        # Doppler shift due to the motion of the satellite [Hz], a Python float,
        # computed once in double precision on the CPU, so that it is the same on every
        # device
        if doppler_shift is None:
            speed = _satellite_doppler_speed(satellite_height, elevation_angle,
                                             torch.float64, "cpu")
            self._doppler_shift = float(speed) * carrier_frequency / SPEED_OF_LIGHT
        else:
            self._doppler_shift = doppler_shift

    @property
    def num_clusters(self) -> int:
        r"""Number of clusters, the specular path of NTN-CDL-C and NTN-CDL-D counted in
        the first cluster"""
        return self._num_clusters

    @property
    def los(self) -> bool:
        r"""`True` for the LOS models NTN-CDL-C and NTN-CDL-D, `False` otherwise"""
        return self._los

    @property
    def k_factor(self) -> torch.Tensor:
        r"""K-factor of the model in linear scale: the power of the specular path over
        the sum of the powers of all Laplacian clusters, eq. (7.7.6-2) of TR 38.901. It
        is that of the table, 10.000 dB for NTN-CDL-C and 7.000 dB for NTN-CDL-D, or
        ``k_factor``. `RuntimeError` for NTN-CDL-A and NTN-CDL-B."""
        if not self._los:
            raise RuntimeError("k_factor is defined for the LOS models 'C' and 'D' "
                               "only")
        return self._k_factor

    @property
    def delays(self) -> torch.Tensor:
        r"""Path delays [s], the normalized delays if ``normalize_delays`` is `False`"""
        return self._delays

    @property
    def mean_powers(self) -> torch.Tensor:
        r"""Path powers in linear scale, the specular path included in the first
        cluster; their sum is one"""
        return self._mean_powers

    @property
    def mean_power_los(self) -> torch.Tensor:
        r"""Power of the specular path in linear scale. `RuntimeError` for NTN-CDL-A and
        NTN-CDL-B."""
        if not self._los:
            raise RuntimeError("mean_power_los is defined for the LOS models 'C' and "
                               "'D' only")
        return self._los_power

    @property
    def delay_spread(self) -> torch.Tensor:
        r"""RMS delay spread [s]"""
        return self._delay_spread

    @property
    def doppler_shift(self) -> float:
        r"""Doppler shift due to the motion of the satellite [Hz], of all clusters"""
        return self._doppler_shift

    @property
    def aoa(self) -> torch.Tensor:
        r"""[num_clusters, 20], `torch.float64`. Azimuth angles of arrival at the UT of
        the rays of the Laplacian clusters [deg], before their random coupling, in the
        coordinate system of the tables, also in the uplink"""
        return torch.tensor(self._angles["aoa"], dtype=torch.float64)

    @property
    def aod(self) -> torch.Tensor:
        r"""[num_clusters, 20], `torch.float64`. Azimuth angles of departure at the
        satellite of the rays of the Laplacian clusters [deg], before their random
        coupling, in the coordinate system of the tables, also in the uplink"""
        return torch.tensor(self._angles["aod"], dtype=torch.float64)

    @property
    def zoa(self) -> torch.Tensor:
        r"""[num_clusters, 20], `torch.float64`. Zenith angles of arrival at the UT of
        the rays of the Laplacian clusters [deg], before their random coupling, in the
        coordinate system of the tables, also in the uplink"""
        return torch.tensor(self._angles["zoa"], dtype=torch.float64)

    @property
    def zod(self) -> torch.Tensor:
        r"""[num_clusters, 20], `torch.float64`. Zenith angles of departure at the
        satellite of the rays of the Laplacian clusters [deg], before their random
        coupling, in the coordinate system of the tables, also in the uplink"""
        return torch.tensor(self._angles["zod"], dtype=torch.float64)

    @property
    def los_aoa(self) -> float:
        r"""Azimuth angle of arrival of the specular path at the UT [deg].
        `RuntimeError` for NTN-CDL-A and NTN-CDL-B."""
        return self._los_angle("los_aoa")

    @property
    def los_aod(self) -> float:
        r"""Azimuth angle of departure of the specular path at the satellite [deg].
        `RuntimeError` for NTN-CDL-A and NTN-CDL-B."""
        return self._los_angle("los_aod")

    @property
    def los_zoa(self) -> float:
        r"""Zenith angle of arrival of the specular path at the UT [deg].
        `RuntimeError` for NTN-CDL-A and NTN-CDL-B."""
        return self._los_angle("los_zoa")

    @property
    def los_zod(self) -> float:
        r"""Zenith angle of departure of the specular path at the satellite [deg].
        `RuntimeError` for NTN-CDL-A and NTN-CDL-B."""
        return self._los_angle("los_zod")

    def __call__(
        self,
        batch_size: int,
        num_time_steps: int,
        sampling_frequency: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _check_device_and_precision(self, self._carrier_frequency)
        for name in ("_ut_orientation", "_bs_orientation", "_ut_velocity"):
            value = getattr(self, name)
            if value is not None and value.dim() == 2 and value.shape[0] != batch_size:
                raise ValueError(f"{name[1:]} has the shape {list(value.shape)}, but "
                                 f"the batch size of the call is {batch_size}")

        # Generate random velocity if not provided
        velocity = self._get_velocity(batch_size)

        # Create topology
        topology = self._create_topology(batch_size, velocity)

        # Create rays from CDL model parameters
        rays = self._create_rays(batch_size)

        # K-factor (only for C and D models)
        k_factor = self._k_factor.reshape(1, 1, 1).expand(batch_size, 1, 1)

        # Generate channel impulse response
        h, delays = self._cir_gen(
            num_time_samples=num_time_steps,
            sampling_frequency=sampling_frequency,
            k_factor=k_factor,
            rays=rays,
            topology=topology,
        )

        # Reshape output to match expected format
        # h: [batch, num_tx, num_rx, num_paths, num_rx_ant, num_tx_ant, num_time_steps]
        # -> [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps]
        h = h.permute(0, 2, 4, 1, 5, 3, 6)

        # delays: [batch, num_tx, num_rx, num_paths]
        # -> [batch, num_rx, num_tx, num_paths]
        delays = delays.permute(0, 2, 1, 3)

        # The Doppler shift due to the motion of the satellite, one factor per sample
        # for all clusters, rays and antennas (TR 38.811 clauses 6.9.1 and 6.9.2),
        # applied after the random draws, which it does not change
        if self._doppler_shift != 0.0:
            h = h * self._doppler_shift_factor(num_time_steps, sampling_frequency)

        # Detach to avoid unnecessary gradient computation
        h = h.detach()
        delays = delays.detach()

        return h, delays

    ########################################
    # Internal utility methods
    ########################################

    def _los_angle(self, name: str) -> float:
        r"""An angle of the specular path [deg]; `RuntimeError` for the NLOS models

        :param name: ``"los_aoa"``, ``"los_aod"``, ``"los_zoa"`` or ``"los_zod"``

        :output: `float`. The angle [deg]
        """
        if not self._los:
            raise RuntimeError(f"{name} is defined for the LOS models 'C' and 'D' only")
        return self._angles[name]

    def _doppler_shift_factor(self, num_time_steps: int,
                              sampling_frequency: Union[float, torch.Tensor]
                              ) -> torch.Tensor:
        r"""Phase factors of the Doppler shift due to the motion of the satellite at the
        samples of a call, in the complex dtype and on the device of the model
        (``_satellite_shift_factor``)

        :param num_time_steps: Number of time steps
        :param sampling_frequency: Sampling frequency [Hz]

        :output factor: [num_time_steps], `torch.complex`. Phase factors
        """
        return _satellite_shift_factor(self._doppler_shift, num_time_steps,
                                       sampling_frequency, self.cdtype, self.device)

    def _load_parameters(self, model: str, elevation_angle: float,
                         k_factor: Optional[float],
                         angular_spreads: Optional[Tuple[float, float]],
                         delay_spread: float) -> None:
        r"""Load the parameters of a model from its parameter file in ``openntn/models``

        The parameter file holds the table of the model with the following keys:
        ``los``, `True` if the first entry is a specular path; ``num_clusters``, the
        number of the other entries; ``delays``, the normalized delays; ``powers``, the
        powers [dB]; ``aod``, ``aoa``, ``zod`` and ``zoa``, the angles of the clusters
        [deg]; ``cASD``, ``cASA``, ``cZSD`` and ``cZSA``, the angular spreads of the
        clusters [deg]; ``xpr``, the XPR [dB]. For the LOS models, the specular path and
        the Laplacian part of the first cluster have the delay 0. The angles of the rays
        are computed for the elevation angle of the link and ``angular_spreads``
        (``_ray_angles``). With ``k_factor``, the powers and the delays are first scaled
        by clause 7.7.6 of TR 38.901. The powers are normalized to a total of one; for
        the LOS models the specular path is taken out and the powers of the Laplacian
        clusters are normalized to a total of one, with the K-factor the power of the
        specular path over their sum.

        :param model: The model, ``"A"`` to ``"D"``
        :param elevation_angle: Elevation angle of the link [deg]
        :param k_factor: Desired K-factor [dB], or `None` for the table's
        :param angular_spreads: Desired ASA and ZSA [deg], or `None` for the model's
        :param delay_spread: RMS delay spread [s]
        """
        source = files(models).joinpath(f"CDL-{model}.json")
        with open(source, encoding="utf-8") as parameter_file:
            params = json.load(parameter_file)

        # LoS scenario?
        self._los = bool(params["los"])

        # Number of clusters, without the specular path
        self._num_clusters = int(params["num_clusters"])

        # The angles of the rays in the frame of the tables [deg], in double precision
        self._angles = _ray_angles(params, elevation_angle, angular_spreads)

        # Powers and delays, in double precision
        powers, delays = params["powers"], params["delays"]
        if k_factor is not None:
            powers, delays = _k_factor_profile(powers, delays, k_factor)
        powers = np.power(10.0, np.asarray(powers, dtype=np.float64) / 10.0)
        powers = powers / np.sum(powers)
        delays = np.asarray(delays, dtype=np.float64)
        if self._los:
            # Extract the specular component (first path)
            los_power = powers[0]
            mean_powers = np.concatenate([powers[:1] + powers[1:2], powers[2:]])
            powers = powers[1:]
            delays = delays[1:]
            # Re-normalize NLoS powers
            norm_fact = np.sum(powers)
            powers = powers / norm_fact
            # K-factor = (power of specular component) / (total NLoS power)
            k = los_power / norm_fact
        else:
            los_power = 0.0
            mean_powers = powers
            k = 0.0

        # Scale delays if normalizing
        if self._normalize_delays:
            delays = delays * delay_spread

        # Register as buffers for CUDAGraph compatibility
        self.register_buffer("_delays", torch.tensor(delays, dtype=self.dtype,
                                                     device=self.device))
        self.register_buffer("_powers", torch.tensor(powers, dtype=self.dtype,
                                                     device=self.device))
        self.register_buffer("_mean_powers", torch.tensor(mean_powers, dtype=self.dtype,
                                                          device=self.device))
        self.register_buffer("_los_power", torch.tensor(los_power, dtype=self.dtype,
                                                        device=self.device))
        self.register_buffer("_k_factor", torch.tensor(k, dtype=self.dtype,
                                                       device=self.device))
        self.register_buffer("_xpr", torch.tensor(
            np.power(10.0, float(params["xpr"]) / 10.0), dtype=self.dtype,
            device=self.device))

        # The angles in radians in the precision of the model, the departure and the
        # arrival angles swapped in the uplink, where the UT transmits
        names = {"aod": "aod", "aoa": "aoa", "zod": "zod", "zoa": "zoa"}
        if self._direction == "uplink":
            names = {"aod": "aoa", "aoa": "aod", "zod": "zoa", "zoa": "zod"}
        for name, source_name in names.items():
            angles = deg_2_rad(torch.tensor(self._angles[source_name],
                                            dtype=torch.float64))
            self.register_buffer(f"_{name}", angles.to(self.dtype).to(self.device))
            # The angle of the specular path, 0 for the NLOS models, as Sionna's CDL
            # gives the topology
            los_angle = torch.zeros([1, 1, 1], dtype=torch.float64)
            if self._los:
                los_angle = los_angle + deg_2_rad(torch.tensor(
                    self._angles[f"los_{source_name}"], dtype=torch.float64))
            self.register_buffer(f"_los_{name}",
                                 los_angle.to(self.dtype).to(self.device))

    def _get_velocity(self, batch_size: int) -> torch.Tensor:
        """Get UT velocity, either from a specified value or random sampling.

        :param batch_size: Batch size

        :output velocity: UT velocity [m/s], shape [batch_size, 3]
        """
        if self._ut_velocity is not None:
            velocity = self._ut_velocity
            if velocity.dim() == 1:
                velocity = velocity.unsqueeze(0).expand(batch_size, -1)
            return velocity

        v_r = (
            rand((batch_size, 1), dtype=self.dtype, device=self.device, generator=self.torch_rng)
            * (self._max_speed - self._min_speed) + self._min_speed
        )
        v_phi = (
            rand((batch_size, 1), dtype=self.dtype, device=self.device, generator=self.torch_rng)
            * 2.0 * PI
        )
        v_theta = (
            rand((batch_size, 1), dtype=self.dtype, device=self.device, generator=self.torch_rng)
            * PI
        )
        velocity = torch.cat(
            [
                v_r * torch.cos(v_phi) * torch.sin(v_theta),
                v_r * torch.sin(v_phi) * torch.sin(v_theta),
                v_r * torch.cos(v_theta),
            ],
            dim=-1,
        )
        return velocity

    def _create_topology(self, batch_size: int, velocity: torch.Tensor) -> Topology:
        r"""Create the topology of the link for the channel coefficient generator

        The package's topology, with the height of the platform and the elevation angle
        of the link, which decide its Faraday rotation, and without the Doppler shift of
        the satellite of the generator, which the call applies instead. The LOS phase
        and the distance are 0: the delays are relative to the first path.

        :param batch_size: Batch size
        :param velocity: UT velocity [m/s], shape [batch_size, 3]

        :output topology: Topology of the link
        """
        los_aoa = self._los_aoa.expand(batch_size, 1, 1)
        los_aod = self._los_aod.expand(batch_size, 1, 1)
        los_zoa = self._los_zoa.expand(batch_size, 1, 1)
        los_zod = self._los_zod.expand(batch_size, 1, 1)

        los = torch.full(
            (batch_size, 1, 1), self._los, dtype=torch.bool, device=self.device
        )

        # Distance and LOS phase
        zeros = torch.zeros((batch_size, 1, 1), dtype=self.dtype, device=self.device)

        # Orientations
        ut_orientation = self._ut_orientation
        if ut_orientation.dim() == 1:
            ut_orientation = ut_orientation.unsqueeze(0).expand(batch_size, -1)
        ut_orientation = ut_orientation.unsqueeze(1)  # Add UT dimension

        bs_orientation = self._bs_orientation
        if bs_orientation.dim() == 1:
            bs_orientation = bs_orientation.unsqueeze(0).expand(batch_size, -1)
        bs_orientation = bs_orientation.unsqueeze(1)  # Add BS dimension

        # Set TX and RX orientations and the moving end based on direction
        if self._direction == "downlink":
            tx_orientations = bs_orientation
            rx_orientations = ut_orientation
            moving_end = "rx"
        else:
            tx_orientations = ut_orientation
            rx_orientations = bs_orientation
            moving_end = "tx"

        topology = Topology(
            velocities=velocity.unsqueeze(1),
            moving_end=moving_end,
            los_aoa=los_aoa,
            los_aod=los_aod,
            los_zoa=los_zoa,
            los_zod=los_zod,
            los=los,
            distance_3d=zeros,
            tx_orientations=tx_orientations,
            rx_orientations=rx_orientations,
            bs_height=self._bs_height,
            elevation_angle=self._elevation_angle,
            doppler_enabled=False,
            los_phase=zeros,
        )

        return topology

    def _create_rays(self, batch_size: int) -> Rays:
        """Create rays from CDL model parameters.

        :param batch_size: Batch size

        :output rays: Rays
        """
        num_clusters = self._num_clusters
        rays_per_cluster = len(_RAY_OFFSETS)

        # Expand to batch and BS/UT dimensions
        # Shape: [batch, num_tx=1, num_rx=1, num_clusters]
        delays = self._delays.reshape(1, 1, 1, num_clusters).expand(batch_size, 1, 1, -1)
        powers = self._powers.reshape(1, 1, 1, num_clusters).expand(batch_size, 1, 1, -1)

        # Angles are [num_clusters, rays_per_cluster]
        # Shape: [batch, num_tx=1, num_rx=1, num_clusters, rays_per_cluster]
        aod = self._aod.reshape(1, 1, 1, num_clusters, rays_per_cluster).expand(
            batch_size, 1, 1, -1, -1
        )
        aoa = self._aoa.reshape(1, 1, 1, num_clusters, rays_per_cluster).expand(
            batch_size, 1, 1, -1, -1
        )
        zod = self._zod.reshape(1, 1, 1, num_clusters, rays_per_cluster).expand(
            batch_size, 1, 1, -1, -1
        )
        zoa = self._zoa.reshape(1, 1, 1, num_clusters, rays_per_cluster).expand(
            batch_size, 1, 1, -1, -1
        )

        # Apply random coupling (TR 38.901 clause 7.7.1, step 2)
        aoa, aod, zoa, zod = self._random_coupling(aoa, aod, zoa, zod)

        # XPR: [batch, num_tx=1, num_rx=1, num_clusters, rays_per_cluster]
        xpr = self._xpr.reshape(1, 1, 1, 1, 1).expand(
            batch_size, 1, 1, num_clusters, rays_per_cluster
        )

        rays = Rays(
            delays=delays,
            powers=powers,
            aoa=aoa,
            aod=aod,
            zoa=zoa,
            zod=zod,
            xpr=xpr,
        )

        return rays

    def _shuffle_angles(self, angles: torch.Tensor) -> torch.Tensor:
        """Randomly shuffle angles of arrival/departure within each cluster.

        :param angles: Angles to shuffle,
            [batch_size, num_tx, num_rx, num_clusters, num_rays]

        :output shuffled_angles: Shuffled angles with the same shape
        """
        # Create randomly shuffled indices by arg-sorting samples from a random
        # normal distribution
        random_numbers = normal(
            angles.shape, dtype=self.dtype, device=self.device, generator=self.torch_rng
        )
        shuffled_indices = torch.argsort(random_numbers, dim=-1)

        # Shuffling the angles using gather
        shuffled_angles = torch.gather(angles, dim=-1, index=shuffled_indices)
        return shuffled_angles

    def _random_coupling(
        self,
        aoa: torch.Tensor,
        aod: torch.Tensor,
        zoa: torch.Tensor,
        zod: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Randomly couple angles within a cluster (TR 38.901 clause 7.7.1, step 2).

        :param aoa: Azimuth angles of arrival (AoA),
            [batch_size, num_tx, num_rx, num_clusters, num_rays]
        :param aod: Azimuth angles of departure (AoD),
            [batch_size, num_tx, num_rx, num_clusters, num_rays]
        :param zoa: Zenith angles of arrival (ZoA),
            [batch_size, num_tx, num_rx, num_clusters, num_rays]
        :param zod: Zenith angles of departure (ZoD),
            [batch_size, num_tx, num_rx, num_clusters, num_rays]

        :output shuffled_aoa: Shuffled azimuth angles of arrival.

        :output shuffled_aod: Shuffled azimuth angles of departure.

        :output shuffled_zoa: Shuffled zenith angles of arrival.

        :output shuffled_zod: Shuffled zenith angles of departure.
        """
        shuffled_aoa = self._shuffle_angles(aoa)
        shuffled_aod = self._shuffle_angles(aod)
        shuffled_zoa = self._shuffle_angles(zoa)
        shuffled_zod = self._shuffle_angles(zod)

        return shuffled_aoa, shuffled_aod, shuffled_zoa, shuffled_zod
