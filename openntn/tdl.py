#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by the Dept. of Communications Engineering of the University of Bremen.
# The code is based on implementations provided by the NVIDIA CORPORATION & AFFILIATES
# in Sionna.
#
"""Tapped delay line models NTN-TDL-A to NTN-TDL-D of 3GPP TR 38.811"""

import json
import math
import numbers
from importlib.resources import files
from typing import List, Optional, Tuple, Union

import numpy as np
import torch

from sionna.phy import PI, SPEED_OF_LIGHT
from sionna.phy.channel import ChannelModel
from sionna.phy.utils import (insert_dims, expand_to_rank, split_dim, flatten_last_dims,
                              rand)

from . import models
from .channel_coefficients import _satellite_doppler_speed
from .utils import _AERIAL_HEIGHTS, _SATELLITE_HEIGHTS

__all__ = ["TDL"]

# The models of TR 38.811 V15.4.0, clause 6.9.2, each in its parameter file
# TDL-<model>.json
_MODELS = ("A", "B", "C", "D")
# TR 38.901 V16.1.0, clause 7.7.2: the specular path of a LOS model adds a peak to the
# Doppler spectrum of its first tap at f_S = 0.7 f_D, that is at the angle of arrival
# arccos(0.7) relative to the direction of motion
_LOS_ANGLE_OF_ARRIVAL = math.acos(0.7)


def _real_number(name: str, value: object) -> float:
    r"""Returns a real number as a float

    :param name: Name of the argument, for the error
    :param value: Value of the argument

    :output: `float`. The value; `TypeError` for a bool or a value that is not a real
        number
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f"{name} must be a real number, not {value!r}")
    return float(value)


def _positive_integer(name: str, value: object) -> int:
    r"""Returns an integer of at least 1 as an int

    :param name: Name of the argument, for the error
    :param value: Value of the argument

    :output: `int`. The value; `TypeError` for a bool or a value that is not an integer,
        `ValueError` for an integer below 1
    """
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        raise TypeError(f"{name} must be an integer, not {value!r}")
    if value < 1:
        raise ValueError(f"{name} must be at least 1, not {value!r}")
    return int(value)


def _k_factor_profile(powers: List[float], delays: List[float],
                      k_factor: float) -> Tuple[np.ndarray, np.ndarray]:
    r"""Powers and normalized delays of a LOS model with a desired K-factor

    TR 38.901 V16.1.0, clause 7.7.6: the K-factor of the model is the power of the
    specular path over the sum of the powers of all Rayleigh taps (eq. (7.7.6-2)); the
    powers of the Rayleigh taps change by eq. (7.7.6-1), so that this ratio becomes the
    desired K-factor, and the delays are divided by the RMS delay spread of the scaled
    profile, the specular path included, so that it is one again.

    :param powers: Powers of the table [dB], the specular path first
    :param delays: Normalized delays of the table, the specular path first
    :param k_factor: Desired K-factor [dB]

    :output powers: [num entries], `numpy.float64`. Powers [dB], the specular path first

    :output delays: [num entries], `numpy.float64`. Normalized delays, the specular path
        first
    """
    table_powers = np.asarray(powers, dtype=np.float64)
    table_delays = np.asarray(delays, dtype=np.float64)
    k_model = table_powers[0] - 10.0 * np.log10(
        np.sum(np.power(10.0, table_powers[1:] / 10.0)))
    scaled = np.concatenate([table_powers[:1], table_powers[1:] - k_factor + k_model])
    linear = np.power(10.0, scaled / 10.0)
    mean_delay = np.sum(linear * table_delays) / np.sum(linear)
    spread = np.sqrt(np.sum(linear * (table_delays - mean_delay) ** 2) / np.sum(linear))
    return scaled, table_delays / spread


def _check_device_and_precision(model: ChannelModel, reference: torch.Tensor) -> None:
    r"""Checks that the tensors of a model are still on the device and in the precision
    of its construction

    A model stays on the device and in the precision of its construction, as Sionna's
    models do: ``.to()`` moves or casts its registered tensors, but neither the device
    and the precision of the model nor the tensors held outside the buffers, so a
    computation would fail inside with an error from PyTorch, or mix precisions. The
    device and the dtype of a buffer registered at the construction are compared with
    those of the model; no tensor value is read.

    :param model: The model, whose class, device and precision the error names
    :param reference: A buffer of the model, registered at its construction

    :output: `None`. Raises `ValueError` if ``reference`` is on another device or of
        another dtype than the model.
    """
    if (reference.device != torch.device(model.device)
            or reference.dtype != model.dtype):
        raise ValueError(
            f"{type(model).__name__} was built on device {model.device!r} in "
            f"{model.precision} precision, but its tensors are {reference.dtype} on "
            f"{reference.device}; build the model with the device and precision it "
            f"is to use instead of moving or casting it")


def _satellite_shift_factor(doppler_shift: float, num_time_steps: int,
                            sampling_frequency: Union[float, torch.Tensor],
                            cdtype: torch.dtype,
                            device: Union[str, torch.device]) -> torch.Tensor:
    r"""Phase factors of the Doppler shift due to the motion of the satellite

    The factor at the sample :math:`n` is :math:`\exp(j 2 \pi f_d n / f_s)`. Its
    phase is computed in double precision as :math:`2 \pi` times the fractional part
    of :math:`n f_d / f_s`, so that it stays exact over long sequences also in single
    precision, where :math:`2 \pi f_d n / f_s` itself would lose its accuracy: at
    681 kHz, the shift of a satellite at 600 km seen at 10 degrees at 30 GHz, the phase
    reaches 4.3e6 rad after one second, where single-precision numbers are 0.5 rad
    apart.

    :param doppler_shift: Doppler shift :math:`f_d` [Hz]
    :param num_time_steps: Number of time steps
    :param sampling_frequency: Sampling frequency :math:`f_s` [Hz]
    :param cdtype: Complex dtype of the factors
    :param device: Device of the factors

    :output factor: [num_time_steps], `torch.complex`. Phase factors
    """
    if isinstance(sampling_frequency, torch.Tensor):
        sampling_frequency = sampling_frequency.to(dtype=torch.float64, device=device)
    samples = torch.arange(num_time_steps, dtype=torch.float64, device=device)
    cycles = samples * (doppler_shift / sampling_frequency)
    phase = (cycles - torch.floor(cycles)) * (2.0 * math.pi)
    return torch.complex(torch.cos(phase), torch.sin(phase)).to(cdtype)


class TDL(ChannelModel):
    # pylint: disable=line-too-long
    r"""Tapped delay line models NTN-TDL-A to NTN-TDL-D of :cite:p:`TR38811`

    The models of TR 38.811 V15.4.0, clause 6.9.2: NTN-TDL-A and NTN-TDL-B represent two
    channel profiles for NLOS links, NTN-TDL-C and NTN-TDL-D two for LOS links, with the
    normalized delays and the powers of Tables 6.9.2-1 to 6.9.2-4, which are given at
    the reference elevation angle of 50 degrees. The first tap of NTN-TDL-C and
    NTN-TDL-D follows a Rician distribution: a specular LOS path adds to the Rayleigh
    fading of the tap. The power delay profile is normalized to a total power of one,
    the specular path included.

    The model applies the procedures of TR 38.901 V16.1.0 :cite:p:`TR38901`, clause 7.7,
    to which clause 6.9.2 refers, as Sionna's TR 38.901 TDL does, and the Doppler shift
    due to the motion of the satellite of clause 6.9.2:

    - The Doppler spectrum of each tap is the classical spectrum of clause 7.7.2, with
      the maximum Doppler frequency :math:`f_D = v f_c / c` of a UT at the speed
      :math:`v`, which is drawn uniformly from [``min_speed``, ``max_speed``] for each
      batch example. The tap coefficients are sums of sinusoids :cite:p:`SoS`. The
      specular path of NTN-TDL-C and NTN-TDL-D adds a peak at :math:`f_S = 0.7 f_D`
      with the default ``los_angle_of_arrival``.
    - The normalized delays are scaled to the RMS delay spread ``delay_spread``
      (clause 7.7.3).
    - A desired K-factor ``k_factor`` of NTN-TDL-C or NTN-TDL-D changes the powers of
      the Rayleigh taps by eq. (7.7.6-1) and divides the normalized delays by the RMS
      delay spread of the scaled profile, the specular path included, so that it is one
      again before the delays are scaled (clause 7.7.6).
    - The Doppler shift due to the motion of the satellite,

      .. math::
          f_{d,\text{shift}} = \frac{v_\text{sat}}{c} \frac{R}{R+h} \cos(\alpha) f_c

      with the speed :math:`v_\text{sat}` of a circular orbit at the height :math:`h`,
      the radius :math:`R` of the Earth and the elevation angle :math:`\alpha` of the
      link, is applied to all taps and all antennas (clause 6.9.2): the path
      coefficients at the sample :math:`n` of a call are multiplied by
      :math:`\exp(j 2 \pi f_{d,\text{shift}} n / f_s)`. The clause writes the angle as
      :math:`\alpha_\text{model}` and defines it as the satellite elevation angle; the
      shift is that of the simulated link, not of the reference elevation angle of the
      tables. It is the largest shift, of an approaching satellite seen from a UT in its
      orbital plane, and it is 0 for a geostationary satellite, which is fixed relative
      to the Earth (clause 5.3.2.3), and for an aerial platform. ``doppler_shift``
      replaces it: 0 for a link whose terminal pre-compensates the shift, as
      3GPP TS 38.101-5 sets the Doppler shift of the satellite motion to zero for the
      performance requirements of its Table 8.2.1.2.2.1.1-3 (clause A.4.4), a negative
      value for a receding satellite, or the shift of a moving aerial platform.
    - Several antennas, ``num_rx_ant`` and ``num_tx_ant``, get the same power delay
      profile, with the spatial correlation of clause 7.7.5.2 given by the correlation
      matrices. The NTN-CDL models of clause 6.9.1 have zero angular spreads of
      departure at the satellite, so a TDL that is consistent with them has a
      correlation of rank one across the antennas of the satellite.

    The shift is contained in the path coefficients at every sample. A channel in the
    frequency domain that is sampled once per OFDM symbol carries its phase from symbol
    to symbol, but not the inter-carrier interference that it causes within a symbol,
    which only the channel in the time domain at the sampling rate of the signal shows.
    The shift can exceed the subcarrier spacing: at 600 km, 50 degrees and 2 GHz it is
    29.63 kHz, 1.98 subcarrier spacings of 15 kHz.

    The speeds of the UT and of the satellite and the elevation angle are constant
    during a call, which suits simulations of a few TTIs, as clause 6.9.2 allows. The
    carrier frequency lies in the S band, 1.9 GHz to 4 GHz, or in the Ka band, 19 GHz to
    40 GHz, the bands of the system-level models of the package; clause 6.9.1 defines
    the NTN-CDL models, from which the NTN-TDL models are derived, for the S and the Ka
    band. Suitable values of ``delay_spread`` and ``k_factor`` for a band, an
    environment and an elevation angle follow from the parameters of clause 6.7.2
    (clause 6.9.1): :func:`~openntn.utils.link_level_parameters` returns the median
    delay spread and the mean K-factor of its tables at the reference elevation angle
    nearest to that of the link.

    The arguments after ``satellite_height`` are taken by keyword only. A wrong value
    raises `ValueError` and a wrong type `TypeError`. The model stays on the device and
    in the precision of its construction: moved with ``.to()`` or cast, it raises
    `ValueError` in its next call.

    :param model: The model, ``"A"``, ``"B"``, ``"C"`` or ``"D"``, for NTN-TDL-A to
        NTN-TDL-D; `ValueError` otherwise
    :param delay_spread: RMS delay spread [s], finite and greater than 0
    :param carrier_frequency: Carrier frequency [Hz], from 1.9 GHz to 4 GHz or from
        19 GHz to 40 GHz
    :param elevation_angle: Elevation angle of the link [deg], from 10 to 90
    :param satellite_height: Height of the satellite, from 160 km to 35 786 km, or of an
        aerial platform, from 8 km to 50 km [m]
    :param doppler_shift: Doppler shift due to the motion of the satellite [Hz], a
        finite number. Defaults to `None`, the shift of clause 6.9.2 at the elevation
        angle of the link, 0 for a geostationary satellite and an aerial platform.
    :param k_factor: Desired K-factor of NTN-TDL-C or NTN-TDL-D [dB], a finite number,
        applied by clause 7.7.6 of TR 38.901; `ValueError` for NTN-TDL-A and NTN-TDL-B.
        Defaults to `None`, the K-factor of the table.
    :param num_sinusoids: Number of sinusoids of the sum-of-sinusoids model, at least 1.
        Defaults to 20.
    :param los_angle_of_arrival: Angle of arrival of the specular path relative to the
        direction of motion [radian]; used by NTN-TDL-C and NTN-TDL-D only. Defaults to
        :math:`\arccos(0.7)`, so that the specular path appears at :math:`0.7 f_D` in
        the Doppler spectrum (clause 7.7.2 of TR 38.901).
    :param min_speed: Minimum speed of the UT [m/s], finite and at least 0. Defaults to
        0.
    :param max_speed: Maximum speed of the UT [m/s], finite and at least ``min_speed``.
        Defaults to `None`, the value of ``min_speed``.
    :param num_rx_ant: Number of receive antennas, at least 1. Defaults to 1.
    :param num_tx_ant: Number of transmit antennas, at least 1. Defaults to 1.
    :param spatial_corr_mat: Spatial correlation matrix of shape
        [num_rx_ant*num_tx_ant, num_rx_ant*num_tx_ant]. If not `None`, ``rx_corr_mat``
        and ``tx_corr_mat`` are ignored. Defaults to `None`.
    :param rx_corr_mat: Spatial correlation matrix of the receiver of shape
        [num_rx_ant, num_rx_ant]. Defaults to `None`, no receive correlation unless
        ``spatial_corr_mat`` is given.
    :param tx_corr_mat: Spatial correlation matrix of the transmitter of shape
        [num_tx_ant, num_tx_ant]. Defaults to `None`, no transmit correlation unless
        ``spatial_corr_mat`` is given.
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

    :output a: [batch size, num_rx=1, num_rx_ant, num_tx=1, num_tx_ant, num_paths, num_time_steps], `torch.complex`.
        Path coefficients.

    :output tau: [batch size, num_rx=1, num_tx=1, num_paths], `torch.float`.
        Path delays [s], relative to the first path.

    .. rubric:: Examples

    .. code-block:: python

        from openntn import TDL

        # NTN-TDL-C at 2 GHz, a satellite at 600 km seen at 50 degrees, a UT at up to
        # 3 m/s
        tdl = TDL("C", delay_spread=100e-9, carrier_frequency=2e9,
                  elevation_angle=50.0, satellite_height=600e3, max_speed=3.0)
        a, tau = tdl(batch_size=16, num_time_steps=14, sampling_frequency=30e3)
        print(a.shape, tau.shape)
        # torch.Size([16, 1, 1, 1, 1, 2, 14]) torch.Size([16, 1, 1, 2])
        print(round(tdl.doppler_shift, 3))
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
        doppler_shift: Optional[float] = None,
        k_factor: Optional[float] = None,
        num_sinusoids: int = 20,
        los_angle_of_arrival: float = _LOS_ANGLE_OF_ARRIVAL,
        min_speed: float = 0.0,
        max_speed: Optional[float] = None,
        num_rx_ant: int = 1,
        num_tx_ant: int = 1,
        spatial_corr_mat: Optional[torch.Tensor] = None,
        rx_corr_mat: Optional[torch.Tensor] = None,
        tx_corr_mat: Optional[torch.Tensor] = None,
        precision: Optional[str] = None,
        device: Optional[str] = None,
    ) -> None:
        if not isinstance(model, str):
            raise TypeError(f"model must be one of 'A', 'B', 'C' and 'D', not "
                            f"{model!r}")
        if model not in _MODELS:
            raise ValueError(f"model must be one of 'A', 'B', 'C' and 'D', not "
                             f"{model!r}")
        delay_spread = self._checked_delay_spread(delay_spread)
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
        num_sinusoids = _positive_integer("num_sinusoids", num_sinusoids)
        num_rx_ant = _positive_integer("num_rx_ant", num_rx_ant)
        num_tx_ant = _positive_integer("num_tx_ant", num_tx_ant)
        min_speed = _real_number("min_speed", min_speed)
        if not (math.isfinite(min_speed) and min_speed >= 0.0):
            raise ValueError(f"min_speed must be finite and at least 0 m/s, not "
                             f"{min_speed!r}")
        if max_speed is not None:
            max_speed = _real_number("max_speed", max_speed)
            if not (math.isfinite(max_speed) and max_speed >= min_speed):
                raise ValueError(f"max_speed must be finite and at least min_speed "
                                 f"({min_speed!r} m/s), not {max_speed!r}")
        super().__init__(precision=precision, device=device)

        # Load model parameters
        self._load_parameters(model, k_factor)

        self._num_rx_ant = num_rx_ant
        self._num_tx_ant = num_tx_ant
        # Register as buffers for CUDAGraph compatibility
        self.register_buffer("_carrier_frequency", torch.tensor(carrier_frequency, dtype=self.dtype, device=self.device))
        self._num_sinusoids = num_sinusoids
        self.register_buffer("_los_angle_of_arrival", torch.tensor(los_angle_of_arrival, dtype=self.dtype, device=self.device))
        self.register_buffer("_delay_spread", torch.tensor(delay_spread, dtype=self.dtype, device=self.device))
        self.register_buffer("_min_speed", torch.tensor(min_speed, dtype=self.dtype, device=self.device))
        if max_speed is None:
            self.register_buffer("_max_speed", self._min_speed.clone())
        else:
            self.register_buffer("_max_speed", torch.tensor(max_speed, dtype=self.dtype, device=self.device))

        # Pre-compute maximum and minimum Doppler shifts
        self._min_doppler = self._compute_doppler(self._min_speed)
        self._max_doppler = self._compute_doppler(self._max_speed)

        # Precompute average angles of arrivals for each sinusoid
        alpha_const = 2. * PI / num_sinusoids * \
                      torch.arange(1., self._num_sinusoids + 1, 1., dtype=self.dtype, device=self.device)
        self._alpha_const = alpha_const.reshape(
            1,  # batch size
            1,  # num rx
            1,  # num rx ant
            1,  # num tx
            1,  # num tx ant
            1,  # num clusters
            1,  # num time steps
            num_sinusoids
        )

        # Precompute square root of spatial covariance matrices
        # Use cholesky_ex for CUDA graph compatibility
        if spatial_corr_mat is not None:
            spatial_corr_mat = torch.as_tensor(spatial_corr_mat, dtype=self.cdtype, device=self.device)
            spatial_corr_mat_sqrt, _ = torch.linalg.cholesky_ex(spatial_corr_mat, check_errors=False)
            spatial_corr_mat_sqrt = expand_to_rank(spatial_corr_mat_sqrt, 7, 0)
            self._spatial_corr_mat_sqrt = spatial_corr_mat_sqrt
        else:
            self._spatial_corr_mat_sqrt = None
            if rx_corr_mat is not None:
                rx_corr_mat = torch.as_tensor(rx_corr_mat, dtype=self.cdtype, device=self.device)
                rx_corr_mat_sqrt, _ = torch.linalg.cholesky_ex(rx_corr_mat, check_errors=False)
                rx_corr_mat_sqrt = expand_to_rank(rx_corr_mat_sqrt, 7, 0)
                self._rx_corr_mat_sqrt = rx_corr_mat_sqrt
            else:
                self._rx_corr_mat_sqrt = None
            if tx_corr_mat is not None:
                tx_corr_mat = torch.as_tensor(tx_corr_mat, dtype=self.cdtype, device=self.device)
                tx_corr_mat_sqrt, _ = torch.linalg.cholesky_ex(tx_corr_mat, check_errors=False)
                tx_corr_mat_sqrt = expand_to_rank(tx_corr_mat_sqrt, 7, 0)
                self._tx_corr_mat_sqrt = tx_corr_mat_sqrt
            else:
                self._tx_corr_mat_sqrt = None

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
        r"""Number of paths (:math:`M`)"""
        return self._num_clusters

    @property
    def los(self) -> bool:
        r"""`True` for the LOS models NTN-TDL-C and NTN-TDL-D, `False` otherwise"""
        return self._los

    @property
    def k_factor(self) -> torch.Tensor:
        r"""K-factor of the model in linear scale: the power of the specular path over
        the sum of the powers of all Rayleigh taps, eq. (7.7.6-2) of TR 38.901. It is
        that of the table, 10.000 dB for NTN-TDL-C and 7.000 dB for NTN-TDL-D, or
        ``k_factor``. The NOTEs of Tables 6.9.2-3 and 6.9.2-4 of TR 38.811 give instead
        the K-factor K1 of the first tap, the power of the specular path over that of
        the Rayleigh part of the first tap, 10.224 dB and 11.707 dB; Sionna's TR 38.901
        TDL reports K1 as its ``k_factor``. `RuntimeError` for NTN-TDL-A and
        NTN-TDL-B."""
        if not self._los:
            raise RuntimeError("k_factor is defined for the LOS models 'C' and 'D' "
                               "only")
        return torch.real(self._los_power / torch.sum(self._mean_powers))

    @property
    def delays(self) -> torch.Tensor:
        r"""Path delays [s]"""
        return self._delays * self._delay_spread

    @property
    def mean_powers(self) -> torch.Tensor:
        r"""Path powers in linear scale, the specular path included in the first path"""
        if self._los:
            mean_powers = torch.cat([self._mean_powers[:1] + self._los_power,
                                     self._mean_powers[1:]], dim=0)
        else:
            mean_powers = self._mean_powers
        return torch.real(mean_powers)

    @property
    def mean_power_los(self) -> torch.Tensor:
        r"""Power of the specular path in linear scale. `RuntimeError` for NTN-TDL-A and
        NTN-TDL-B."""
        if not self._los:
            raise RuntimeError("mean_power_los is defined for the LOS models 'C' and "
                               "'D' only")
        return torch.real(self._los_power)

    @property
    def delay_spread(self) -> torch.Tensor:
        r"""RMS delay spread [s]"""
        return self._delay_spread

    @delay_spread.setter
    def delay_spread(self, value: float) -> None:
        value = self._checked_delay_spread(value)
        # Register as buffer for CUDAGraph compatibility
        self.register_buffer("_delay_spread", torch.tensor(value, dtype=self.dtype, device=self.device))

    @property
    def doppler_shift(self) -> float:
        r"""Doppler shift due to the motion of the satellite [Hz], of all taps"""
        return self._doppler_shift

    def __call__(
        self,
        batch_size: int,
        num_time_steps: int,
        sampling_frequency: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        _check_device_and_precision(self, self._carrier_frequency)

        # Time steps
        sample_times = torch.arange(num_time_steps, dtype=self.dtype, device=self.device) \
            / sampling_frequency
        sample_times = insert_dims(sample_times, 6, 0).unsqueeze(-1)

        # Generate random maximum Doppler shifts for each sample
        # The Doppler shift is different for each TX-RX link, but shared by
        # all RX ant and TX ant couple for a given link.
        doppler = rand(
            (batch_size, 1, 1, 1, 1, 1, 1, 1),
            dtype=self.dtype,
            device=self.device,
            generator=self.torch_rng,
        ) * (self._max_doppler - self._min_doppler) + self._min_doppler

        # Eq. (7) in the paper [TDL] (see class docstring)
        # The angle of arrival is different for each TX-RX link.
        theta = rand(
            (batch_size, 1, 1, 1, 1, self._num_clusters, 1, self._num_sinusoids),
            dtype=self.dtype,
            device=self.device,
            generator=self.torch_rng,
        ) * (2 * PI / self._num_sinusoids) - PI / self._num_sinusoids
        alpha = self._alpha_const + theta

        # Eq. (6a)-(6c) in the paper [TDL] (see class docstring)
        phi = rand(
            (batch_size, 1, self._num_rx_ant, 1, self._num_tx_ant,
             self._num_clusters, 1, self._num_sinusoids),
            dtype=self.dtype,
            device=self.device,
            generator=self.torch_rng,
        ) * 2 * PI - PI

        argument = doppler * sample_times * torch.cos(alpha) + phi

        # Eq. (6a) in the paper [SoS]
        h = torch.complex(torch.cos(argument), torch.sin(argument))
        normalization_factor = 1. / torch.sqrt(torch.tensor(self._num_sinusoids, dtype=self.dtype, device=self.device))
        h = torch.complex(normalization_factor, torch.tensor(0., dtype=self.dtype, device=self.device)) \
            * h.sum(dim=-1)

        # Scaling by average power
        mean_powers = insert_dims(self._mean_powers, 5, 0).unsqueeze(-1)
        h = torch.sqrt(mean_powers) * h

        # Add specular component to first tap Eq. (11) in [SoS] if LoS
        if self._los:
            # The first tap follows a Rician
            # distribution

            # Specular component phase shift
            phi_0 = rand(
                (batch_size, 1, 1, 1, 1, 1, 1),
                dtype=self.dtype,
                device=self.device,
                generator=self.torch_rng,
            ) * 2 * PI - PI

            # Remove the sinusoids dim
            doppler = doppler.squeeze(-1)
            sample_times = sample_times.squeeze(-1)
            arg_spec = doppler * sample_times * torch.cos(self._los_angle_of_arrival) \
                + phi_0
            h_spec = torch.complex(torch.cos(arg_spec), torch.sin(arg_spec))

            # Update the first tap with the specular component
            h = torch.cat([h_spec * torch.sqrt(self._los_power) + h[:, :, :, :, :, :1, :],
                           h[:, :, :, :, :, 1:, :]],
                          dim=5)  # Path dim

        # Delays
        delays = self._delays * self._delay_spread
        delays = insert_dims(delays, 3, 0)
        delays = delays.expand(batch_size, -1, -1, -1)

        # Apply spatial correlation if required
        if self._spatial_corr_mat_sqrt is not None:
            h = h.permute(0, 1, 3, 5, 6, 2, 4)  # [..., num_rx_ant, num_tx_ant]
            h = flatten_last_dims(h, 2)  # [..., num_rx_ant*num_tx_ant]
            h = h.unsqueeze(-1)  # [..., num_rx_ant*num_tx_ant, 1]
            h = torch.matmul(self._spatial_corr_mat_sqrt, h)
            h = h.squeeze(-1)
            h = split_dim(h, [self._num_rx_ant, self._num_tx_ant],
                          h.dim() - 1)  # [..., num_rx_ant, num_tx_ant]
            h = h.permute(0, 1, 5, 2, 6, 3, 4)
        else:
            if ((self._rx_corr_mat_sqrt is not None)
                    or (self._tx_corr_mat_sqrt is not None)):
                h = h.permute(0, 1, 3, 5, 6, 2, 4)
                if self._rx_corr_mat_sqrt is not None:
                    h = torch.matmul(self._rx_corr_mat_sqrt, h)
                if self._tx_corr_mat_sqrt is not None:
                    h = torch.matmul(h, self._tx_corr_mat_sqrt.mH)
                h = h.permute(0, 1, 5, 2, 6, 3, 4)

        # The Doppler shift due to the motion of the satellite, one factor per sample
        # for all paths and antennas (TR 38.811 clause 6.9.2), applied after the random
        # draws, which it does not change. Without the shift the path coefficients are
        # those of the sum of sinusoids above, bit for bit.
        if self._doppler_shift != 0.0:
            h = h * self._doppler_shift_factor(num_time_steps, sampling_frequency)

        # Detach to avoid unnecessary gradient computation
        h = h.detach()
        delays = delays.detach()

        return h, delays

    ###########################################
    # Internal utility functions
    ###########################################

    @staticmethod
    def _checked_delay_spread(value: object) -> float:
        r"""Returns the RMS delay spread [s] as a float

        :param value: RMS delay spread [s]

        :output: `float`. The value; `TypeError` for a bool or a value that is not a
            real number, `ValueError` for a value that is not finite and greater than 0
        """
        value = _real_number("delay_spread", value)
        if not (math.isfinite(value) and value > 0.0):
            raise ValueError(f"delay_spread must be finite and greater than 0 s, not "
                             f"{value!r}")
        return value

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

    def _compute_doppler(self, speed: torch.Tensor) -> torch.Tensor:
        r"""Compute the maximum Doppler frequency in radians per second for a given
        speed [m/s].

        The maximum Doppler frequency :math:`\omega_d` is calculated as:

        .. math::
            \omega_d = 2\pi  \frac{v}{c} f_c

        where :math:`v` [m/s] is the speed of the receiver relative to the
        transmitter, :math:`c` [m/s] is the speed of light and,
        :math:`f_c` [Hz] the carrier frequency.

        :param speed: Speed [m/s]

        :output doppler: Maximum Doppler frequency [rad/s]
        """
        return 2. * PI * speed / SPEED_OF_LIGHT * self._carrier_frequency

    def _load_parameters(self, model: str, k_factor: Optional[float]) -> None:
        r"""Load the parameters of a model from its parameter file in ``openntn/models``

        The parameter file holds the table of the model with the following keys:
        ``los``, `True` if the first entry is a specular path; ``num_clusters``, the
        number of the other entries; ``delays``, the normalized delays; ``powers``, the
        powers [dB]. For the LOS models, the specular path and the Rayleigh part of the
        first tap have the delay 0, and the power of the specular path is stored
        separately. With ``k_factor``, the powers and the delays are first scaled by
        clause 7.7.6 of TR 38.901. The powers are normalized to a total of one, the
        specular path included.

        :param model: The model, ``"A"`` to ``"D"``
        :param k_factor: Desired K-factor [dB], or `None` for the table's
        """
        source = files(models).joinpath(f"TDL-{model}.json")
        with open(source, encoding="utf-8") as parameter_file:
            params = json.load(parameter_file)

        # LoS scenario ?
        self._los = bool(params['los'])

        # Loading cluster delays and mean powers
        self._num_clusters = int(params['num_clusters'])

        # Retrieve power and delays
        powers = params['powers']
        delays = params['delays']
        if k_factor is not None:
            powers, delays = _k_factor_profile(powers, delays, k_factor)
        delays = torch.tensor(delays, dtype=self.dtype, device=self.device)
        mean_powers = np.power(10.0, np.array(powers) / 10.0)
        mean_powers = torch.tensor(mean_powers, dtype=self.cdtype, device=self.device)

        if self._los:
            # The power of the specular component of the first path is stored
            # separately
            self._los_power = mean_powers[0]
            mean_powers = mean_powers[1:]
            # The first two paths have 0 delays as they correspond to the
            # specular and reflected components of the first path.
            # We need to keep only one.
            delays = delays[1:]

        # Normalize the PDP
        if self._los:
            norm_factor = torch.sum(mean_powers) + self._los_power
            self._los_power = self._los_power / norm_factor
            mean_powers = mean_powers / norm_factor
        else:
            norm_factor = torch.sum(mean_powers)
            mean_powers = mean_powers / norm_factor

        self._delays = delays
        self._mean_powers = mean_powers
