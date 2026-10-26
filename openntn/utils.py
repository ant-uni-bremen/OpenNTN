#
# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by the Dept. of Communications Engineering of the University of Bremen.
# The code is based on implementations provided by the NVIDIA CORPORATION & AFFILIATES
# in Sionna.
#
"""Utility functions for the channel module"""

import math
import warnings

import numpy as np
import torch

from sionna.phy import PI, config, dtypes
from sionna.phy.utils import expand_to_rank, randint
from sionna.phy.utils.random import uniform


def subcarrier_frequencies(num_subcarriers, subcarrier_spacing,
                           precision=None, device=None):
    # pylint: disable=line-too-long
    r"""
    Compute the baseband frequencies of ``num_subcarrier`` subcarriers spaced by
    ``subcarrier_spacing``, i.e.,

    >>> # If num_subcarrier is even:
    >>> frequencies = [-num_subcarrier/2, ..., 0, ..., num_subcarrier/2-1] * subcarrier_spacing
    >>>
    >>> # If num_subcarrier is odd:
    >>> frequencies = [-(num_subcarrier-1)/2, ..., 0, ..., (num_subcarrier-1)/2] * subcarrier_spacing


    Input
    ------
    num_subcarriers : int
        Number of subcarriers

    subcarrier_spacing : float
        Subcarrier spacing [Hz]

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
        frequencies : [``num_subcarrier``], torch.float
            Baseband frequencies of subcarriers
    """

    if precision is None:
        rdtype = config.dtype
    else:
        rdtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device

    if num_subcarriers % 2 == 0:
        start = -num_subcarriers // 2
        limit = num_subcarriers // 2
    else:
        start = -(num_subcarriers - 1) // 2
        limit = (num_subcarriers - 1) // 2 + 1

    frequencies = torch.arange(start, limit, dtype=rdtype, device=device)
    frequencies = frequencies * subcarrier_spacing
    return frequencies

def time_frequency_vector(num_samples, sample_duration, precision=None,
                          device=None):
    # pylint: disable=line-too-long
    r"""
    Compute the time and frequency vector for a given number of samples
    and duration per sample in normalized time unit.

    >>> t = torch.linspace(-n_min, n_max, num_samples) * sample_duration
    >>> f = torch.linspace(-n_min, n_max, num_samples) * 1/(sample_duration*num_samples)

    Input
    ------
        num_samples : int
            Number of samples

        sample_duration : float
            Sample duration in normalized time

        precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
        t : [``num_samples``], ``torch.float``
            Time vector

        f : [``num_samples``], ``torch.float``
            Frequency vector
    """

    if precision is None:
        rdtype = config.dtype
    else:
        rdtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device

    num_samples = int(num_samples)

    if num_samples % 2 == 0:  # if even
        n_min = -num_samples // 2
        n_max = num_samples // 2 - 1
    else:  # if odd
        n_min = -(num_samples - 1) // 2
        n_max = (num_samples + 1) // 2 - 1

    # Time vector
    indices = torch.linspace(n_min, n_max, num_samples, dtype=rdtype,
                             device=device)
    t = indices * sample_duration

    # Frequency vector
    df = 1.0 / sample_duration / num_samples
    f = indices * df

    return t, f

def time_lag_discrete_time_channel(bandwidth, maximum_delay_spread=3e-6):
    # pylint: disable=line-too-long
    r"""
    Compute the smallest and largest time-lag for the descrete complex baseband
    channel, i.e., :math:`L_{\text{min}}` and :math:`L_{\text{max}}`.

    The smallest time-lag (:math:`L_{\text{min}}`) returned is always -6, as this value
    was found small enough for all models included in Sionna.

    The largest time-lag (:math:`L_{\text{max}}`) is computed from the ``bandwidth``
    and ``maximum_delay_spread`` as follows:

    .. math::
        L_{\text{max}} = \lceil W \tau_{\text{max}} \rceil + 6

    where :math:`L_{\text{max}}` is the largest time-lag, :math:`W` the ``bandwidth``,
    and :math:`\tau_{\text{max}}` the ``maximum_delay_spread``.

    The default value for the ``maximum_delay_spread`` is 3us, which was found
    to be large enough to include most significant paths with all channel models
    included in Sionna assuming a nominal delay spread of 100ns.

    Note
    ----
    The values of :math:`L_{\text{min}}` and :math:`L_{\text{max}}` computed
    by this function are only recommended values.
    :math:`L_{\text{min}}` and :math:`L_{\text{max}}` should be set according to
    the considered channel model. For OFDM systems, one also needs to be careful
    that the effective length of the complex baseband channel is not larger than
    the cyclic prefix length.

    Input
    ------
    bandwidth : float
        Bandwith (:math:`W`) [Hz]

    maximum_delay_spread : float
        Maximum delay spread [s]. Defaults to 3us.

    Output
    -------
    l_min : int
        Smallest time-lag (:math:`L_{\text{min}}`) for the descrete complex baseband
        channel. Set to -6, , as this value was found small enough for all models
        included in Sionna.

    l_max : int
        Largest time-lag (:math:`L_{\text{max}}`) for the descrete complex baseband
        channel
    """
    l_min = -6
    l_max = int(math.ceil(maximum_delay_spread * bandwidth) + 6)
    return l_min, l_max

def cir_to_ofdm_channel(frequencies, a, tau, normalize=False):
    # pylint: disable=line-too-long
    r"""
    Compute the frequency response of the channel at ``frequencies``.

    Given a channel impulse response
    :math:`(a_{m}, \tau_{m}), 0 \leq m \leq M-1` (inputs ``a`` and ``tau``),
    the channel frequency response for the frequency :math:`f`
    is computed as follows:

    .. math::
        \widehat{h}(f) = \sum_{m=0}^{M-1} a_{m} e^{-j2\pi f \tau_{m}}

    Input
    ------
    frequencies : [fft_size], torch.float
        Frequencies at which to compute the channel response

    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], torch.complex
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths] or [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths], torch.float
        Path delays

    normalize : bool
        If set to `True`, the channel is normalized over the resource grid
        to ensure unit average energy per resource element. Defaults to `False`.

    Output
    -------
    h_f : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size], torch.complex
        Channel frequency responses at ``frequencies``
    """

    real_dtype = tau.dtype

    if tau.dim() == 4:
        # Expand dims to broadcast with h. Add the following dimensions:
        #  - number of rx antennas (2)
        #  - number of tx antennas (4)
        tau = tau.unsqueeze(2).unsqueeze(4)
        # Broadcast is not supported yet for such high rank tensors.
        # We therefore do part of it manually
        tau = tau.expand(-1, -1, -1, -1, a.shape[4], -1)

    # Add a time samples dimension for broadcasting
    tau = tau.unsqueeze(6)

    # Bring all tensors to broadcastable shapes
    tau = tau.unsqueeze(-1)
    h = a.unsqueeze(-1)
    frequencies = frequencies.to(device=tau.device, dtype=real_dtype)
    frequencies = expand_to_rank(frequencies, tau.dim(), axis=0)

    ## Compute the Fourier transforms of all cluster taps
    # Exponential component
    e = torch.exp(torch.complex(torch.zeros_like(tau),
                                -2 * PI * frequencies * tau))
    h_f = h * e
    # Sum over all clusters to get the channel frequency responses
    h_f = h_f.sum(dim=-3)

    if normalize:
        # Normalization is performed such that for each batch example and
        # link the energy per resource grid is one.
        # Average over TX antennas, RX antennas, OFDM symbols and
        # subcarriers.
        c = h_f.abs().square().mean(dim=(2, 4, 5, 6), keepdim=True)
        c = torch.complex(c.sqrt(), torch.zeros_like(c))
        h_f = torch.where(c.abs() > 0, h_f / c, h_f)

    return h_f

def cir_to_time_channel(bandwidth, a, tau, l_min, l_max, normalize=False):
    # pylint: disable=line-too-long
    r"""
    Compute the channel taps forming the discrete complex-baseband
    representation of the channel from the channel impulse response
    (``a``, ``tau``).

    This function assumes that a sinc filter is used for pulse shaping and receive
    filtering. Therefore, given a channel impulse response
    :math:`(a_{m}(t), \tau_{m}), 0 \leq m \leq M-1`, the channel taps
    are computed as follows:

    .. math::
        \bar{h}_{b, \ell}
        = \sum_{m=0}^{M-1} a_{m}\left(\frac{b}{W}\right)
            \text{sinc}\left( \ell - W\tau_{m} \right)

    for :math:`\ell` ranging from ``l_min`` to ``l_max``, and where :math:`W` is
    the ``bandwidth``.

    Input
    ------
    bandwidth : float
        Bandwidth [Hz]

    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], torch.complex
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths] or [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths], torch.float
        Path delays [s]

    l_min : int
        Smallest time-lag for the discrete complex baseband channel (:math:`L_{\text{min}}`)

    l_max : int
        Largest time-lag for the discrete complex baseband channel (:math:`L_{\text{max}}`)

    normalize : bool
        If set to `True`, the channel is normalized over the block size
        to ensure unit average energy per time step. Defaults to `False`.

    Output
    -------
    hm :  [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1], torch.complex
        Channel taps coefficients
    """

    real_dtype = tau.dtype

    if tau.dim() == 4:
        # Expand dims to broadcast with h. Add the following dimensions:
        #  - number of rx antennas (2)
        #  - number of tx antennas (4)
        tau = tau.unsqueeze(2).unsqueeze(4)
        # Broadcast is not supported for such high rank tensors.
        # We therefore do part of it manually
        tau = tau.expand(-1, -1, -1, -1, a.shape[4], -1)

    # Add a time samples dimension for broadcasting
    tau = tau.unsqueeze(6)

    # Time lags for which to compute the channel taps
    l = torch.arange(l_min, l_max + 1, dtype=real_dtype, device=tau.device)

    # Bring tau and l to broadcastable shapes
    tau = tau.unsqueeze(-1)
    l = expand_to_rank(l, tau.dim(), axis=0)

    # sinc pulse shaping (torch.sinc is normalized: sinc(x) = sin(pi*x)/(pi*x))
    g = torch.sinc(l - tau * bandwidth)
    g = torch.complex(g, torch.zeros_like(g))
    a = a.unsqueeze(-1)

    # For every tap, sum the sinc-weighted coefficients
    hm = (a * g).sum(dim=-3)

    if normalize:
        # Normalization is performed such that for each batch example and
        # link the energy per block is one.
        # The total energy of a channel response is the sum of the squared
        # norm over the channel taps.
        # Average over block size, RX antennas, and TX antennas
        c = hm.abs().square().sum(dim=6, keepdim=True).mean(dim=(2, 4, 5),
                                                            keepdim=True)
        c = torch.complex(c.sqrt(), torch.zeros_like(c))
        hm = torch.where(c.abs() > 0, hm / c, hm)

    return hm

def time_to_ofdm_channel(h_t, rg, l_min):
    # pylint: disable=line-too-long
    r"""
    Compute the channel frequency response from the discrete complex-baseband
    channel impulse response.

    Given a discrete complex-baseband channel impulse response
    :math:`\bar{h}_{b,\ell}`, for :math:`\ell` ranging from :math:`L_\text{min}\le 0`
    to :math:`L_\text{max}`, the discrete channel frequency response is computed as

    .. math::

        \hat{h}_{b,n} = \sum_{k=0}^{L_\text{max}} \bar{h}_{b,k} e^{-j \frac{2\pi kn}{N}} + \sum_{k=L_\text{min}}^{-1} \bar{h}_{b,k} e^{-j \frac{2\pi n(N+k)}{N}}, \quad n=0,\dots,N-1

    where :math:`N` is the FFT size and :math:`b` is the time step.

    This function only produces one channel frequency response per OFDM symbol, i.e.,
    only values of :math:`b` corresponding to the start of an OFDM symbol (after
    cyclic prefix removal) are considered.

    Input
    ------
    h_t : [...num_time_steps,l_max-l_min+1], torch.complex
        Tensor of discrete complex-baseband channel impulse responses

    rg : :class:`~sionna.phy.ofdm.ResourceGrid`
        Resource grid

    l_min : int
        Smallest time-lag for the discrete complex baseband
        channel impulse response (:math:`L_{\text{min}}`)

    Output
    ------
    h_f : [...,num_ofdm_symbols,fft_size], torch.complex
        Tensor of discrete complex-baseband channel frequency responses

    Note
    ----
    Note that the result of this function is generally different from the
    output of :meth:`~sionna.phy.channel.utils.cir_to_ofdm_channel` because
    the discrete complex-baseband channel impulse response is truncated
    (see :meth:`~sionna.phy.channel.utils.cir_to_time_channel`). This effect
    can be observed in the example below.

    Examples
    --------
    .. code-block:: Python

        # Setup resource grid and channel model
        torch.manual_seed(4)
        sm = StreamManagement(np.array([[1]]), 1)
        rg = ResourceGrid(num_ofdm_symbols=1,
                          fft_size=1024,
                          subcarrier_spacing=15e3)
        tdl = TDL("A", 100e-9, 3.5e9)

        # Generate CIR
        cir = tdl(batch_size=1, num_time_steps=1, sampling_frequency=rg.bandwidth)

        # Generate OFDM channel from CIR
        frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        h_freq = cir_to_ofdm_channel(frequencies, *cir, normalize=True).squeeze()

        # Generate time channel from CIR
        l_min, l_max = time_lag_discrete_time_channel(rg.bandwidth)
        h_time = cir_to_time_channel(rg.bandwidth, *cir, l_min=l_min, l_max=l_max, normalize=True)

        # Generate OFDM channel from time channel
        h_freq_hat = time_to_ofdm_channel(h_time, rg, l_min).squeeze()

        # Visualize results
        plt.figure()
        plt.plot(np.real(h_freq), "-")
        plt.plot(np.real(h_freq_hat), "--")
        plt.plot(np.imag(h_freq), "-")
        plt.plot(np.imag(h_freq_hat), "--")
        plt.xlabel("Subcarrier index")
        plt.ylabel(r"Channel frequency response")
        plt.legend(["OFDM Channel (real)", "OFDM Channel from time (real)", "OFDM Channel (imag)", "OFDM Channel from time (imag)"])
    """
    # Totla length of an OFDM symbol including cyclic prefix
    ofdm_length = rg.fft_size + rg.cyclic_prefix_length

    # Downsample the impulse respons to one sample per OFDM symbol
    h_t = h_t[..., rg.cyclic_prefix_length:rg.num_time_samples:ofdm_length, :]

    # Pad channel impulse response with zeros to the FFT size
    pad_dims = rg.fft_size - h_t.shape[-1]
    if pad_dims > 0:
        pad_shape = list(h_t.shape[:-1]) + [pad_dims]
        h_t = torch.cat([h_t, torch.zeros(pad_shape, dtype=h_t.dtype,
                                          device=h_t.device)], dim=-1)

    # Circular shift of negative time lags so that the channel impulse reponse
    # starts with h_{b,0}
    h_t = torch.roll(h_t, shifts=l_min, dims=-1)

    # Compute FFT
    h_f = torch.fft.fft(h_t)

    # Move the zero subcarrier to the center of the spectrum
    h_f = torch.fft.fftshift(h_f, dim=-1)

    return h_f


def deg_2_rad(x):
    r"""
    Convert degree to radian

    Input
    ------
        x : Tensor
            Angles in degree

    Output
    -------
        y : Tensor
            Angles ``x`` converted to radian
    """
    return x * (PI / 180.0)

def rad_2_deg(x):
    r"""
    Convert radian to degree

    Input
    ------
        x : Tensor
            Angles in radian

    Output
    -------
        y : Tensor
            Angles ``x`` converted to degree
    """
    return x * (180.0 / PI)

def wrap_angle_0_360(angle):
    r"""
    Wrap ``angle`` to (0,360)

    Input
    ------
        angle : Tensor
            Input to wrap

    Output
    -------
        y : Tensor
            ``angle`` wrapped to (0,360)
    """
    return angle % 360.0

def sample_bernoulli(shape, p, precision=None, device=None):
    r"""
    Sample a tensor with shape ``shape`` from a Bernoulli distribution with
    probability ``p``

    Input
    --------
    shape : Tensor shape
        Shape of the tensor to sample

    p : Broadcastable with ``shape``, torch.float
        Probability

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    --------
    : Tensor of shape ``shape``, bool
        Binary samples
    """
    if precision is None:
        rdtype = config.dtype
    else:
        rdtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device

    generator = config.torch_rng(device)
    z = uniform(shape, low=0.0, high=1.0, dtype=rdtype, device=device,
                generator=generator)
    z = z < p
    return z

def drop_uts_in_sector(batch_size, num_ut, bs_height, elevation_angle, isd,
                       precision=None, device=None):
    r"""
    Sample UT locations around one point at the elevation angle of the BS.

    The BS is assumed to be located above the origin (0,0) of the coordinate
    system, at the height ``bs_height``. The UTs are dropped uniformly within
    +/- ``isd``/2 in x and in y around one point, drawn once for all batch
    examples, at the horizontal distance ``bs_height``/tan(``elevation_angle``)
    from the origin, so that the BS appears at about the elevation angle from
    the UTs.

    Input
    --------
    batch_size : int
        Batch size

    num_ut : int
        Number of UTs to sample per batch example

    bs_height : torch.float
        Height of the BS (satellite or aerial platform) [m]

    elevation_angle : float
        Elevation angle of the BS as seen from the UTs [deg]

    isd : torch.float
        Inter-site distance [m]. The UTs are dropped within +/- ``isd``/2 in x
        and in y around the drop point.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    device : `None` (default) | str
        Device on which the UT locations are created, for example ``"cuda:0"``.
        If `None`, :attr:`~sionna.phy.config.Config.device` is used.

    Output
    ------
    ut_loc : [batch_size, num_ut, 2], torch.float
        UTs locations in the X-Y plan
    """

    if precision is None:
        rdtype = config.dtype
    else:
        rdtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device

    generator = config.torch_rng(device)

    # Gets the actual distance from the UTs to the BS based on the BS height and elevation angle
    # Samples a random value for an xy center location with the corret distance and then places UTs around
    # the center with some random variation based on the isd

    bs_height = torch.as_tensor(bs_height, dtype=rdtype, device=device)
    elevation_angle = torch.as_tensor(elevation_angle, dtype=rdtype,
                                      device=device)

    actual_bs_ut_distance = bs_height \
        / torch.sin(deg_2_rad(elevation_angle))
    distance_center_to_ut = torch.sqrt(torch.square(actual_bs_ut_distance)
                                       - torch.square(bs_height))
    x_base = uniform([1], low=0.0, high=distance_center_to_ut, dtype=rdtype,
                     device=device, generator=generator)
    y_base = torch.sqrt(torch.square(actual_bs_ut_distance)
                        - torch.square(bs_height) - torch.square(x_base))

    # side is not used. It is kept because it takes random numbers from the generator of
    # the draws after it (config.torch_rng), so that the drops under a seed stay as they
    # are: without it, every later draw would change
    side = sample_bernoulli([batch_size, num_ut],
                            torch.as_tensor(0.5, dtype=rdtype, device=device),
                            precision=precision, device=device)
    side = side.to(rdtype)
    side = 2. * side + 1.

    # Set UT location in X-Y coordinate system
    x_dis = uniform([batch_size, num_ut], low=-isd / 2.0, high=isd / 2.0,
                    dtype=rdtype, device=device, generator=generator)
    y_dis = uniform([batch_size, num_ut], low=-isd / 2.0, high=isd / 2.0,
                    dtype=rdtype, device=device, generator=generator)

    ut_loc = torch.stack([x_base + x_dis,
                          y_base + y_dis], dim=-1)

    return ut_loc

def set_3gpp_scenario_parameters(   scenario,
                                    isd=None,
                                    bs_height=None,
                                    elevation_angle=None,
                                    min_ut_height=None,
                                    max_ut_height=None,
                                    indoor_probability = None,
                                    min_ut_velocity=None,
                                    max_ut_velocity=None,
                                    precision=None,
                                    device=None):
    r"""
    Set valid parameters for a specified 3GPP system level ``scenario``
    (DenseUrban, Urban, or SubUrban).

    If a parameter is given, then it is returned. If it is set to `None`,
    then a parameter valid according to the chosen scenario is returned
    (see [TR38811]_).
    Input
    --------
    scenario : str
        System level model scenario. Must be one of "sur", "urb", or "dur".

    isd : None or torch.float
        Inter-site distance [m]

    bs_height : None or torch.float
        BS elevation [m]

    elevation_angle : None or torch.float
        elevation angle of UTs and BS. Assumed constant for all UTs

    min_ut_height : None or torch.float
        Minimum UT elevation [m]

    max_ut_height : None or torch.float
        Maximum UT elevation [m]

    indoor_probability : None or torch.float
        Probability of a UT to be indoor, returned as given (default 0). The
        topology generators ignore it and place all UTs outdoor: the models
        accept outdoor UTs only, for satellites and aerial platforms. TR 38.811
        considers only outdoor UTs for satellites and indoor UTs as well for
        aerial platforms (clause 6.1.2), but the models have neither the
        building entry loss of clause 6.6.3 nor O2I clusters.

    min_ut_velocity : None or torch.float
        Minimum UT velocity [m/s]

    max_ut_velocity : None or torch.float
        Maximim UT velocity [m/s]

    precision : str, `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`,
        :py:attr:`~sionna.phy.config.precision` is used.
    Output
    --------
    isd : torch.float
        Inter-site distance [m]

    bs_height : torch.float
        BS elevation [m]

    elevation_angle : None or torch.float
        elevation angle of UTs and BS. Assumed constant for all UTs

    min_ut_height : torch.float
        Minimum UT elevation [m]

    max_ut_height : torch.float
        Maximum UT elevation [m]

    indoor_probability : torch.float
        Probability of a UT to be indoor, as given (default 0); the topology
        generators ignore it

    min_ut_velocity : torch.float
        Minimum UT velocity [m/s]

    max_ut_velocity : torch.float
        Maximim UT velocity [m/s]
    """

    if not (scenario in ('urb', 'dur', 'sur')):
        raise ValueError("`scenario` must be one of 'urb', 'dur', 'sur'")

    if precision is None:
        rdtype = config.dtype
    else:
        rdtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device

    def _const(value):
        return torch.tensor(value, dtype=rdtype, device=device)

    # Default values for scenario parameters.
    # Partially taken from TR38.901, sections 7.2 and 7.4.
    # If not specified, a LEO satellite and elevation angle of 80 degrees are assumed.
    # All distances and heights are in meters
    # All velocities are in meters per second.
    # The elevation angle is in degrees
    default_scenario_par = {'dur' : {
                                'isd' : _const(200.),
                                'bs_height' : _const(600000.0),
                                'elevation_angle' : _const(80.),
                                'min_ut_height' : _const(1.5),
                                'max_ut_height' : _const(1.5),
                                'indoor_probability' : _const(0.0),#Zero: the models accept outdoor UTs only
                                'min_ut_velocity' : _const(0.0),
                                'max_ut_velocity' : _const(0.0)
                            },
                            'urb' : {
                                'isd' : _const(500.),
                                'elevation_angle' : _const(80.),
                                'bs_height' : _const(600000.0),
                                'min_ut_height' : _const(1.5),
                                'max_ut_height' : _const(1.5),
                                'indoor_probability' : _const(0.0),#Zero: the models accept outdoor UTs only
                                'min_ut_velocity' : _const(0.0),
                                'max_ut_velocity' : _const(0.0),
                            },
                            'sur' : {
                                'isd' : _const(5000.),
                                'elevation_angle' : _const(80.),
                                'bs_height' : _const(600000.0),
                                'min_ut_height' : _const(1.5),
                                'max_ut_height' : _const(1.5),
                                'indoor_probability' : _const(0.0),#Zero: the models accept outdoor UTs only
                                'min_ut_velocity' : _const(0.0),
                                'max_ut_velocity' : _const(0.0)
                            }
                        }

    # Setting the scenario parameters
    if isd is None:
        isd = default_scenario_par[scenario]['isd']
    if bs_height is None:
        bs_height = default_scenario_par[scenario]['bs_height']
    if elevation_angle is None:
        elevation_angle = default_scenario_par[scenario]['elevation_angle']
    if min_ut_height is None:
        min_ut_height = default_scenario_par[scenario]['min_ut_height']
    if max_ut_height is None:
        max_ut_height = default_scenario_par[scenario]['max_ut_height']
    if indoor_probability is None:
        indoor_probability =default_scenario_par[scenario]['indoor_probability']
    if min_ut_velocity is None:
        min_ut_velocity = default_scenario_par[scenario]['min_ut_velocity']
    if max_ut_velocity is None:
        max_ut_velocity = default_scenario_par[scenario]['max_ut_velocity']

    return isd, bs_height, elevation_angle, min_ut_height, max_ut_height,\
            indoor_probability, min_ut_velocity, max_ut_velocity

def relocate_uts(ut_loc, sector_id, cell_loc):
    # pylint: disable=line-too-long
    r"""
    Relocate the UTs by rotating them into the sector with index ``sector_id``
    and transposing them to the cell centered on ``cell_loc``.

    ``sector_id`` gives the index of the sector to which the UTs are
    rotated to. Sector ``k`` is sector 0 rotated by ``k`` x 120 degrees about
    the center of the cell, from the x-axis toward the y-axis.

    If ``sector_id`` is a scalar, then all UTs are relocated to the same
    sector indexed by ``sector_id``.
    If ``sector_id`` is a tensor, it should be broadcastable with
    [``batch_size``, ``num_ut``], and give the sector in which each UT or
    batch example is relocated to.

    When calling the function, ``ut_loc`` gives the locations of the UTs to
    relocate, which are all assumed to be in sector with index 0, and in the
    cell centered on the origin (0,0).

    Input
    --------
    ut_loc : [batch_size, num_ut, 2], torch.float
        UTs locations in the X-Y plan

    sector_id : Tensor broadcastable with [batch_size, num_ut], int
        Indexes of the sector to which to relocate the UTs

    cell_loc : Tensor broadcastable with [batch_size, num_ut], torch.float
        Center of the cell to which to transpose the UTs

    Output
    ------
    ut_loc : [batch_size, num_ut, 2], torch.float
        Relocated UTs locations in the X-Y plan
    """

    # Expand the rank of sector_id such that is is broadcastable with
    # (batch size, num_ut)
    sector_id = torch.as_tensor(sector_id, dtype=ut_loc.dtype,
                                device=ut_loc.device)
    sector_id = expand_to_rank(sector_id, 2, 0)

    # Expant
    cell_loc = torch.as_tensor(cell_loc, dtype=ut_loc.dtype,
                               device=ut_loc.device)
    cell_loc = expand_to_rank(cell_loc, ut_loc.dim(), 0)

    # Rotation matrix tensor, broadcastable with [batch size, num uts, 2, 2]
    rotation_angle = sector_id * 2. * PI / 3.0
    rotation_matrix = torch.stack([torch.cos(rotation_angle),
                                   -torch.sin(rotation_angle),
                                   torch.sin(rotation_angle),
                                   torch.cos(rotation_angle)],
                                  dim=-1)
    new_shape = list(rotation_angle.shape) + [2, 2]
    rotation_matrix = rotation_matrix.reshape(new_shape)
    rotation_matrix = rotation_matrix.to(ut_loc.dtype)

    # Applying the rotation matrix
    ut_loc = ut_loc.unsqueeze(-1)
    ut_loc_rotated = (rotation_matrix @ ut_loc).squeeze(-1)

    # Translate to the BS location
    ut_loc_rotated_translated = ut_loc_rotated + cell_loc

    return ut_loc_rotated_translated

# The models accept outdoor UTs only (set_topology rejects indoor UTs): they have neither
# the building entry loss of TR38.811 6.6.3 nor O2I clusters, so all UTs are outdoor here.
def generate_uts_topology(  batch_size,
                            num_ut,
                            drop_area,
                            cell_loc_xy,
                            bs_height,
                            elevation_angle,
                            isd,
                            min_ut_height,
                            max_ut_height,
                            indoor_probability,
                            min_ut_velocity,
                            max_ut_velocity,
                            precision=None,
                            device=None):
    # pylint: disable=line-too-long
    r"""
    Sample UTs location from a sector or a cell

    Each UT faces a BS at (0, 0, ``bs_height``): its bearing is atan2(y, x), the
    azimuth of its location (x, y), and its downtilt the elevation angle + 180 deg,
    which together turn the boresight of the UT (the x-axis of its local coordinate
    system, TR 38.901, eq. (7.1-4)) horizontally toward the origin and upward at the
    elevation angle. Its slant is the elevation angle + 90 deg.

    Input
    --------
    batch_size : int
        Batch size

    num_ut : int
        Number of UTs to sample per batch example

    drop_area : str
        'sector' or 'cell'. If set to 'sector', UTs are sampled from the
        sector with index 0; if set to 'cell', each UT is rotated into a sector
        of the cell drawn uniformly from the three (see
        :meth:`~openntn.utils.relocate_uts`, which indexes the sectors)

    cell_loc_xy : Tensor broadcastable with[batch_size, num_ut, 3], torch.float
        Center of the cell(s)

    bs_height : None or torch.float
        Height of the BS [m]

    elevation_angle : None or torch.float
        elevation angle of UTs and BS. Assumed constant for all UTs

    isd : None or torch.float
        Inter-site distance [m]

    min_ut_height : None or torch.float
        Minimum UT elevation [m]

    max_ut_height : None or torch.float
        Maximum UT elevation [m]

    indoor_probability : None or torch.float
        Ignored: the models accept outdoor UTs only, for satellites and aerial
        platforms, so the probability of a UT to be indoor is 0. TR 38.811
        considers only outdoor UTs for satellites and indoor UTs as well for
        aerial platforms (clause 6.1.2), but the models have neither the
        building entry loss of clause 6.6.3 nor O2I clusters.

    min_ut_velocity : None or torch.float
        Minimum UT velocity [m/s]

    max_ut_velocity : None or torch.float
        Maximum UT velocity [m/s]

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    ut_loc : [batch_size, num_ut, 3], torch.float
        UTs locations

    ut_orientations : [batch_size, num_ut, 3], torch.float
        UTs orientations [radian]: bearing, downtilt and slant, each UT facing the
        BS

    ut_velocities : [batch_size, num_ut, 3], torch.float
        UTs velocities [m/s]

    in_state : [batch_size, num_ut], torch.float
        Indoor/outdoor state of UTs. `True` means indoor, `False` means
        outdoor. Always `False`: the models accept outdoor UTs only.
    """

    if not (drop_area in ('sector', 'cell')):
        raise ValueError("Drop area must be either 'sector' or 'cell'")

    if precision is None:
        rdtype = config.dtype
    else:
        rdtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device

    generator = config.torch_rng(device)

    # Randomly generating the UT locations
    ut_loc_xy = drop_uts_in_sector(batch_size,
                                   num_ut,
                                   bs_height,
                                   elevation_angle,
                                   isd,
                                   precision=precision,
                                   device=device)
    if drop_area == 'sector':
        sectors = torch.tensor(0, dtype=torch.int32, device=device)
    elif drop_area == 'cell':
        sectors = randint(0, 3, [batch_size, num_ut], dtype=torch.int32,
                          device=device, generator=generator)
    ut_loc_xy = relocate_uts(ut_loc_xy,
                             sectors,
                             cell_loc_xy)

    ut_loc_z = uniform([batch_size, num_ut, 1],
                       low=min_ut_height,
                       high=max_ut_height,
                       dtype=rdtype,
                       device=device,
                       generator=generator)
    ut_loc = torch.cat([    ut_loc_xy,
                            ut_loc_z], dim=-1)

    # Randomly generating the UT indoor/outdoor state
    # The models accept outdoor UTs only (set_topology rejects indoor UTs), so the
    # probability is forced to be 0, whatever the caller passes. The draw stays for
    # an implementation of indoor UTs.
    indoor_probability = 0.0
    in_state = sample_bernoulli([batch_size, num_ut], indoor_probability,
                                precision=precision, device=device)

    # Randomly generate the UT velocities
    ut_vel_angle = uniform([batch_size, num_ut],
                           low=-PI,
                           high=PI,
                           dtype=rdtype,
                           device=device,
                           generator=generator)
    ut_vel_norm = uniform([batch_size, num_ut],
                          low=min_ut_velocity,
                          high=max_ut_velocity,
                          dtype=rdtype,
                          device=device,
                          generator=generator)
    ut_velocities = torch.stack([   ut_vel_norm * torch.cos(ut_vel_angle),
                                    ut_vel_norm * torch.sin(ut_vel_angle),
                                    torch.zeros([batch_size, num_ut],
                                                dtype=rdtype, device=device)],
                                dim=-1)

    elevation_angle = torch.as_tensor(elevation_angle, dtype=rdtype,
                                      device=device)
    ut_downtilt = torch.zeros([batch_size, num_ut], dtype=rdtype,
                              device=device) \
        + deg_2_rad(elevation_angle + 180.0)
    ut_slant = torch.zeros([batch_size, num_ut], dtype=rdtype, device=device) \
        + deg_2_rad(elevation_angle + 90.0)
    # Each UT faces the BS at (0, 0, bs_height): the bearing atan2(y, x), the azimuth of
    # the UT's location, with the downtilt of the elevation angle + 180 deg turns the
    # boresight of TR 38.901 eq. (7.1-4) horizontally toward the origin and upward at
    # the elevation angle. atan(y/x) would lose the half plane: it differs from that
    # bearing by 180 deg for the UTs at x < 0. The slant is the elevation angle + 90 deg.
    ut_bearing = torch.atan2(ut_loc_xy[:, :, 1], ut_loc_xy[:, :, 0])
    ut_orientations = torch.stack([ut_bearing, ut_downtilt, ut_slant], dim=-1)

    return ut_loc, ut_orientations, ut_velocities, in_state

def gen_single_sector_topology( batch_size,
                                num_ut,
                                scenario,
                                bs_height=None,
                                elevation_angle=None,
                                isd=None,
                                min_ut_height=None,
                                max_ut_height=None,
                                indoor_probability = None,
                                min_ut_velocity=None,
                                max_ut_velocity=None,
                                precision=None,
                                device=None):
    # pylint: disable=line-too-long
    r"""
    Generate a batch of topologies consisting of a single BS above the origin and
    ``num_ut`` UTs dropped around one point at the elevation angle of the BS.

    The UTs are dropped as :meth:`~openntn.utils.drop_uts_in_sector` describes.

    The BS is placed at (0, 0, ``bs_height``) and faces the nadir: its orientation
    is (0, 90 deg, 0), the bearing 0, the downtilt 90 deg and the slant 0. Each UT
    faces the BS: its bearing is atan2(y, x), the azimuth of its location (x, y),
    and its downtilt the elevation angle + 180 deg, which together turn the
    boresight of the UT (the x-axis of its local coordinate system, TR 38.901,
    eq. (7.1-4)) horizontally toward the origin and upward at the elevation angle.
    Its slant is the elevation angle + 90 deg.

    The drop configuration can be controlled through the optional parameters.
    Parameters set to `None` are set to valid values according to the chosen
    ``scenario`` (see [TR38901]_).

    The returned batch of topologies can be used as-is with the
    :meth:`set_topology` method of the system level models, i.e.
    :class:`~openntn.DenseUrban`, :class:`~openntn.Urban`,
    and :class:`~openntn.SubUrban`.

    Example
    --------
    >>> # Create antenna arrays
    >>> bs_array = PanelArray(num_rows_per_panel = 4,
    ...                      num_cols_per_panel = 4,
    ...                      polarization = 'dual',
    ...                      polarization_type = 'VH',
    ...                      antenna_pattern = '38.901',
    ...                      carrier_frequency = 3.5e9)
    >>>
    >>> ut_array = PanelArray(num_rows_per_panel = 1,
    ...                       num_cols_per_panel = 1,
    ...                       polarization = 'single',
    ...                       polarization_type = 'V',
    ...                       antenna_pattern = 'omni',
    ...                       carrier_frequency = 3.5e9)
    >>> # Create channel model
    >>> channel_model = Urban(carrier_frequency = 3.5e9,
    ...                     ut_array = ut_array,
    ...                     bs_array = bs_array,
    ...                     direction = 'uplink',
    ...                     elevation_angle = 80.0)
    >>> # Generate the topology
    >>> topology = gen_single_sector_topology(batch_size = 100,
    ...                                       num_ut = 4,
    ...                                       scenario = 'urb',
    ...                                       elevation_angle = 80.0)
    >>> # Set the topology
    >>> ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = topology
    >>> channel_model.set_topology(ut_loc,
    ...                            bs_loc,
    ...                            ut_orientations,
    ...                            bs_orientations,
    ...                            ut_velocities,
    ...                            in_state)
    >>> channel_model.show_topology()

    Input
    --------
    batch_size : int
        Batch size

    num_ut : int
        Number of UTs to sample per batch example

    scenario : str
        System leven model scenario. Must be one of "dur", "urb", or "sur".

    isd : None or torch.float
        Inter-site distance [m]

    bs_height : None or torch.float
        Height of the BS [m]: a satellite from 160 km to 35 786 km or an aerial
        platform from 8 km to 50 km, as the ``set_topology`` method of the models
        accepts. TR 38.811, clause 3.1, places aerial platforms at 8 km to 50 km and
        LEO satellites from 500 km on; satellites from 160 km, very low orbits
        included, are accepted. Table 4.5-1 gives typical altitudes (LEO 600 km to
        1500 km, MEO 7000 km to 20 000 km, GEO 35 786 km), Table 5.1-1 the reference
        deployments (GEO, non-GEO down to 600 km, UAS 8 km to 50 km). Defaults to
        600 km.

    elevation_angle : None or torch.float
        elevation angle of UTs and BS. Assumed constant for all UTs

    min_ut_height : None or torch.float
        Minimum UT elevation [m]

    max_ut_height : None or torch.float
        Maximum UT elevation [m]

    indoor_probability : None or torch.float
        Ignored: the models accept outdoor UTs only, for satellites and aerial
        platforms, so the probability of a UT to be indoor is 0. TR 38.811
        considers only outdoor UTs for satellites and indoor UTs as well for
        aerial platforms (clause 6.1.2), but the models have neither the
        building entry loss of clause 6.6.3 nor O2I clusters.

    min_ut_velocity : None or torch.float
        Minimum UT velocity [m/s]

    max_ut_velocity : None or torch.float
        Maximim UT velocity [m/s]

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    ut_loc : [batch_size, num_ut, 3], torch.float
        UTs locations

    bs_loc : [batch_size, 1, 3], torch.float
        BS location. Set to (0, 0, ``bs_height``) for all batch examples.

    ut_orientations : [batch_size, num_ut, 3], torch.float
        UTs orientations [radian]: bearing, downtilt and slant, each UT facing the
        BS

    bs_orientations : [batch_size, 1, 3], torch.float
        BS orientations [radian]: (0, pi/2, 0), facing the nadir

    ut_velocities : [batch_size, num_ut, 3], torch.float
        UTs velocities [m/s]

    in_state : [batch_size, num_ut], torch.float
        Indoor/outdoor state of UTs. `True` means indoor, `False` means
        outdoor. Always `False`: the models accept outdoor UTs only.
    """

    params = set_3gpp_scenario_parameters(  scenario,
                                            isd,
                                            bs_height,
                                            elevation_angle,
                                            min_ut_height,
                                            max_ut_height,
                                            indoor_probability,
                                            min_ut_velocity,
                                            max_ut_velocity,
                                            precision=precision,
                                            device=device)
    isd, bs_height, elevation_angle, min_ut_height, max_ut_height,\
            indoor_probability, min_ut_velocity, max_ut_velocity = params

    # The height is validated by set_topology of the models: a satellite from 160 km to
    # 35 786 km or an aerial platform from 8 km to 50 km

    if precision is None:
        rdtype = config.dtype
    else:
        rdtype = dtypes[precision]["torch"]["dtype"]

    if device is None:
        device = config.device

    bs_height = torch.as_tensor(bs_height, dtype=rdtype, device=device)

    # Setting BS to (0,0,bs_height)
    bs_loc = torch.stack([  torch.zeros([batch_size, 1], dtype=rdtype,
                                        device=device),
                            torch.zeros([batch_size, 1], dtype=rdtype,
                                        device=device),
                            bs_height.expand(batch_size, 1)], dim=-1)

    # The BS faces the nadir: bearing 0, downtilt 90 deg, slant 0
    bs_yaw = torch.tensor(0.0, dtype=rdtype, device=device)
    bs_downtilt = torch.tensor(0.5 * PI, dtype=rdtype, device=device)
    #bs_downtilt = PI
    bs_orientation = torch.stack([  bs_yaw.expand(batch_size, 1),
                                    bs_downtilt.expand(batch_size, 1),
                                    torch.zeros([batch_size, 1], dtype=rdtype,
                                                device=device)], dim=-1)

    # Generating the UTs
    ut_topology = generate_uts_topology(    batch_size,
                                            num_ut,
                                            'sector',
                                            torch.zeros([2], dtype=rdtype,
                                                        device=device),
                                            bs_height,
                                            elevation_angle,
                                            isd,
                                            min_ut_height,
                                            max_ut_height,
                                            indoor_probability,
                                            min_ut_velocity,
                                            max_ut_velocity,
                                            precision,
                                            device)
    ut_loc, ut_orientations, ut_velocities, in_state = ut_topology

    return ut_loc, bs_loc, ut_orientations, bs_orientation, ut_velocities,\
            in_state

def _toeplitz(col, row):
    """
    Construct a Toeplitz matrix from the first column and first row.

    :param col: First column of the Toeplitz matrix, shape [..., n]
    :param row: First row of the Toeplitz matrix, shape [..., m]
        (row[..., 0] should equal col[..., 0])

    :output result: Toeplitz matrix, shape [..., n, m]
    """
    n = col.shape[-1]
    m = row.shape[-1]

    # Create the values array: [row[m-1], ..., row[1], col[0], col[1], ..., col[n-1]]
    # We reverse row[1:] and concatenate with col
    row_reversed = row[..., 1:].flip(dims=[-1])
    vals = torch.cat([row_reversed, col], dim=-1)

    # Create index matrix for Toeplitz structure
    # T[i,j] = vals[m - 1 - j + i]
    i_idx = torch.arange(n, device=col.device).unsqueeze(1)  # [n, 1]
    j_idx = torch.arange(m, device=col.device).unsqueeze(0)  # [1, m]
    indices = (m - 1) - j_idx + i_idx  # [n, m]

    # Expand indices for batch dimensions
    batch_shape = col.shape[:-1]
    for _ in range(len(batch_shape)):
        indices = indices.unsqueeze(0)
    indices = indices.expand(*batch_shape, n, m)

    # Gather values using the indices
    result = torch.gather(vals.unsqueeze(-2).expand(*batch_shape, n, -1), -1,
                          indices)

    return result


def exp_corr_mat(a, n, precision=None, device=None):
    r"""Generate exponential correlation matrices.

    This function computes for every element :math:`a` of a complex-valued
    tensor :math:`\mathbf{a}` the corresponding :math:`n\times n` exponential
    correlation matrix :math:`\mathbf{R}(a,n)`, defined as (Eq. 1, [MAL2018]_):

    .. math::
        \mathbf{R}(a,n)_{i,j} = \begin{cases}
                    1 & \text{if } i=j\\
                    a^{i-j}  & \text{if } i>j\\
                    (a^\star)^{j-i}  & \text{if } j<i, j=1,\dots,n\\
                  \end{cases}

    where :math:`|a|<1` and :math:`\mathbf{R}\in\mathbb{C}^{n\times n}`.

    Input
    -----
    a : [n_0, ..., n_k], torch.complex
        A tensor of arbitrary rank whose elements
        have an absolute value smaller than one.

    n : int
        Number of dimensions of the output correlation matrices.

    precision : `None` (default) | "single" | "double"
        Precision of the output, `torch.complex64` for "single" and
        `torch.complex128` for "double". If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    device : `None` (default) | str
        Device of the output (e.g., ``"cpu"``, ``"cuda:0"``). If `None`,
        :attr:`~sionna.phy.config.Config.device` is used.

    Output
    ------
    R : [n_0, ..., n_k, n, n], torch.complex
        A tensor of the complex dtype of ``precision``.
    """
    if precision is None:
        cdtype = config.cdtype
    else:
        cdtype = dtypes[precision]["torch"]["cdtype"]

    if device is None:
        device = config.device

    # Cast to desired output dtype and expand last dimension for broadcasting
    a = torch.as_tensor(a, dtype=cdtype, device=device)
    a = a.unsqueeze(-1)

    # Check that a is valid
    if torch.any(torch.abs(a) >= 1):
        raise ValueError(
            "The absolute value of the elements of `a` must be smaller than one")

    # Vector of exponents, adapt dtype and dimensions for broadcasting
    exp = torch.arange(0, n, device=device)
    exp = exp.to(cdtype)
    exp = expand_to_rank(exp, a.dim(), 0)

    # First column of R
    col = torch.pow(a, exp)

    # For a=0, one needs to remove the resulting ,nans due to 0**0=nan
    cond = torch.isnan(col.real)
    col = torch.where(cond, torch.ones_like(col), col)

    # First row of R (equal to complex-conjugate of the first column)
    row = torch.conj(col)

    # Build Toeplitz matrix
    r = _toeplitz(col, row)

    return r

def one_ring_corr_mat(phi_deg, num_ant, d_h=0.5, sigma_phi_deg=15,
                      precision=None, device=None):
    r"""Generate covariance matrices from the one-ring model.

    This function generates approximate covariance matrices for the
    so-called `one-ring` model (Eq. 2.24) [BHS2017]_. A uniform
    linear array (ULA) with uniform antenna spacing is assumed. The elements
    of the covariance matrices are computed as:

    .. math::
        \mathbf{R}_{\ell,m} =
              \exp\left( j2\pi d_\text{H} (\ell -m)\sin(\varphi) \right)
              \exp\left( -\frac{\sigma_\varphi^2}{2}
              \left( 2\pi d_\text{H}(\ell -m)\cos(\varphi) \right)^2 \right)

    for :math:`\ell,m = 1,\dots, M`, where :math:`M` is the number of antennas,
    :math:`\varphi` is the angle of arrival, :math:`d_\text{H}` is the antenna
    spacing in multiples of the wavelength,
    and :math:`\sigma^2_\varphi` is the angular standard deviation.

    Input
    -----
    phi_deg : [n_0, ..., n_k], torch.float
        A tensor of arbitrary rank containing azimuth angles (deg) of arrival.

    num_ant : int
        Number of antennas

    d_h : float
        Antenna spacing in multiples of the wavelength. Defaults to 0.5.

    sigma_phi_deg : float
        Angular standard deviation (deg). Defaults to 15 (deg). Values greater
        than 15 should not be used as the approximation becomes invalid.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    R : [n_0, ..., n_k, num_ant, nun_ant], `dtype`
        Tensor containing the covariance matrices of the desired dtype.
    """

    if precision is None:
        rdtype = config.dtype
        cdtype = config.cdtype
    else:
        rdtype = dtypes[precision]["torch"]["dtype"]
        cdtype = dtypes[precision]["torch"]["cdtype"]

    if device is None:
        device = config.device

    if sigma_phi_deg>15:
        warnings.warn("sigma_phi_deg should be smaller than 15.")

    # Convert all inputs to radians
    phi_deg = torch.as_tensor(phi_deg, dtype=rdtype, device=device)
    sigma_phi_deg = torch.as_tensor(sigma_phi_deg, dtype=rdtype, device=device)
    phi = deg_2_rad(phi_deg)
    sigma_phi = deg_2_rad(sigma_phi_deg)

    # Add dimensions for broadcasting
    phi = phi.unsqueeze(-1)
    sigma_phi = sigma_phi.unsqueeze(-1) if sigma_phi.dim() > 0 else sigma_phi

    # Compute first column
    c = 2 * PI * d_h
    d = c * torch.arange(0, num_ant, dtype=rdtype, device=device)
    d = expand_to_rank(d, phi.dim(), 0)

    a = torch.complex(torch.zeros_like(d * torch.sin(phi)), d * torch.sin(phi))
    exp_a = torch.exp(a) # First exponential term

    b = -0.5 * (sigma_phi * d * torch.cos(phi)) ** 2
    exp_b = torch.exp(b).to(cdtype) # Second exponetial term

    col = exp_a*exp_b # First column

    # First row is just the complex conjugate of first column
    row = torch.conj(col)

    # Build Toeplitz matrix
    r = _toeplitz(col, row)

    return r

# ITU-R P.676-10 (09/2013), Annex 2: the specific attenuations of dry air and water
# vapour and their equivalent heights in closed form, for frequencies from 1 GHz to
# 350 GHz


def _p676_10_dry_air(f: float, rp: float, rt: float) -> float:
    r"""Specific attenuation of dry air [dB/km] by ITU-R P.676-10, Annex 2, eqs. (22a)
    to (22u)

    :param f: Frequency [GHz], 1 to 350
    :param rp: Pressure ratio r_p, the total air pressure divided by 1013 hPa
    :param rt: Temperature ratio r_t = 288/(273 + t), the temperature t in deg C

    :output: `float`. gamma_o [dB/km]
    """
    def phi(a: float, b: float, c: float, d: float) -> float:
        return rp**a*rt**b*math.exp(c*(1.0 - rp) + d*(1.0 - rt))               # (22u)

    if f <= 54.0:
        xi1 = phi(0.0717, -1.8132, 0.0156, -1.6515)                            # (22g)
        xi2 = phi(0.5146, -4.6368, -0.1921, -5.7416)                           # (22h)
        xi3 = phi(0.3414, -6.5851, 0.2130, -8.5854)                            # (22i)
        return (7.2*rt**2.8/(f**2 + 0.34*rp**2*rt**1.6)
                + 0.62*xi3/((54.0 - f)**(1.16*xi1) + 0.83*xi2))*f**2*rp**2*1e-3  # (22a)
    if f <= 66.0:
        gamma_54 = 2.192*phi(1.8286, -1.9487, 0.4051, -2.8509)                 # (22n)
        gamma_58 = 12.59*phi(1.0045, 3.5610, 0.1588, 1.2834)                   # (22o)
        gamma_60 = 15.0*phi(0.9003, 4.1335, 0.0427, 1.6088)                    # (22p)
        gamma_62 = 14.28*phi(0.9886, 3.4176, 0.1827, 1.3429)                   # (22q)
        gamma_64 = 6.819*phi(1.4320, 0.6258, 0.3177, -0.5914)                  # (22r)
        gamma_66 = 1.908*phi(2.0717, -4.1404, 0.4910, -4.8718)                 # (22s)
        if f <= 60.0:
            return math.exp(math.log(gamma_54)/24.0*(f - 58.0)*(f - 60.0)
                            - math.log(gamma_58)/8.0*(f - 54.0)*(f - 60.0)
                            + math.log(gamma_60)/12.0*(f - 54.0)*(f - 58.0))    # (22b)
        if f <= 62.0:
            return gamma_60 + (gamma_62 - gamma_60)*(f - 60.0)/2.0              # (22c)
        return math.exp(math.log(gamma_62)/8.0*(f - 64.0)*(f - 66.0)
                        - math.log(gamma_64)/4.0*(f - 62.0)*(f - 66.0)
                        + math.log(gamma_66)/8.0*(f - 62.0)*(f - 64.0))         # (22d)
    if f <= 120.0:
        xi4 = phi(-0.0112, 0.0092, -0.1033, -0.0009)                           # (22j)
        xi5 = phi(0.2705, -2.7192, -0.3016, -4.1033)                           # (22k)
        xi6 = phi(0.2445, -5.9191, 0.0422, -8.0719)                            # (22l)
        xi7 = phi(-0.1833, 6.5589, -0.2402, 6.131)                             # (22m)
        return (3.02e-4*rt**3.5 + 0.283*rt**3.8/((f - 118.75)**2 + 2.91*rp**2*rt**1.6)
                + 0.502*xi6*(1.0 - 0.0163*xi7*(f - 66.0))
                / ((f - 66.0)**(1.4346*xi4) + 1.15*xi5))*f**2*rp**2*1e-3      # (22e)
    delta = -0.00306*phi(3.211, -14.94, 1.583, -16.37)                         # (22t)
    return (3.02e-4/(1.0 + 1.9e-5*f**1.5)
            + 0.283*rt**0.3/((f - 118.75)**2 + 2.91*rp**2*rt**1.6)
            )*f**2*rp**2*rt**3.5*1e-3 + delta                                  # (22f)


def _p676_10_water_vapour(f: float, rp: float, rt: float, rho: float) -> float:
    r"""Specific attenuation of water vapour [dB/km] by ITU-R P.676-10, Annex 2, eqs.
    (23a) to (23d)

    :param f: Frequency [GHz], 1 to 350
    :param rp: Pressure ratio r_p, the total air pressure divided by 1013 hPa
    :param rt: Temperature ratio r_t = 288/(273 + t), the temperature t in deg C
    :param rho: Water vapour density [g/m^3]

    :output: `float`. gamma_w [dB/km]
    """
    eta1 = 0.955*rp*rt**0.68 + 0.006*rho                                       # (23b)
    eta2 = 0.735*rp*rt**0.5 + 0.0353*rt**4*rho                                 # (23c)

    def g(f_i: float) -> float:
        return 1.0 + ((f - f_i)/(f + f_i))**2                                  # (23d)

    return (3.98*eta1*math.exp(2.23*(1.0 - rt))/((f - 22.235)**2 + 9.42*eta1**2)*g(22.0)
            + 11.96*eta1*math.exp(0.7*(1.0 - rt))/((f - 183.31)**2 + 11.14*eta1**2)
            + 0.081*eta1*math.exp(6.44*(1.0 - rt))/((f - 321.226)**2 + 6.29*eta1**2)
            + 3.66*eta1*math.exp(1.6*(1.0 - rt))/((f - 325.153)**2 + 9.22*eta1**2)
            + 25.37*eta1*math.exp(1.09*(1.0 - rt))/(f - 380.0)**2
            + 17.4*eta1*math.exp(1.46*(1.0 - rt))/(f - 448.0)**2
            + 844.6*eta1*math.exp(0.17*(1.0 - rt))/(f - 557.0)**2*g(557.0)
            + 290.0*eta1*math.exp(0.41*(1.0 - rt))/(f - 752.0)**2*g(752.0)
            + 8.3328e4*eta2*math.exp(0.99*(1.0 - rt))/(f - 1780.0)**2*g(1780.0)
            )*f**2*rt**2.5*rho*1e-4                                            # (23a)


def _p676_10_dry_air_height(f: float, rp: float) -> float:
    r"""Equivalent height of dry air [km] by ITU-R P.676-10, Annex 2, eqs. (25a) to
    (25e)

    :param f: Frequency [GHz], 1 to 350
    :param rp: Pressure ratio r_p, the total air pressure divided by 1013 hPa

    :output: `float`. h_o [km]
    """
    t1 = (4.64/(1.0 + 0.066*rp**-2.3)
          * math.exp(-((f - 59.7)/(2.87 + 12.4*math.exp(-7.9*rp)))**2))         # (25b)
    t2 = 0.14*math.exp(2.12*rp)/((f - 118.75)**2 + 0.031*math.exp(2.2*rp))     # (25c)
    t3 = (0.0114/(1.0 + 0.14*rp**-2.6)*f*(-0.0247 + 0.0001*f + 1.61e-6*f**2)
          / (1.0 - 0.0169*f + 4.1e-5*f**2 + 3.2e-7*f**3))                      # (25d)
    h_o = 6.1/(1.0 + 0.17*rp**-1.1)*(1.0 + t1 + t2 + t3)                       # (25a)
    if f < 70.0:
        h_o = min(h_o, 10.7*rp**0.3)                                           # (25e)
    return h_o


def _p676_10_water_vapour_height(f: float, rp: float) -> float:
    r"""Equivalent height of water vapour [km] by ITU-R P.676-10, Annex 2, eqs. (26a)
    and (26b)

    :param f: Frequency [GHz], 1 to 350
    :param rp: Pressure ratio r_p, the total air pressure divided by 1013 hPa

    :output: `float`. h_w [km]
    """
    sigma_w = 1.013/(1.0 + math.exp(-8.6*(rp - 0.57)))                         # (26b)
    return 1.66*(1.0 + 1.39*sigma_w/((f - 22.235)**2 + 2.56*sigma_w)
                 + 3.37*sigma_w/((f - 183.31)**2 + 4.69*sigma_w)
                 + 1.58*sigma_w/((f - 325.1)**2 + 2.89*sigma_w))               # (26a)


def _compute_pathloss_gas(self):

        r"""Calculates the pathloss of a scenario based on 3GPP TR38.811 6.6.4

        TR 38.811, clause 6.6.4, takes for system level simulations "The method of
        Annex 2 in ITU-R P.676", the approximate method for frequencies up to 350 GHz;
        Annex 1 gives the complete line-by-line method that Annex 2 approximates. The
        loss is computed by Annex 2 of ITU-R P.676-10 (09/2013) as printed, the last
        revision whose Annex 2 gives the specific attenuations and the equivalent
        heights in closed form: the specific attenuation of dry air of eqs. (22a) to
        (22u) and of water vapour of eqs. (23a) to (23d), the equivalent heights of dry
        air of eqs. (25a) to (25e) and of water vapour of eqs. (26a) and (26b), and the
        zenith attenuation of eq. (27), for the UT at the surface (0 km) and the
        elevation angle of the scenario (eq. (28); TR 38.811, eq. (6.6-8)),

        .. math::
            PL_g = \frac{\gamma_o h_o + \gamma_w h_w}{\sin\alpha}

        The atmosphere is the scenario's: the temperature T, the water vapour density
        rho and the atmospheric pressure, which is the dry air pressure p, as TR 38.811
        clause 6.6.4 calls 1013.25 hPa ("p the dry air pressure"). Annex 2 takes the
        pressure ratio r_p = p_tot/1013 with the total air pressure p_tot = p + e, where
        e = rho T / 216.7 is the water vapour partial pressure (Annex 1, eq. (4)), and
        the temperature ratio r_t = 288/(273 + t) with the temperature t = T - 273.15 in
        deg C.

        The carrier frequency must lie between 1 GHz and 350 GHz, the range of Annex 2
        (`ValueError` otherwise). Clause 6.6.4 excludes UE altitudes above 10 km and
        frequencies within 0.5 GHz of the centres of resonance lines from the method;
        no error is raised for them.

        """

        T = float(self._temperature)
        p = float(self._atmospheric_pressure)
        rho = float(self._water_vapor_density)
        f = float(self._carrier_frequency)/1e9
        if not 1.0 <= f <= 350.0:
            raise ValueError(f"carrier frequency {f} GHz outside 1 GHz to 350 GHz, the "
                             f"range of ITU-R P.676-10, Annex 2")

        # Pressure ratio of the total pressure p + e, with the water vapour partial
        # pressure of Annex 1, eq. (4); temperature ratio of the temperature in deg C
        e = rho*T/216.7
        rp = (p + e)/1013.0
        t = T - 273.15
        rt = 288.0/(273.0 + t)

        # Zenith attenuation, eq. (27), and the slant path, eq. (28) and TR 38.811
        # eq. (6.6-8)
        A_zenith = (_p676_10_dry_air(f, rp, rt)*_p676_10_dry_air_height(f, rp)
                    + _p676_10_water_vapour(f, rp, rt, rho)
                    * _p676_10_water_vapour_height(f, rp))
        Elev = self._elevation_angle
        GasAtt = A_zenith/math.sin(math.radians(Elev))

        pl_g = torch.zeros_like(self.los, dtype=self.dtype) + GasAtt

        self._update_attr("_pl_g", pl_g)


# TR 38.811 V15.4.0, Table 6.6.6.2.1-1: tropospheric attenuation [dB] with 99 %
# probability at 20 GHz in Toulouse, per reference elevation angle [deg]
_TROPOSPHERIC_SCINTILLATION = {10: 1.08, 20: 0.48, 30: 0.30, 40: 0.22, 50: 0.17, 60: 0.13,
                               70: 0.12, 80: 0.12, 90: 0.12}


def _compute_pathloss_scintillation(self):
        r"""Calculates the scintillation loss :math:`PL_S` of a scenario based on 3GPP
        TR38.811 6.6.6

        From 6 GHz on, the loss is the tropospheric scintillation of clause 6.6.6.2.1:
        "As baseline for system-level simulations above 6 GHz, the fading due to
        scintillation PL_S is equal to the tropospheric attenuation level at 99% of the
        time derived from Figure 6.6.6.2.1-1 and summarized in Table 6.6.6.2.1-1". The
        table is given for 20 GHz in Toulouse; it is used at every carrier frequency from
        6 GHz on, as TR 38.821 V16.2.0 does in its link budgets at 20 GHz and 30 GHz
        (Tables 6.1.3.2-1 and 6.1.3.3-1). The value is that of the reference elevation
        angle nearest to the elevation angle of the scenario, as for the other tables of
        TR 38.811 (clauses 6.6.1 and 6.6.2); an elevation angle midway between two
        reference angles takes the higher one. The loss does not depend on the
        atmospheric parameters of the scenario.

        Below 6 GHz, the loss is the ionospheric scintillation of clause 6.6.6.1.4: for
        system-level simulations below 6 GHz, PL_S is A_IS of eq. (6.6-13) for latitudes
        within 20 deg of the magnetic equator, and PL_S = 0 at higher latitudes (between
        20 and 60 deg, and above 60 deg, where the model does not apply). The latitude is
        that of the scenario, the latitude of the UTs relative to the magnetic equator,
        around which the clause places the region of the model; it applies to every
        link to a satellite. :math:`A_{IS} = P_{fluc}/\sqrt{2}`, with :math:`P_{fluc}` =
        1.1 dB at 4 GHz, the value of the example of the clause at 2 GHz, multiplied by
        :math:`(f/4)^{-1.5}`, f in GHz. Links to an aerial platform (8 km to 50 km) get
        PL_S = 0 below 6 GHz at every latitude: Table 6.10.1-1 gives the scintillation as
        negligible for the deployment of UAS including HAPS below 6 GHz, whose links stay
        below the ionosphere.

        """

        f = self._carrier_frequency/(10.0**9.0)
        if f >= 6.0:
            # Nearest reference elevation angle; an angle midway between two takes the
            # higher one
            reference_angle = _reference_elevation(self._elevation_angle)
            pl_s = torch.zeros_like(self.los, dtype=self.dtype) \
                + _TROPOSPHERIC_SCINTILLATION[reference_angle]
            self._update_attr("_pl_s", pl_s)

        #For carrier frequencies below 6GHz we only consider Ionospheric scintillation
        elif abs(self._latitude) <= 20.0:
            #Pathloss at 4GHz
            Pluc_4GHz = 1.1
            #Exponent to scale the frequency loss below 6GHz
            n = -1.5
            #Scale P_fluc at 4GHz to the carrier frequency, step 2 of 3GPP TR38.811 6.6.6.1.4
            AIs = Pluc_4GHz * (f/4)**(n)
            #Get final loss by dividing by sqrt(2) according to 3GPP TR38.811 6.6-13
            pl_s = torch.zeros_like(self.los, dtype=self.dtype) + (AIs/math.sqrt(2.0))
            # TR 38.811 V15.4.0, Table 6.10.1-1: the scintillation of the deployment of
            # UAS including HAPS below 6 GHz is negligible, their links stay below the
            # ionosphere; one platform per call, a satellite keeps A_IS
            satellite = _is_satellite(self.h_bs).unsqueeze(2)
            pl_s = torch.where(satellite, pl_s, torch.zeros_like(pl_s))
            self._update_attr("_pl_s", pl_s)
        else:
            # Outside +-20 deg of the magnetic equator PL_S = 0 (TR 38.811 clause
            # 6.6.6.1.4)
            self._update_attr("_pl_s", torch.zeros_like(self.los, dtype=self.dtype))



def _compute_pathloss_entry(self):
        # The building entry loss of TR38.811 6.6.3 is 0 for every link: the models
        # accept outdoor UTs only (set_topology rejects indoor UTs)
        pl_e = torch.zeros_like(self.los, dtype=self.dtype)
        self._update_attr("_pl_e", pl_e)

def _compute_pathloss_basic(self):
        r"""Computes the basic path loss PL_b in dB of TR38.811 6.6.2 without its shadow
        fading: PL_b = FSPL + CL, with the free space path loss of eq. (6.6-2) and, for
        NLOS links, the clutter loss CL of Tables 6.6.2-1 to 6.6.2-3 (0 dB for LOS
        links). Eq. (6.6-4) writes the shadow fading SF into PL_b; the models apply it
        once, as the large scale parameter SF of step 4 of TR38.901 7.5, in step 12
        (Tables 6.7.2 of TR38.811 give its cross-correlations and correlation
        distance)."""
        distance_3d = self._distance_3d
        fc = self._carrier_frequency/(10**9)
        angle_str = str(_reference_elevation(self._elevation_angle))

        fspl = 32.45 + 20*math.log10(fc) + 20*torch.log10(distance_3d)
        self._update_attr("_fspl", fspl)
        cl = self._params_nlos["CL" + '_' + angle_str]

        pl_b = torch.where(self.los, fspl, fspl + cl)

        self._update_attr("_pl_b", pl_b)

# Heights of the platforms that the channel models accept [m]. TR 38.811 V15.4.0,
# clause 3.1, places aerial platforms "typically at an altitude between 8 to 50 km" and
# LEO satellites "typically at an altitude between 500 km to 2000 km"; the models accept
# satellites from 160 km, very low orbits included, to GEO at 35 786 km. Table 4.5-1
# gives typical altitudes (LEO 600 km to 1500 km, MEO 7000 km to 20 000 km, GEO
# 35 786 km), Table 5.1-1 the reference deployments (GEO, non-GEO down to 600 km, UAS
# 8 km to 50 km).
_SATELLITE_HEIGHTS = (160e3, 35786e3)
_AERIAL_HEIGHTS = (8e3, 50e3)
# Lowest height of the Faraday rotation [m]. TR 38.811 V15.4.0, clause 6.8.2, gives the
# rotation "In case of propagation for spaceborne BS above the ionosphere". 600 km is
# the lowest satellite altitude of TR 38.811 (Table 4.5-1, LEO; Table 5.1-1, non-GEO),
# a convention of the specification, not a boundary of the ionosphere: a satellite
# below it flies within the F region of the ionosphere, below much of its electron
# content, and TR 38.811 gives no Faraday rotation for it.
_FARADAY_ROTATION_HEIGHT = 600e3


def _is_satellite(height: torch.Tensor) -> torch.Tensor:
    r"""Satellite or not, by the height of the platform

    A platform from 160 km to 35 786 km is a satellite. The accepted heights, the zero
    departure spreads at the satellite (NOTE 8 of TR 38.811 Tables 6.7.2-1a to 6.7.2-8b,
    with the cluster ASD) and the satellite Doppler shift use this range; the Faraday
    rotation applies from 600 km only (``_has_faraday_rotation``).

    :param height: Heights of the platforms [m]

    :output: `torch.bool` tensor of the shape of ``height``, `True` for a satellite
    """
    return (height >= _SATELLITE_HEIGHTS[0]) & (height <= _SATELLITE_HEIGHTS[1])


def _has_faraday_rotation(height: torch.Tensor) -> torch.Tensor:
    r"""Faraday rotation of TR 38.811 clause 6.8.2 or not, by the height of the platform

    A satellite from 600 km on gets the rotation; a satellite below 600 km, which flies
    within the ionosphere, and an aerial platform do not (``_FARADAY_ROTATION_HEIGHT``).

    :param height: Heights of the platforms [m]

    :output: `torch.bool` tensor of the shape of ``height``, `True` with the rotation
    """
    return _is_satellite(height) & (height >= _FARADAY_ROTATION_HEIGHT)


def _validate_platform(bs_loc: torch.Tensor) -> None:
    r"""Checks that ``bs_loc`` describes one satellite or aerial platform at one height

    The channel models simulate one platform per call, with one nominal elevation angle
    and one height for every batch example.

    :param bs_loc: Locations of the BSs [m], shape [batch size, num_bs, 3]

    :output: `None`. Raises `ValueError` for more than one BS, for heights that differ
        across the batch and for heights outside 160 km to 35 786 km (satellite) and
        8 km to 50 km (aerial platform).
    """
    if bs_loc.shape[1] != 1:
        raise ValueError(f"one satellite or aerial platform per call: bs_loc has "
                         f"{bs_loc.shape[1]} BSs")
    height = bs_loc[:, 0, 2]
    lowest, highest = (float(v) for v in torch.aminmax(height))
    if lowest != highest:
        raise ValueError(f"one height per call: the BS heights range from {lowest} m to "
                         f"{highest} m across the batch")
    satellite = _SATELLITE_HEIGHTS[0] <= lowest <= _SATELLITE_HEIGHTS[1]
    aerial = _AERIAL_HEIGHTS[0] <= lowest <= _AERIAL_HEIGHTS[1]
    if not (satellite or aerial):
        raise ValueError(f"BS height {lowest} m: a satellite is at 160 km to 35 786 km, "
                         f"an aerial platform at 8 km to 50 km")


def _reference_elevation(elevation_angle: float) -> int:
    r"""Reference elevation angle of the tables of TR 38.811 for an elevation angle

    TR 38.811 V15.4.0 gives its tables at the reference elevation angles 10 deg to 90 deg
    in steps of 10 deg, and a UT takes the values of the reference angle nearest to its
    elevation angle (clauses 6.6.1 and 6.6.2). The clauses give no rule for an angle
    midway between two reference angles; such an angle takes the higher one (round half
    up), so 25, 45, 65 and 85 deg take 30, 50, 70 and 90 deg, and 15, 35, 55 and 75 deg
    take 20, 40, 60 and 80 deg.

    :param elevation_angle: Elevation angle [deg]

    :output: `int`. The reference elevation angle [deg]
    """
    return 10*math.floor(elevation_angle/10.0 + 0.5)


#Function that computes the satellite speed as a function of its height.
def compute_satellite_speed(h_sat):
    G = 6.6743 * (10 ** (-11)) #Gravitational constant of Earth in m^3/(kg*s^2)
    M = 5.972 * (10 ** 24) #Mass of Earth in kg
    r_earth = 6371000.0#Earths radius in m

    #Equation (8.41) from Curtis, H. D. (2005). Orbital Mechanics for Engineering Students. Butterworth-Heinemann. ISBN: 0 7506 6169 0.
    v_sat = torch.sqrt(torch.as_tensor((G*M)/((r_earth)+(h_sat))))#speed of the satellite in given Orbit around Earth in m/s
    return v_sat
