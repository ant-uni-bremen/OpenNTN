#
# This file has been created by the Dept. of Communications Engineering of the University of Bremen.
# The code is based on implementations provided by the NVIDIA CORPORATION & AFFILIATES
#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Utility functions for the channel module"""

import tensorflow as tf
import warnings

from sionna.phy.constants import PI
from sionna.phy.utils import expand_to_rank
import math 
from sionna.phy.utils import log10
import numpy as np
from sionna.phy import PI, config, dtypes


def subcarrier_frequencies(num_subcarriers, subcarrier_spacing,
                           precision=None):
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
        frequencies : [``num_subcarrier``], tf.float
            Baseband frequencies of subcarriers
    """

    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    if tf.equal(tf.math.floormod(num_subcarriers, 2), 0):
        start=-num_subcarriers/2
        limit=num_subcarriers/2
    else:
        start=-(num_subcarriers-1)/2
        limit=(num_subcarriers-1)/2+1

    frequencies = tf.range( start=start,
                            limit=limit,
                            dtype=rdtype)
    frequencies = frequencies*subcarrier_spacing
    return frequencies

def time_frequency_vector(num_samples, sample_duration, precision=None):
    # pylint: disable=line-too-long
    r"""
    Compute the time and frequency vector for a given number of samples
    and duration per sample in normalized time unit.

    >>> t = tf.cast(tf.linspace(-n_min, n_max, num_samples), dtype) * sample_duration
    >>> f = tf.cast(tf.linspace(-n_min, n_max, num_samples), dtype) * 1/(sample_duration*num_samples)

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
        t : [``num_samples``], ``tf.float``
            Time vector

        f : [``num_samples``], ``tf.float``
            Frequency vector
    """

    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]
    num_samples = int(num_samples)

    num_samples = int(num_samples)

    if tf.math.mod(num_samples, 2) == 0:  # if even
        n_min = tf.cast(-(num_samples) / 2, dtype=tf.int32)
        n_max = tf.cast((num_samples) / 2 - 1, dtype=tf.int32)
    else:  # if odd
        n_min = tf.cast(-(num_samples-1) / 2, dtype=tf.int32)
        n_max = tf.cast((num_samples+1) / 2 - 1, dtype=tf.int32)

    # Time vector
    t = tf.cast(tf.linspace(n_min, n_max, num_samples), rdtype) \
        * tf.cast(sample_duration, rdtype)

    # Frequency vector
    df = tf.cast(1.0/sample_duration, rdtype)/tf.cast(num_samples, rdtype)
    f = tf.cast(tf.linspace(n_min, n_max, num_samples), rdtype) \
        * tf.cast(df, rdtype)

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
    l_min = tf.cast(-6, tf.int32)
    l_max = tf.math.ceil(maximum_delay_spread*bandwidth) + 6
    l_max = tf.cast(l_max, tf.int32)
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
    frequencies : [fft_size], tf.float
        Frequencies at which to compute the channel response

    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths] or [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths], tf.float
        Path delays

    normalize : bool
        If set to `True`, the channel is normalized over the resource grid
        to ensure unit average energy per resource element. Defaults to `False`.

    Output
    -------
    h_f : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, fft_size], tf.complex
        Channel frequency responses at ``frequencies``
    """

    real_dtype = tau.dtype

    if len(tau.shape) == 4:
        # Expand dims to broadcast with h. Add the following dimensions:
        #  - number of rx antennas (2)
        #  - number of tx antennas (4)
        tau = tf.expand_dims(tf.expand_dims(tau, axis=2), axis=4)
        # Broadcast is not supported yet by TF for such high rank tensors.
        # We therefore do part of it manually
        tau = tf.tile(tau, [1, 1, 1, 1, a.shape[4], 1])

    # Add a time samples dimension for broadcasting
    tau = tf.expand_dims(tau, axis=6)

    # Bring all tensors to broadcastable shapes
    tau = tf.expand_dims(tau, axis=-1)
    h = tf.expand_dims(a, axis=-1)
    frequencies = expand_to_rank(frequencies, tf.rank(tau), axis=0)

    ## Compute the Fourier transforms of all cluster taps
    # Exponential component
    e = tf.exp(tf.complex(tf.constant(0, real_dtype),
        -2*PI*frequencies*tau))
    h_f = h*e
    # Sum over all clusters to get the channel frequency responses
    h_f = tf.reduce_sum(h_f, axis=-3)

    if normalize:
        # Normalization is performed such that for each batch example and
        # link the energy per resource grid is one.
        # Average over TX antennas, RX antennas, OFDM symbols and
        # subcarriers.
        c = tf.reduce_mean( tf.square(tf.abs(h_f)), axis=(2,4,5,6),
                            keepdims=True)
        c = tf.complex(tf.sqrt(c), tf.constant(0., real_dtype))
        h_f = tf.math.divide_no_nan(h_f, c)

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

    a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex
        Path coefficients

    tau : [batch size, num_rx, num_tx, num_paths] or [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths], tf.float
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
    hm :  [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_time_steps, l_max - l_min + 1], tf.complex
        Channel taps coefficients
    """

    real_dtype = tau.dtype

    if len(tau.shape) == 4:
        # Expand dims to broadcast with h. Add the following dimensions:
        #  - number of rx antennas (2)
        #  - number of tx antennas (4)
        tau = tf.expand_dims(tf.expand_dims(tau, axis=2), axis=4)
        # Broadcast is not supported by TF for such high rank tensors.
        # We therefore do part of it manually
        tau = tf.tile(tau, [1, 1, 1, 1, a.shape[4], 1])

    # Add a time samples dimension for broadcasting
    tau = tf.expand_dims(tau, axis=6)

    # Time lags for which to compute the channel taps
    l = tf.range(l_min, l_max+1, dtype=real_dtype)

    # Bring tau and l to broadcastable shapes
    tau = tf.expand_dims(tau, axis=-1)
    l = expand_to_rank(l, tau.shape.rank, axis=0)

    # sinc pulse shaping
    g = tf.experimental.numpy.sinc(l-tau*bandwidth)
    g = tf.complex(g, tf.constant(0., real_dtype))
    a = tf.expand_dims(a, axis=-1)

    # For every tap, sum the sinc-weighted coefficients
    hm = tf.reduce_sum(a*g, axis=-3)

    if normalize:
        # Normalization is performed such that for each batch example and
        # link the energy per block is one.
        # The total energy of a channel response is the sum of the squared
        # norm over the channel taps.
        # Average over block size, RX antennas, and TX antennas
        c = tf.reduce_mean(tf.reduce_sum(tf.square(tf.abs(hm)),
                                         axis=6, keepdims=True),
                           axis=(2,4,5), keepdims=True)
        c = tf.complex(tf.sqrt(c), tf.constant(0., real_dtype))
        hm = tf.math.divide_no_nan(hm, c)

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
    h_t : [...num_time_steps,l_max-l_min+1], tf.complex
        Tensor of discrete complex-baseband channel impulse responses

    resource_grid : :class:`~sionna.ofdm.ResourceGrid`
        Resource grid

    l_min : int
        Smallest time-lag for the discrete complex baseband
        channel impulse response (:math:`L_{\text{min}}`)

    Output
    ------
    h_f : [...,num_ofdm_symbols,fft_size], tf.complex
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
        tf.random.set_seed(4)
        sm = StreamManagement(np.array([[1]]), 1)
        rg = ResourceGrid(num_ofdm_symbols=1,
                          fft_size=1024,
                          subcarrier_spacing=15e3)
        tdl = TDL("A", 100e-9, 3.5e9)

        # Generate CIR
        cir = tdl(batch_size=1, num_time_steps=1, sampling_frequency=rg.bandwidth)

        # Generate OFDM channel from CIR
        frequencies = subcarrier_frequencies(rg.fft_size, rg.subcarrier_spacing)
        h_freq = tf.squeeze(cir_to_ofdm_channel(frequencies, *cir, normalize=True))

        # Generate time channel from CIR
        l_min, l_max = time_lag_discrete_time_channel(rg.bandwidth)
        h_time = cir_to_time_channel(rg.bandwidth, *cir, l_min=l_min, l_max=l_max, normalize=True)

        # Generate OFDM channel from time channel
        h_freq_hat = tf.squeeze(time_to_ofdm_channel(h_time, rg, l_min))

        # Visualize results
        plt.figure()
        plt.plot(np.real(h_freq), "-")
        plt.plot(np.real(h_freq_hat), "--")
        plt.plot(np.imag(h_freq), "-")
        plt.plot(np.imag(h_freq_hat), "--")
        plt.xlabel("Subcarrier index")
        plt.ylabel(r"Channel frequency response")
        plt.legend(["OFDM Channel (real)", "OFDM Channel from time (real)", "OFDM Channel (imag)", "OFDM Channel from time (imag)"])

    .. image:: ../figures/time_to_ofdm_channel.png
    """
    # Totla length of an OFDM symbol including cyclic prefix
    ofdm_length = rg.fft_size + rg.cyclic_prefix_length

    # Downsample the impulse respons to one sample per OFDM symbol
    h_t = h_t[...,rg.cyclic_prefix_length:rg.num_time_samples:ofdm_length, :]

    # Pad channel impulse response with zeros to the FFT size
    pad_dims = rg.fft_size - tf.shape(h_t)[-1]
    pad_shape = tf.concat([tf.shape(h_t)[:-1], [pad_dims]], axis=-1)
    h_t = tf.concat([h_t, tf.zeros(pad_shape, dtype=h_t.dtype)], axis=-1)

    # Circular shift of negative time lags so that the channel impulse reponse
    # starts with h_{b,0}
    h_t = tf.roll(h_t, l_min, axis=-1)

    # Compute FFT
    h_f = tf.signal.fft(h_t)

    # Move the zero subcarrier to the center of the spectrum
    h_f = tf.signal.fftshift(h_f, axes=-1)

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
    return x*tf.constant(PI/180.0, x.dtype)

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
    return x*tf.constant(180.0/PI, x.dtype)

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
    return tf.math.mod(angle, 360.)

def sample_bernoulli(shape, p, precision=None):
    r"""
    Sample a tensor with shape ``shape`` from a Bernoulli distribution with
    probability ``p``

    Input
    --------
    shape : Tensor shape
        Shape of the tensor to sample

    p : Broadcastable with ``shape``, tf.float
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
        rdtype = config.tf_rdtype
    elif precision is tf.float32 or precision == "single":
        precision = "single"
        rdtype = dtypes[precision]["tf"]["rdtype"]
    elif precision is tf.float64 or precision == "double":
        precision = "double"
        rdtype = dtypes[precision]["tf"]["rdtype"]
    z = config.tf_rng.uniform(shape=shape, minval=0.0, maxval=1.0, dtype=rdtype)
    z = tf.math.less(z, p)
    return z

def drop_uts_in_sector(batch_size, num_ut, bs_height, elevation_angle, isd,
                       precision=None):
    r"""
    Uniformly sample UT locations from a sector.

    The sector from which UTs are sampled is shown in the following figure.
    The BS is assumed to be located at the origin (0,0) of the coordinate
    system.

    .. figure:: ../figures/drop_uts_in_sector.png
        :align: center
        :scale: 30%

    Input
    --------
    batch_size : int
        Batch size

    num_ut : int
        Number of UTs to sample per batch example

    bs_height : tf.float
        Minimum BS-UT distance [m]

    isd : tf.float
        Inter-site distance, i.e., the distance between two adjacent BSs [m]

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    ut_loc : [batch_size, num_ut, 2], tf.float
        UTs locations in the X-Y plan
    """

    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    # Gets the actual distance from the UTs to the BS based on the BS height and elevation angle
    # Samples a random value for an xy center location with the corret distance and then places UTs around
    # the center with some random variation based on the isd

    
    actual_bs_ut_distance = tf.cast(bs_height, rdtype) / tf.cast(tf.math.sin(deg_2_rad(tf.cast(elevation_angle,rdtype))),rdtype)
    distance_center_to_ut = tf.math.sqrt(tf.math.square(actual_bs_ut_distance) - tf.math.square(bs_height))
    x_base = tf.random.uniform(shape = [1], minval=0,maxval=distance_center_to_ut)
    y_base = tf.math.sqrt(tf.math.square(actual_bs_ut_distance) - tf.math.square(bs_height) - tf.math.square(x_base))

    # Randomly assign the UTs to one of the two half of the sector
    side = sample_bernoulli([batch_size, num_ut],
                            tf.cast(0.5, rdtype),
                            rdtype)
    side = tf.cast(side, rdtype)
    side = 2.*side+1.

    # Set UT location in X-Y coordinate system
    x_dis = tf.random.uniform(shape=[batch_size, num_ut],
                                   minval=-isd/2.0,
                                   maxval=isd/2.0,
                                   dtype=rdtype)
    y_dis = tf.random.uniform(shape=[batch_size, num_ut],
                                   minval=-isd/2.0,
                                   maxval=isd/2.0,
                                   dtype=rdtype)

    ut_loc = tf.stack([x_base+x_dis,
                       y_base+y_dis], axis=-1)

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
                                    precision=None):
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

    min_bs_ut_dist : None or tf.float
        Minimum BS-UT distance [m]

    isd : None or tf.float
        Inter-site distance [m]

    bs_height : None or tf.float
        BS elevation [m]

    elevation_angle : None or tf.float
        elevation angle of UTs and BS. Assumed constant for all UTs

    min_ut_height : None or tf.float
        Minimum UT elevation [m]

    max_ut_height : None or tf.float
        Maximum UT elevation [m]

    indoor_probability : None or tf.float
        Probability of a UT to be indoor. Currently 0 for NTN

    min_ut_velocity : None or tf.float
        Minimum UT velocity [m/s]

    max_ut_velocity : None or tf.float
        Maximim UT velocity [m/s]

    precision : str, `None` (default) | 'single' | 'double'
        Precision used for internal calculations and outputs.
        If set to `None`,
        :py:attr:`~sionna.phy.config.precision` is used.
    Output
    --------
    min_bs_ut_dist : tf.float
        Minimum BS-UT distance [m]

    isd : tf.float
        Inter-site distance [m]

    bs_height : tf.float
        BS elevation [m]
    
    elevation_angle : None or tf.float
        elevation angle of UTs and BS. Assumed constant for all UTs

    min_ut_height : tf.float
        Minimum UT elevation [m]

    max_ut_height : tf.float
        Maximum UT elevation [m]

    indoor_probability : tf.float
        Probability of a UT to be indoor

    min_ut_velocity : tf.float
        Minimum UT velocity [m/s]

    max_ut_velocity : tf.float
        Maximim UT velocity [m/s]
    """

    assert scenario in ('urb', 'dur', 'sur'),\
        "`scenario` must be one of 'urb', 'dur', 'sur'"

    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    # Default values for scenario parameters.
    # Partially taken from TR38.901, sections 7.2 and 7.4.
    # If not specified, a LEO satellite and elevation angle of 80 degrees are assumed.
    # All distances and heights are in meters
    # All velocities are in meters per second.
    # The elevation angle is in degrees
    default_scenario_par = {'dur' : {
                                'isd' : tf.constant(200., rdtype),
                                'bs_height' : tf.constant(600000.0, rdtype),
                                'elevation_angle' : tf.constant(80., rdtype),
                                'min_ut_height' : tf.constant(1.5, rdtype),
                                'max_ut_height' : tf.constant(1.5, rdtype),
                                'indoor_probability' : tf.constant(0.0,#Zero as HAPS are currently not considered
                                                                    rdtype),
                                'min_ut_velocity' : tf.constant(0.0,
                                                                    rdtype),
                                'max_ut_velocity' :tf.constant(0.0, rdtype)
                            },
                            'urb' : {
                                'isd' : tf.constant(500., rdtype),
                                'elevation_angle' : tf.constant(80., rdtype),
                                'bs_height' : tf.constant(600000.0, rdtype),
                                'min_ut_height' : tf.constant(1.5, rdtype),
                                'max_ut_height' : tf.constant(1.5, rdtype),
                                'indoor_probability' : tf.constant(0.0,#Zero as HAPS are currently not considered
                                                                    rdtype),
                                'min_ut_velocity' : tf.constant(0.0,
                                                                    rdtype),
                                'max_ut_velocity' : tf.constant(0.0,
                                                                    rdtype),
                            },
                            'sur' : {
                                'isd' : tf.constant(5000., rdtype),
                                'elevation_angle' : tf.constant(80., rdtype),
                                'bs_height' : tf.constant(600000.0, rdtype),
                                'min_ut_height' : tf.constant(1.5, rdtype),
                                'max_ut_height' : tf.constant(1.5, rdtype),
                                'indoor_probability' : tf.constant(0.0,#Zero as HAPS are currently not considered
                                                                    rdtype),
                                'min_ut_velocity' : tf.constant(0.0,
                                                                    rdtype),
                                'max_ut_velocity' : tf.constant(0.0,
                                                                    rdtype)
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
    rotated to. The picture below shows how the three sectors of a cell are
    indexed.

    .. figure:: ../figures/panel_array_sector_id.png
        :align: center
        :scale: 30%

        Indexing of sectors

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
    ut_loc : [batch_size, num_ut, 2], tf.float
        UTs locations in the X-Y plan

    sector_id : Tensor broadcastable with [batch_size, num_ut], int
        Indexes of the sector to which to relocate the UTs

    cell_loc : Tensor broadcastable with [batch_size, num_ut], tf.float
        Center of the cell to which to transpose the UTs

    Output
    ------
    ut_loc : [batch_size, num_ut, 2], tf.float
        Relocated UTs locations in the X-Y plan
    """

    # Expand the rank of sector_id such that is is broadcastable with
    # (batch size, num_ut)
    sector_id = tf.cast(sector_id, ut_loc.dtype)
    sector_id = expand_to_rank(sector_id, 2, 0)

    # Expant
    cell_loc = tf.cast(cell_loc, ut_loc.dtype)
    cell_loc = expand_to_rank(cell_loc, tf.rank(ut_loc), 0)

    # Rotation matrix tensor, broadcastable with [batch size, num uts, 2, 2]
    rotation_angle = sector_id*2.*PI/3.0
    rotation_matrix = tf.stack([tf.math.cos(rotation_angle),
                                -tf.math.sin(rotation_angle),
                                tf.math.sin(rotation_angle),
                                tf.math.cos(rotation_angle)],
                               axis=-1)
    rotation_matrix = tf.reshape(rotation_matrix,
                                 tf.concat([tf.shape(rotation_angle),
                                            [2,2]], axis=-1))
    rotation_matrix = tf.cast(rotation_matrix, ut_loc.dtype)

    # Applying the rotation matrix
    ut_loc = tf.expand_dims(ut_loc, axis=-1)
    ut_loc_rotated = tf.squeeze(rotation_matrix@ut_loc, axis=-1)

    # Translate to the BS location
    ut_loc_rotated_translated = ut_loc_rotated + cell_loc

    return ut_loc_rotated_translated

# TODO A future version will support the indoor states. However, as the standard has not yet
# included the indoor state, this cannot yet be done.
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
                            precision=None):
    # pylint: disable=line-too-long
    r"""
    Sample UTs location from a sector or a cell

    Input
    --------
    batch_size : int
        Batch size

    num_ut : int
        Number of UTs to sample per batch example

    drop_area : str
        'sector' or 'cell'. If set to 'sector', UTs are sampled from the
        sector with index 0 in the figure below

        .. figure:: ../figures/panel_array_sector_id.png
            :align: center
            :scale: 30%

    Indexing of sectors

    cell_loc_xy : Tensor broadcastable with[batch_size, num_ut, 3], tf.float
        Center of the cell(s)

    bs_height : None or tf.float
        Height of the BS [m]
    
    elevation_angle : None or tf.float
        elevation angle of UTs and BS. Assumed constant for all UTs

    isd : None or tf.float
        Inter-site distance [m]

    min_ut_height : None or tf.float
        Minimum UT elevation [m]

    max_ut_height : None or tf.float
        Maximum UT elevation [m]

    indoor_probability : None or tf.float
        Probability of a UT to be indoor

    min_ut_velocity : None or tf.float
        Minimum UT velocity [m/s]

    max_ut_velocity : None or tf.float
        Maximum UT velocity [m/s]

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    ut_loc : [batch_size, num_ut, 3], tf.float
        UTs locations

    ut_orientations : [batch_size, num_ut, 3], tf.float
        UTs orientations [radian]

    ut_velocities : [batch_size, num_ut, 3], tf.float
        UTs velocities [m/s]

    in_state : [batch_size, num_ut], tf.float
        Indoor/outdoor state of UTs. `True` means indoor, `False` means
        outdoor.
    """

    assert drop_area in ('sector', 'cell'),\
        "Drop area must be either 'sector' or 'cell'"

    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    # Randomly generating the UT locations
    ut_loc_xy = drop_uts_in_sector(batch_size,
                                   num_ut,
                                   bs_height,
                                   elevation_angle,
                                   isd,
                                   precision=precision)
    if drop_area == 'sector':
        sectors = tf.constant(0, tf.int32)
    elif drop_area == 'cell':
        sectors = tf.random.uniform(shape=[batch_size, num_ut],
                                    minval=0,
                                    maxval=3,
                                    dtype=tf.int32)
    ut_loc_xy = relocate_uts(ut_loc_xy,
                             sectors,
                             cell_loc_xy)

    ut_loc_z = tf.random.uniform(   shape=[batch_size, num_ut, 1],
                                    minval=min_ut_height,
                                    maxval=max_ut_height,
                                    dtype=rdtype)
    ut_loc = tf.concat([    ut_loc_xy,
                            ut_loc_z], axis=-1)

    # Randomly generating the UT indoor/outdoor state
    #Currently indoor scenarios are not supported and therefore the probability is forced to be 0. This line already exists to support future implementations.
    indoor_probability = 0.0
    in_state = sample_bernoulli(   [batch_size, num_ut], indoor_probability,
                                    rdtype)

    # Randomly generate the UT velocities
    ut_vel_angle = tf.random.uniform(   [batch_size, num_ut],
                                        minval=-PI,
                                        maxval=PI,
                                        dtype=rdtype)
    ut_vel_norm = tf.random.uniform(    [batch_size, num_ut],
                                        minval=min_ut_velocity,
                                        maxval=max_ut_velocity,
                                        dtype=rdtype)
    ut_velocities = tf.stack([  ut_vel_norm*tf.math.cos(ut_vel_angle),
                                ut_vel_norm*tf.math.sin(ut_vel_angle),
                                tf.zeros([batch_size, num_ut], rdtype)],
                                axis=-1)

    
    ut_downtilt = tf.zeros([batch_size, num_ut]) + tf.cast(deg_2_rad(tf.cast(elevation_angle + 180.0, rdtype)),rdtype)
    ut_slant = tf.zeros([batch_size, num_ut]) + tf.cast(deg_2_rad(tf.cast(elevation_angle + 90.0, rdtype)),rdtype)
    #Always point at the BS, which is at [0,0,bs_height]
    ut_bearing = tf.math.atan(ut_loc_xy[:,:,1]/ut_loc_xy[:,:,0])
    ut_orientations = tf.stack([ut_bearing, ut_downtilt, ut_slant], axis=-1)

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
                                precision=None):
    # pylint: disable=line-too-long
    r"""
    Generate a batch of topologies consisting of a single BS located at the
    origin and ``num_ut`` UTs randomly and uniformly dropped in a cell sector.

    The following picture shows the sector from which UTs are sampled.

    .. figure:: ../figures/drop_uts_in_sector.png
        :align: center
        :scale: 30%

    UTs orientations are randomly and uniformly set, whereas the BS orientation
    is set such that the it is oriented towards the center of the sector.

    The drop configuration can be controlled through the optional parameters.
    Parameters set to `None` are set to valid values according to the chosen
    ``scenario`` (see [TR38901]_).

    The returned batch of topologies can be used as-is with the
    :meth:`set_topology` method of the system level models, i.e.
    :class:`~sionna.phy.channel.tr38811.DenseUrban`, :class:`~sionna.phy.channel.tr38811.Urban`,
    and :class:`~sionna.phy.channel.tr38811.SubUrban`.

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
    ...                     o2i_model = 'low',
    ...                     ut_array = ut_array,
    ...                     bs_array = bs_array,
    ...                     direction = 'uplink')
    >>> # Generate the topology
    >>> topology = gen_single_sector_topology(batch_size = 100,
    ...                                       num_ut = 4,
    ...                                       scenario = 'urb')
    >>> # Set the topology
    >>> ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = topology
    >>> channel_model.set_topology(ut_loc,
    ...                            bs_loc,
    ...                            ut_orientations,
    ...                            bs_orientations,
    ...                            ut_velocities,
    ...                            in_state)
    >>> channel_model.show_topology()

    .. image:: ../figures/drop_uts_in_sector_topology.png

    Input
    --------
    batch_size : int
        Batch size

    num_ut : int
        Number of UTs to sample per batch example

    scenario : str
        System leven model scenario. Must be one of "dur", "urb", or "sur".

    min_bs_ut_dist : None or tf.float
        Minimum BS-UT distance [m]

    isd : None or tf.float
        Inter-site distance [m]

    bs_height : None or tf.float
        BS elevation [m]

    elevation_angle : None or tf.float
        elevation angle of UTs and BS. Assumed constant for all UTs

    min_ut_height : None or tf.float
        Minimum UT elevation [m]

    max_ut_height : None or tf.float
        Maximum UT elevation [m]

    indoor_probability : None or tf.float
        Probability of a UT to be indoor

    min_ut_velocity : None or tf.float
        Minimum UT velocity [m/s]

    max_ut_velocity : None or tf.float
        Maximim UT velocity [m/s]

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    ut_loc : [batch_size, num_ut, 3], tf.float
        UTs locations

    bs_loc : [batch_size, 1, 3], tf.float
        BS location. Set to (0,0,0) for all batch examples.

    ut_orientations : [batch_size, num_ut, 3], tf.float
        UTs orientations [radian]

    bs_orientations : [batch_size, 1, 3], tf.float
        BS orientations [radian]. Oriented towards the center of the sector.

    ut_velocities : [batch_size, num_ut, 3], tf.float
        UTs velocities [m/s]

    in_state : [batch_size, num_ut], tf.float
        Indoor/outdoor state of UTs. `True` means indoor, `False` means
        outdoor.
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
                                            precision=precision)
    isd, bs_height, elevation_angle, min_ut_height, max_ut_height,\
            indoor_probability, min_ut_velocity, max_ut_velocity = params

    # TODO here used ot be a check doing two things. First, height in int would be cast fo float, as they should both be acceptable and 
    # improve the user interfce. Second, detecting insufficient base station height wuold throw an error. However, both of these things would 
    # mess up eager executions. A future version will incorporate both functionalities again without loosing eager execution compatability
    #Cast height to float
    #disabled this here temporarily to not messup XLA
    #if isinstance(bs_height, int):
    #    bs_height = float(bs_height)

    # Currently only satellites are supported, which start at 600'000m according to 3GPP TR38.811 Table 4.5-1: Typical characteristics of Airborne or Space-borne vehicles
    
    #temporarily removed verification to check XLA compatibility
    #tf.debugging.assert_equal(tf.logical_and(tf.greater_equal(bs_height, 600000.0), tf.less_equal(bs_height, 36000000.0)), tf.constant(True), message="The base station height must be at least 600'000m and not more than 36'000'000m, as only satellites are currently supported")        
    #bs_height >= 600000.0 and bs_height <= 36000000.0, \
        

    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    # Setting BS to (0,0,bs_height)
    bs_loc = tf.stack([ tf.zeros([batch_size, 1], rdtype),
                        tf.zeros([batch_size, 1], rdtype),
                        tf.fill( [batch_size, 1], bs_height)], axis=-1)

    # BS always faces exactly down in simulations
    bs_yaw = tf.constant(0.0, rdtype)
    bs_downtilt = tf.constant(0.5*PI, rdtype)
    #bs_downtilt = PI
    bs_orientation = tf.stack([ tf.fill([batch_size, 1], bs_yaw),
                                tf.fill([batch_size, 1], bs_downtilt),
                                tf.zeros([batch_size, 1], rdtype)], axis=-1)

    # Generating the UTs
    ut_topology = generate_uts_topology(    batch_size,
                                            num_ut,
                                            'sector',
                                            tf.zeros([2], rdtype),
                                            bs_height,
                                            elevation_angle,
                                            isd,
                                            min_ut_height,
                                            max_ut_height,
                                            indoor_probability,
                                            min_ut_velocity,
                                            max_ut_velocity,
                                            precision)
    ut_loc, ut_orientations, ut_velocities, in_state = ut_topology

    return ut_loc, bs_loc, ut_orientations, bs_orientation, ut_velocities,\
            in_state

#TODO currently unused in NTN cases. A future version will include these as well.
def gen_single_sector_topology_interferers( batch_size,
                                            num_ut,
                                            num_interferer,
                                            scenario,
                                            min_bs_ut_dist=None,
                                            isd=None,
                                            bs_height=None,
                                            min_ut_height=None,
                                            max_ut_height=None,
                                            indoor_probability = None,
                                            min_ut_velocity=None,
                                            max_ut_velocity=None,
                                            precision=None):
    # pylint: disable=line-too-long
    r"""
    Generate a batch of topologies consisting of a single BS located at the
    origin, ``num_ut`` UTs randomly and uniformly dropped in a cell sector, and
    ``num_interferer`` interfering UTs randomly dropped in the adjacent cells.

    The following picture shows how UTs are sampled

    .. figure:: ../figures/drop_uts_in_sector_interferers.png
        :align: center
        :scale: 30%

    UTs orientations are randomly and uniformly set, whereas the BS orientation
    is set such that it is oriented towards the center of the sector it
    serves.

    The drop configuration can be controlled through the optional parameters.
    Parameters set to `None` are set to valid values according to the chosen
    ``scenario`` (see [TR38901]_).

    The returned batch of topologies can be used as-is with the
    :meth:`set_topology` method of the system level models, i.e.
    :class:`~sionna.phy.channel.tr38811.DenseUrban`, :class:`~sionna.phy.channel.tr38811.Urban`,
    and :class:`~sionna.phy.channel.tr38811.SubUrban`.

    In the returned ``ut_loc``, ``ut_orientations``, ``ut_velocities``, and
    ``in_state`` tensors, the first ``num_ut`` items along the axis with index
    1 correspond to the served UTs, whereas the remaining ``num_interferer``
    items correspond to the interfering UTs.

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
    ...                     o2i_model = 'low',
    ...                     ut_array = ut_array,
    ...                     bs_array = bs_array,
    ...                     direction = 'uplink')
    >>> # Generate the topology
    >>> topology = gen_single_sector_topology_interferers(batch_size = 100,
    ...                                                   num_ut = 4,
    ...                                                   num_interferer = 4,
    ...                                                   scenario = 'urb')
    >>> # Set the topology
    >>> ut_loc, bs_loc, ut_orientations, bs_orientations, ut_velocities, in_state = topology
    >>> channel_model.set_topology(ut_loc,
    ...                            bs_loc,
    ...                            ut_orientations,
    ...                            bs_orientations,
    ...                            ut_velocities,
    ...                            in_state)
    >>> channel_model.show_topology()

    .. image:: ../figures/drop_uts_in_sector_topology_inter.png

    Input
    --------
    batch_size : int
        Batch size

    num_ut : int
        Number of UTs to sample per batch example

    num_interferer : int
        Number of interfeering UTs per batch example

    scenario : str
        System leven model scenario. Must be one of "dur", "urb", or "sur".

    min_bs_ut_dist : None or tf.float
        Minimum BS-UT distance [m]

    isd : None or tf.float
        Inter-site distance [m]

    bs_height : None or tf.float
        BS elevation [m]

    min_ut_height : None or tf.float
        Minimum UT elevation [m]

    max_ut_height : None or tf.float
        Maximum UT elevation [m]

    indoor_probability : None or tf.float
        Probability of a UT to be indoor

    min_ut_velocity : None or tf.float
        Minimum UT velocity [m/s]

    max_ut_velocity : None or tf.float
        Maximim UT velocity [m/s]

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Output
    ------
    ut_loc : [batch_size, num_ut, 3], tf.float
        UTs locations. The first ``num_ut`` items along the axis with index
        1 correspond to the served UTs, whereas the remaining
        ``num_interferer`` items correspond to the interfeering UTs.

    bs_loc : [batch_size, 1, 3], tf.float
        BS location. Set to (0,0,0) for all batch examples.

    ut_orientations : [batch_size, num_ut, 3], tf.float
        UTs orientations [radian]. The first ``num_ut`` items along the
        axis with index 1 correspond to the served UTs, whereas the
        remaining ``num_interferer`` items correspond to the interfeering
        UTs.

    bs_orientations : [batch_size, 1, 3], tf.float
        BS orientation [radian]. Oriented towards the center of the sector.

    ut_velocities : [batch_size, num_ut, 3], tf.float
        UTs velocities [m/s]. The first ``num_ut`` items along the axis
        with index 1 correspond to the served UTs, whereas the remaining
        ``num_interferer`` items correspond to the interfeering UTs.

    in_state : [batch_size, num_ut], tf.float
        Indoor/outdoor state of UTs. `True` means indoor, `False` means
        outdoor. The first ``num_ut`` items along the axis with
        index 1 correspond to the served UTs, whereas the remaining
        ``num_interferer`` items correspond to the interfeering UTs.
    """

    params = set_3gpp_scenario_parameters(  scenario,
                                            min_bs_ut_dist,
                                            isd,
                                            bs_height,
                                            min_ut_height,
                                            max_ut_height,
                                            indoor_probability,
                                            min_ut_velocity,
                                            max_ut_velocity,
                                            precision)
    min_bs_ut_dist, isd, bs_height, min_ut_height, max_ut_height,\
            indoor_probability, min_ut_velocity, max_ut_velocity = params

    if precision is None:
        rdtype = config.tf_rdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]

    # Setting BS to (0,0,bs_height)
    bs_loc = tf.stack([ tf.zeros([batch_size, 1], rdtype),
                        tf.zeros([batch_size, 1], rdtype),
                        tf.fill( [batch_size, 1], bs_height)], axis=-1)

    # Setting the BS orientation such that it is downtilted towards the center
    # of the sector
    sector_center = (min_bs_ut_dist + 0.5*isd)*0.5
    bs_downtilt = 0.5*PI - tf.math.atan(sector_center/bs_height)
    bs_yaw = tf.constant(PI/3.0, rdtype)
    bs_orientation = tf.stack([ tf.fill([batch_size, 1], bs_yaw),
                                tf.fill([batch_size, 1], bs_downtilt),
                                tf.zeros([batch_size, 1], rdtype)], axis=-1)

    # Generating the UTs located in the UTs served by the BS
    ut_topology = generate_uts_topology(    batch_size,
                                            num_ut,
                                            'sector',
                                            tf.zeros([2], rdtype),
                                            min_bs_ut_dist,
                                            isd,
                                            min_ut_height,
                                            max_ut_height,
                                            indoor_probability,
                                            min_ut_velocity,
                                            max_ut_velocity,
                                            dtype)
    ut_loc, ut_orientations, ut_velocities, in_state = ut_topology


    ## Generating the UTs located in the adjacent cells

    # Users are randomly dropped in one of the two adjacent cells
    inter_cell_center = tf.stack([[0.0, isd],
                                  [isd*tf.math.cos(PI/6.0),
                                   isd*tf.math.sin(PI/6.0)]],
                                 axis=0)
    cell_index = tf.random.uniform(shape=[batch_size, num_interferer],
                                  minval=0, maxval=2, dtype=tf.int32)
    inter_cells = tf.gather(inter_cell_center, cell_index)

    inter_topology = generate_uts_topology(     batch_size,
                                                num_interferer,
                                                'cell',
                                                inter_cells,
                                                min_bs_ut_dist,
                                                isd,
                                                min_ut_height,
                                                max_ut_height,
                                                indoor_probability,
                                                min_ut_velocity,
                                                max_ut_velocity,
                                                dtype)
    inter_loc, inter_orientations, inter_velocities, inter_in_state \
        = inter_topology

    ut_loc = tf.concat([ut_loc, inter_loc], axis=1)
    ut_orientations = tf.concat([ut_orientations, inter_orientations], axis=1)
    ut_velocities = tf.concat([ut_velocities, inter_velocities], axis=1)
    in_state = tf.concat([in_state, inter_in_state], axis=1)

    return ut_loc, bs_loc, ut_orientations, bs_orientation, ut_velocities,\
            in_state



def exp_corr_mat(a, n, precision=None):
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
    a : [n_0, ..., n_k], tf.complex
        A tensor of arbitrary rank whose elements
        have an absolute value smaller than one.

    n : int
        Number of dimensions of the output correlation matrices.

    dtype : tf.complex64, tf.complex128
        The dtype of the output.

    Output
    ------
    R : [n_0, ..., n_k, n, n], tf.complex
        A tensor of the same dtype as the input tensor :math:`\mathbf{a}`.
    """
    if precision is None:
        cdtype = config.tf_cdtype
    else:
        cdtype = dtypes[precision]["tf"]["cdtype"]
    
    # Cast to desired output dtype and expand last dimension for broadcasting
    a = tf.cast(a, dtype=cdtype)
    a = tf.expand_dims(a, -1)

    # Check that a is valid
    msg = "The absolute value of the elements of `a` must be smaller than one"
    tf.debugging.assert_less(tf.abs(a), tf.cast(1, a.dtype.real_dtype), msg)

    # Vector of exponents, adapt dtype and dimensions for broadcasting
    exp = tf.range(0, n)
    exp = tf.cast(exp, dtype=cdtype)
    exp = expand_to_rank(exp, tf.rank(a), 0)

    # First column of R
    col = tf.math.pow(a, exp)

    # For a=0, one needs to remove the resulting ,nans due to 0**0=nan
    cond = tf.math.is_nan(tf.math.real(col))
    col = tf.where(cond, tf.ones_like(col), col)

    # First row of R (equal to complex-conjugate of the first column)
    row = tf.math.conj(col)

    # Create Toeplitz operator
    operator = tf.linalg.LinearOperatorToeplitz(col, row)

    # Generate dense tensor from operator
    r = operator.to_dense()

    return r

def one_ring_corr_mat(phi_deg, num_ant, d_h=0.5, sigma_phi_deg=15,
                      precision=None):
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
    phi_deg : [n_0, ..., n_k], tf.float
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
        rdtype = config.tf_rdtype
        cdtype = config.tf_cdtype
    else:
        rdtype = dtypes[precision]["tf"]["rdtype"]
        cdtype = dtypes[precision]["tf"]["cdtype"]

    if sigma_phi_deg>15:
        warnings.warn("sigma_phi_deg should be smaller than 15.")

    # Convert all inputs to radians
    phi_deg = tf.cast(phi_deg, dtype=rdtype)
    sigma_phi_deg = tf.cast(sigma_phi_deg, dtype=rdtype)
    phi = deg_2_rad(phi_deg)
    sigma_phi = deg_2_rad(sigma_phi_deg)

    # Add dimensions for broadcasting
    phi = tf.expand_dims(phi, -1)
    sigma_phi = tf.expand_dims(sigma_phi, -1)

    # Compute first column
    c = tf.constant(2*PI*d_h, dtype=rdtype)
    d = c*tf.range(0, num_ant, dtype=rdtype)
    d = expand_to_rank(d, tf.rank(phi), 0)

    a = tf.complex(tf.cast(0, dtype=rdtype), d*tf.sin(phi))
    exp_a = tf.exp(a) # First exponential term

    b = -tf.cast(0.5, dtype=rdtype)*(sigma_phi*d*tf.cos(phi))**2
    exp_b = tf.cast(tf.exp(b), dtype=cdtype) # Second exponetial term

    col = exp_a*exp_b # First column

    # First row is just the complex conjugate of first column
    row = tf.math.conj(col)

    # Create Toeplitz operator
    operator = tf.linalg.LinearOperatorToeplitz(col, row)

    # Generate dense tensor from operator
    r = operator.to_dense()

    return r

def compute_pathloss_gas(self):
        
        r"""Calculates the pathloss of a scenario based on 3GPP TR38.811 6.6.4

        The mathematical process is described in more detail in ITU-R P.676. The loss is defined for
        frequencies between 1GHz and 350GHz. 

        """

        # Constants set during scenario creation 
        T=self._temperature
        p=self._atmospheric_pressure

        f = self._carrier_frequency/(10.0**9.0)
        pw=self._water_vapor_density#water vapor density
        
        # Calculate the water vapor specific attenuation(Yw)
        # For more detail see ITU-R P.676-5 (23a)
        rt=288/T
        rp=p/1013
        n1=0.955*rp*(rt**0.68)+(0.006*pw)
        n2=0.735*rp*(rt**0.5)+(0.0353*(rt**4)*pw)
        Yw = (((3.98*n1*tf.math.exp(2.23*(1-rt))) / (((f-22.235)**2) + 9.42*(n1**2))) * (1+((f-22) / (f+22))**2) + \
            ((11.96*n1*tf.math.exp(0.7*(1-rt))) / (((f-183.31)**2) + 11.14*(n1**2))) + \
            ((0.081*n1*tf.math.exp(6.44*(1-rt))) / (((f-321.226)**2) + (6.29*(n1**2)))) + \
            ((3.66*n1*tf.math.exp(1.6*(1-rt))) / (((f-325.153)**2) + 9.22*(n1**2))) + \
            ((25.37*n1*tf.math.exp(1.09*(1-rt))) / ((f-380)**2)) + \
            ((17.4*n1*tf.math.exp(1.46*(1-rt))) / ((f-448)**2)) + \
            ((844.6*n1*tf.math.exp(0.17*(1-rt))) / ((f-557)**2)) * (1+((f-557)/(f+557))**2) + \
            ((290*n1*tf.math.exp(0.41*(1-rt))) / ((f-752)**2)) * (1+((f-752)/(f+752))**2) + \
            ((83328*n2*tf.math.exp(0.99*(1-rt))) / ((f-1780)**2)) * (1+((f-1780)/(f+1780))**2)) * \
            ((f**2) * (rt**2.5) * (pw*10**(1-5)))

        # calculation of the path length for water vapor contents(hw) (km) 
        conw=1.013/(1+tf.math.exp((0-8.1)*(rp-0.57)))
        hw=1.66*(1+((1.39*conw)/(((f-22.235)**2)+(2.56*conw)))+((3.37*conw)/(((f-183.31)**2)+(4.69*conw)))+((1.5*conw)/(((f-325.1)**2)+(2.89*conw))))

        #water vapor attenuation in zenith angle path (dB)
        Aw=Yw*hw 

        ###constants
        ee1=(rp**0.0717)*(rt**(-1.8132))*tf.math.exp((0.0156*(1-rp))+(-1.6515*(1-rt)))
        ee2=(rp**0.5146)*(rt**(-4.6368))*tf.math.exp(((-0.1921)*(1-rp))+(-5.7416*(1-rt)))
        ee3=(rp**0.3414)*(rt**(0-6.5851))*tf.math.exp(((0.2130)*(1-rp))+(-8.5854*(1-rt)))
        ee4=(rp**(-0.0112))*(rt**(0.0092))*tf.math.exp((-0.1033*(1-rp))+(-0.0009*(1-rt)))
        ee5=(rp**0.2705)*(rt**(-2.7192))*tf.math.exp((-0.3016*(1-rp))+(-4.1033*(1-rt)))
        ee6=(rp**0.2445)*(rt**(-5.9191))*tf.math.exp((0.0422*(1-rp))+(-8.0719*(1-rt)))
        ee7=(rp**(-0.1833))*(rt**(6.5589))*tf.math.exp((-0.2402*(1-rp))+(6.131*(1-rt)))

        g54 = 2.136 * rp**1.4975 * rt**(-1.5852) * math.exp((-2.5196)*(1-rt))
        g57 = 9.984 * rp**0.9313 * rt**(2.6732) * math.exp((0.8563)*(1-rt))
        g60 = 15.42 * rp**0.8595 * rt**(3.6178) * math.exp((1.1521)*(1-rt))
        g63 = 10.63 * rp**0.9298 * rt**(2.3284) * math.exp((0.6287)*(1-rt))
        g66 = 1.944 * rp**1.6657 * rt**(-3.3714) * math.exp((-4.1643)*(1-rt))

        
        # Calculation of the dry air specific attenuation 
        # For more detail see ITU-R P.676-5 (22a)
        if tf.less_equal(f,60.0):
            N = 0.0
        else:
            N = -15.0
        
        if tf.less_equal(f, 54.0):
            Yo=(((7.2*(rt**2.8))/((f**2)+(0.34*(rp**2)*(rt**1.6))))+((0.62*ee3)/(((54-f)**(1.16*ee1))+(0.83*ee2))))*((f**2)*(rp**2)*(10**(-3)))

        elif tf.less(54.0,f) and tf.less_equal(f, 66.0):
            Yo = tf.math.exp((
                            ((54.0**(-N)) * tf.math.log(g54) * (f - 57.0) * (f - 60.0) * (f - 63.0) * (f - 66.0) / 1944.0) -
                            ((57.0**(-N)) * tf.math.log(g57) * (f - 54.0) * (f - 60.0) * (f - 63.0) * (f - 66.0) / 486.0) +
                            ((60.0**(-N)) * tf.math.log(g60) * (f - 54.0) * (f - 57.0) * (f - 63.0) * (f - 66.0) / 324.0) -
                            ((63.0**(-N)) * tf.math.log(g63) * (f - 54.0) * (f - 57.0) * (f - 60.0) * (f - 66.0) / 486.0) +
                            ((66.0**(-N)) * tf.math.log(g66) * (f - 54.0) * (f - 57.0) * (f - 60.0) * (f - 63.0) / 1944.0) 
                            )* f**N)
        
        elif tf.less(66.0,f) and tf.less_equal(f, 120.0):
            Yo = 3.02 * 10.0**(-4) * rt**(3.5) + ((0.283*rt**3.8)/((f-118.75)**2.0 +2.91 * rp**2.0 * rt**1.6)) + ((0.502 * ee6 * (1 - 0.0163 * ee7 * (f - 66.0)))/((f - 66.0) ** (1.4346 * ee4 + 1.15 * ee5)))
        # TODO In eager execution this code requires Yo to always be set. The best solution would be to throw an exception if illeagal frequencies are encoutered. However, this
        # currently breaks eager execution and thus uses this value instead. A future version will properly cath this case and throw an exceotion.For now this is not 
        # necessary, as the frequency range is not supported by the standard.
        else:
            Yo = 0.0

        # Calculation of the equivalent height
        t1=(4.64/(1+(0.066*(rp**(-2.3)))))*tf.math.exp(-(((f-59.7)/(2.87+(12.4*tf.math.exp((-7.9)*rp))))**2.0))
        t2=(0.14*tf.math.exp(2.12*rp))/(((f-118.75)**2)+0.031*tf.math.exp(2.2*rp))
        t3=(0.0114/(1+(0.14*(rp**(-2.6)))))*f*((-0.0247+(0.0001*f)+(1.61*(10.0**(-6.0))*f**2.0))/(1.0-(0.0169*f)+(4.1*(10.0**(-5.0))*f**2)+ (3.2*(10.0**(-7.0))*f**3.0)))
        ho=(6.1/(1.0+(0.17*(rp**(-1.1)))))*(1.0+t1+t2+t3)

        # Dry air attenuation in zenith angle path (dB)
        Ao=Yo*ho
            
        # Calculate the total gases attenuation for the given elevation
        Elev = self._elevation_angle
        Atotx = (Ao + Aw) / tf.math.sin(math.radians(Elev))
        GasAtt = abs(Atotx)

        pl_g = tf.zeros(shape = tf.shape(self.los)) + GasAtt

        self._pl_g = pl_g
            

def compute_pathloss_scintilation(self):

        f = self._carrier_frequency/(10.0**9.0)
        #For frequencies above 6GHz, we only consider the troposhperic scintillation. 
        #Below 6GHz we only consider Ionospheric scintillation. 
        #6GHz exactly is not covered by either model. However, the tropospheric model is calculated for 1-350GHz, thus we add this
        #edge case to the troposhperic scintillation
        if f >= 6.0:
            T=self._temperature
            #  Calculate the saturation water vapor pressure
            T_celcius = T-273
            

            es = 6.1121 * math.exp((17.502 * T_celcius) / (T_celcius + 240.97))

            # Compute the wet term of the radio refractivity
            H = self._relative_humidity # Relative humidity in percentage
            Nwet = 3732 * H * es / ((T_celcius + 273)**2)

            #  Calculate the standard deviation of the signal amplitude
            ref = 3.6e-3 + (Nwet * 1e-4) # in dB

            #  Calculate the effective path length
            hL = 1000 # Height of turbulent layer in meters
            Elev = self._elevation_angle
            L = (2 * hL) / (tf.math.sqrt(tf.math.sin(math.radians(Elev))**2 + 2.35e-4 + tf.math.sin(math.radians(Elev))))
                
            # Step 5: Calculate the effective antenna diameter
            D = self._diameter_earth_antenna # antenna diameter in meters
            efficiency = self._antenna_efficiency # Antenna efficiency
            Deff = math.sqrt(efficiency) * D
            # Step 6: Calculate the antenna averaging factor

            x = (1.22 * Deff**2 * f) / L
            g = tf.math.sqrt(3.86 * (x**2 + 1)**(11/12) * tf.math.sin((11/6) * tf.math.atan(1/x)) - 7.08 * x**(5/6))

            # Step 7: Calculate the standard deviation of the signal for the period and propagation path
            g_x = (ref * f**(7/12) * g) / (math.sin(math.radians(Elev))**1.2)
            # Step 8: Calculate the time percentage factor
            p = 0.01 # Time percentage in #
            a_p = -0.061 * (math.log10(p))**3 + 0.072 * (math.log10(p))**2 - 1.71 * math.log10(p) + 3.0

            # Step 9: Calculate the scintillation fade depth
            TroScin = a_p * g_x

            TScin=abs(TroScin)
            pl_s = tf.zeros(shape = tf.shape(self.los)) + TScin
            self._pl_s = pl_s
            
        #For carrier frequencies below 6GHz we only consider Ionospheric scintillation    
        else:
             #TODO: Add check that the latidue is valid (between 20 and 60 degrees above or below the equator)
            #Pathloss at 4GHz
            Pluc_4GHz = 1.1
            #Exponent to scale the frequeny loss below 6GHz TODO: check the applied frequencies
            n = -1.5
            #Scale scintillation pathloss at 4GHz to used carrier frequency by applying 3GPP TR38.811 6.6-12
            AIs = Pluc_4GHz * (f/4)**(n)
            #Get final loss by dividing by sqrt(2) according to 3GPP TR38.811 6.6-13
            pl_s = tf.zeros(shape = tf.shape(self.los)) + (AIs/math.sqrt(2.0))
            self._pl_s = pl_s
        


def compute_pathloss_entry(self):
        #Currently only 0, as no HAPS are considered an thus no building cases
        pl_e = tf.zeros(shape = tf.shape(self.los))
        self._pl_e = pl_e
        #ntnu.compute_pathloss_gas(self)

def compute_pathloss_basic(self):
        r"""Computes the basic pathloss based on TR38.811 6.6-1 in dB."""
        distance_3d = self._distance_3d
        fc = self._carrier_frequency/(10**9)
        angle_str = str(round(self._elevation_angle/10.0)*10)

        fspl = 32.45 + 20*log10(fc) + 20*log10(distance_3d)
        self._fspl = fspl
        cl = self._params_nlos["CL" + '_' + angle_str]
        sigmaSF_los = self._params_los["sigmaSF" + '_' + angle_str]
        sigmaSF_nlos = self._params_nlos["sigmaSF" + '_' + angle_str]

        SF_los = tf.random.normal(shape = tf.shape(self.los), mean = 0.0, stddev = sigmaSF_los)
        SF_nlos = tf.random.normal(shape = tf.shape(self.los), mean = 0.0, stddev = sigmaSF_nlos)

        pl_los = fspl + SF_los
        pl_nlos = fspl + SF_nlos + cl
        pl_b = tf.where(self.los, pl_los, pl_nlos)

        self._pl_b = pl_b

# TODO this function computes additional pathlosses, which are usually considered 0 in many callibration cases of 3GPP. However, we still have implemented them and will
# verify their use. For now, we run into not yet understood issues in eager execution, which will be fixed in a future version.
# Additionally, the temperature and lattitude will become topology parameters, so that these can be adapted freely in the user interface, instead
# of using the standard ITU values
def compute_pathloss_additional(self):
        r"""Cmoputes the PL due to rain and cloud attenuation based on 3GPP TR38.811 6.6.5"""
        ###Cloud attenuation(based on ITU-R P.840-3)
        f = self._carrier_frequency/(10**9)
        T=self._temperature
        LWC=self._lwc
        Lat=self._latitude
        #Step 1: Calculation of the principal & secondary relaxation frequencies: 
        theta=300/T#constant
        fp=20.09-142*(theta-1)+294*(theta-1)**2 #Primary relaxation frequency (GHz)
        fs=590-1500*(theta-1) # Secondary relaxation frequency (GHz)

        #Step 2:Calculation of the complex dielectric permittivity of water:

        eo=77.6+103.3*(theta-1)#constant
        e1=5.48#constant
        e2=3.51#constant
        eta2=(f/fp)*((eo-e1)/(1+(f/fp)** 2))+(f/fs)*((e1-e2)/(1+(f/fs)**2))
        eta1=((eo-e1)/(1+(f/fp)**2))+((e1-e2)/(1+(f/fs)**2))+e2 #Complex dielectric permittivity of water
        n=(2+eta1)/eta2

        #Step 3: cloud specific attenuation coefficient ((dB/km)/(g/m**3)) 

        Kl=0.819*f/(eta2*(1+n**2)) #Specific Cloud attenuation coefficient((dB/km)/(g/m**3)) 

        #Step 4:cloud attenuation
        Elev = self._elevation_angle
        CAtten=LWC*Kl/math.sin(math.radians(Elev)) #cloud attenuation in dB

        #  disp(['Cloud attenuation= ', num2str(CAtten),' dB'])
        if self._direction == "downlink":
            hs = self._bs_loc[:,:,2]
        else:
            hs = 0

        ### Rain Attenuation
        hr = 5 - 0.075*(Lat - 23) # rain height for latitude more than 23 deg
        x = Elev*math.pi/180 # convert elevation angle to radians
        ls = (hr-hs)/math.sin(x) # slant path length
        lg = ((ls/1000)*math.cos(x)) # horizontal projection length in km
        # Initialize variables
        log_k = 0

        # Constants for each value of j
        a = [-5.33980, -0.35351, -0.23789, -0.94158]
        b = [-0.10008, 1.26970, 0.86036, 0.64552]
        c = [1.13098, 0.45400, 0.15354, 0.16817]
        m = -0.18961
        d = 0.71147

        # Iterate over j values from 1 to 4 and calculate the sum
        for j in range(len(a)):
            log_k = log_k + (a[j] * math.exp(-((log10(f) - b[j]) / c[j])**2))

        # Add (m * log10(f)) and d to the final log_k value
        log_k = log_k + (m * log10(f)) + d

        # Calculate k by taking the antilogarithm (10**x) of log_k
        kH = 10**log_k

        # Initialize variables
        log_k1 = 0

        # Constants for each value of j
        a1= [-3.80595,-3.44965 ,-0.39902 ,0.50167  ]
        b1 = [0.56934,-0.22911  ,0.73042 ,1.07319  ]
        c1 = [0.81061, 0.51059, 0.11899, 0.27195]
        x1 = -0.16398
        d1 = 0.63297

        # Iterate over j values from 1 to 4 and calculate the sum
        for l in range(len(a1)):
            log_k1 = log_k1 + (a1[l] * math.exp(-((log10(f) - b1[l]) / c1[l])**2))

        # Add (m * log10(f)) and d to the final log_k value
        log_k1 = log_k1 + (x1 * log10(f)) + d1

        # Calculate k by taking the antilogarithm (10**x) of log_k
        kV = 10**log_k1

        # Initialize variables
        log_k2 = 0

        # Constants for each value of j
        a2= [-0.14318,0.29591 ,0.32177 ,-5.37610 ,16.1721   ]
        b2 = [1.82442 ,0.77564  ,0.63773 ,-0.96230, -3.29980 ]
        c2 = [-0.55187,0.19822,0.13164,1.47828,3.43990]
        x2 = 0.67849
        d2 = -1.95537

        # Iterate over j values from 1 to 5 and calculate the sum
        for i in range(len(a2)):
            log_k2= log_k2 + (a2[i] * math.exp(-((log10(f) - b2[i]) / c2[i])**2))

        # Add (m * log10(f)) and d to the final log_k value
        log_k2 = log_k2 + (x2 * log10(f)) + d2

        # Calculate k by taking the antilogarithm (10**x) of log_k
        aH = log_k2

        # Initialize variables
        log_k3 = 0

        # Constants for each value of j
        a3= [-0.07771,0.56727 ,-0.20238,-48.2991 ,48.5833    ]
        b3 = [2.33840 ,0.95545,1.14520,0.791669 ,0.791459  ]
        c3 = [-0.76284,0.54039,0.26809,0.116226,0.116479]
        x3 = -0.053739
        d3 = 0.83433

        # Iterate over j values from 1 to 5 and calculate the sum
        for n in range(len(a3)):
            log_k3= log_k3 + (a3[n] * math.exp(-((log10(f) - b3[n]) / c3[n])**2))

        # Add (m * log10(f)) and d to the final log_k value
        log_k3 = log_k3 + (x3 * log10(f)) + d3

        # Calculate k by taking the antilogarithm (10**x) of log_k
        aV = log_k3
        
        y = math.cos(x)**2
        y1 = math.cos(2*45*math.pi/180)
        y2 = y*y1
        
        #k and a are frequency dependent constants
        
        k = (kH + kV + (kH - kV )*y2)/2 
        a = (kH*aH + kV*aV +(kH*aH - kV*aV)*y2) / (2*k)
        #TODO The parameter r here is believed to be a constant, but might depend on the scenario/topology. If so, it needs to be added to the parameterlist.
        #Rain rate in mm/h
        r = 40
        #specific attenuation
        yr = k*r**a 
        # horizontal reduction factor(hrf)
        m1 = lg*yr/f
        m2 = 0.38*(1-math.exp(-2*lg))
        m3 = 0.78*math.sqrt(m1-m2)
        hrf = 1/(1+m3)
        # vertical adjustment factor
        z1 = (hr-hs)/(lg*hrf)
        z2 = z1*math.pi/180
        # TODO Here the inversion of the matrix does not work breaking the entire simulation. It might be due to the large numbers in NTN
        # causing a null matrix which cant be inverted, but this is not yet understood fully yet.
        #z = tf.linalg.inv(tf.math.tan(z2))
        z = math.tan(z2)
        if z>x:
            lr =  lg*hrf/math.cos(x)
        else:
            lr=ls    
        vrf = 1/(1+math.sqrt(math.sin(x))*(31*(1-math.exp(-x))*(math.sqrt(lr*yr)/f**2)-0.45))

        #   effective path length 
        le = lr*vrf

        # predicted attenuation for 0.01# excedence 
        att = le*yr
        Ratt=abs(att)

        total_att = Ratt + CAtten

        pl_a = tf.zeros(shape = tf.shape(self.los)) + total_att

        self._pl_a = pl_a

#Function that computes the satellite speed as a function of its height.
def compute_satellite_speed(h_sat):
    G = 6.6743 * (10 ** (-11)) #Gravitational constant of Earth in m^3/(kg*s^2)
    M = 5.972 * (10 ** 24) #Mass of Earth in kg
    r_earth = 6371000.0#Earths radius in m

    #Equation (8.41) from Curtis, H. D. (2005). Orbital Mechanics for Engineering Students. Butterworth-Heinemann. ISBN: 0 7506 6169 0.
    v_sat = tf.math.sqrt((G*M)/((r_earth)+(h_sat)))#speed of the satellite in given Orbit around Earth in km/s
    return v_sat

def compute_stallite_doppler(h_sat, elevation_angle, carrier_frequency):
        r"""Compute the maximum radian Doppler frequency [Hz] for a given
        speed [m/s].

        The maximum radian Doppler frequency :math:`\omega_d` is calculated
        as:

        .. math::
            \omega_d = 2\pi  \frac{v}{c} f_c

        where :math:`v` [m/s] is the speed of the receiver relative to the
        transmitter, :math:`c` [m/s] is the speed of light and,
        :math:`f_c` [Hz] the carrier frequency.

        Input
        ------
        speed : float
            Speed [m/s]

        Output
        --------
        doppler_shift : float
            Doppler shift [Hz]
        """
        #Experimental addition according to 811 6.9.2
        #G = 6.6743 * (10 ** (-11)) #Gravitational constant of Earth in m^3/(kg*s^2)
        #M = 5.972 * (10 ** 24) #Mass of Earth in kg
        
        v_light = 299792458.0# speed of light in m/s (as m/s * 3.6 to get km/h)
        r_earth = 6371000.0#Earths radius in m

        #Equation (8.41) from Curtis, H. D. (2005). Orbital Mechanics for Engineering Students. Butterworth-Heinemann. ISBN: 0 7506 6169 0.
        #v_sat = tf.math.sqrt((G*M)/((r_earth)+(h_sat)))#speed of the satellite in given Orbit around Earth in km/s
        v_sat = compute_satellite_speed(h_sat)
        additional_doppler_shift_811 = (v_sat/v_light) * ((r_earth/(r_earth+h_sat)) * np.cos(np.deg2rad(elevation_angle))) * carrier_frequency

        return additional_doppler_shift_811 + carrier_frequency