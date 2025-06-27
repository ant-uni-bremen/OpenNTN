#
# This file has been created by the Dept. of Communications Engineering of the University of Bremen.
# The code is based on implementations provided by the NVIDIA CORPORATION & AFFILIATES
#
# Based on open source framework Sionna of 2021-2023 NVIDIA CORPORATION & AFFILIATES
# Based on implementation of scenario UMa in Sionna TR38.901
#
"""Dense Urban channel model from 3GPP TR38.811 specification"""

import tensorflow as tf

from . import SystemLevelChannel
from . import DenseUrbanScenario


class DenseUrban(SystemLevelChannel):
    # pylint: disable=line-too-long
    r"""DenseUrban(carrier_frequency, ut_array, bs_array, direction, enable_pathloss=True, enable_shadow_fading=True, always_generate_lsp=False, dtype=tf.complex64)

    Dense Urban channel model from 3GPP [TR38811]_ specification.

    Setting up a Dense Urban model requires configuring the network topology, i.e., the
    UTs and BSs locations, UTs velocities, etc. This is achieved using the
    :meth:`~sionna.phy.channel.tr38811.DenseUrban.set_topology` method. Setting a different
    topology for each batch example is possible. The batch size used when setting up the network topology
    is used for the link simulations.

    The following code snippet shows how to setup an DenseUrban channel model assuming
    an OFDM waveform:

    >>> # UT and BS panel arrays
    >>> bs_array = PanelArray(num_rows_per_panel = 4,
    ...                       num_cols_per_panel = 4,
    ...                       polarization = 'dual',
    ...                       polarization_type = 'cross',
    ...                       antenna_pattern = '38.901',
    ...                       carrier_frequency = 3.5e9)
    >>> ut_array = PanelArray(num_rows_per_panel = 1,
    ...                       num_cols_per_panel = 1,
    ...                       polarization = 'single',
    ...                       polarization_type = 'V',
    ...                       antenna_pattern = 'omni',
    ...                       carrier_frequency = 3.5e9)
    >>> # Instantiating DenseUrban channel model
    >>> channel_model = DenseUrban(carrier_frequency = 3.5e9,
    ...                     ut_array = ut_array,
    ...                     bs_array = bs_array,
    ...                     direction = 'uplink')
    >>> # Setting up network topology
    >>> # ut_loc: UTs locations
    >>> # bs_loc: BSs locations
    >>> # ut_orientations: UTs array orientations
    >>> # bs_orientations: BSs array orientations
    >>> # in_state: Indoor/outdoor states of UTs
    >>> channel_model.set_topology(ut_loc,
    ...                            bs_loc,
    ...                            ut_orientations,
    ...                            bs_orientations,
    ...                            ut_velocities,
    ...                            in_state)
    >>> # Instanting the OFDM channel
    >>> channel = OFDMChannel(channel_model = channel_model,
    ...                       resource_grid = rg)

    where ``rg`` is an instance of :class:`~sionna.ofdm.ResourceGrid`.

    Parameters
    -----------

    carrier_frequency : float
        Carrier frequency [Hz]

    rx_array : PanelArray
        Panel array used by the receivers. All receivers share the same
        antenna array configuration.

    tx_array : PanelArray
        Panel array used by the transmitters. All transmitters share the
        same antenna array configuration.

    direction : str
        Link direction. Either "uplink" or "downlink".

    elevation_angle : float
        elevation angle of the LOS path of the satellite/HAPS vs. ground horizon in degrees

    enable_pathloss : bool
        If `True`, apply pathloss. Otherwise doesn't. Defaults to `True`.

    enable_shadow_fading : bool
        If `True`, apply shadow fading. Otherwise doesn't.
        Defaults to `True`.

    average_street_width : float
        Average street width [m]. Defaults to 5m.

    average_building_height : float
        Average building height [m]. Defaults to 20m.

    always_generate_lsp : bool
        If `True`, new large scale parameters (LSPs) are generated for every
        new generation of channel impulse responses. Otherwise, always reuse
        the same LSPs, except if the topology is changed. Defaults to
        `False`.

    precision : `None` (default) | "single" | "double"
        Precision used for internal calculations and outputs.
        If set to `None`,
        :attr:`~sionna.phy.config.Config.precision` is used.

    Input
    -----

    num_time_steps : int
        Number of time steps

    sampling_frequency : float
        Sampling frequency [Hz]

    Output
    -------
        a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_steps], tf.complex
            Path coefficients

        tau : [batch size, num_rx, num_tx, num_paths], tf.float
            Path delays [s]
    """

    def __init__(self, carrier_frequency, ut_array, bs_array,
        direction, elevation_angle, enable_pathloss=True, enable_shadow_fading=True,
        average_street_width=20.0, average_building_height=5.0,
        always_generate_lsp=False, doppler_enabled=True, precision=None):

        # dense urban scenario
        scenario = DenseUrbanScenario(carrier_frequency, ut_array, bs_array,
            direction, elevation_angle, enable_pathloss, enable_shadow_fading,
            average_street_width, average_building_height, doppler_enabled, precision=precision)

        super().__init__(scenario, always_generate_lsp)
