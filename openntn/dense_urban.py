#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by the Dept. of Communications Engineering of the University of Bremen.
# Based on the implementation of scenario UMa in Sionna TR38.901.
#
"""Dense Urban channel model from 3GPP TR38.811 specification"""

from . import SystemLevelChannel
from . import DenseUrbanScenario


class DenseUrban(SystemLevelChannel):
    # pylint: disable=line-too-long
    r"""DenseUrban(carrier_frequency, ut_array, bs_array, direction, elevation_angle, enable_pathloss=True, enable_shadow_fading=True, always_generate_lsp=False, doppler_enabled=True, precision=None, device=None)

    Dense Urban channel model from 3GPP [TR38811]_ specification.

    Setting up a Dense Urban model requires configuring the network topology, i.e., the
    UTs and BSs locations, UTs velocities, etc. This is achieved using the
    :meth:`~openntn.DenseUrban.set_topology` method. Setting a different
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
    ...                     direction = 'uplink',
    ...                     elevation_angle = 30.0)
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

    where ``rg`` is an instance of :class:`~sionna.phy.ofdm.ResourceGrid`.

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
        elevation angle of the LOS path of the satellite/HAPS vs. ground horizon in degrees.
        The tables of TR 38.811 are given at the reference elevation angles 10 to 90
        degrees in steps of 10 degrees; the model takes the values of the reference
        angle nearest to the elevation angle, and an angle midway between two reference
        angles takes the higher one (25 degrees takes 30 degrees).

    enable_pathloss : bool
        If `True`, apply pathloss. Otherwise doesn't. Defaults to `True`.

    enable_shadow_fading : bool
        If `True`, apply shadow fading. Otherwise doesn't.
        Defaults to `True`.

    always_generate_lsp : bool
        If `True`, new large scale parameters (LSPs) are generated for every
        new generation of channel impulse responses. Otherwise, always reuse
        the same LSPs, except if the topology is changed. Defaults to
        `False`. Must be `True` or `False`; `TypeError` otherwise.

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
        construction is used. ``ut_array`` and ``bs_array`` must be built on
        this device (their argument ``device``); `ValueError` otherwise. The
        model stays on this device and in its precision: moved with ``.to()``
        or cast, it raises `ValueError` in its next ``set_topology`` or call.

    Input
    -----

    num_time_samples : int
        Number of time samples

    sampling_frequency : float
        Sampling frequency [Hz]

    Output
    -------
        a : [batch size, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_samples], torch.complex
            Path coefficients

        tau : [batch size, num_rx, num_tx, num_paths], torch.float
            Path delays [s]

    The path coefficients and delays are detached from the autograd graph, as in
    Sionna, so gradients with respect to the topology exist only for the quantities
    of the scenario.
    """

    def __init__(self, carrier_frequency, ut_array, bs_array,
        direction, elevation_angle, enable_pathloss=True, enable_shadow_fading=True,
        always_generate_lsp=False, doppler_enabled=True, precision=None, device=None):

        # dense urban scenario
        scenario = DenseUrbanScenario(carrier_frequency, ut_array, bs_array,
            direction, elevation_angle, enable_pathloss, enable_shadow_fading,
            doppler_enabled, precision=precision, device=device)

        super().__init__(scenario, always_generate_lsp)
