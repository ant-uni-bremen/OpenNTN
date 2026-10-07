#
# SPDX-FileCopyrightText: Copyright (c) 2021-2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Modified by the Dept. of Communications Engineering of the University of Bremen.
# Parts of the code reuse implementations provided by the NVIDIA CORPORATION & AFFILIATES
# in Sionna.
#

"""
DISCLAIMER: THE TDL MODEL IS NOT YET FULLY DEFINED BY THE 3GPP. THUS, THE IMPLEMENTATION IS NOT YET PROVIDED.
THIS FILE ONLY EXISTS TO BE A REFERENCE FOR A LATER IMPLEMENTATION, AS SOON AS THE STANDARD IS UPDATED.

Sionna 2.0 / PyTorch migration note: the original TensorFlow template body (a copy of the TR38.901
version) was removed so the package imports cleanly under the TensorFlow-free Sionna 2.0.0 stack.
TDL stays out of scope and unimplemented; when the standard is updated, use the git history and the
Sionna 2.0 ``sionna.phy.channel.tr38901.TDL`` as a torch reference for the implementation.
"""

"""Tapped delay line (TDL) channel model from 3GPP TR38.811 specification (not implemented)."""


class TDL:
    """Placeholder for the TR38.811 TDL model. Not implemented (see module docstring)."""

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(
            "The TR38.811 TDL model is not implemented in OpenNTN. "
            "It is defined only as a placeholder until 3GPP finalizes the standard."
        )
