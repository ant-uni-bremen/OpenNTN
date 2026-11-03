# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# This file is a characterization test of the tapped delay line model TDL. It compares
# its outputs, the path coefficients a and the delays tau, for fixed seeds on the CPU
# with reference outputs stored in reference_outputs/tdl.npz. The references were
# produced by the implementation itself, so they do not prove conformance with
# TR 38.811, which tdl_test.py checks; they make every change of numerical results
# visible, also a change that keeps the distributions, such as another order of the
# random draws. A change that alters these outputs must come with a documented
# explanation, and the references are rebuilt with the following command, with openntn
# importable (in a source checkout, set PYTHONPATH to the repository root):
#
#     python <path to this file> --regenerate
#
# It writes reference_outputs/tdl.npz only. Environment: the references are made in the
# environment of reference_outputs/environment.json, which the references of the
# system-level models record (reference_outputs_test.py); the regeneration refuses to
# write in another environment, so that the file describes both sets.
#
# Precision and tolerance: those of reference_outputs_test.py. The outputs are computed
# in double precision, where the rounding differences between CPUs and their vectorized
# kernels stay orders of magnitude below the tolerance, and every stored array is
# compared with |actual - reference| <= RTOL * |reference| + ATOL_SCALE *
# max|reference|, with RTOL = 1e-5 and ATOL_SCALE = 1e-6.
#
# The configurations, each with its own seed, a batch of 2 and 14 time steps at 30 kHz:
# - NTN-TDL-A, 100 ns, 2 GHz, 50 degrees, a satellite at 600 km, the defaults: the shift
#   of the satellite, 29 634.973 Hz, and a UT at rest;
# - NTN-TDL-B, 300 ns, 30 GHz, 30 degrees, 1200 km, a UT at up to 30 m/s;
# - NTN-TDL-C, 50 ns, 2 GHz, 70 degrees, 600 km, a UT at 3 m/s;
# - NTN-TDL-D, 30 ns, 20 GHz, 10 degrees, a geostationary satellite (no shift), the
#   desired K-factor 3 dB, a UT at 30 m/s;
# - NTN-TDL-C, 5 ns, 2 GHz, 50 degrees, 600 km, the desired K-factor 7.95 dB, no shift,
#   a UT at 30 m/s: the conditions of the test profile NTN-TDLC5-200 of TS 38.101-5;
# - NTN-TDL-A, 100 ns, 3.5 GHz, 40 degrees, an aerial platform at 20 km with the given
#   shift -500 Hz, 2 x 2 antennas with the receive and transmit correlation matrices of
#   exp_corr_mat(0.5, 2).
import json
import os
import sys
import unittest

import numpy as np
import pytest
from sionna.phy import config

from openntn import TDL
from openntn.utils import exp_corr_mat
from reference_outputs_test import (ATOL_SCALE, ENVIRONMENT_FILE, PRECISION,
                                    REFERENCE_DIR, RTOL, _describe, environment)

REFERENCE_FILE = os.path.join(REFERENCE_DIR, "tdl.npz")

# Lets CI run this test only in the environment of the references.
pytestmark = pytest.mark.reference

BATCH_SIZE = 2
NUM_TIME_STEPS = 14
SAMPLING_FREQUENCY = 30e3

# name -> (seed, model, delay spread [s], carrier frequency [Hz], elevation angle [deg],
# height [m], keyword arguments; the correlation matrices as the argument of
# exp_corr_mat)
CONFIGURATIONS = {
    "a_satellite_shift": (101, "A", 100e-9, 2.0e9, 50.0, 600e3, {}),
    "b_ka_band_moving_ut": (102, "B", 300e-9, 30.0e9, 30.0, 1200e3,
                            {"max_speed": 30.0}),
    "c_moving_ut": (103, "C", 50e-9, 2.0e9, 70.0, 600e3, {"min_speed": 3.0}),
    "d_geostationary_k_factor": (104, "D", 30e-9, 20.0e9, 10.0, 35786e3,
                                 {"k_factor": 3.0, "min_speed": 30.0}),
    "c_k_factor_without_shift": (105, "C", 5e-9, 2.0e9, 50.0, 600e3,
                                 {"k_factor": 7.95, "doppler_shift": 0.0,
                                  "min_speed": 30.0}),
    "a_aerial_platform_correlated_antennas": (106, "A", 100e-9, 3.5e9, 40.0, 20e3,
                                              {"doppler_shift": -500.0, "num_rx_ant": 2,
                                               "num_tx_ant": 2, "rx_corr_mat": 0.5,
                                               "tx_corr_mat": 0.5}),
}


def compute_case(seed, model, delay_spread, carrier_frequency, elevation_angle, height,
                 options):
    """The path coefficients and the delays of one configuration as NumPy arrays"""
    options = dict(options)
    for name in ("rx_corr_mat", "tx_corr_mat"):
        if name in options:
            options[name] = exp_corr_mat(options[name], 2, precision=PRECISION,
                                         device="cpu")
    tdl = TDL(model, delay_spread, carrier_frequency, elevation_angle, height,
              precision=PRECISION, device="cpu", **options)
    config.seed = seed
    a, tau = tdl(BATCH_SIZE, NUM_TIME_STEPS, SAMPLING_FREQUENCY)
    return a.cpu().numpy(), tau.cpu().numpy()


def compute_all():
    """The outputs of all configurations"""
    previous_device = config.device
    previous_precision = config.precision
    config.device = "cpu"
    config.precision = PRECISION
    try:
        results = {}
        for name, configuration in CONFIGURATIONS.items():
            a, tau = compute_case(*configuration)
            results[f"{name}__a"] = a
            results[f"{name}__tau"] = tau
        return results
    finally:
        config.device = previous_device
        config.precision = previous_precision


def regenerate():
    with open(ENVIRONMENT_FILE, encoding="utf-8") as f:
        recorded = json.load(f)
    if environment() != recorded:
        sys.exit(f"not written: the environment ({_describe(environment())}) is not "
                 f"that of {ENVIRONMENT_FILE} ({_describe(recorded)})")
    results = compute_all()
    np.savez_compressed(REFERENCE_FILE, **results)
    print(f"wrote {REFERENCE_FILE} ({len(results)} arrays)")


class TDLReferenceOutputs(unittest.TestCase):

    def test_tdl(self):
        with np.load(REFERENCE_FILE) as stored:
            reference = {k: stored[k] for k in stored.files}
        with open(ENVIRONMENT_FILE, encoding="utf-8") as f:
            produced_with = _describe(json.load(f))
        actual = compute_all()
        self.assertEqual(sorted(actual), sorted(reference),
                         "the set of stored arrays changed")
        for key in sorted(reference):
            with self.subTest(array=key):
                ref = reference[key]
                act = actual[key]
                self.assertEqual(act.shape, ref.shape)
                self.assertEqual(act.dtype, ref.dtype)
                finite = np.abs(ref[np.isfinite(ref)])
                atol = ATOL_SCALE * (finite.max() if finite.size else 0.0)
                np.testing.assert_allclose(
                    act, ref, rtol=RTOL, atol=atol, equal_nan=True,
                    err_msg=(f"tdl/{key} differs from the reference (produced with "
                             f"{produced_with}; now {_describe(environment())})"))


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        unittest.main()
