# This file is a characterization test. It compares the outputs of the channel models
# for fixed seeds on the CPU with reference outputs stored in reference_outputs/*.npz.
# The references were produced by the implementation itself, so they do not prove
# conformance with TR 38.811; they make every change of numerical results visible.
# A change that alters these outputs must come with a documented explanation, and the
# references are rebuilt with:
#
#     python <path to this file> --regenerate
#
# Environment: reference_outputs/environment.json records the Python, PyTorch, Sionna
# and NumPy versions, the CPU and the thread count with which the references were
# produced. PyTorch does not guarantee identical results across releases or platforms,
# and a different random number stream changes every output, so the comparison is only
# meaningful in that environment; elsewhere a failure does not by itself indicate a
# model change.
#
# Tolerance: |actual - reference| <= RTOL * |reference| + ATOL_SCALE * max|reference|,
# evaluated per stored array, with RTOL = 1e-5 and ATOL_SCALE = 1e-6. The computation
# runs in single precision (machine epsilon 1.2e-7), so RTOL allows about 80 units in
# the last place. This absorbs small rounding differences, for example from another
# thread count, but not a model change: a shift of 0.002 dB in a 180 dB path loss, or
# of 0.001 % in a delay, already fails. The absolute term only matters for entries that
# are close to zero, such as the powers of unused clusters; for the normalized time
# correlations, which are bounded by 1, it is ATOL_SCALE itself. LoS states must be
# equal.
import json
import os
import platform
import sys
import unittest

import numpy as np
import sionna
import torch
from sionna.phy import config

from openntn import Antenna, AntennaArray, DenseUrban, SubUrban, Urban
from openntn import utils

REFERENCE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference_outputs")
ENVIRONMENT_FILE = os.path.join(REFERENCE_DIR, "environment.json")

RTOL = 1e-5
ATOL_SCALE = 1e-6

# Small topologies keep the reference files small; the coverage comes from the grid of
# scenarios, bands, directions and elevation angles.
BATCH_SIZE = 2
NUM_UT = 3
BS_HEIGHT = 600000.0
NUM_TIME_SAMPLES = 14
SAMPLING_FREQUENCY = 30e3

# The seeds are fixed per stage so that a change in the number of random draws of one
# stage (for example the topology) does not shift the random streams of the next one.
SEED_TOPOLOGY = 7
SEED_CHANNEL = 11

SCENARIOS = {
    "dense_urban": (DenseUrban, "dur"),
    "urban": (Urban, "urb"),
    "sub_urban": (SubUrban, "sur"),
}
# Same carrier frequencies as the other tests of the suite.
CARRIER_FREQUENCY = {
    ("s", "downlink"): 2.2e9,
    ("s", "uplink"): 2.0e9,
    ("ka", "downlink"): 20.0e9,
    ("ka", "uplink"): 30.0e9,
}
BANDS = ("s", "ka")
DIRECTIONS = ("downlink", "uplink")
ELEVATION_ANGLES = (10.0, 30.0, 60.0, 90.0)

LSP_NAMES = ("ds", "asd", "asa", "sf", "k_factor", "zsa", "zsd")


def _case_key(band, direction, elevation_angle):
    return f"{band}_{'dl' if direction == 'downlink' else 'ul'}_{int(elevation_angle)}"


def _arrays(carrier_frequency):
    ut_array = Antenna(polarization="single",
                       polarization_type="V",
                       antenna_pattern="38.901",
                       carrier_frequency=carrier_frequency)
    bs_array = AntennaArray(num_rows=1,
                            num_cols=4,
                            polarization="dual",
                            polarization_type="VH",
                            antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency)
    return ut_array, bs_array


def _np(x):
    return x.detach().cpu().numpy()


def compute_case(scenario, band, direction, elevation_angle):
    """Return the characterization outputs of one configuration as NumPy arrays."""
    model_class, scenario_key = SCENARIOS[scenario]
    carrier_frequency = CARRIER_FREQUENCY[(band, direction)]

    config.seed = SEED_TOPOLOGY
    ut_array, bs_array = _arrays(carrier_frequency)
    model = model_class(carrier_frequency=carrier_frequency,
                        ut_array=ut_array,
                        bs_array=bs_array,
                        direction=direction,
                        elevation_angle=elevation_angle,
                        enable_pathloss=True,
                        enable_shadow_fading=True)
    topology = utils.gen_single_sector_topology(batch_size=BATCH_SIZE,
                                                num_ut=NUM_UT,
                                                scenario=scenario_key,
                                                elevation_angle=elevation_angle,
                                                bs_height=BS_HEIGHT)
    model.set_topology(*topology)

    sc = model._scenario
    out = {
        "los": _np(sc.los),
        "distance_3d": _np(sc.distance_3d),
        "pl_free_space": _np(sc.free_space_pathloss),
        "pl_basic": _np(sc.basic_pathloss),
        "pl_gas": _np(sc.gas_pathloss),
        "pl_scintillation": _np(sc.scintillation_pathloss),
        "pl_entry": _np(sc.entry_pathloss),
        "pl_total": _np(model._lsp_sampler.sample_pathloss()),
    }
    for name in LSP_NAMES:
        out["lsp_" + name] = _np(getattr(model._lsp, name))

    config.seed = SEED_CHANNEL
    h, tau = model(NUM_TIME_SAMPLES, SAMPLING_FREQUENCY)
    # h: [batch, num_rx, num_rx_ant, num_tx, num_tx_ant, num_paths, num_time_samples]
    power = torch.abs(h.to(torch.complex128)) ** 2
    out["tau"] = _np(tau)
    # Mean power per path over antennas and time: the power delay profile, path loss
    # included.
    out["path_power"] = _np(power.mean(dim=(2, 4, 6)))
    # Power per antenna pair, summed over paths and averaged over time: depends on the
    # antenna patterns, the polarization and the angles.
    out["antenna_power"] = _np(power.sum(dim=5).mean(dim=-1))
    # Complex correlation between the last and the first time sample of each link,
    # normalized: its phase depends on the Doppler shifts. Real and imaginary parts are
    # stored instead of the phase, which would jump at +-pi.
    h64 = h.to(torch.complex128)
    first = h64[..., 0]
    last = h64[..., -1]
    corr = (last * first.conj()).sum(dim=(2, 4, 5))
    norm = torch.sqrt((torch.abs(first) ** 2).sum(dim=(2, 4, 5))
                      * (torch.abs(last) ** 2).sum(dim=(2, 4, 5)))
    out["time_correlation_re"] = _np((corr / norm).real)
    out["time_correlation_im"] = _np((corr / norm).imag)
    return out


def compute_all(scenario):
    """Characterization outputs of all configurations of one scenario."""
    previous_device = config.device
    config.device = "cpu"
    try:
        results = {}
        for band in BANDS:
            for direction in DIRECTIONS:
                for elevation_angle in ELEVATION_ANGLES:
                    case = _case_key(band, direction, elevation_angle)
                    for name, value in compute_case(scenario, band, direction,
                                                    elevation_angle).items():
                        results[f"{case}__{name}"] = value
        return results
    finally:
        config.device = previous_device


def _cpu_model():
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine()


def environment():
    """The software and hardware that determine the reference outputs."""
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "sionna": sionna.__version__,
        "numpy": np.__version__,
        "cpu": _cpu_model(),
        "machine": platform.machine(),
        "system": f"{platform.system()} {platform.release()}",
        "torch_num_threads": torch.get_num_threads(),
        "device": "cpu",
        "precision": "single",
    }


def _describe(env):
    return ", ".join(f"{k} {v}" for k, v in env.items())


def regenerate():
    os.makedirs(REFERENCE_DIR, exist_ok=True)
    for scenario in SCENARIOS:
        results = compute_all(scenario)
        path = os.path.join(REFERENCE_DIR, scenario + ".npz")
        np.savez_compressed(path, **results)
        print(f"wrote {path} ({len(results)} arrays)")
    with open(ENVIRONMENT_FILE, "w", encoding="utf-8") as f:
        json.dump(environment(), f, indent=1)
        f.write("\n")
    print(f"wrote {ENVIRONMENT_FILE}")


class ReferenceOutputs(unittest.TestCase):

    def _check_scenario(self, scenario):
        path = os.path.join(REFERENCE_DIR, scenario + ".npz")
        with np.load(path) as stored:
            reference = {k: stored[k] for k in stored.files}
        with open(ENVIRONMENT_FILE, encoding="utf-8") as f:
            produced_with = _describe(json.load(f))
        actual = compute_all(scenario)
        self.assertEqual(sorted(actual), sorted(reference),
                         "the set of stored arrays changed")
        for key in sorted(reference):
            with self.subTest(array=key):
                ref = reference[key]
                act = actual[key]
                self.assertEqual(act.shape, ref.shape)
                if ref.dtype == bool:
                    np.testing.assert_array_equal(act, ref)
                    continue
                if key.endswith(("_re", "_im")):
                    # Normalized correlations are bounded by 1, which sets their scale.
                    scale = 1.0
                else:
                    finite = np.abs(ref[np.isfinite(ref)])
                    scale = finite.max() if finite.size else 0.0
                atol = ATOL_SCALE * scale
                np.testing.assert_allclose(
                    act, ref, rtol=RTOL, atol=atol, equal_nan=True,
                    err_msg=(f"{scenario}/{key} differs from the reference "
                             f"(produced with {produced_with}; now {_describe(environment())})"))

    def test_dense_urban(self):
        self._check_scenario("dense_urban")

    def test_urban(self):
        self._check_scenario("urban")

    def test_sub_urban(self):
        self._check_scenario("sub_urban")


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        regenerate()
    else:
        unittest.main()
