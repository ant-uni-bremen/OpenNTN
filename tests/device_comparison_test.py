# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the deterministic quantities of the channel models on the CPU and on a GPU.
#
# The same model built on two devices from one topology computes the same deterministic
# quantities: the 3D distances, the LOS angles and the LOS phase, the path loss
# components, the parameters of the scenario, the correlation factors of the large scale
# parameters, the antenna positions and the antenna fields. (The random quantities are
# checked only statistically, by the other tests: the generators of the CPU and of CUDA
# give different streams for one seed.) In double precision the two devices can still
# differ by rounding, since their kernels and libraries order the same operations
# differently. Every entry is therefore compared with a bound,
#
#     |a - b| <= K * EPS * S,
#
# with EPS = 2**-52, the spacing of double-precision numbers at 1, and S a scale per
# quantity and comparison (one model configuration, or one element pattern at one slant
# angle):
# - the four LOS angles, in degrees: 180. The zenith angle of arrival is 180 degrees minus
#   the zenith angle of departure, so near the zenith it keeps the absolute rounding of a
#   value near 180 degrees on a value near 0, where its own magnitude is no scale;
# - the two correlation factors of the large scale parameters: 1;
# - the antenna fields: the largest magnitude of both components of the pattern at that
#   slant angle, since a component that is zero in exact arithmetic, such as f_theta at
#   the slant angle pi/2, has only rounding as its own magnitude;
# - every other quantity: its largest magnitude in the comparison.
# Magnitudes are taken over the finite entries on both devices; NaN equals NaN, and equal
# infinities are equal.
#
# The bound comes from a measurement in the configurations below: an NVIDIA GeForce RTX
# 4090 (driver 535.309.01, CUDA 12.6) against the CPU of the same machine, with PyTorch
# 2.14.1 and with Sionna 2.0.1 and 2.2.0. 64 of the 76 quantities agreed bit for bit. The
# others differed by at most 4 EPS S (the cross correlation factor); the LOS angles by at
# most 1.42 EPS S, the antenna fields by 1.13 EPS S and the spatial correlation factor by
# 0.5 EPS S. K = 64 leaves a margin of 16 over the largest of these, for other GPUs and
# other versions of CUDA and PyTorch, which can round differently by a few units in the
# last place. K * EPS = 2**-46, about 1.4e-14, so a quantity computed in single precision
# on one of the devices, an error of about 6e-8 S, fails by far. The bound is not widened
# to make a test pass.
#
# The tapped delay line model TDL is compared in the same way, with the same bound, in
# double precision: its delays, its mean powers, the power of its specular path and its
# K-factor, the square roots of its correlation matrices, and the phase factors of the
# Doppler shift of the satellite at the samples of a call. The shift is computed once in
# double precision on the CPU, its phase in double precision on the device of the model.
#
# So is the clustered delay line model CDL: the delays of a call, its delays, its mean
# powers, the power of its specular path and its K-factor, the angles of its rays and of
# its specular path in degrees, which it computes in double precision on the CPU, the
# angles in radians in the precision of the model on its device, and the phase factors
# of the Doppler shift of the satellite.
import math

import pytest
import torch
from sionna.phy import config

from openntn import CDL, TDL, Antenna, AntennaArray, DenseUrban, SubUrban, Urban
from openntn.antenna import AntennaElement
from openntn.utils import exp_corr_mat, gen_single_sector_topology

EPS = 2.0 ** -52
K = 64

# The configurations of the measurement: the three scenarios; both links in the S band
# (2.2 GHz downlink, 2.0 GHz uplink) and in the Ka band (20 GHz downlink, 30 GHz uplink);
# the elevation angles 10, 30, 60 and 90 degrees; a satellite at 600 km; 4 batch examples
# of 4 UTs, the second UT of the first batch example at the location of the first; the
# LOS state forced to LOS and to NLOS in turn: 96 model configurations. Then the four
# element patterns at three slant angles, on a grid of 37 zenith and 73 azimuth angles.
SCENARIOS = ((DenseUrban, "dur"), (Urban, "urb"), (SubUrban, "sur"))
CARRIER_FREQUENCY = {("S", "downlink"): 2.2e9, ("S", "uplink"): 2.0e9,
                     ("Ka", "downlink"): 20.0e9, ("Ka", "uplink"): 30.0e9}
ELEVATION_ANGLES = (10.0, 30.0, 60.0, 90.0)
SATELLITE_HEIGHT = 600e3
BATCH_SIZE = 4
NUM_UT = 4
SEED = 7
PATTERNS = ("omni", "38.901", "aperture", "dlp")
SLANT_ANGLES = (0.0, math.pi / 4, math.pi / 2)

SCENARIO_QUANTITIES = ("distance_3d", "los_zod", "los_zoa", "los_aod", "los_aoa",
                       "los_phase", "free_space_pathloss", "basic_pathloss", "gas_pathloss",
                       "scintillation_pathloss", "entry_pathloss")
LOS_ANGLES = ("los_zod", "los_zoa", "los_aod", "los_aoa")
CORRELATION_FACTORS = ("_cross_lsp_correlation_matrix_sqrt",
                       "_spatial_lsp_correlation_matrix_sqrt")
FIELD_COMPONENTS = ("f_theta", "f_phi")


def _model(scenario, direction, carrier_frequency, elevation_angle, device):
    # A single-polarized UT antenna and a 1 x 4 dual-polarized array at the satellite,
    # both with the element pattern of TR 38.901
    ut_array = Antenna(polarization="single", polarization_type="V",
                       antenna_pattern="38.901", carrier_frequency=carrier_frequency,
                       precision="double", device=device)
    bs_array = AntennaArray(num_rows=1, num_cols=4, polarization="dual",
                            polarization_type="VH", antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency, precision="double",
                            device=device)
    return scenario(carrier_frequency=carrier_frequency, ut_array=ut_array,
                    bs_array=bs_array, direction=direction,
                    elevation_angle=elevation_angle, precision="double", device=device)


def _topology(key, elevation_angle):
    config.seed = SEED
    topology = list(gen_single_sector_topology(
        batch_size=BATCH_SIZE, num_ut=NUM_UT, scenario=key, elevation_angle=elevation_angle,
        bs_height=SATELLITE_HEIGHT, precision="double", device="cpu"))
    ut_loc = topology[0].clone()
    ut_loc[0, 1, :2] = ut_loc[0, 0, :2]
    topology[0] = ut_loc
    return topology


def _parameter_names(scenario):
    # The parameters that the LOS and the NLOS parameter files share, and the scaling
    # factors of the NLOS cluster angles
    def names(parameters):
        return {k.rsplit("_", 1)[0] for k in parameters if k.rsplit("_", 1)[-1].isdigit()}
    return sorted(names(scenario._params_los) & names(scenario._params_nlos)) + \
        ["CPhiNLoS", "CThetaNLoS"]


def _quantities(model):
    scenario = model._scenario
    quantities = {name: getattr(scenario, name) for name in SCENARIO_QUANTITIES}
    for name in _parameter_names(scenario):
        quantities[f"get_param({name})"] = scenario.get_param(name)
    for name in CORRELATION_FACTORS:
        quantities[name] = getattr(model._lsp_sampler, name)
    quantities["ut_array.ant_pos"] = scenario.ut_array.ant_pos
    quantities["bs_array.ant_pos"] = scenario.bs_array.ant_pos
    return quantities


def _model_comparison(scenario, key, band, direction, elevation_angle, los, device):
    # The quantities of the model built on the CPU and on the device, from one topology
    # made on the CPU, which set_topology moves to the device of each model. The LOS
    # state is forced, since it is drawn at random.
    topology = _topology(key, elevation_angle)
    values = []
    for model_device in ("cpu", device):
        model = _model(scenario, direction, CARRIER_FREQUENCY[(band, direction)],
                       elevation_angle, model_device)
        model.set_topology(*topology, los=los)
        assert model._scenario.distance_3d.device == torch.device(model_device)
        values.append(_quantities(model))
    label = (f"{scenario.__name__} {direction} {band} {elevation_angle:.0f} deg "
             f"{'LOS' if los else 'NLOS'}")
    return label, {name: (values[0][name], values[1][name]) for name in values[0]}


def _field_comparison(pattern, slant_angle, device):
    # Both components of the field of an element on the CPU and on the device, on a grid
    # made on the CPU
    theta = torch.linspace(0.0, math.pi, 37, dtype=torch.float64)
    phi = torch.linspace(-math.pi, math.pi, 73, dtype=torch.float64)
    phi, theta = torch.meshgrid(phi, theta, indexing="xy")
    fields = [AntennaElement(pattern, slant_angle, precision="double",
                             device=element_device).field(theta, phi)
              for element_device in ("cpu", device)]
    return (f"{pattern} slant {slant_angle:.4f}",
            {component: (fields[0][i], fields[1][i])
             for i, component in enumerate(FIELD_COMPONENTS)})


def _comparisons(device):
    for scenario, key in SCENARIOS:
        for band, direction in CARRIER_FREQUENCY:
            for elevation_angle in ELEVATION_ANGLES:
                for los in (True, False):
                    yield _model_comparison(scenario, key, band, direction,
                                            elevation_angle, los, device)
    for pattern in PATTERNS:
        for slant_angle in SLANT_ANGLES:
            yield _field_comparison(pattern, slant_angle, device)


def _magnitude(*tensors):
    values = torch.cat([t.detach().to("cpu", torch.float64).abs().flatten()
                        for t in tensors])
    values = values[torch.isfinite(values)]
    return float(values.max()) if values.numel() else 0.0


def _scale(name, pairs):
    if name in LOS_ANGLES:
        return 180.0
    if name in CORRELATION_FACTORS:
        return 1.0
    if name in FIELD_COMPONENTS:
        return _magnitude(*pairs["f_theta"], *pairs["f_phi"])
    return _magnitude(*pairs[name])


def _largest_difference(a, b):
    # A NaN against a number counts as an infinite difference
    a = a.detach().to("cpu", torch.float64)
    b = b.detach().to("cpu", torch.float64)
    assert a.shape == b.shape
    same = (a == b) | (torch.isnan(a) & torch.isnan(b))
    if bool(same.all()):
        return 0.0
    return float((a - b).abs()[~same].nan_to_num(nan=math.inf).max())


def _beyond_the_bound(comparisons):
    # The quantities, by comparison, with an entry that differs by more than K * EPS * S
    found = []
    for label, pairs in comparisons:
        for name, (a, b) in pairs.items():
            bound = K * EPS * _scale(name, pairs)
            difference = _largest_difference(a, b)
            if difference > bound:
                found.append(f"{name} ({label}): largest |a - b| {difference:.3g}, "
                             f"bound {bound:.3g}")
    return found


@pytest.mark.slow
def test_cpu_against_itself():
    # The same model built twice on the CPU, and each element twice: no entry beyond the
    # bound
    comparisons = list(_comparisons("cpu"))
    assert len(comparisons) == 96 + 12
    found = _beyond_the_bound(comparisons)
    assert not found, "\n".join(found)


@pytest.mark.parametrize("name", ["distance_3d", "los_zoa",
                                  "_cross_lsp_correlation_matrix_sqrt", "f_theta"])
@pytest.mark.parametrize("multiple,beyond", [(2.0, True), (0.5, False)])
def test_perturbed_entry(name, multiple, beyond):
    # One entry perturbed by 2 * K * EPS * S is beyond the bound, one perturbed by
    # K * EPS * S / 2 is within it. At 90 degrees the zenith angle of arrival is close to
    # 0 degrees, and its scale is still 180 degrees; f_theta of the co-phased dual linear
    # polarized pattern ("dlp") at the slant angle pi/2 is zero in exact arithmetic, and
    # its scale comes from f_phi.
    if name == "f_theta":
        label, pairs = _field_comparison("dlp", math.pi / 2, "cpu")
    else:
        label, pairs = _model_comparison(Urban, "urb", "S", "downlink", 90.0, True, "cpu")
    assert not _beyond_the_bound([(label, pairs)])
    a, b = pairs[name]
    perturbed = b.detach().clone().reshape(-1)
    perturbed[0] += multiple * K * EPS * _scale(name, pairs)
    pairs[name] = (a, perturbed.reshape(b.shape))
    found = _beyond_the_bound([(label, pairs)])
    assert [entry.split(" (")[0] for entry in found] == ([name] if beyond else [])


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU available")
def test_cpu_against_gpu():
    # The same model built on the CPU and on cuda:0, and each element on both: every
    # entry within the bound
    comparisons = list(_comparisons("cuda:0"))
    assert len(comparisons) == 96 + 12
    found = _beyond_the_bound(comparisons)
    assert not found, "\n".join(found)


# The tapped delay line model: the four models; the S band and the Ka band; satellites
# at 600 km and 1200 km, a geostationary one and an aerial platform at 20 km with a
# given shift; desired K-factors; 2 x 2 antennas with receive and transmit correlation;
# the phase factors of the shift at 2048 samples at 30.72 MHz
TDL_CONFIGURATIONS = (
    ("A", 100e-9, 2.0e9, 50.0, 600e3, {}),
    ("B", 300e-9, 30.0e9, 30.0, 1200e3, {}),
    ("C", 50e-9, 2.0e9, 70.0, 600e3, {"k_factor": 3.0}),
    ("D", 30e-9, 20.0e9, 10.0, 35786e3, {"k_factor": 3.0}),
    ("A", 100e-9, 30.0e9, 10.0, 600e3, {"num_rx_ant": 2, "num_tx_ant": 2,
                                         "rx_corr_mat": 0.5, "tx_corr_mat": 0.3}),
    ("D", 100e-9, 3.5e9, 40.0, 20e3, {"doppler_shift": -500.0}),
)
TDL_SAMPLES = 2048
TDL_SAMPLING_FREQUENCY = 30.72e6


def _tdl_quantities(tdl):
    quantities = {"delays": tdl.delays, "mean_powers": tdl.mean_powers}
    if tdl.los:
        quantities["mean_power_los"] = tdl.mean_power_los
        quantities["k_factor"] = tdl.k_factor
    for name in ("_rx_corr_mat_sqrt", "_tx_corr_mat_sqrt"):
        if getattr(tdl, name, None) is not None:
            quantities[name + ".real"] = getattr(tdl, name).real
            quantities[name + ".imag"] = getattr(tdl, name).imag
    factor = tdl._doppler_shift_factor(TDL_SAMPLES, TDL_SAMPLING_FREQUENCY)
    quantities["shift_factor.real"] = factor.real
    quantities["shift_factor.imag"] = factor.imag
    return quantities


def _tdl_comparison(model, delay_spread, carrier_frequency, elevation_angle, height,
                    options, device):
    # The model built on the CPU and on the device, the correlation matrices of
    # exp_corr_mat made on each
    values = []
    for model_device in ("cpu", device):
        built = dict(options)
        for name in ("rx_corr_mat", "tx_corr_mat"):
            if name in built:
                built[name] = exp_corr_mat(built[name], 2, precision="double",
                                           device=model_device)
        tdl = TDL(model, delay_spread, carrier_frequency, elevation_angle, height,
                  precision="double", device=model_device, **built)
        values.append(_tdl_quantities(tdl))
    label = (f"TDL {model} {delay_spread * 1e9:g} ns {carrier_frequency / 1e9:g} GHz "
             f"{elevation_angle:g} deg {height / 1e3:g} km {sorted(options)}")
    return label, {name: (values[0][name], values[1][name]) for name in values[0]}


def test_tdl_cpu_against_itself():
    # The tapped delay line model built twice on the CPU: no entry beyond the bound
    comparisons = [_tdl_comparison(*configuration, "cpu")
                   for configuration in TDL_CONFIGURATIONS]
    found = _beyond_the_bound(comparisons)
    assert not found, "\n".join(found)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU available")
def test_tdl_cpu_against_gpu():
    # The tapped delay line model built on the CPU and on cuda:0: every entry within the
    # bound
    comparisons = [_tdl_comparison(*configuration, "cuda:0")
                   for configuration in TDL_CONFIGURATIONS]
    found = _beyond_the_bound(comparisons)
    assert not found, "\n".join(found)


# The clustered delay line model: the four models in both directions; the S band and the
# Ka band; satellites at 600 km and 1200 km, a geostationary one and an aerial platform
# at 20 km with a given shift; desired angular spreads and K-factors; a dual-polarized
# UT antenna and a 1 x 2 dual-polarized array at the satellite; the phase factors of
# the shift at the samples of the tapped delay line model
CDL_CONFIGURATIONS = (
    ("A", 100e-9, 2.0e9, 50.0, 600e3, "downlink", {}),
    ("B", 300e-9, 30.0e9, 30.0, 1200e3, "uplink", {}),
    ("C", 50e-9, 2.0e9, 70.0, 600e3, "downlink",
     {"angular_spreads": (30.0, 10.0), "k_factor": 3.0}),
    ("D", 30e-9, 20.0e9, 10.0, 35786e3, "uplink", {"k_factor": 3.0}),
    ("A", 100e-9, 30.0e9, 10.0, 600e3, "uplink", {"angular_spreads": (45.0, 5.0)}),
    ("D", 100e-9, 3.5e9, 40.0, 20e3, "downlink", {"doppler_shift": -500.0}),
)
CDL_ANGLES = ("aoa", "aod", "zoa", "zod")


def _cdl_quantities(cdl):
    config.seed = SEED
    _, tau = cdl(2, 3, TDL_SAMPLING_FREQUENCY)
    quantities = {"tau": tau, "delays": cdl.delays, "mean_powers": cdl.mean_powers}
    if cdl.los:
        quantities["mean_power_los"] = cdl.mean_power_los
        quantities["k_factor"] = cdl.k_factor
        for name in CDL_ANGLES:
            quantities[f"los_{name}"] = torch.tensor(getattr(cdl, f"los_{name}"),
                                                     dtype=torch.float64)
            quantities[f"_los_{name}"] = getattr(cdl, f"_los_{name}")
    for name in CDL_ANGLES:
        quantities[name] = getattr(cdl, name)
        quantities[f"_{name}"] = getattr(cdl, f"_{name}")
    factor = cdl._doppler_shift_factor(TDL_SAMPLES, TDL_SAMPLING_FREQUENCY)
    quantities["shift_factor.real"] = factor.real
    quantities["shift_factor.imag"] = factor.imag
    return quantities


def _cdl_comparison(model, delay_spread, carrier_frequency, elevation_angle, height,
                    direction, options, device):
    # The model built on the CPU and on the device, with arrays built on each
    values = []
    for model_device in ("cpu", device):
        ut_array = Antenna("dual", "cross", "omni", carrier_frequency,
                           precision="double", device=model_device)
        bs_array = AntennaArray(1, 2, "dual", "VH", "38.901", carrier_frequency,
                                precision="double", device=model_device)
        cdl = CDL(model, delay_spread, carrier_frequency, elevation_angle, height,
                  ut_array=ut_array, bs_array=bs_array, direction=direction,
                  precision="double", device=model_device, **options)
        values.append(_cdl_quantities(cdl))
    label = (f"CDL {model} {direction} {delay_spread * 1e9:g} ns "
             f"{carrier_frequency / 1e9:g} GHz {elevation_angle:g} deg "
             f"{height / 1e3:g} km {sorted(options)}")
    return label, {name: (values[0][name], values[1][name]) for name in values[0]}


def test_cdl_cpu_against_itself():
    # The clustered delay line model built twice on the CPU: no entry beyond the bound
    comparisons = [_cdl_comparison(*configuration, "cpu")
                   for configuration in CDL_CONFIGURATIONS]
    found = _beyond_the_bound(comparisons)
    assert not found, "\n".join(found)


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="No CUDA GPU available")
def test_cdl_cpu_against_gpu():
    # The clustered delay line model built on the CPU and on cuda:0: every entry within
    # the bound
    comparisons = [_cdl_comparison(*configuration, "cuda:0")
                   for configuration in CDL_CONFIGURATIONS]
    found = _beyond_the_bound(comparisons)
    assert not found, "\n".join(found)
