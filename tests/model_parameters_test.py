# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# This file compares every parameter of the scenario parameter files in openntn/models
# with the tables of 3GPP TR 38.811 V15.4.0:
#   - Tables 6.7.2-1a to 6.7.2-6b: large scale parameters, cross-correlations, delay
#     scaling, XPR, number of clusters and rays, cluster spreads, per cluster
#     shadowing and correlation distances;
#   - Tables 6.6.2-1 to 6.6.2-3: shadow fading standard deviation and clutter loss;
#   - Table 6.6.1-1: LOS probability.
# The angular scaling factors C_phi^NLOS and C_theta^NLOS of Tables 6.7.2-1aa and
# 6.7.2-1ab depend on the number of clusters, which changes with the elevation angle and
# the link state. They are not part of the parameter files; the scenario selects them in
# get_param, which is checked against the tables for every scenario, band, direction,
# link state and elevation angle.
# The table values are stored as printed in the document in
# spec_tables/tr38811_v15.4.0.json. Values the tables give as N/A, or rows they do not
# have, are not specified; for those only the presence of the parameter is checked.
# This concerns only the K-factor rows of the NLOS files (mean, standard deviation and
# cross-correlations, which the files set to 0, and the correlation distance, 1 m).
#
# All 24 files hold the tables as printed, in both directions. NOTE 8 of Tables 6.7.2-1a
# to 6.7.2-8b ("For satellite (e.g. GEO/LEO), the departure angle spreads are zeros, i.e.
# mu_lgASD and mu_lgZSD are -inf, and corresponding standard deviations are zeros") is
# applied by the scenario for satellites (get_param), not in the files; aerial platforms
# use the table values.
#
# The files CDL-A to CDL-D and TDL-A to TDL-D hold the link-level models NTN-CDL-A to D
# and NTN-TDL-A to D of TR 38.811 V15.4.0, Tables 6.9.1-1 to 6.9.1-4 and 6.9.2-1 to
# 6.9.2-4, given at the elevation angle alpha_model = 50 degrees, in the format of
# Sionna's TR 38.901 CDL and TDL files. Every value of these files is compared with the
# table cell it comes from: the normalized delays, powers, AOD, AOA, ZOD and ZOA of the
# clusters, the per-cluster parameters c_ASD, c_ASA, c_ZSD, c_ZSA and XPR, and for the
# TDL files the normalized delays and powers of the taps. The files do not hold the
# columns "Cluster #" and "Tap #" (the order of the entries), "Cluster PAS" and "Fading
# distribution" (the flag los marks a first entry that is the specular LOS path, and
# num_clusters counts the other entries, as in Sionna's files), nor the K-factor of the
# NOTE of Tables 6.9.2-3 and 6.9.2-4, which equals the power difference of the first two
# taps. Clause 6.9.1 writes "resulting in ZODs equal to 190" while the tables print a
# ZOD of 140 degrees, 90 degrees + alpha_model as the same clause defines it; the files
# hold 140. No model of the package reads these files.
import json
import math
import os
import re

import pytest
import torch

import openntn
from openntn import Antenna, AntennaArray, DenseUrban, SubUrban, Urban
from openntn import utils

MODELS_DIR = os.path.join(os.path.dirname(openntn.__file__), "models")
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_tables",
                       "tr38811_v15.4.0.json"), encoding="ascii") as _f:
    SPEC = json.load(_f)
TABLES = SPEC["tables"]
ELEVATIONS = SPEC["elevation_angles_deg"]

FILE_PATTERN = re.compile(r"^(Dense_Urban|Urban|Sub_Urban)_(LOS|NLOS)_(S|Ka)_band_(UL|DL)\.json$")
# Scenario -> (column of Table 6.6.1-1, parameter table per (state, band))
SCENARIOS = {
    "Dense_Urban": ("dense_urban", {("LOS", "S"): "6.7.2-1a", ("LOS", "Ka"): "6.7.2-1b",
                                    ("NLOS", "S"): "6.7.2-2a", ("NLOS", "Ka"): "6.7.2-2b"}),
    "Urban": ("urban", {("LOS", "S"): "6.7.2-3a", ("LOS", "Ka"): "6.7.2-3b",
                        ("NLOS", "S"): "6.7.2-4a", ("NLOS", "Ka"): "6.7.2-4b"}),
    "Sub_Urban": ("suburban_rural", {("LOS", "S"): "6.7.2-5a", ("LOS", "Ka"): "6.7.2-5b",
                                     ("NLOS", "S"): "6.7.2-6a", ("NLOS", "Ka"): "6.7.2-6b"}),
}
# The NTN-CDL and NTN-TDL files and their tables
NTN_FILES = {"CDL-A.json": "6.9.1-1", "CDL-B.json": "6.9.1-2", "CDL-C.json": "6.9.1-3",
             "CDL-D.json": "6.9.1-4", "TDL-A.json": "6.9.2-1", "TDL-B.json": "6.9.2-2",
             "TDL-C.json": "6.9.2-3", "TDL-D.json": "6.9.2-4"}
# Other files in openntn/models. They hold no TR 38.811 table values and are not read by
# the models.
NOT_COMPARED = {
    "TDL-A30.json": "TDL profile not defined in TR 38.811",
    "TDL-B100.json": "TDL profile not defined in TR 38.811",
    "TDL-C300.json": "TDL profile not defined in TR 38.811",
}

# Model parameter -> row of Tables 6.7.2-xx
TABLE_ROWS = {
    "muDS": "mu_lgDS", "sigmaDS": "sigma_lgDS", "muASD": "mu_lgASD", "sigmaASD": "sigma_lgASD",
    "muASA": "mu_lgASA", "sigmaASA": "sigma_lgASA", "muZSA": "mu_lgZSA", "sigmaZSA": "sigma_lgZSA",
    "muZSD": "mu_lgZSD", "sigmaZSD": "sigma_lgZSD", "muK": "mu_K", "sigmaK": "sigma_K",
    "muXPR": "mu_XPR", "sigmaXPR": "sigma_XPR", "rTau": "r_tau", "numClusters": "N",
    "zeta": "zeta", "cDS": "c_DS", "cASD": "c_ASD", "cASA": "c_ASA", "cZSA": "c_ZSA",
}
for _lsp in ("DS", "ASD", "ASA", "SF", "K", "ZSA", "ZSD"):
    TABLE_ROWS["corrDist" + _lsp] = "corr_dist_" + _lsp
for _a, _b in (("ASD", "DS"), ("ASA", "DS"), ("ASA", "SF"), ("ASD", "SF"), ("DS", "SF"),
               ("ASD", "ASA"), ("ASD", "K"), ("ASA", "K"), ("DS", "K"), ("SF", "K"),
               ("ZSD", "SF"), ("ZSA", "SF"), ("ZSD", "K"), ("ZSA", "K"), ("ZSD", "DS"),
               ("ZSA", "DS"), ("ZSD", "ASD"), ("ZSA", "ASD"), ("ZSD", "ASA"), ("ZSA", "ASA"),
               ("ZSD", "ZSA")):
    TABLE_ROWS[f"corr{_a}vs{_b}"] = f"corr_{_a}_vs_{_b}"
PER_ELEVATION = set(TABLE_ROWS) | {"sigmaSF", "CL", "LoS_p"}
# Selected in the code, not stored in the parameter files
SCALING_FACTORS = {"CPhiNLoS": "6.7.2-1aa", "CThetaNLoS": "6.7.2-1ab"}

NOT_SPECIFIED = "not specified"


def _parts(name):
    match = FILE_PATTERN.match(name)
    assert match, name
    return match.groups()


def _files(scenario, state):
    return [f"{scenario}_{state}_{band}_band_{direction}.json"
            for band in ("S", "Ka") for direction in ("UL", "DL")]


def _parse(text):
    """Value as printed in the table (or stored in the model file) as a float."""
    if isinstance(text, str):
        text = text.strip().replace("\u2212", "-")
        if text == "N/A":
            return NOT_SPECIFIED
        if text.endswith("%"):
            return float(text[:-1]) / 100.0
    return float(text)


def _model_files():
    return sorted(f for f in os.listdir(MODELS_DIR) if f.endswith(".json"))


def _scenario_files():
    return [f for f in _model_files() if FILE_PATTERN.match(f)]


def _load(name):
    with open(os.path.join(MODELS_DIR, name), encoding="utf-8") as f:
        return json.load(f)


def _expected_keys(name):
    state = _parts(name)[1]
    per_elevation = PER_ELEVATION - ({"CL"} if state == "LOS" else set())
    return {f"{k}_{e}" for k in per_elevation for e in ELEVATIONS}


def _parameters(name):
    state = _parts(name)[1]
    return sorted(PER_ELEVATION - ({"CL"} if state == "LOS" else set()))


def _table_values(name, parameter):
    """Values of the tables per elevation angle (a float or NOT_SPECIFIED each)."""
    scenario, state, band, _ = _parts(name)
    los_column, parameter_tables = SCENARIOS[scenario]
    table = TABLES[parameter_tables[(state, band)]]
    if parameter in TABLE_ROWS:
        row = table["rows"].get(TABLE_ROWS[parameter])
        return [NOT_SPECIFIED] * 9 if row is None else [_parse(v) for v in row]
    if parameter in ("sigmaSF", "CL"):
        column = f"{band}_band_{state}_{'sigma_SF' if parameter == 'sigmaSF' else 'CL'}"
        return [_parse(v) for v in TABLES[table["sigma_SF_table"]]["rows"][column]]
    if parameter == "LoS_p":
        return [_parse(v) for v in TABLES["6.6.1-1"]["rows"][los_column]]
    raise KeyError(parameter)


def scaling_factor(scenario, state, band, parameter, elevation):
    """C_phi^NLOS or C_theta^NLOS of Tables 6.7.2-1aa and 6.7.2-1ab for the number of
    clusters N of the parameter table of the scenario, link state and band."""
    table = TABLES[SCENARIOS[scenario][1][(state, band)]]
    num_clusters = table["rows"]["N"][ELEVATIONS.index(elevation)]
    scaling = TABLES[SCALING_FACTORS[parameter]]
    factors = dict(zip(scaling["number_of_clusters"], next(iter(scaling["rows"].values()))))
    return _parse(factors[num_clusters])


def expected_values(name, parameter):
    """Expected values per elevation angle (a float or NOT_SPECIFIED each)."""
    return _table_values(name, parameter)


def actual_values(name, parameter):
    data = _load(name)
    return [_parse(data[f"{parameter}_{e}"]) for e in ELEVATIONS]


def _equal(actual, expected):
    if math.isinf(expected) or math.isinf(actual):
        return actual == expected
    return math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12)


def _check(name, parameter, elevations):
    actual = actual_values(name, parameter)
    expected = expected_values(name, parameter)
    errors = []
    for i, e in enumerate(ELEVATIONS):
        if e not in elevations or expected[i] is NOT_SPECIFIED:
            continue
        if not _equal(actual[i], expected[i]):
            errors.append(f"{e} deg: model {actual[i]}, table {expected[i]}")
    assert not errors, f"{name} {parameter}: " + "; ".join(errors)


def test_model_files_accounted_for():
    files = _model_files()
    scenario_files = _scenario_files()
    assert len(scenario_files) == 3 * 2 * 2 * 2
    unknown = sorted(set(files) - set(scenario_files) - set(NTN_FILES) - set(NOT_COMPARED))
    assert not unknown, f"model files that this test does not know: {unknown}"


def ntn_mismatches(name, data):
    """Differences between the values of an NTN-CDL or NTN-TDL file and its table."""
    table = TABLES[NTN_FILES[name]]
    rows = table["rows"]
    # Tables 6.9.1-1 and 6.9.1-2 have no "Cluster PAS" column: NLOS models
    first = (rows.get("cluster_pas") or rows.get("fading_distribution") or [""])[0]
    los = first in ("Specular(LOS path)", "LOS path")
    errors = []

    def compare(key, actual, expected):
        if not _equal(_parse(actual), _parse(expected)):
            errors.append(f"{key}: file {actual}, table {expected}")

    if data["los"] != int(los):
        errors.append(f"los: file {data['los']}, table {int(los)}")
    entries = len(rows["normalized_delay"])
    if data["num_clusters"] != entries - int(los):
        errors.append(f"num_clusters: file {data['num_clusters']}, table {entries - int(los)}")
    columns = {"delays": "normalized_delay", "powers": "power_dB"}
    if "cluster" in rows:
        columns.update({"aod": "AOD_deg", "aoa": "AOA_deg", "zod": "ZOD_deg",
                        "zoa": "ZOA_deg"})
        for key, cell in (("cASD", "c_ASD_deg"), ("cASA", "c_ASA_deg"), ("cZSD", "c_ZSD_deg"),
                          ("cZSA", "c_ZSA_deg"), ("xpr", "XPR_dB")):
            compare(key, data[key], table["per_cluster"][cell])
    else:
        if not data["scale_delays"]:
            errors.append("scale_delays: the table gives normalized delays")
        if los:
            # NOTE: "The first tap follows a Ricean distribution with a K-factor of K1"
            k_factor = float(re.search(r"K1 = (-?\d+\.\d+) dB", table["note"]).group(1))
            compare("K-factor of the NOTE", _parse(rows["power_dB"][0])
                    - _parse(rows["power_dB"][1]), k_factor)
    for key, column in columns.items():
        if len(data[key]) != entries:
            errors.append(f"{key}: {len(data[key])} entries, table {entries}")
            continue
        for i, (actual, expected) in enumerate(zip(data[key], rows[column])):
            compare(f"{key}[{i}]", actual, expected)
    unexpected = set(data) - {"los", "num_clusters", "scale_delays", "cASD", "cASA", "cZSD",
                              "cZSA", "xpr"} - set(columns)
    if unexpected:
        errors.append(f"keys without a table cell: {sorted(unexpected)}")
    return errors


@pytest.mark.parametrize("name", sorted(NTN_FILES))
def test_ntn_link_level_file(name):
    errors = ntn_mismatches(name, _load(name))
    assert not errors, f"{name} against Table {NTN_FILES[name]}: " + "; ".join(errors)


@pytest.mark.parametrize("name", _scenario_files())
def test_parameter_names(name):
    keys = set(_load(name))
    assert not keys & set(SCALING_FACTORS), "the scaling factors are selected in the code"
    expected = _expected_keys(name)
    assert keys == expected, (f"missing: {sorted(expected - keys)}, "
                              f"unexpected: {sorted(keys - expected)}")


@pytest.mark.parametrize("name,parameter",
                         [(n, p) for n in _scenario_files() for p in _parameters(n)])
def test_parameter(name, parameter):
    _check(name, parameter, ELEVATIONS)


MODEL_CLASSES = {"Dense_Urban": (DenseUrban, "dur"), "Urban": (Urban, "urb"),
                 "Sub_Urban": (SubUrban, "sur")}
CARRIER_FREQUENCIES = {("S", "DL"): 2.2e9, ("S", "UL"): 2.0e9, ("Ka", "DL"): 20.0e9,
                       ("Ka", "UL"): 30.0e9}


@pytest.mark.parametrize("scenario,band,direction,elevation", [
    (s, b, d, e) for s in SCENARIOS for b in ("S", "Ka") for d in ("UL", "DL")
    for e in ELEVATIONS])
def test_scaling_factors(scenario, band, direction, elevation):
    model_class, scenario_key = MODEL_CLASSES[scenario]
    carrier_frequency = CARRIER_FREQUENCIES[(band, direction)]
    antenna = Antenna(polarization="single", polarization_type="V",
                      antenna_pattern="38.901", carrier_frequency=carrier_frequency)
    model = model_class(carrier_frequency=carrier_frequency, ut_array=antenna,
                        bs_array=antenna,
                        direction="uplink" if direction == "UL" else "downlink",
                        elevation_angle=float(elevation))
    topology = list(utils.gen_single_sector_topology(
        batch_size=1, num_ut=2, scenario=scenario_key, elevation_angle=float(elevation),
        bs_height=600000.0))
    topology[5] = torch.zeros_like(topology[5])  # outdoor, so that los applies
    errors = []
    for state in ("LOS", "NLOS"):
        model.set_topology(*topology, los=state == "LOS")
        for parameter in SCALING_FACTORS:
            actual = model._scenario.get_param(parameter).flatten().tolist()
            expected = scaling_factor(scenario, state, band, parameter, elevation)
            if not all(math.isclose(a, expected, rel_tol=1e-6) for a in actual):
                errors.append(f"{state} {parameter}: {actual}, table {expected}")
    assert not errors, "; ".join(errors)


@pytest.mark.parametrize("model_class,scenario", [(DenseUrban, "Dense_Urban"),
                                                  (Urban, "Urban"),
                                                  (SubUrban, "Sub_Urban")])
def test_rays_per_cluster(model_class, scenario):
    # Number of rays per cluster M of Tables 6.7.2-xx; it is not part of the
    # parameter files.
    expected = {int(v) for table in SCENARIOS[scenario][1].values()
                for v in TABLES[table]["rows"]["M"]}
    assert len(expected) == 1
    carrier_frequency = 2.2e9
    ut_array = Antenna(polarization="single", polarization_type="V",
                       antenna_pattern="38.901", carrier_frequency=carrier_frequency)
    bs_array = AntennaArray(num_rows=1, num_cols=4, polarization="dual",
                            polarization_type="VH", antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency)
    model = model_class(carrier_frequency=carrier_frequency, ut_array=ut_array,
                        bs_array=bs_array, direction="downlink", elevation_angle=50.0)
    assert int(model._scenario.rays_per_cluster) == expected.pop()
