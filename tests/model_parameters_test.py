# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# This file compares every parameter of the scenario parameter files in openntn/models
# with the tables of 3GPP TR 38.811 V15.4.0:
#   - Tables 6.7.2-1a to 6.7.2-6b: large scale parameters, cross-correlations, delay
#     scaling, XPR, number of clusters and rays, cluster spreads, per cluster
#     shadowing and correlation distances;
#   - Tables 6.6.2-1 to 6.6.2-3: shadow fading standard deviation and clutter loss;
#   - Table 6.6.1-1: LOS probability;
#   - Tables 6.7.2-1aa and 6.7.2-1ab: angular scaling factors C_phi^NLOS and
#     C_theta^NLOS for the number of clusters of each elevation angle.
# The table values are stored as printed in the document in
# spec_tables/tr38811_v15.4.0.json. Values the tables give as N/A, or rows they do not
# have, are not specified; for those only the presence of the parameter is checked.
# This concerns only the K-factor rows of the NLOS files (mean, standard deviation and
# cross-correlations, which the files set to 0, and the correlation distance, 1 m).
#
# Downlink convention. NOTE 8 of Tables 6.7.2-1a to 6.7.2-8b: "For satellite (e.g.
# GEO/LEO), the departure angle spreads are zeros, i.e. mu_lgASD and mu_lgZSD are -inf,
# and corresponding standard deviations are zeros." The downlink files apply this note
# (the satellite transmits) and set the cross-correlations of ASD and ZSD to zero,
# which has no effect once their spreads are zero. The uplink files use the tables'
# lgASD and lgZSD rows as printed. This test checks both as they are; whether the
# uplink files should apply NOTE 8 as well is an open question, not a test failure.
#
# Known deviations from the tables are tested separately as strict expected failures.
import json
import math
import os
import re

import pytest

import openntn
from openntn import Antenna, AntennaArray, DenseUrban, SubUrban, Urban

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
# Other files in openntn/models. They hold no TR 38.811 table values and are not read by
# the scenario models; the TDL and CDL classes that would use the TDL and CDL files are
# not implemented.
NOT_COMPARED = {
    "CDL-A.json": "CDL profile of TR 38.901", "CDL-B.json": "CDL profile of TR 38.901",
    "CDL-C.json": "CDL profile of TR 38.901", "CDL-D.json": "CDL profile of TR 38.901",
    "TDL-A.json": "TDL profile of TR 38.901", "TDL-B.json": "TDL profile of TR 38.901",
    "TDL-C.json": "TDL profile of TR 38.901", "TDL-D.json": "TDL profile of TR 38.901",
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
SINGLE_VALUE = {"CPhiNLoS", "CThetaNLoS"}

NOT_SPECIFIED = "not specified"


def _parts(name):
    match = FILE_PATTERN.match(name)
    assert match, name
    return match.groups()


def _files(scenario, state):
    return [f"{scenario}_{state}_{band}_band_{direction}.json"
            for band in ("S", "Ka") for direction in ("UL", "DL")]


# (file, parameter, elevation) -> reason. A known deviation is left out of
# test_parameter and tested in test_known_deviation instead.
KNOWN_DEVIATIONS = {}


def _known(files, parameters, elevations, reason):
    for name in files:
        for parameter in parameters:
            for elevation in elevations:
                KNOWN_DEVIATIONS[(name, parameter, elevation)] = reason


_known(["Sub_Urban_NLOS_Ka_band_UL.json"], ["muZSD"], [80],
       "mu_lgZSD at 80 degrees is -3.30; TR 38.811 Table 6.7.2-6b gives -3.20")
# At these elevation angles the factors in the files belong to a different number of
# clusters than the one of the table.
_SCALING = ["CPhiNLoS", "CThetaNLoS"]
_SCALING_REASON = ("One C_phi^NLOS and one C_theta^NLOS per file, while the number of "
                   "clusters of TR 38.811 Tables 6.7.2-3a to 6.7.2-6b changes with the "
                   "elevation angle; Tables 6.7.2-1aa and 6.7.2-1ab give the factors per "
                   "number of clusters")
_known(_files("Urban", "LOS"), _SCALING, [10], _SCALING_REASON)
_known(_files("Urban", "NLOS"), _SCALING, [70, 80, 90], _SCALING_REASON)
_known(_files("Sub_Urban", "LOS"), _SCALING, [70, 80, 90], _SCALING_REASON)
_known(_files("Sub_Urban", "NLOS"), _SCALING, [60, 70, 80, 90], _SCALING_REASON)
_known(["Urban_NLOS_Ka_band_UL.json", "Urban_NLOS_Ka_band_DL.json"], ["cASA"],
       [30, 40, 50],
       "c_ASA at 30, 40 and 50 degrees is 18.14, 16.04 and 17.86; TR 38.811 "
       "Table 6.7.2-4b gives 16.4, 17.86 and 19.74")


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
    return {f"{k}_{e}" for k in per_elevation for e in ELEVATIONS} | SINGLE_VALUE


def _parameters(name):
    state = _parts(name)[1]
    return sorted((PER_ELEVATION - ({"CL"} if state == "LOS" else set())) | SINGLE_VALUE)


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
    if parameter in SINGLE_VALUE:
        scaling = TABLES["6.7.2-1aa" if parameter == "CPhiNLoS" else "6.7.2-1ab"]
        factors = dict(zip(scaling["number_of_clusters"], next(iter(scaling["rows"].values()))))
        return [_parse(factors[n]) if n in factors else NOT_SPECIFIED for n in table["rows"]["N"]]
    raise KeyError(parameter)


def _note_8(parameter):
    """Value of a downlink parameter under NOTE 8, or None if NOTE 8 does not apply."""
    if parameter in ("muASD", "muZSD"):
        return -math.inf
    if parameter in ("sigmaASD", "sigmaZSD"):
        return 0.0
    if parameter.startswith("corr") and not parameter.startswith("corrDist") \
            and ("ASD" in parameter or "ZSD" in parameter):
        return 0.0
    return None


def expected_values(name, parameter):
    """Expected values per elevation angle (a float or NOT_SPECIFIED each)."""
    values = _table_values(name, parameter)
    if _parts(name)[3] == "DL" and _note_8(parameter) is not None:
        values = [v if v is NOT_SPECIFIED else _note_8(parameter) for v in values]
    return values


def actual_values(name, parameter):
    data = _load(name)
    if parameter in SINGLE_VALUE:
        return [_parse(data[parameter])] * 9
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
    unknown = sorted(set(files) - set(scenario_files) - set(NOT_COMPARED))
    assert not unknown, f"model files that this test does not know: {unknown}"


@pytest.mark.parametrize("name", _scenario_files())
def test_parameter_names(name):
    keys = set(_load(name))
    expected = _expected_keys(name)
    assert keys == expected, (f"missing: {sorted(expected - keys)}, "
                              f"unexpected: {sorted(keys - expected)}")


@pytest.mark.parametrize("name,parameter",
                         [(n, p) for n in _scenario_files() for p in _parameters(n)])
def test_parameter(name, parameter):
    elevations = [e for e in ELEVATIONS if (name, parameter, e) not in KNOWN_DEVIATIONS]
    _check(name, parameter, elevations)


@pytest.mark.parametrize("name,parameter,elevation", [
    pytest.param(n, p, e, marks=pytest.mark.xfail(strict=True, raises=AssertionError,
                                                   reason=reason))
    for (n, p, e), reason in sorted(KNOWN_DEVIATIONS.items())])
def test_known_deviation(name, parameter, elevation):
    _check(name, parameter, [elevation])


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
