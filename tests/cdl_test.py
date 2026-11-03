# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the clustered delay line models NTN-CDL-A to NTN-CDL-D (openntn.CDL) against
# values computed in the test from 3GPP TR 38.811 V15.4.0, clauses 6.8 and 6.9, and
# TR 38.901 V16.1.0, clauses 7.5 and 7.7 and Annex A.
#
# - The tables: at the reference elevation angle of 50 degrees and with the angular
#   spreads of the model, the delays, the powers, normalized to a total power of one
#   with the specular path, the power of the specular path, and the angles of the rays
#   are those of Tables 6.9.1-1 to 6.9.1-4 (spec_tables/tr38811_v15.4.0.json). The rays
#   of a cluster are its angles plus the angular spreads of the clusters times the
#   offsets of TR 38.901 Table 7.5-3 (eq. (7.7-0a)); four rays of the last cluster of
#   NTN-CDL-B lie above 180 degrees in the table and are reflected by the rule of
#   TR 38.901 clause 7.5, step 7, which clause 6.9.1 names.
# - The K-factor of TR 38.901 eq. (7.7.6-2), the power of the specular path over the sum
#   of the powers of all Laplacian clusters, is 10.000 dB for NTN-CDL-C and 7.000 dB for
#   NTN-CDL-D; with a desired K-factor (clause 7.7.6) the powers follow eq. (7.7.6-1)
#   and the RMS delay spread of the delays, the specular path included, is the desired
#   delay spread; the angles do not change.
# - The angles at the elevation angle of the link, by eqs. (6.9-1) and (6.9-2) of
#   TR 38.811 and eq. (7.7-5) of TR 38.901, with the mean angles and the angular spreads
#   of the model of Annex A of TR 38.901 computed in the test over the rays of the table
#   and the specular path: the zenith angles of departure 90 degrees plus the elevation
#   angle; the zenith angles of arrival shifted by the difference of the elevation angle
#   to 50 degrees and reflected, the specular path at 90 degrees minus the elevation
#   angle; with desired angular spreads the scaled angles, the differences of the
#   azimuth angles to their mean wrapped to [-180, 180] degrees.
# - The statistics of generated coefficients, within 4 standard errors of the expected
#   distribution (statistical_checks.py), in both directions, with vertically polarized
#   isotropic antennas, the UT with the orientation [pi, 0, 0] and the satellite with
#   [0, 0, 0], where the polarization of every ray keeps its direction: the mean power
#   of every cluster, below 600 km P_n / (K + 1), and K / (K + 1) more for the first
#   cluster of a LOS model, with P_n the powers of the Laplacian clusters normalized to
#   a sum of one and K = 0 for an NLOS model; from 600 km, with the Faraday rotation
#   psi = 108 / f^2 degrees of TR 38.811 clause 6.8.2, (P_n / (K + 1)) (cos^2 psi +
#   sin^2 psi / kappa) with the XPR kappa of the table, and K / (K + 1) cos^2 psi more
#   for the first cluster. And the time correlation of every cluster of NTN-CDL-A for a
#   moving UT, E[h(t + tau) h*(t)] = P (1/400) sum_i sum_j exp(j 2 pi r(theta_j,
#   phi_i)^T v tau / lambda_0) over the zenith and the azimuth angles of arrival of the
#   rays of the cluster at the UT: the mean of the Doppler term of TR 38.901 clause 7.5,
#   step 11, over the random coupling of the rays (clause 7.7.1, step 2). The standard
#   errors follow from the random phases of the rays and the random coupling.
# - The Doppler shift due to the motion of the satellite of TR 38.811 clause 6.9.2,
#   f_d = (v/c) (R/(R+h)) cos(alpha) f_c with the speed v = sqrt(G M / (R + h)) of a
#   circular orbit and the constants of openntn.utils.compute_satellite_speed, computed
#   in the test: with the UT at rest every cluster of every antenna pair advances by
#   2 pi f_d / f_s per sample; 0 for a geostationary satellite and an aerial platform,
#   the channel constant; a given doppler_shift is used as it is; with a moving UT the
#   coefficients are those of the same draws without the shift times exp(j 2 pi f_d
#   n / f_s).
# - The checks of the arguments and of the antenna arrays, the arguments taken by
#   keyword only, the default orientations, the outputs in both precisions and
#   directions, and the ValueError of a model moved or cast after its construction.
import inspect
import json
import math
import os
import warnings

import numpy as np
import pytest
import torch
from sionna.phy import SPEED_OF_LIGHT, config

from openntn import CDL, Antenna, AntennaArray
from statistical_checks import Z, assert_mean

with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_tables",
                       "tr38811_v15.4.0.json"), encoding="ascii") as _f:
    TABLES = json.load(_f)["tables"]
CDL_TABLES = {"A": "6.9.1-1", "B": "6.9.1-2", "C": "6.9.1-3", "D": "6.9.1-4"}
MODELS = "ABCD"
NUM_CLUSTERS = {"A": 3, "B": 4, "C": 2, "D": 3}
# TR 38.901 V16.1.0, Table 7.5-3: the ray offset angles for an RMS angular spread of 1,
# rays 1 to 20
OFFSETS = np.array([0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715,
                    0.5129, -0.5129, 0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481,
                    1.5195, -1.5195, 2.1551, -2.1551])
ALPHA_MODEL = 50.0

R_EARTH = 6371e3
# G and the mass of the Earth of openntn.utils.compute_satellite_speed
GM = 6.6743e-11 * 5.972e24
DELAY_SPREAD = 100e-9
CARRIER_FREQUENCY = 2e9
ELEVATION_ANGLE = 50.0
SATELLITE_HEIGHT = 600e3
SAMPLING_FREQUENCY = 1e6


def _arrays(carrier_frequency=CARRIER_FREQUENCY, precision="double", dual=True):
    """A UT antenna and a 1 x 2 array at the satellite, dual-polarized, or one
    vertically polarized isotropic antenna at each end"""
    if dual:
        return {"ut_array": Antenna("dual", "cross", "omni", carrier_frequency,
                                    precision=precision),
                "bs_array": AntennaArray(1, 2, "dual", "VH", "38.901",
                                         carrier_frequency, precision=precision)}
    return {"ut_array": Antenna("single", "V", "omni", carrier_frequency,
                                precision=precision),
            "bs_array": Antenna("single", "V", "omni", carrier_frequency,
                                precision=precision)}


def _cdl(model, delay_spread=DELAY_SPREAD, carrier_frequency=CARRIER_FREQUENCY,
         elevation_angle=ELEVATION_ANGLE, satellite_height=SATELLITE_HEIGHT,
         precision="double", dual=True, **kwargs):
    arrays = _arrays(carrier_frequency, precision, dual)
    return CDL(model, delay_spread, carrier_frequency, elevation_angle,
               satellite_height, precision=precision, **{**arrays, **kwargs})


def _table(model):
    """The table of a model: the normalized delays, the powers [dB], the angles of its
    entries [deg], the angular spreads of the clusters [deg], the XPR [dB] and whether
    the first entry is the specular path"""
    table = TABLES[CDL_TABLES[model]]
    rows = table["rows"]
    los = (rows.get("cluster_pas") or [""])[0] == "Specular(LOS path)"
    angles = {name: np.array([float(v) for v in rows[f"{name.upper()}_deg"]])
              for name in ("aod", "aoa", "zod", "zoa")}
    spreads = {name: float(table["per_cluster"][key]) for name, key in
               (("aod", "c_ASD_deg"), ("aoa", "c_ASA_deg"), ("zod", "c_ZSD_deg"),
                ("zoa", "c_ZSA_deg"))}
    return (np.array([float(v) for v in rows["normalized_delay"]]),
            np.array([float(v) for v in rows["power_dB"]]), angles, spreads,
            float(table["per_cluster"]["XPR_dB"]), los)


def _rms_delay_spread(delays, powers):
    mean = np.sum(powers * delays) / np.sum(powers)
    return math.sqrt(np.sum(powers * (delays - mean) ** 2) / np.sum(powers))


def _profile(model, k_factor=None):
    """The delays, the powers normalized to a total of one with the first cluster
    holding the specular path, and the power of the specular path (0 for NLOS models),
    in the test's own computation; with k_factor [dB], TR 38.901 clause 7.7.6"""
    delays, powers, _, _, _, los = _table(model)
    if k_factor is not None:
        k_model = powers[0] - 10 * math.log10(np.sum(10 ** (powers[1:] / 10)))
        powers = np.concatenate([powers[:1], powers[1:] - k_factor + k_model])
        delays = delays / _rms_delay_spread(delays, 10 ** (powers / 10))
    linear = 10 ** (powers / 10)
    linear = linear / linear.sum()
    if los:
        first = linear[0] + linear[1]
        return delays[1:], np.concatenate([[first], linear[2:]]), linear[0]
    return delays, linear, 0.0


def _table_rays(model):
    """The rays of the Laplacian clusters of the table by eq. (7.7-0a) [deg], [clusters,
    20], their powers P_n / 20 with the powers of the table normalized to one, and the
    angles and the power of the specular path"""
    _, powers, angles, spreads, _, los = _table(model)
    first = 1 if los else 0
    linear = 10 ** (powers / 10)
    linear = linear / linear.sum()
    rays = {name: angles[name][first:, None] + spreads[name] * OFFSETS
            for name in angles}
    ray_powers = np.repeat(linear[first:, None] / 20, 20, axis=1)
    specular = ({name: angles[name][0] for name in angles}, linear[0]) if los else None
    return rays, ray_powers, specular


def _annex_a(angles, powers):
    """TR 38.901 Annex A: the mean angle of eq. (A-2) and the RMS angular spread of
    eq. (A-1) [deg]"""
    z = np.sum(powers * np.exp(1j * np.radians(angles))) / np.sum(powers)
    return math.degrees(math.atan2(z.imag, z.real)), math.degrees(
        math.sqrt(-2 * math.log(abs(z))))


def _model_statistics(model, name):
    """The mean angle and the angular spread of the model by Annex A over the rays of
    the table and the specular path"""
    rays, ray_powers, specular = _table_rays(model)
    angles, powers = rays[name].ravel(), ray_powers.ravel()
    if specular is not None:
        angles = np.concatenate([[specular[0][name]], angles])
        powers = np.concatenate([[specular[1]], powers])
    return _annex_a(angles, powers)


def _reflect(theta):
    """TR 38.901 clause 7.5, step 7: wrapped to [0, 360) degrees, and set to 360 degrees
    minus the angle from 180 degrees on"""
    theta = np.mod(theta, 360.0)
    return np.where(theta >= 180.0, 360.0 - theta, theta)


def _wrap(phi):
    """Wrapped to [-180, 180) degrees"""
    return np.mod(phi + 180.0, 360.0) - 180.0


def _expected_angles(model, elevation_angle, spreads=None):
    """The angles of the rays and of the specular path at the elevation angle of the
    link [deg], by eqs. (6.9-1), (6.9-2) and (7.7-5), the mean angles and the angular
    spreads of the model by Annex A, the desired means those of the model"""
    rays, _, specular = _table_rays(model)
    mean_aoa, asa = _model_statistics(model, "aoa")
    mean_zoa, zsa = _model_statistics(model, "zoa")
    desired_asa, desired_zsa = spreads if spreads is not None else (None, zsa)
    delta_alpha = elevation_angle - ALPHA_MODEL

    def zoa(theta):
        return _reflect(desired_zsa / zsa * (theta - mean_zoa) + mean_zoa - delta_alpha)

    def aoa(phi):
        if desired_asa is None:
            return phi
        return desired_asa / asa * _wrap(phi - mean_aoa) + mean_aoa

    expected = {"aod": rays["aod"], "aoa": aoa(rays["aoa"]),
                "zod": np.full_like(rays["zod"], 90.0 + elevation_angle),
                "zoa": zoa(rays["zoa"])}
    if specular is not None:
        expected.update({"los_aod": specular[0]["aod"],
                         "los_aoa": aoa(specular[0]["aoa"]),
                         "los_zod": 90.0 + elevation_angle,
                         "los_zoa": zoa(specular[0]["zoa"])})
    return expected


def _assert_angles(cdl, expected, atol=1e-9):
    for name in ("aod", "aoa", "zod", "zoa"):
        np.testing.assert_allclose(getattr(cdl, name).numpy(), expected[name], rtol=0.0,
                                   atol=atol, err_msg=name)
    if cdl.los:
        for name in ("los_aod", "los_aoa", "los_zod", "los_zoa"):
            expected_angle = float(expected[name])
            assert getattr(cdl, name) == pytest.approx(expected_angle, abs=atol), name


def _satellite_shift(height, elevation_angle, carrier_frequency):
    """f_d of clause 6.9.2; 0 for a geostationary satellite and an aerial platform"""
    if height >= 35786e3 or height <= 50e3:
        return 0.0
    v_sat = math.sqrt(GM / (R_EARTH + height))
    return (v_sat / SPEED_OF_LIGHT * R_EARTH / (R_EARTH + height)
            * math.cos(math.radians(elevation_angle)) * carrier_frequency)


# The tables

@pytest.mark.parametrize("model", MODELS)
def test_model_holds_its_table(model):
    # At 50 degrees with the angular spreads of the model: the powers and the delays of
    # the table, and the rays of eq. (7.7-0a), the zenith angles of arrival reflected
    # into [0, 180] degrees
    delays, powers, specular = _profile(model)
    cdl = _cdl(model)
    assert cdl.los == (model in "CD")
    assert cdl.num_clusters == NUM_CLUSTERS[model] == len(delays)
    np.testing.assert_allclose(cdl.delays.cpu().numpy(), delays * DELAY_SPREAD,
                               rtol=1e-12, atol=0.0)
    np.testing.assert_allclose(cdl.mean_powers.cpu().numpy(), powers, rtol=1e-12,
                               atol=0.0)
    if cdl.los:
        assert float(cdl.mean_power_los) == pytest.approx(specular, rel=1e-12)
    normalized = _cdl(model, normalize_delays=False)
    np.testing.assert_allclose(normalized.delays.cpu().numpy(), delays, rtol=1e-12,
                               atol=0.0)

    rays, _, specular_path = _table_rays(model)
    expected = dict(rays, zoa=_reflect(rays["zoa"]))
    if specular_path is not None:
        expected.update({f"los_{name}": value
                         for name, value in specular_path[0].items()})
    _assert_angles(cdl, expected)
    assert set(np.unique(cdl.zod.numpy())) == {140.0}
    assert set(np.unique(cdl.aod.numpy())) == {0.0}


def test_reflected_rays_of_ntn_cdl_b():
    # The last cluster of NTN-CDL-B, at the zenith angle 174.1 degrees with the cluster
    # ZSA 7 degrees, has four rays above 180 degrees in the table, 174.1 + 7 alpha_m for
    # alpha_m = 0.8844, 1.1481, 1.5195 and 2.1551: they arrive at 360 degrees minus
    # their angle, at their azimuth angle
    zoa = _cdl("B").zoa.numpy()[3]
    above = 174.1 + 7 * OFFSETS
    assert int(np.sum(above > 180.0)) == 4
    np.testing.assert_allclose(zoa[above > 180.0], 360.0 - above[above > 180.0],
                               rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(sorted(zoa[above > 180.0]),
                               [170.8143, 175.2635, 177.8633, 179.7092], atol=1e-4)
    np.testing.assert_allclose(zoa[above <= 180.0], above[above <= 180.0], rtol=0.0,
                               atol=1e-9)


@pytest.mark.parametrize("model,k_model", [("C", 10.000), ("D", 7.000)])
def test_k_factors_of_the_tables(model, k_model):
    # The K-factor of eq. (7.7.6-2), the power of the specular path over the sum of the
    # powers of all Laplacian clusters, which the property k_factor gives in linear
    # scale
    _, powers, _, _, _, los = _table(model)
    assert los
    k_table = powers[0] - 10 * math.log10(np.sum(10 ** (powers[1:] / 10)))
    assert round(k_table, 3) == k_model
    cdl = _cdl(model)
    assert 10 * math.log10(float(cdl.k_factor)) == pytest.approx(k_table, abs=1e-9)


@pytest.mark.parametrize("model", "AB")
def test_specular_path_of_nlos_models(model):
    cdl = _cdl(model)
    for name in ("k_factor", "mean_power_los", "los_aoa", "los_aod", "los_zoa",
                 "los_zod"):
        with pytest.raises(RuntimeError, match=name):
            getattr(cdl, name)


@pytest.mark.parametrize("k_factor", [-3.0, 3.0, 7.95, 13.0])
@pytest.mark.parametrize("model", "CD")
def test_desired_k_factor(model, k_factor):
    # TR 38.901 clause 7.7.6: the powers of eq. (7.7.6-1), the K-factor of
    # eq. (7.7.6-2) equal to the desired one, and the delays renormalized so that the
    # RMS delay spread of the output delays, with the specular path, is delay_spread;
    # the angles are those of the model
    delays, powers, specular = _profile(model, k_factor)
    cdl = _cdl(model, k_factor=k_factor)
    np.testing.assert_allclose(cdl.mean_powers.cpu().numpy(), powers, rtol=1e-12,
                               atol=0.0)
    assert float(cdl.mean_power_los) == pytest.approx(specular, rel=1e-12)
    assert 10 * math.log10(float(cdl.k_factor)) == pytest.approx(k_factor, abs=1e-9)
    _, tau = cdl(2, 1, SAMPLING_FREQUENCY)
    for example in tau.cpu().numpy()[:, 0, 0]:
        np.testing.assert_allclose(example, delays * DELAY_SPREAD, rtol=1e-12, atol=0.0)
        assert _rms_delay_spread(example, powers) == pytest.approx(DELAY_SPREAD,
                                                                   rel=1e-12)
    _assert_angles(cdl, _expected_angles(model, ELEVATION_ANGLE), atol=0.0)


# The angles at the elevation angle of the link

@pytest.mark.parametrize("elevation_angle", [10.0, 30.0, 50.0, 70.0, 90.0])
@pytest.mark.parametrize("model", MODELS)
def test_angles_at_elevation_angle(model, elevation_angle):
    # Eq. (6.9-1): the zenith angle of departure 90 degrees plus the elevation angle;
    # eq. (6.9-2) with the ZSA and the mean ZOA of the model: the zenith angles of
    # arrival shifted by 50 degrees minus the elevation angle and reflected, the
    # specular path at 90 degrees minus the elevation angle; the azimuth angles those of
    # the table
    cdl = _cdl(model, elevation_angle=elevation_angle)
    expected = _expected_angles(model, elevation_angle)
    _assert_angles(cdl, expected)
    if cdl.los:
        assert cdl.los_zoa == pytest.approx(90.0 - elevation_angle, abs=1e-9)
        assert cdl.los_zod == 90.0 + elevation_angle
    assert bool((cdl.zoa >= 0.0).all()) and bool((cdl.zoa <= 180.0).all())


@pytest.mark.parametrize("elevation_angle,spreads", [(50.0, (30.0, 10.0)),
                                                     (20.0, (60.0, 15.0)),
                                                     (80.0, (5.0, 3.0))])
@pytest.mark.parametrize("model", MODELS)
def test_desired_angular_spreads(model, elevation_angle, spreads):
    # Eq. (7.7-5) for the azimuth angles of arrival, the differences to the mean wrapped
    # to [-180, 180] degrees, and eq. (6.9-2) for the zenith angles of arrival, with the
    # desired ASA and ZSA, the desired means those of the model, on the rays and on the
    # specular path; the angles of departure unchanged
    cdl = _cdl(model, elevation_angle=elevation_angle, angular_spreads=spreads)
    _assert_angles(cdl, _expected_angles(model, elevation_angle, spreads))


def test_azimuth_differences_wrapped():
    # The mean AOA of NTN-CDL-A is -176.418 degrees, its first cluster at 178.8 degrees:
    # with half the ASA of the model the cluster moves to -178.809 degrees, while the
    # differences taken without the wrap to [-180, 180] degrees would move it to the
    # other side of the circle, to 1.191 degrees
    mean_aoa, asa = _model_statistics("A", "aoa")
    _, zsa = _model_statistics("A", "zoa")
    assert mean_aoa == pytest.approx(-176.418, abs=1e-3)
    cdl = _cdl("A", angular_spreads=(asa / 2, zsa))
    first = float(np.mean(cdl.aoa.numpy()[0]))
    assert first == pytest.approx(mean_aoa + (_wrap(178.8 - mean_aoa)) / 2, abs=1e-9)
    assert first == pytest.approx(-178.809, abs=1e-3)
    assert mean_aoa + (178.8 - mean_aoa) / 2 == pytest.approx(1.191, abs=1e-3)


@pytest.mark.parametrize("model", "AD")
def test_angles_in_the_frame_of_the_tables(model):
    # The properties give the angles in the coordinate system of the tables, departure
    # at the satellite and arrival at the UT, in both directions
    downlink = _cdl(model, elevation_angle=30.0, angular_spreads=(30.0, 10.0))
    uplink = _cdl(model, elevation_angle=30.0, angular_spreads=(30.0, 10.0),
                  direction="uplink")
    for name in ("aod", "aoa", "zod", "zoa"):
        assert torch.equal(getattr(downlink, name), getattr(uplink, name)), name


def test_default_orientations():
    # By default the satellite has the orientation [0, alpha, 0] and the UT
    # [pi, -alpha, 0], alpha the elevation angle in radians: the outputs equal those of
    # these orientations given, bit for bit. Their boresights, the x-axis of the local
    # coordinate system rotated by TR 38.901 eq. (7.1-4), point along the specular path:
    # the satellite to the zenith angle 90 + alpha degrees at the azimuth 0, the UT to
    # 90 - alpha degrees at the azimuth 180 degrees
    alpha = 30.0
    rad = math.radians(alpha)
    for direction in ("downlink", "uplink"):
        outputs = []
        for orientations in ({}, {"bs_orientation": [0.0, rad, 0.0],
                                  "ut_orientation": [math.pi, -rad, 0.0]}):
            cdl = _cdl("C", elevation_angle=alpha, direction=direction, max_speed=3.0,
                       **orientations)
            config.seed = 3
            outputs.append(cdl(2, 3, SAMPLING_FREQUENCY)[0])
        assert torch.equal(outputs[0], outputs[1]), direction

    def boresight(bearing, downtilt):
        return np.array([math.cos(bearing) * math.cos(downtilt),
                         math.sin(bearing) * math.cos(downtilt), -math.sin(downtilt)])

    def direction_of(zenith, azimuth):
        theta, phi = math.radians(zenith), math.radians(azimuth)
        return np.array([math.sin(theta) * math.cos(phi),
                         math.sin(theta) * math.sin(phi), math.cos(theta)])

    cdl = _cdl("C", elevation_angle=alpha)
    np.testing.assert_allclose(boresight(0.0, rad),
                               direction_of(cdl.los_zod, cdl.los_aod), atol=1e-12)
    np.testing.assert_allclose(boresight(math.pi, -rad),
                               direction_of(cdl.los_zoa, cdl.los_aoa), atol=1e-12)


# The arguments

ARGUMENTS = dict(model="A", delay_spread=DELAY_SPREAD,
                 carrier_frequency=CARRIER_FREQUENCY, elevation_angle=ELEVATION_ANGLE,
                 satellite_height=SATELLITE_HEIGHT, precision="double")
INVALID = [
    (dict(model="E"), ValueError, "model"),
    (dict(model=1), TypeError, "model"),
    (dict(delay_spread=0.0), ValueError, "delay_spread"),
    (dict(delay_spread="100e-9"), TypeError, "delay_spread"),
    (dict(carrier_frequency=10e9), ValueError, "carrier_frequency"),
    (dict(carrier_frequency=None), TypeError, "carrier_frequency"),
    (dict(elevation_angle=9.9), ValueError, "elevation_angle"),
    (dict(elevation_angle=True), TypeError, "elevation_angle"),
    (dict(satellite_height=100e3), ValueError, "satellite_height"),
    (dict(satellite_height="600e3"), TypeError, "satellite_height"),
    (dict(ut_array="omni"), TypeError, "ut_array"),
    (dict(bs_array=None), TypeError, "bs_array"),
    (dict(direction="sideways"), ValueError, "direction"),
    (dict(direction=1), TypeError, "direction"),
    (dict(doppler_shift=math.nan), ValueError, "doppler_shift"),
    (dict(doppler_shift="0"), TypeError, "doppler_shift"),
    (dict(k_factor=3.0), ValueError, "k_factor"),
    (dict(model="C", k_factor=math.inf), ValueError, "k_factor"),
    (dict(model="C", k_factor="3"), TypeError, "k_factor"),
    (dict(angular_spreads=(30.0,)), ValueError, "angular_spreads"),
    (dict(angular_spreads=(30.0, 0.0)), ValueError, "angular_spreads"),
    (dict(angular_spreads=(math.inf, 10.0)), ValueError, "angular_spreads"),
    (dict(angular_spreads="30, 10"), TypeError, "angular_spreads"),
    (dict(angular_spreads=(30.0, "10")), TypeError, "angular_spreads"),
    (dict(ut_orientation=[0.0, 0.0]), ValueError, "ut_orientation"),
    (dict(ut_orientation=[0.0, math.nan, 0.0]), ValueError, "ut_orientation"),
    (dict(ut_orientation="pi 0 0"), TypeError, "ut_orientation"),
    (dict(bs_orientation=torch.zeros(2, 2)), ValueError, "bs_orientation"),
    (dict(bs_orientation=[True, False, True]), TypeError, "bs_orientation"),
    (dict(ut_velocity=[0.0, math.inf, 0.0]), ValueError, "ut_velocity"),
    (dict(ut_velocity=torch.zeros(3, dtype=torch.complex64)), TypeError, "ut_velocity"),
    (dict(min_speed=-1.0), ValueError, "min_speed"),
    (dict(min_speed="0"), TypeError, "min_speed"),
    (dict(min_speed=2.0, max_speed=1.0), ValueError, "max_speed"),
    (dict(max_speed=math.inf), ValueError, "max_speed"),
    (dict(normalize_delays=1), TypeError, "normalize_delays"),
]


@pytest.mark.parametrize("change,error,name", INVALID,
                         ids=[f"{list(c)[-1]}-{c[list(c)[-1]]!r}"
                              for c, _, _ in INVALID])
def test_invalid_argument(change, error, name):
    arguments = {**ARGUMENTS, **_arrays(), **change}
    with pytest.raises(error, match=name):
        CDL(**arguments)


@pytest.mark.parametrize("change", [
    dict(carrier_frequency=1.9e9), dict(carrier_frequency=40e9),
    dict(elevation_angle=10), dict(elevation_angle=90.0),
    dict(satellite_height=160e3), dict(satellite_height=35786e3),
    dict(satellite_height=8e3), dict(satellite_height=50e3),
    dict(min_speed=0, max_speed=0.0), dict(model="D", k_factor=-10),
    dict(doppler_shift=0), dict(angular_spreads=[1e-3, 1e-3]),
    dict(ut_orientation=np.zeros((2, 3)), ut_velocity=torch.ones(2, 3)),
    dict(normalize_delays=False),
])
def test_accepted_boundaries(change):
    arrays = _arrays(change.get("carrier_frequency", CARRIER_FREQUENCY))
    a, _ = CDL(**{**ARGUMENTS, **arrays, **change})(2, 3, SAMPLING_FREQUENCY)
    assert bool(torch.isfinite(a).all())


@pytest.mark.parametrize("problem", ["precision", "device", "carrier_frequency"])
@pytest.mark.parametrize("name", ["ut_array", "bs_array"])
def test_arrays_of_the_model(name, problem):
    # The arrays must be in the precision and on the device of the model and built for
    # its carrier frequency. On the CPU, an array whose device is set to cuda:0 stands
    # in for an array built on a GPU.
    arrays = _arrays()
    message = r"precision\. Build the arrays in the precision of the model"
    if problem == "precision":
        arrays[name] = _arrays(precision="single")[name]
    elif problem == "device":
        arrays[name]._device_str = "cuda:0"
        message = rf"is on device 'cpu', but its {name} \(\w+\) is on device 'cuda:0'"
    else:
        arrays[name] = _arrays(carrier_frequency=2.2e9)[name]
        message = rf"{name} \(\w+\) of CDL is built for another carrier frequency"
    with pytest.raises(ValueError, match=message):
        CDL(**{**ARGUMENTS, **arrays}, device="cpu")


def test_keyword_only_arguments():
    # The arguments after satellite_height are taken by keyword only, so that a call in
    # the order of Sionna's TR 38.901 CDL, which takes ut_array, bs_array and direction
    # after carrier_frequency, raises instead of taking the arrays for the elevation
    # angle and the height
    parameters = inspect.signature(CDL).parameters
    assert [name for name, p in parameters.items()
            if p.kind == p.POSITIONAL_OR_KEYWORD] == ["model", "delay_spread",
                                                       "carrier_frequency",
                                                       "elevation_angle",
                                                       "satellite_height"]
    assert all(p.kind == p.KEYWORD_ONLY for name, p in parameters.items()
               if name not in ("model", "delay_spread", "carrier_frequency",
                               "elevation_angle", "satellite_height"))
    assert parameters["ut_array"].default is inspect.Parameter.empty
    assert parameters["bs_array"].default is inspect.Parameter.empty
    arrays = _arrays()
    ut_array, bs_array = arrays["ut_array"], arrays["bs_array"]
    with pytest.raises(TypeError):
        CDL("A", DELAY_SPREAD, CARRIER_FREQUENCY, ut_array, bs_array)
    with pytest.raises(TypeError):
        CDL("A", DELAY_SPREAD, CARRIER_FREQUENCY, ut_array, bs_array, "downlink")


def test_batch_size_of_the_call():
    # An orientation or a velocity of shape [batch size, 3] fixes the batch size of the
    # call
    for name in ("ut_orientation", "bs_orientation", "ut_velocity"):
        cdl = _cdl("A", **{name: np.zeros((2, 3))})
        cdl(2, 1, SAMPLING_FREQUENCY)
        with pytest.raises(ValueError, match=rf"{name} has the shape \[2, 3\], but the "
                                             rf"batch size of the call is 3"):
            cdl(3, 1, SAMPLING_FREQUENCY)


@pytest.mark.parametrize("direction", ["downlink", "uplink"])
@pytest.mark.parametrize("precision", ["single", "double"])
@pytest.mark.parametrize("model", MODELS)
def test_outputs(model, precision, direction):
    # The UT has 2 antennas and the satellite 4: the receiver is the UT in the downlink
    # and the satellite in the uplink
    cdl = _cdl(model, precision=precision, direction=direction, max_speed=3.0)
    a, tau = cdl(4, 5, SAMPLING_FREQUENCY)
    num_rx_ant, num_tx_ant = (2, 4) if direction == "downlink" else (4, 2)
    num_clusters = NUM_CLUSTERS[model]
    assert a.shape == (4, 1, num_rx_ant, 1, num_tx_ant, num_clusters, 5)
    assert tau.shape == (4, 1, 1, num_clusters)
    real, complex_ = ((torch.float32, torch.complex64) if precision == "single"
                      else (torch.float64, torch.complex128))
    assert (a.dtype, tau.dtype) == (complex_, real)
    assert a.device == tau.device == torch.device(cdl.device)
    assert not a.requires_grad and not tau.requires_grad
    assert bool(torch.isfinite(a).all())
    assert bool((tau == cdl.delays).all())


# A cast or a move to the device meta (it stands in for a GPU), with the precision of
# the model and the dtype and the device of its tensors after the move or the cast
MOVES = (("single", torch.float64, r"torch\.float64 on cpu"),
         ("double", torch.float32, r"torch\.float32 on cpu"),
         ("single", "meta", r"torch\.float32 on meta"))


@pytest.mark.parametrize("precision,move,tensors", MOVES,
                         ids=["single-float64", "double-float32", "single-meta"])
def test_moved_or_cast_model(precision, move, tensors):
    # A model stays on the device and in the precision of its construction: .to() moves
    # or casts its registered tensors, but not its device and its precision, so the next
    # call raises ValueError
    config.device = "cpu"
    cdl = _cdl("C", precision=precision)
    cdl(2, 3, SAMPLING_FREQUENCY)
    with warnings.catch_warnings():
        # A cast to another floating point dtype also casts the complex tensors to real,
        # which PyTorch warns about
        warnings.simplefilter("ignore")
        cdl.to(move)
    with pytest.raises(ValueError, match=rf"was built on device 'cpu' in {precision} "
                                         rf"precision, but its tensors are {tensors}"):
        cdl(2, 3, SAMPLING_FREQUENCY)


# The Doppler shift due to the motion of the satellite

@pytest.mark.parametrize("height,elevation_angle,carrier_frequency", [
    (600e3, 50.0, 2e9), (600e3, 10.0, 30e9), (1200e3, 30.0, 30e9), (600e3, 70.0, 2e9),
    (10000e3, 30.0, 20e9), (160e3, 90.0, 3.5e9),
])
def test_phase_advance_at_rest(height, elevation_angle, carrier_frequency):
    # With the UT at rest, every cluster of every antenna pair advances per sample by
    # 2 pi f_d / f_s, in double precision to 1e-9 rad: 29 634.973 Hz at 600 km,
    # 50 degrees and 2 GHz; 681 051.185 Hz at 600 km, 10 degrees and 30 GHz
    f_d = _satellite_shift(height, elevation_angle, carrier_frequency)
    cdl = _cdl("C", carrier_frequency=carrier_frequency,
               elevation_angle=elevation_angle, satellite_height=height)
    assert cdl.doppler_shift == pytest.approx(f_d, rel=1e-12, abs=1e-6)
    a = cdl(2, 6, SAMPLING_FREQUENCY)[0].cpu().numpy()
    step = a[..., 1:] * np.conj(a[..., :-1])
    expected = np.exp(2j * np.pi * f_d / SAMPLING_FREQUENCY)
    worst = float(np.abs(np.angle(step * np.conj(expected))).max())
    assert worst <= 1e-9, f"phase step off by {worst:.3g} rad"


@pytest.mark.parametrize("height", [35786e3, 20e3, 8e3, 50e3])
def test_no_shift_without_orbital_motion(height):
    # No Doppler shift for a geostationary satellite (TR 38.811 clause 5.3.2.3) and for
    # an aerial platform: with the UT at rest the channel is constant
    cdl = _cdl("D", satellite_height=height)
    assert cdl.doppler_shift == 0.0
    a = cdl(2, 4, SAMPLING_FREQUENCY)[0]
    assert torch.equal(a[..., 1:], a[..., :1].expand_as(a[..., 1:]))


@pytest.mark.parametrize("doppler_shift", [-500.0, 0.0, 1234.5, 70e3])
def test_given_doppler_shift(doppler_shift):
    # A given shift is used as it is, also for an aerial platform: a receding satellite,
    # a terminal that pre-compensates the shift, the motion of an aerial platform
    for height in (600e3, 20e3):
        cdl = _cdl("A", satellite_height=height, doppler_shift=doppler_shift)
        assert cdl.doppler_shift == doppler_shift
        a = cdl(2, 6, SAMPLING_FREQUENCY)[0].cpu().numpy()
        step = a[..., 1:] * np.conj(a[..., :-1])
        expected = np.exp(2j * np.pi * doppler_shift / SAMPLING_FREQUENCY)
        assert float(np.abs(np.angle(step * np.conj(expected))).max()) <= 1e-9


@pytest.mark.parametrize("direction", ["downlink", "uplink"])
@pytest.mark.parametrize("model", "AC")
def test_shift_with_moving_ut(model, direction):
    # The shift is applied after the random draws: with a moving UT the same draws
    # without it give the same coefficients up to the factor exp(j 2 pi f_d n / f_s)
    f_d = _satellite_shift(SATELLITE_HEIGHT, ELEVATION_ANGLE, CARRIER_FREQUENCY)
    outputs = []
    for doppler_shift in (0.0, None):
        cdl = _cdl(model, direction=direction, min_speed=3.0, max_speed=30.0,
                   doppler_shift=doppler_shift)
        config.seed = 7
        outputs.append(cdl(3, 8, SAMPLING_FREQUENCY)[0].cpu().numpy())
    factor = np.exp(2j * np.pi * f_d * np.arange(8) / SAMPLING_FREQUENCY)
    np.testing.assert_allclose(outputs[1], outputs[0] * factor, rtol=0.0, atol=1e-12)
    assert np.abs(outputs[0][..., 1:] - outputs[0][..., :1]).max() > 1e-6


# Statistics of generated coefficients

STATISTICS_BATCH = 10000
CORRELATION_BATCH = 10000
UT_VELOCITY = (30.0, 0.0, 0.0)                    # f_D = 200.138 Hz at 2 GHz
CORRELATION_SAMPLING = 400.0
CORRELATION_STEPS = 5                             # lags 2.5, 5, 7.5 and 10 ms
# The orientations under which the polarization of every ray keeps its direction
FIXED_ORIENTATIONS = {"ut_orientation": [math.pi, 0.0, 0.0],
                      "bs_orientation": [0.0, 0.0, 0.0]}


def _assert_within(estimate, expected, standard_error, label):
    """An estimate within Z standard errors of the expected distribution"""
    assert abs(estimate - expected) <= Z * standard_error, (
        f"{label}: {estimate:.6g}, expected {expected:.6g} +- {Z * standard_error:.3g} "
        f"({Z:g} standard errors)")


def _power_moments(specular, phasors):
    """E|h|^2 and Var|h|^2 of h = s + sum_k a_k exp(j psi_k) with |s|^2 = specular,
    |a_k|^2 = phasors[k] and independent uniform phases psi_k:
    Var|h|^2 = 2 |s|^2 sum_k b_k + (sum_k b_k)^2 - sum_k b_k^2 with b_k = |a_k|^2"""
    total = float(np.sum(phasors))
    return (specular + total,
            2 * specular * total + total ** 2 - float(np.sum(np.square(phasors))))


@pytest.mark.parametrize("direction", ["downlink", "uplink"])
@pytest.mark.parametrize("model", MODELS)
def test_mean_power_per_cluster(model, direction):
    # At the first sample of independent batch examples, with a vertically polarized
    # isotropic antenna at each end. Each ray of a cluster of power P is sqrt(P / 20)
    # exp(j Phi_theta,theta) with its random phase, and from 600 km with the Faraday
    # rotation sqrt(P / 20) (cos(psi) exp(j Phi_theta,theta) +- sin(psi) / sqrt(kappa)
    # times a phasor of another random phase); the specular path is cos(psi)
    # sqrt(K / (K + 1)), its phase fixed
    _, powers, specular = _profile(model)
    _, _, _, _, xpr, _ = _table(model)
    kappa = 10 ** (xpr / 10)
    for height in (550e3, 600e3):
        cdl = _cdl(model, satellite_height=height, direction=direction, dual=False,
                   **FIXED_ORIENTATIONS)
        a, _ = cdl(STATISTICS_BATCH, 1, SAMPLING_FREQUENCY)
        samples = np.abs(a[:, 0, 0, 0, 0, :, 0].cpu().numpy()) ** 2
        psi = math.radians(108.0 / (CARRIER_FREQUENCY / 1e9) ** 2) if height >= 600e3 \
            else 0.0
        for k in range(len(powers)):
            laplacian = powers[k] - (specular if k == 0 else 0.0)
            phasors = np.concatenate([
                np.full(20, laplacian * math.cos(psi) ** 2 / 20),
                np.full(20, laplacian * math.sin(psi) ** 2 / kappa / 20)])
            mean, variance = _power_moments(
                (specular if k == 0 else 0.0) * math.cos(psi) ** 2, phasors)
            assert_mean(samples[:, k], mean, math.sqrt(variance),
                        f"NTN-CDL-{model} {direction} {height / 1e3:g} km mean power "
                        f"of cluster {k + 1}")


def _coupling_moments(g):
    """For S = sum_i g[i, sigma(i)] with a uniform random permutation sigma, the moments
    E[S], E|S|^2 (real), E[S^2] and E[T] of T = sum_i g[i, sigma(i)]^2: E[S] =
    sum g / N, and the pairs of rays i != k take the pairs j != l of columns with the
    probability 1 / (N (N - 1)); |g| = 1"""
    n = g.shape[0]

    def pairs(x, y):
        # The sum over i != k and j != l of x[i, j] y[k, l]
        return (x.sum() * y.sum() - np.sum(x.sum(axis=1) * y.sum(axis=1))
                - np.sum(x.sum(axis=0) * y.sum(axis=0)) + np.sum(x * y))
    mean_t = np.sum(g ** 2) / n
    return (g.sum() / n, n + float(pairs(g, np.conj(g)).real) / (n * (n - 1)),
            mean_t + pairs(g, g) / (n * (n - 1)), mean_t)


@pytest.mark.parametrize("direction", ["downlink", "uplink"])
def test_time_correlation(direction):
    # NTN-CDL-A at 50 degrees and 550 km (no Faraday rotation), without the shift of the
    # satellite, the UT at [30, 0, 0] m/s with a vertically polarized isotropic antenna.
    # The coefficient of a cluster of power P is the sum over its 20 rays of
    # sqrt(P / 20) exp(j Phi_m) exp(j w_m t) with w_m = 2 pi r_m^T v / lambda_0, r_m the
    # direction of ray m at the UT, which the random coupling makes the pair of the ZOA
    # theta_sigma1(m) and the AOA phi_sigma2(m) for independent uniform permutations.
    # For W = h(t + tau) h*(t): E[W] = P (1/400) sum_i sum_j g(i, j) with g(i, j) =
    # exp(j 2 pi r(theta_j, phi_i)^T v tau / lambda_0), and with b = P / 20 and
    # S = sum_m exp(j w_m tau), T = sum_m exp(2 j w_m tau): E|W|^2 = b^2 (N^2 - N +
    # E|S|^2) and E[W^2] = b^2 (2 E[S^2] - E[T]), N = 20. In the uplink the velocity
    # turns the rays by their angles of departure, which are those at the UT.
    _, powers, _ = _profile("A")
    rays, _, _ = _table_rays("A")
    cdl = _cdl("A", satellite_height=550e3, direction=direction, dual=False,
               doppler_shift=0.0, ut_velocity=list(UT_VELOCITY), **FIXED_ORIENTATIONS)
    a, _ = cdl(CORRELATION_BATCH, CORRELATION_STEPS, CORRELATION_SAMPLING)
    h = a[:, 0, 0, 0, 0].cpu().numpy()                    # [batch, cluster, time]
    wavelength = SPEED_OF_LIGHT / CARRIER_FREQUENCY
    n = 20
    for k in range(len(powers)):
        theta = np.radians(rays["zoa"][k])[None, :]
        phi = np.radians(rays["aoa"][k])[:, None]
        projection = (np.sin(theta) * np.cos(phi) * UT_VELOCITY[0]
                      + np.sin(theta) * np.sin(phi) * UT_VELOCITY[1]
                      + np.cos(theta) * UT_VELOCITY[2])           # [aoa i, zoa j]
        b = powers[k] / n
        for step in range(1, CORRELATION_STEPS):
            lag = step / CORRELATION_SAMPLING
            g = np.exp(2j * np.pi * projection * lag / wavelength)
            mean_s, abs2_s, square_s, mean_t = _coupling_moments(g)
            mean = b * mean_s
            abs2 = b ** 2 * (n * n - n + abs2_s)
            square = b ** 2 * (2 * square_s - mean_t)
            w = h[:, k, step] * np.conj(h[:, k, 0])
            label = f"NTN-CDL-A {direction} cluster {k + 1}, lag {lag * 1e3:g} ms"
            sigma_re = math.sqrt((abs2 + square.real) / 2 - mean.real ** 2)
            sigma_im = math.sqrt((abs2 - square.real) / 2 - mean.imag ** 2)
            assert_mean(w.real, mean.real, sigma_re, label + ", real part")
            assert_mean(w.imag, mean.imag, sigma_im, label + ", imaginary part")
