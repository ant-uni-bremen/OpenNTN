# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the cluster and ray angles of steps 7 and 8 of TR 38.901 V16.1.0, clause 7.5,
# which TR 38.811 V15.4.0, clause 6.7.2, uses, against values computed in the test from
# the positions and the specification.
import json
import os

import numpy as np
import pytest
from sionna.phy import config

from openntn import Antenna, DenseUrban, SubUrban, Urban
from openntn.utils import gen_single_sector_topology
from statistical_checks import assert_mean, assert_proportion

BS_HEIGHT = 600000.0
ELEVATION_ANGLES = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0]
SCENARIOS = [(DenseUrban, "dur"), (Urban, "urb"), (SubUrban, "sur")]
DIRECTIONS = ["downlink", "uplink"]
# TR 38.901 V16.1.0, Table 7.5-3: ray offset angles alpha_m of rays 1 to 20
RAY_OFFSETS = np.array([v for a in (0.0447, 0.1413, 0.2492, 0.3715, 0.5129, 0.6797,
                                    0.8844, 1.1481, 1.5195, 2.1551) for v in (a, -a)])
with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_tables",
                       "tr38811_v15.4.0.json"), encoding="ascii") as _f:
    TABLES = json.load(_f)["tables"]
# TR 38.811 V15.4.0, parameter tables of the S band for LOS and NLOS links
S_BAND_TABLES = {(DenseUrban, True): "6.7.2-1a", (DenseUrban, False): "6.7.2-2a",
                 (Urban, True): "6.7.2-3a", (Urban, False): "6.7.2-4a",
                 (SubUrban, True): "6.7.2-5a", (SubUrban, False): "6.7.2-6a"}
# The S-band and Ka-band tables (a and b) and a carrier frequency of each band
BANDS = {"S": ("a", 2.2e9), "Ka": ("b", 20e9)}


def _model(model_class, direction, elevation_angle, carrier_frequency=2.2e9):
    antenna = Antenna(polarization="single",
                      polarization_type="V",
                      antenna_pattern="omni",
                      carrier_frequency=carrier_frequency)
    return model_class(carrier_frequency=carrier_frequency,
                       ut_array=antenna,
                       bs_array=antenna,
                       direction=direction,
                       elevation_angle=elevation_angle,
                       enable_pathloss=False,
                       enable_shadow_fading=False)


def _set_topology(model, scenario, elevation_angle, los, batch_size=4, num_ut=3,
                  height=BS_HEIGHT):
    topology = gen_single_sector_topology(batch_size=batch_size,
                                          num_ut=num_ut,
                                          scenario=scenario,
                                          elevation_angle=elevation_angle,
                                          bs_height=height)
    model.set_topology(*topology, los=los)


def _rays(model):
    """Rays of steps 5 to 9 for the current topology; angles in degrees, links seen
    from the satellite (the scenario's BS) to the UT, as the generator produces them
    in both directions; cluster powers of TR 38.901 V16.1.0, eq. (7.5-6)."""
    rays = model._ray_sampler(model._lsp)
    out = {name: np.degrees(getattr(rays, name).cpu().numpy())
           for name in ("aoa", "aod", "zoa", "zod")}
    out["powers"] = rays.powers.cpu().numpy()
    return out


def _los_departure_angles(model):
    """LOS AOD and ZOD [deg] from the satellite to each UT, from the positions (TR 38.901
    V16.1.0, clause 7.1: zenith angle from the +z axis): [batch, BSs, UTs]."""
    sc = model._scenario
    ut_loc = sc.ut_loc.cpu().numpy()
    bs_loc = sc.bs_loc.cpu().numpy()
    delta = ut_loc[:, None, :, :] - bs_loc[:, :, None, :]
    aod = np.degrees(np.arctan2(delta[..., 1], delta[..., 0]))
    zod = np.degrees(np.arctan2(np.hypot(delta[..., 0], delta[..., 1]), delta[..., 2]))
    return aod, zod


def _circular_difference(a, b):
    """a - b in degrees, in (-180, 180]."""
    return -np.remainder(b - a + 180.0, 360.0) + 180.0


def _circular_mean(angles, axis):
    """Circular mean in degrees."""
    radians = np.radians(angles)
    return np.degrees(np.arctan2(np.sin(radians).mean(axis=axis),
                                 np.cos(radians).mean(axis=axis)))


@pytest.mark.parametrize("model_class,scenario", SCENARIOS)
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("los", [True, False])
def test_zod_offset_is_zero(model_class, scenario, direction, los):
    # TR 38.811 V15.4.0, clause 6.7.2, replaces the parameter tables of TR 38.901
    # V16.1.0 with its own tables, which have no row for the ZOD offset mu_offset,ZOD of
    # eq. (7.5-19), so the offset is 0 for every link.
    config.precision = "double"
    worst = (0.0, None)
    for elevation_angle in ELEVATION_ANGLES:
        model = _model(model_class, direction, elevation_angle)
        _set_topology(model, scenario, elevation_angle, los)
        offset = np.abs(model._scenario.zod_offset.cpu().numpy()).max()
        if offset > worst[0]:
            worst = (offset, elevation_angle)
    assert worst[0] == 0.0, (f"{direction}, LOS {los}: ZOD offset up to {worst[0]:.3g} deg "
                             f"(at {worst[1]:g} deg)")


@pytest.mark.parametrize("model_class,scenario", SCENARIOS)
@pytest.mark.parametrize("los", [True, False])
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("band", ["S", "Ka"])
@pytest.mark.parametrize("height", [160e3, 600e3, 35786e3])
def test_departure_angles_at_satellite(model_class, scenario, los, direction, band, height):
    # The departure angle spreads at a satellite are zero (TR 38.811 V15.4.0, NOTE 8 of
    # Tables 6.7.2-1a to 6.7.2-8b: mu_lgASD and mu_lgZSD are -inf), in both directions:
    # the satellite is the departure end of the rays as the generator draws them, and the
    # large scale parameters ASD and ZSD of every link are 0. With a zero ZOD offset, TR
    # 38.901 V16.1.0, eqs. (7.5-19) and (7.5-20), and, on LOS links, (7.5-17), give every
    # ray the LOS ZOD. The cluster ASD at a satellite is 0 as well (clause 6.9.1: "all
    # cluster AOD and ZOD angles the same"), so eqs. (7.5-11) to (7.5-13) applied to the
    # departure give every ray the LOS AOD, also in urban, whose tables print a cluster
    # ASD c_ASD of 0.08 to 6.63 degrees.
    config.precision = "double"
    _, carrier_frequency = BANDS[band]
    worst = {"ZOD": (0.0, None), "AOD": (0.0, None), "ASD": (0.0, None),
             "ZSD": (0.0, None)}
    for elevation_angle in ELEVATION_ANGLES:
        model = _model(model_class, direction, elevation_angle, carrier_frequency)
        _set_topology(model, scenario, elevation_angle, los, height=height)
        rays = _rays(model)
        aod_los, zod_los = _los_departure_angles(model)
        errors = {"ZOD": np.abs(rays["zod"] - zod_los[..., None, None]).max(),
                  "AOD": np.abs(_circular_difference(rays["aod"],
                                                     aod_los[..., None, None])).max(),
                  "ASD": np.abs(model._lsp.asd.cpu().numpy()).max(),
                  "ZSD": np.abs(model._lsp.zsd.cpu().numpy()).max()}
        for name, error in errors.items():
            if error > worst[name][0]:
                worst[name] = (error, elevation_angle)
    label = f"{height / 1e3:g} km, {direction}, {band} band, LOS {los}"
    for name in ("ASD", "ZSD"):
        error, where = worst[name]
        assert error == 0.0, (f"{label}: the large scale parameter {name} is {error:.3g} "
                              f"deg (at {where:g} deg)")
    for name in ("ZOD", "AOD"):
        error, where = worst[name]
        assert error <= 1e-12, (f"{label}: ray {name} off the LOS {name} by {error:.3g} "
                                f"deg (at {where:g} deg)")


@pytest.mark.parametrize("los", [True, False])
@pytest.mark.parametrize("direction", DIRECTIONS)
def test_cluster_asd_of_an_aerial_platform(los, direction):
    # An aerial platform keeps the cluster ASD of the tables (the zero spreads of NOTE 8
    # concern satellites): the rays of each cluster leave it at the cluster AOD plus
    # c_ASD alpha_m (TR 38.901 V16.1.0, eq. (7.5-13) applied to the departure), with the
    # c_ASD of the urban Tables 6.7.2-3a (LOS) and 6.7.2-4a (NLOS), S band, and the
    # offsets alpha_m of Table 7.5-3. The cluster AOD is the circular mean of its rays.
    config.precision = "double"
    table = TABLES[S_BAND_TABLES[(Urban, los)]]["rows"]["c_ASD"]
    worst = (0.0, None)
    for k, elevation_angle in enumerate(ELEVATION_ANGLES):
        model = _model(Urban, direction, elevation_angle)
        _set_topology(model, "urb", elevation_angle, los, height=20e3)
        rays = _rays(model)
        c = float(table[k])
        offsets, _ = _ray_offsets_from_center(rays["aod"], "aod", c)
        error = np.abs(np.sort(offsets, axis=-1) - np.sort(c * RAY_OFFSETS)).max()
        if error > worst[0]:
            worst = (error, elevation_angle)
    assert worst[0] <= 1e-9, (f"20 km, {direction}, LOS {los}: ray AOD offsets off c_ASD "
                              f"alpha_m by {worst[0]:.3g} deg (at {worst[1]:g} deg)")


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("los", [True, False])
def test_cluster_signs_drawn_per_link(direction, los):
    # TR 38.901 V16.1.0, clause 7.5, step 7: the sign X_n and the variation Y_n of
    # eq. (7.5-11) are random variables of each BS-UT link; the procedure shares results
    # only between co-sited sectors and one UT. Both are symmetric, so the sign of the
    # offset of a cluster AOA from the LOS AOA is +1 or -1 with probability 1/2,
    # independently for two UTs of the same satellite. The cluster AOA is the circular
    # mean of its 20 rays, whose offsets c_ASA alpha_m are symmetric (Table 7.5-3). The
    # strongest cluster (phi'_n = 0), the first cluster of a LOS link (set to the LOS
    # direction) and clusters without power are left out.
    config.precision = "double"
    elevation_angle = 30.0
    model = _model(DenseUrban, direction, elevation_angle)
    _set_topology(model, "dur", elevation_angle, los, batch_size=2000, num_ut=3)
    rays = _rays(model)
    aod_los, _ = _los_departure_angles(model)
    aoa_los = aod_los + 180.0
    offset = _circular_difference(_circular_mean(rays["aoa"], axis=-1),
                                  aoa_los[..., None])                # [batch, BSs, UTs, clusters]
    powers = rays["powers"]
    used = powers > 0.0
    used &= np.arange(powers.shape[-1]) != np.argmax(powers, axis=-1)[..., None]
    if los:
        used[..., 0] = False
    both = used[:, 0, 0] & used[:, 0, 1]
    agree = np.sign(offset[:, 0, 0]) == np.sign(offset[:, 0, 1])
    assert_proportion(int(np.sum(agree & both)), int(np.sum(both)), 0.5,
                      f"{direction}, LOS {los}: sign of the cluster AOA offset equal for "
                      f"two UTs")


@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("los", [True, False])
def test_ray_coupling_drawn_per_link(direction, los):
    # TR 38.901 V16.1.0, clause 7.5, step 8: the rays of each link are coupled at random.
    # The position of each ray's AOA offset in the order of the 20 offsets of its
    # cluster is then a random permutation per link, and the number of rays at the
    # same position for two UTs is the number of fixed points of a random permutation,
    # with mean 1 and standard deviation 1. The two strongest clusters are left out:
    # their rays are coupled within the sub-clusters of Table 7.5-5 (step 8).
    config.precision = "double"
    elevation_angle = 30.0
    model = _model(DenseUrban, direction, elevation_angle)
    _set_topology(model, "dur", elevation_angle, los, batch_size=2000, num_ut=3)
    rays = _rays(model)
    rank = np.argsort(np.argsort(rays["aoa"], axis=-1), axis=-1)  # [batch, BSs, UTs, clusters, rays]
    same = np.sum(rank[:, 0, 0] == rank[:, 0, 1], axis=-1)        # [batch, clusters]
    powers = rays["powers"]
    strongest = np.argsort(-powers, axis=-1)[..., :2]
    other = np.ones(powers.shape, dtype=bool)
    np.put_along_axis(other, strongest, False, axis=-1)
    other = other[:, 0, 0] & other[:, 0, 1]
    assert_mean(same[other], 1.0, 1.0,
                f"{direction}, LOS {los}: rays at the same position of the AOA order for "
                f"two UTs")


# TR 38.901 V16.1.0, Table 7.5-5: rays of the sub-clusters R1, R2 and R3 (rays 1 to 20
# as indices 0 to 19)
SUB_CLUSTERS = ([0, 1, 2, 3, 4, 5, 6, 7, 18, 19], [8, 9, 10, 11, 16, 17], [12, 13, 14, 15])
SUB_CLUSTER_DELAYS = (0.0, 1.28, 2.56)


def _intra_cluster_spreads(model_class, direction, los, elevation_angle):
    """Intra-cluster spreads [deg] of the four angles, in the frame of the generated rays
    (departure at the satellite), from the TR 38.811 tables: c_ASA and c_ZSA at the UT;
    at a satellite, in both directions, the cluster ASD is 0 and the ZOD spread (3/8)
    10^mu_lgZSD of TR 38.901 V16.1.0, eq. (7.5-20), is 0 (NOTE 8 of Tables 6.7.2-1a to
    6.7.2-8b: mu_lgZSD is -inf). The tests use a satellite at 600 km."""
    rows = TABLES[S_BAND_TABLES[(model_class, los)]]["rows"]
    k = ELEVATION_ANGLES.index(elevation_angle)
    return {"aoa": float(rows["c_ASA"][k]), "aod": 0.0,
            "zoa": float(rows["c_ZSA"][k]), "zod": 0.0}


def _ray_offsets_from_center(angles, name, c):
    """Ray angle minus cluster angle [deg] and a mask of the clusters for which this is
    well defined. The cluster angle is the circular mean (azimuth) or the mean (zenith)
    of its 20 rays, whose offsets c alpha_m are symmetric. Zenith rays are wrapped to
    [0, 360] and reflected into [0, 180] (TR 38.901 V16.1.0, after eq. (7.5-18)); a
    cluster whose rays straddle 0 or 180 degrees is left out, a cluster reflected as a
    whole keeps its offsets up to the sign, which the symmetric offsets of each
    sub-cluster do not distinguish."""
    if name in ("aoa", "aod"):
        center = _circular_mean(angles, axis=-1)
        offsets = _circular_difference(angles, center[..., None])
        valid = np.ones(center.shape, dtype=bool)
    else:
        center = angles.mean(axis=-1)
        offsets = angles - center[..., None]
        margin = RAY_OFFSETS.max() * c
        valid = (angles.min(axis=-1) > margin) & (angles.max(axis=-1) < 180.0 - margin)
    return offsets, valid


def _subcluster_coupled(offsets, c, tolerance=1e-9):
    """True for the clusters whose rays of each sub-cluster R_k carry exactly the offsets
    c alpha_m of R_k (Tables 7.5-3 and 7.5-5): [..., clusters]."""
    result = None
    for group in SUB_CLUSTERS:
        actual = np.sort(offsets[..., group], axis=-1)
        expected = np.sort(c * RAY_OFFSETS[group])
        match = np.all(np.abs(actual - expected) <= tolerance, axis=-1)
        result = match if result is None else result & match
    return result


@pytest.mark.parametrize("model_class,scenario", SCENARIOS)
@pytest.mark.parametrize("direction", DIRECTIONS)
@pytest.mark.parametrize("los", [True, False])
def test_rays_coupled_within_sub_clusters(model_class, scenario, direction, los):
    # TR 38.901 V16.1.0, clause 7.5, step 8: the rays are coupled "within a cluster n, or
    # within a sub-cluster in the case of two strongest clusters (see Step 11 and Table
    # 7.5-3)"; step 11 maps the 20 rays of a cluster to the sub-clusters of Table 7.5-5,
    # with the offsets of Table 7.5-3. So in each of the two clusters with the largest
    # powers of eq. (7.5-6), the rays with the indices of R_k carry the offsets alpha_m of
    # R_k, for every angle with a non-zero intra-cluster spread; in the other clusters
    # each ray carries one of the 20 offsets, each once. Links whose second and third
    # cluster powers are equal are left out of the first check (no unique pair).
    config.precision = "double"
    checked = {"strongest": 0, "other": 0}
    for elevation_angle in (10.0, 30.0, 60.0, 90.0):
        model = _model(model_class, direction, elevation_angle)
        _set_topology(model, scenario, elevation_angle, los, batch_size=50, num_ut=3)
        rays = _rays(model)
        powers = rays["powers"]
        order = np.argsort(-powers, axis=-1, kind="stable")
        sorted_powers = np.take_along_axis(powers, order, axis=-1)
        unique = sorted_powers[..., 1] > sorted_powers[..., 2]
        strongest = np.zeros(powers.shape, dtype=bool)
        np.put_along_axis(strongest, order[..., :2], True, axis=-1)
        spreads = _intra_cluster_spreads(model_class, direction, los, elevation_angle)
        for name, c in spreads.items():
            if c == 0.0:
                continue
            offsets, valid = _ray_offsets_from_center(rays[name], name, c)
            coupled = _subcluster_coupled(offsets, c)
            selected = strongest & unique[..., None] & valid
            assert np.all(coupled[selected]), (
                f"{direction}, LOS {los}, {elevation_angle:g} deg, {name}: the rays of "
                f"{np.sum(~coupled[selected])} of {np.sum(selected)} strongest clusters "
                f"are not coupled within the sub-clusters of Table 7.5-5")
            checked["strongest"] += int(np.sum(selected))
            expected = np.sort(c * RAY_OFFSETS)
            others = ~strongest & valid
            error = np.abs(np.sort(offsets, axis=-1) - expected)[others]
            assert error.size == 0 or error.max() <= 1e-9, (
                f"{direction}, LOS {los}, {elevation_angle:g} deg, {name}: ray offsets of "
                f"the other clusters off Table 7.5-3 by {error.max():.3g} deg")
            checked["other"] += int(np.sum(others))
    assert checked["strongest"] > 0 and checked["other"] > 0, checked


def _run_reduce_nlos(model):
    """Runs the channel and returns the rays and the delays that step 11 receives and
    produces for the NLOS part."""
    sampler = model._cir_sampler
    original = sampler._step_11_reduce_nlos
    captured = {}

    def record(h_full, rays, c_ds):
        h_nlos, delays_nlos = original(h_full, rays, c_ds)
        captured.update(rays=rays, c_ds=c_ds, delays_nlos=delays_nlos)
        return h_nlos, delays_nlos

    sampler._step_11_reduce_nlos = record
    try:
        model(1, 1.0)
    finally:
        del sampler._step_11_reduce_nlos
    return captured


@pytest.mark.parametrize("direction", DIRECTIONS)
def test_step_11_splits_the_coupled_clusters(direction):
    # TR 38.901 V16.1.0, clause 7.5, steps 8 and 11: the two clusters that step 11 splits
    # into three sub-clusters with the delays tau_n + {0, 1.28, 2.56} c_DS (eq. (7.5-26))
    # are the two clusters whose rays step 8 coupled within the sub-clusters, also on
    # links with fewer than two clusters of non-zero power, where the powers do not
    # determine the pair (sub-urban, LOS, 30 degrees).
    config.precision = "double"
    elevation_angle = 30.0
    model = _model(SubUrban, direction, elevation_angle)
    _set_topology(model, "sur", elevation_angle, los=True, batch_size=700, num_ut=3)
    captured = _run_reduce_nlos(model)
    rays = captured["rays"]
    delays = rays.delays.cpu().numpy()                     # [batch, tx, rx, clusters]
    c_ds = captured["c_ds"].cpu().numpy()[..., None]       # [batch, tx, rx, 1]
    delays_nlos = captured["delays_nlos"].cpu().numpy()    # [batch, tx, rx, paths]
    # A cluster is split if its first sub-cluster delay offset appears among the paths
    shifted = delays + SUB_CLUSTER_DELAYS[1] * c_ds
    split = np.any(np.abs(shifted[..., None] - delays_nlos[..., None, :])
                   <= 1e-6 * c_ds[..., None], axis=-1)
    # The UT-side azimuth: AOA in the downlink, AOD in the uplink (tx and rx swapped)
    name = "aoa" if direction == "downlink" else "aod"
    angles = np.degrees(getattr(rays, name).cpu().numpy())
    c = _intra_cluster_spreads(SubUrban, direction, True, elevation_angle)["aoa"]
    offsets, _ = _ray_offsets_from_center(angles, "aoa", c)
    coupled = _subcluster_coupled(offsets, c)
    assert np.all(np.sum(split, axis=-1) == 2)
    few = np.sum(rays.powers.cpu().numpy() > 0.0, axis=-1) < 2
    assert np.any(few), "no link with fewer than two clusters of non-zero power"
    mismatch = np.any(split != coupled, axis=-1)
    assert not np.any(mismatch), (
        f"{direction}: step 11 splits other clusters than step 8 coupled within the "
        f"sub-clusters in {np.sum(mismatch)} of {mismatch.size} links "
        f"({np.sum(mismatch & few)} of {np.sum(few)} links with fewer than two clusters "
        f"of non-zero power)")
