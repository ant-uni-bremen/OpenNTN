# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the orientations that the topology generators set.
#
# The orientation of an antenna, its bearing alpha, downtilt beta and slant gamma, turns
# its local coordinate system against the global one by the rotation matrix R of TR
# 38.901 V16.1.0, eq. (7.1-4). The boresight of the antenna, the x-axis of its local
# coordinate system, is the first column of R: (cos(alpha) cos(beta), sin(alpha)
# cos(beta), -sin(beta)). The generators place the BS (the satellite or aerial platform)
# at (0, 0, h) facing the nadir, with the orientation (0, 90 deg, 0), and turn each UT
# toward it: the bearing atan2(y, x) of the UT's location (x, y) with the downtilt of the
# elevation angle + 180 deg points the boresight of the UT horizontally toward the
# origin and upward at the elevation angle; the slant is the elevation angle + 90 deg.
# The tests build R from the orientations and compare the boresights with the
# directions computed from the locations.
import numpy as np
import pytest
import torch

from openntn.utils import gen_single_sector_topology, generate_uts_topology

SATELLITE_HEIGHT = 600e3
# Largest angle between the boresight of a UT and the direction from the UT to the BS
# [deg]. The boresight points upward at the elevation angle of the drop; a UT dropped a
# few hundred metres from the point seen at that angle sees a satellite at 600 km at an
# elevation that differs by a few hundredths of a degree at most.
MAX_ANGLE = 0.1


def _rotation(orientations):
    """R of TR 38.901 V16.1.0, eq. (7.1-4), for orientations [..., 3] (bearing,
    downtilt, slant) [rad], as an array [..., 3, 3]"""
    a, b, c = orientations[..., 0], orientations[..., 1], orientations[..., 2]
    ca, sa, cb, sb, cc, sc = np.cos(a), np.sin(a), np.cos(b), np.sin(b), np.cos(c), np.sin(c)
    return np.stack([
        np.stack([ca * cb, ca * sb * sc - sa * cc, ca * sb * cc + sa * sc], axis=-1),
        np.stack([sa * cb, sa * sb * sc + ca * cc, sa * sb * cc - ca * sc], axis=-1),
        np.stack([-sb, cb * sc, cb * cc], axis=-1)], axis=-2)


@pytest.mark.parametrize("elevation_angle", [10.0, 30.0, 60.0])
def test_uts_face_the_bs(elevation_angle):
    # With drop_area="cell" the UTs are rotated into the three sectors of the cell, so
    # about a third of them lie at negative x. The boresight of every UT points at the
    # BS at (0, 0, h).
    double = torch.float64
    ut_loc, ut_orientations, _, _ = generate_uts_topology(
        batch_size=16, num_ut=16, drop_area="cell",
        cell_loc_xy=torch.zeros(2, dtype=double),
        bs_height=torch.tensor(SATELLITE_HEIGHT, dtype=double),
        elevation_angle=torch.tensor(elevation_angle, dtype=double),
        isd=torch.tensor(500.0, dtype=double), min_ut_height=1.5, max_ut_height=1.5,
        indoor_probability=0.0, min_ut_velocity=0.0, max_ut_velocity=0.0,
        precision="double")
    loc = ut_loc.numpy()
    negative = loc[..., 0] < 0
    assert negative.any() and not negative.all()
    to_bs = np.array([0.0, 0.0, SATELLITE_HEIGHT]) - loc
    to_bs /= np.linalg.norm(to_bs, axis=-1, keepdims=True)
    boresight = _rotation(ut_orientations.numpy())[..., :, 0]
    angle = np.degrees(np.arccos(np.clip((boresight * to_bs).sum(axis=-1), -1.0, 1.0)))
    assert angle.max() <= MAX_ANGLE, (
        f"{elevation_angle:g} deg: boresights {angle[negative].min():.4f} to "
        f"{angle[negative].max():.4f} deg off the direction to the BS for the "
        f"{int(negative.sum())} UTs at x < 0, up to {angle[~negative].max():.4f} deg for "
        f"the others")


@pytest.mark.parametrize("bs_height", [SATELLITE_HEIGHT, 20e3])
@pytest.mark.parametrize("scenario", ["dur", "urb", "sur"])
def test_single_sector_orientations(scenario, bs_height):
    # The BS at (0, 0, h) faces the nadir. The boresight of each UT points horizontally
    # toward the origin, the foot point of the BS, and upward at the elevation angle; at
    # 90 deg it points to the zenith. The slant of the UTs is the elevation angle + 90 deg.
    batch_size, num_ut = 16, 16
    for elevation_angle in (10.0, 30.0, 60.0, 80.0, 90.0):
        ut_loc, bs_loc, ut_orientations, bs_orientations, _, _ = \
            gen_single_sector_topology(batch_size=batch_size, num_ut=num_ut,
                                       scenario=scenario, elevation_angle=elevation_angle,
                                       bs_height=bs_height, precision="double")
        where = f"{scenario}, {bs_height / 1e3:g} km, {elevation_angle:g} deg"
        np.testing.assert_array_equal(
            bs_loc.numpy(), np.broadcast_to([0.0, 0.0, bs_height], (batch_size, 1, 3)))
        np.testing.assert_allclose(
            _rotation(bs_orientations.numpy())[..., :, 0],
            np.broadcast_to([0.0, 0.0, -1.0], (batch_size, 1, 3)), rtol=0, atol=1e-15,
            err_msg=where)

        boresight = _rotation(ut_orientations.numpy())[..., :, 0]
        np.testing.assert_allclose(np.degrees(np.arcsin(boresight[..., 2])),
                                   elevation_angle, rtol=0, atol=1e-6, err_msg=where)
        # The horizontal part of the boresight against the direction from the UT to the
        # origin, where the boresight has a horizontal part (below 90 deg)
        loc = ut_loc.numpy()
        horizontal = np.hypot(boresight[..., 0], boresight[..., 1])
        tilted = horizontal > 1e-9
        if elevation_angle < 90.0:
            assert tilted.all(), where
        cross = boresight[..., 0] * -loc[..., 1] - boresight[..., 1] * -loc[..., 0]
        dot = boresight[..., 0] * -loc[..., 0] + boresight[..., 1] * -loc[..., 1]
        azimuth_error = np.abs(np.arctan2(cross, dot))[tilted]
        assert azimuth_error.size == 0 or azimuth_error.max() <= 1e-9, (
            f"{where}: horizontal boresight up to {np.degrees(azimuth_error.max()):.4f} deg "
            f"off the direction to the origin")
        np.testing.assert_allclose(ut_orientations.numpy()[..., 2],
                                   np.radians(elevation_angle + 90.0), rtol=0, atol=1e-12,
                                   err_msg=where)
