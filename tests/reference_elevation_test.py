# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the reference elevation angle that selects the table values.
#
# TR 38.811 V15.4.0 gives its tables at the reference elevation angles 10 to 90 degrees
# in steps of 10 degrees: "For an UE-to-satellite or UE-to-HAPS link, the LOS probability
# is taken from the nearest reference elevation angle" (clause 6.6.1), and "The UE in a
# particular scenario should take the values corresponding to the reference angle
# nearest to its elevation angle" (clause 6.6.2). The clauses give no rule for an angle
# midway between two reference angles; the models take the higher one (round half up),
# so 25, 45, 65 and 85 degrees take the values of 30, 50, 70 and 90 degrees. The
# expected reference angle is computed in the test, and the expected values are read
# from the parameter files, which tests/model_parameters_test.py checks against the
# tables of TR 38.811.
import json
import os
from importlib.resources import files

import pytest
import torch
from sionna.phy import config

import openntn
from openntn import Antenna, DenseUrban, SubUrban, Urban, models
from openntn.utils import gen_single_sector_topology

SCENARIOS = {"Dense_Urban": (DenseUrban, "dur"), "Urban": (Urban, "urb"),
             "Sub_Urban": (SubUrban, "sur")}
BANDS = {"S": 2.2e9, "Ka": 20e9}
# (elevation angle, reference angle): the ties 25, 45, 65 and 85 degrees, which go up;
# the ties 15, 35, 55 and 75 degrees, which Python's round (ties to even) also rounds
# up; and two angles that are not ties
ANGLES = [(25.0, 30), (45.0, 50), (65.0, 70), (85.0, 90),
          (15.0, 20), (35.0, 40), (55.0, 60), (75.0, 80),
          (24.0, 20), (26.0, 30)]


def _parameters(scenario, state, band):
    source = files(models).joinpath(f"{scenario}_{state}_{band}_band_DL.json")
    with open(source, encoding="ascii") as f:
        return json.load(f)


@pytest.mark.parametrize("elevation_angle,reference_angle", ANGLES)
@pytest.mark.parametrize("band", sorted(BANDS))
@pytest.mark.parametrize("scenario", sorted(SCENARIOS))
def test_reference_elevation_angle(scenario, band, elevation_angle, reference_angle):
    config.precision = "double"
    model_class, key = SCENARIOS[scenario]
    antenna = Antenna(polarization="single", polarization_type="V", antenna_pattern="omni",
                      carrier_frequency=BANDS[band])
    model = model_class(carrier_frequency=BANDS[band], ut_array=antenna, bs_array=antenna,
                        direction="downlink", elevation_angle=elevation_angle)
    topology = gen_single_sector_topology(batch_size=2, num_ut=2, scenario=key,
                                          elevation_angle=elevation_angle,
                                          bs_height=600e3)
    suffix = f"_{reference_angle}"
    los_file = _parameters(scenario, "LOS", band)
    nlos_file = _parameters(scenario, "NLOS", band)
    where = f"{scenario}, {band} band, {elevation_angle:g} deg"
    for los, values in ((True, los_file), (False, nlos_file)):
        model.set_topology(*topology, los=los)
        sc = model._scenario
        # The parameters of the link state that both files hold (those of the
        # departure spreads at the satellite are set to zero for satellites and not
        # compared here; the clutter loss follows below)
        for name in sorted(k[:-len(suffix)] for k in values if k.endswith(suffix)):
            if "ASD" in name or "ZSD" in name or name + suffix not in los_file:
                continue
            actual = sc.get_param(name)
            expected = torch.full_like(actual, values[name + suffix])
            assert torch.equal(actual, expected), f"{where}, LOS {los}: {name}"
        # The basic path loss: FSPL, plus the clutter loss of the reference angle for
        # NLOS links
        cl = 0.0 if los else nlos_file["CL" + suffix]
        assert torch.equal(sc.basic_pathloss, sc.free_space_pathloss + cl), \
            f"{where}, LOS {los}: clutter loss"
    assert sc.num_clusters_los == los_file["numClusters" + suffix], where
    assert sc.num_clusters_nlos == nlos_file["numClusters" + suffix], where
    probability = sc.los_probability
    assert torch.equal(probability, torch.full_like(probability, los_file["LoS_p" + suffix])), \
        f"{where}: LOS probability"


def test_one_rounding_of_the_elevation_angle():
    # The reference angle is computed in one place: no other rounding of the elevation
    # angle in the package.
    package = os.path.dirname(os.path.abspath(openntn.__file__))
    found = []
    for name in sorted(os.listdir(package)):
        if not name.endswith(".py"):
            continue
        with open(os.path.join(package, name), encoding="utf-8") as f:
            for number, line in enumerate(f, start=1):
                code = line.split("#")[0]
                if "elevation" in code and any(call in code for call in (
                        "round(", "floor(", "ceil(", "int(")):
                    found.append(f"{name}:{number}: {code.strip()}")
    assert len(found) == 1 and found[0].startswith("utils.py:") and found[0].endswith(
        ": return 10*math.floor(elevation_angle/10.0 + 0.5)"), "\n".join(found)
