# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the indoor/outdoor state of the UTs.
#
# TR 38.811 V15.4.0 considers only outdoor UTs for satellites: "Only outdoor conditions
# are considered for satellite operations, since performance requirements are not
# expected to be met with the available link budget for indoor communications" (clause
# 6.1.2), and Table 5.1-1 gives "100% Outdoors" for the deployments with GEO and
# non-GEO satellites, "Indoor and Outdoor" only for UAS including HAPS. The models have
# neither the building entry loss of clause 6.6.3 nor clusters for indoor UTs, so
# set_topology accepts outdoor UTs only: an indoor UT raises ValueError before anything
# is stored.
import pytest
import torch
from sionna.phy import config

from openntn import Antenna, AntennaArray, DenseUrban
from openntn.utils import gen_single_sector_topology

CARRIER_FREQUENCY = 2e9
ELEVATION_ANGLE = 30.0
STORED = ("ut_loc", "bs_loc", "indoor", "los", "distance_2d", "distance_3d",
          "basic_pathloss", "gas_pathloss", "scintillation_pathloss")
LSPS = ("ds", "asd", "asa", "zsa", "zsd", "k_factor", "sf")


def _model():
    ut_array = Antenna(polarization="single", polarization_type="V",
                       antenna_pattern="38.901", carrier_frequency=CARRIER_FREQUENCY)
    bs_array = AntennaArray(num_rows=1, num_cols=2, polarization="dual",
                            polarization_type="VH", antenna_pattern="38.901",
                            carrier_frequency=CARRIER_FREQUENCY)
    return DenseUrban(carrier_frequency=CARRIER_FREQUENCY, ut_array=ut_array,
                      bs_array=bs_array, direction="downlink",
                      elevation_angle=ELEVATION_ANGLE)


def _topology():
    return list(gen_single_sector_topology(batch_size=2, num_ut=4, scenario="dur",
                                           elevation_angle=ELEVATION_ANGLE,
                                           bs_height=600e3))


@pytest.mark.parametrize("target", ["channel model", "scenario"])
@pytest.mark.parametrize("with_topology", [True, False])
def test_indoor_ut_rejected(target, with_topology):
    # One indoor UT, passed with new topology tensors or alone: ValueError, and the
    # stored topology, the losses and the large scale parameters are unchanged.
    config.precision = "double"
    model = _model()
    model.set_topology(*_topology())
    scenario = model._scenario
    before = {name: getattr(scenario, name).clone() for name in STORED}
    lsp_before = {name: getattr(model._lsp, name).clone() for name in LSPS}
    topology = _topology()
    in_state = torch.zeros_like(topology[5], dtype=torch.bool)
    in_state[0, 0] = True
    owner = model if target == "channel model" else scenario
    with pytest.raises(ValueError):
        if with_topology:
            topology[5] = in_state
            owner.set_topology(*topology)
        else:
            owner.set_topology(in_state=in_state)
    for name in STORED:
        assert torch.equal(getattr(scenario, name), before[name]), name
    for name in LSPS:
        assert torch.equal(getattr(model._lsp, name), lsp_before[name]), name


def test_outdoor_uts():
    # Outdoor UTs, LOS and NLOS forced: accepted, and every channel coefficient is
    # finite.
    config.precision = "double"
    model = _model()
    topology = _topology()
    topology[5] = torch.zeros_like(topology[5], dtype=torch.bool)
    for los in (True, False):
        model.set_topology(*topology, los=los)
        h, tau = model(4, 30e3)
        assert torch.isfinite(h).all() and torch.isfinite(tau).all(), f"LOS {los}"
