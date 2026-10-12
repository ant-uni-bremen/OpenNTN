# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the public names of the package against explicit lists.
#
# The lists hold the classes that openntn/__init__.py imports and the public functions
# and classes defined in openntn.utils. A change of the public API changes these lists
# in the same commit. Removed before 2.0.0, because they had no effect or did not work:
# the placeholders TDL and CDL with their modules openntn.tdl and openntn.cdl (they
# raised NotImplementedError), utils.compute_pathloss_additional (the rain and cloud
# attenuation, which no model called) with the scenario property additional_pathloss,
# utils.gen_single_sector_topology_interferers (it failed in every scenario), and the
# set_topology parameters lwc and rain_rate (read only by the rain and cloud
# attenuation) and relative_humidity, diameter_earth_antenna and antenna_efficiency
# (read only by the ITU-R P.618-13 computation of the tropospheric scintillation, which
# TR 38.811 V15.4.0, Table 6.6.6.2.1-1 replaced).
import importlib
import importlib.util
import inspect

import pytest

import openntn
from openntn import Antenna, DenseUrban, utils
from openntn.utils import gen_single_sector_topology

EXPORTS = [
    "Antenna", "AntennaArray", "AntennaElement", "AntennaPanel",
    "ChannelCoefficientsGenerator", "DenseUrban", "DenseUrbanScenario", "LSP",
    "LSPGenerator", "PanelArray", "Rays", "RaysGenerator", "SubUrban", "SubUrbanScenario",
    "SystemLevelChannel", "SystemLevelScenario", "Topology", "Urban", "UrbanScenario",
]
UTILS = [
    "cir_to_ofdm_channel", "cir_to_time_channel", "compute_pathloss_basic",
    "compute_pathloss_entry", "compute_pathloss_gas", "compute_pathloss_scintilation",
    "compute_satellite_speed", "deg_2_rad", "drop_uts_in_sector", "exp_corr_mat",
    "gen_single_sector_topology", "generate_uts_topology", "one_ring_corr_mat",
    "rad_2_deg", "relocate_uts", "sample_bernoulli", "set_3gpp_scenario_parameters",
    "subcarrier_frequencies", "time_frequency_vector", "time_lag_discrete_time_channel",
    "time_to_ofdm_channel", "wrap_angle_0_360",
]
REMOVED_EXPORTS = ["TDL", "CDL"]
REMOVED_MODULES = ["openntn.tdl", "openntn.cdl"]
REMOVED_UTILS = ["compute_pathloss_additional", "gen_single_sector_topology_interferers"]
REMOVED_PROPERTIES = ["additional_pathloss", "lwc", "rain_rate", "relative_humidity",
                      "diameter_earth_antenna", "antenna_efficiency"]
REMOVED_PARAMETERS = ["lwc", "rain_rate", "relative_humidity", "diameter_earth_antenna",
                      "antenna_efficiency"]


def test_exports():
    exported = sorted(name for name, value in vars(openntn).items()
                      if not name.startswith("_") and not inspect.ismodule(value))
    assert exported == sorted(EXPORTS)


def test_utils():
    defined = sorted(name for name, value in vars(utils).items()
                     if not name.startswith("_")
                     and (inspect.isfunction(value) or inspect.isclass(value))
                     and value.__module__ == utils.__name__)
    assert defined == sorted(UTILS)


def test_removed_names():
    for name in REMOVED_EXPORTS:
        assert not hasattr(openntn, name), name
    for name in REMOVED_UTILS:
        assert not hasattr(utils, name), name
    for name in REMOVED_MODULES:
        assert importlib.util.find_spec(name) is None, name
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(name)
    for name in REMOVED_PROPERTIES:
        assert not hasattr(openntn.SystemLevelScenario, name), name


@pytest.mark.parametrize("parameter", REMOVED_PARAMETERS)
def test_removed_parameters(parameter):
    antenna = Antenna(polarization="single", polarization_type="V", antenna_pattern="omni",
                      carrier_frequency=20e9)
    model = DenseUrban(carrier_frequency=20e9, ut_array=antenna, bs_array=antenna,
                       direction="downlink", elevation_angle=30.0)
    topology = gen_single_sector_topology(batch_size=2, num_ut=2, scenario="dur",
                                          elevation_angle=30.0, bs_height=600e3)
    with pytest.raises(TypeError):
        model.set_topology(*topology, **{parameter: 1.0})
    with pytest.raises(TypeError):
        model._scenario.set_topology(*topology, **{parameter: 1.0})
