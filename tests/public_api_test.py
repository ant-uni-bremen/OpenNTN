# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the public names of the package against explicit lists.
#
# The lists hold the classes that openntn/__init__.py imports, the public functions and
# classes defined in openntn.utils, and the names, modules, properties, parameters and
# arguments that the package must not have; a call that passes such a parameter or
# argument raises TypeError. The tests also check the name of the property
# AntennaPanel.polarization, that the flags always_generate_lsp, doppler_enabled,
# enable_pathloss and enable_shadow_fading accept True and False only, and that
# set_topology takes the parameters after latitude by keyword only. A change of the
# public API changes these lists in the same commit and is described in CHANGELOG.md.
import importlib
import importlib.util
import inspect

import pytest
import torch

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
    "cir_to_ofdm_channel", "cir_to_time_channel", "compute_satellite_speed", "deg_2_rad",
    "drop_uts_in_sector", "exp_corr_mat", "gen_single_sector_topology",
    "generate_uts_topology", "one_ring_corr_mat", "rad_2_deg", "relocate_uts",
    "sample_bernoulli", "set_3gpp_scenario_parameters", "subcarrier_frequencies",
    "time_frequency_vector", "time_lag_discrete_time_channel", "time_to_ofdm_channel",
    "wrap_angle_0_360",
]
REMOVED_EXPORTS = ["TDL", "CDL"]
REMOVED_MODULES = ["openntn.tdl", "openntn.cdl"]
REMOVED_UTILS = ["compute_pathloss_additional", "gen_single_sector_topology_interferers",
                 "compute_pathloss_basic", "compute_pathloss_entry", "compute_pathloss_gas",
                 "compute_pathloss_scintilation"]
REMOVED_PROPERTIES = ["additional_pathloss", "lwc", "rain_rate", "relative_humidity",
                      "diameter_earth_antenna", "antenna_efficiency"]
REMOVED_PANEL_PROPERTIES = ["porlarization"]
REMOVED_PARAMETERS = ["lwc", "rain_rate", "relative_humidity", "diameter_earth_antenna",
                      "antenna_efficiency"]
MODELS = ["DenseUrban", "Urban", "SubUrban"]
SCENARIOS = ["DenseUrbanScenario", "UrbanScenario", "SubUrbanScenario"]
REMOVED_MODEL_ARGUMENTS = ["average_street_width", "average_building_height"]
REMOVED_SCENARIO_PROPERTIES = ["average_street_width", "average_building_height"]


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
    for name in REMOVED_PANEL_PROPERTIES:
        assert not hasattr(openntn.AntennaPanel, name), name
    for scenario in SCENARIOS:
        for name in REMOVED_SCENARIO_PROPERTIES:
            assert not hasattr(getattr(openntn, scenario), name), (scenario, name)


@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_antenna_panel_polarization(polarization):
    panel = openntn.AntennaPanel(num_rows=2, num_cols=2, polarization=polarization,
                                 vertical_spacing=0.5, horizontal_spacing=0.5)
    assert panel.polarization == polarization


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


def _arguments():
    """The arguments of a model or scenario without defaults, in their order"""
    antenna = Antenna(polarization="single", polarization_type="V", antenna_pattern="omni",
                      carrier_frequency=20e9)
    return dict(carrier_frequency=20e9, ut_array=antenna, bs_array=antenna,
                direction="downlink", elevation_angle=30.0)


@pytest.mark.parametrize("cls", MODELS + SCENARIOS)
@pytest.mark.parametrize("argument", REMOVED_MODEL_ARGUMENTS)
def test_removed_model_arguments(cls, argument):
    with pytest.raises(TypeError):
        getattr(openntn, cls)(**_arguments(), **{argument: 10.0})


@pytest.mark.parametrize("cls", MODELS + SCENARIOS)
def test_old_positional_order(cls):
    # A call that passes an average street width and building height, here 20 m and
    # 5 m, by position after enable_shadow_fading: the street width comes to
    # always_generate_lsp (models) or doppler_enabled (scenarios), which reject a
    # number, and the building height to doppler_enabled (models) or precision
    # (scenarios)
    flag = "always_generate_lsp" if cls in MODELS else "doppler_enabled"
    with pytest.raises(TypeError, match=flag):
        getattr(openntn, cls)(*_arguments().values(), True, True, 20.0)
    with pytest.raises(TypeError, match="doppler_enabled"):
        getattr(openntn, cls)(*_arguments().values(), True, True, 20.0, 5.0)


@pytest.mark.parametrize("value", [1, 0.0])
@pytest.mark.parametrize("cls,flag", [(cls, "always_generate_lsp") for cls in MODELS]
                         + [(cls, flag) for flag in ("doppler_enabled", "enable_pathloss",
                                                     "enable_shadow_fading")
                            for cls in MODELS + SCENARIOS])
def test_flags_are_bool(cls, flag, value):
    with pytest.raises(TypeError, match=flag):
        getattr(openntn, cls)(**_arguments(), **{flag: value})


@pytest.mark.parametrize("target", ["model", "scenario"])
def test_atmospheric_parameters_by_keyword(target):
    # A call that passes two more values by position after los and latitude, here a
    # liquid water content of 0.5 g/m^3 and a rain rate of 100 mm/h, would give them to
    # atmospheric_pressure and temperature, which take any number: the parameters after
    # latitude are taken by keyword only, and the same values pass by keyword
    antenna = Antenna(polarization="single", polarization_type="V", antenna_pattern="omni",
                      carrier_frequency=20e9)
    model = DenseUrban(carrier_frequency=20e9, ut_array=antenna, bs_array=antenna,
                       direction="downlink", elevation_angle=30.0)
    topology = gen_single_sector_topology(batch_size=2, num_ut=2, scenario="dur",
                                          elevation_angle=30.0, bs_height=600e3)
    obj = model if target == "model" else model._scenario
    with pytest.raises(TypeError, match="positional argument"):
        obj.set_topology(*topology, None, 47.0, 0.5, 100.0)
    obj.set_topology(*topology, None, 47.0, atmospheric_pressure=0.5, temperature=100.0)
    scenario = model._scenario
    assert (scenario.latitude, scenario.atmospheric_pressure,
            scenario.temperature) == (47.0, 0.5, 100.0)


@pytest.mark.parametrize("value", [1, 0.0])
def test_doppler_enabled_of_set_topology_is_bool(value):
    scenario = openntn.DenseUrbanScenario(**_arguments())
    topology = gen_single_sector_topology(batch_size=2, num_ut=2, scenario="dur",
                                          elevation_angle=30.0, bs_height=600e3)
    with pytest.raises(TypeError, match="doppler_enabled"):
        scenario.set_topology(*topology, doppler_enabled=value)


@pytest.mark.parametrize("value", [1, 0.0])
def test_doppler_enabled_of_topology_is_bool(value):
    zeros = torch.zeros(2, 1, 1)
    with pytest.raises(TypeError, match="doppler_enabled"):
        openntn.Topology(velocities=torch.zeros(2, 1, 3), moving_end="rx", los_aoa=zeros,
                         los_aod=zeros, los_zoa=zeros, los_zod=zeros,
                         los=torch.ones(2, 1, 1, dtype=torch.bool), distance_3d=zeros,
                         tx_orientations=torch.zeros(2, 1, 3),
                         rx_orientations=torch.zeros(2, 1, 3),
                         bs_height=torch.tensor([600e3]), elevation_angle=30.0,
                         doppler_enabled=value, los_phase=zeros)
