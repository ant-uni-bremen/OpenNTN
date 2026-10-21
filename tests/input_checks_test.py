# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the checks of the inputs of the antenna classes, the models, their scenarios
# and the topology generators: an input of the wrong type raises TypeError and an
# invalid value ValueError, with the message of its check. The checks are if
# statements, so that they also act under python -O, which removes assert statements;
# one test runs a few of them in a subprocess under -O.
import os
import re
import subprocess
import sys

import pytest

import openntn
from openntn import Antenna, AntennaElement, AntennaPanel, DenseUrban, PanelArray
from openntn.utils import (gen_single_sector_topology, generate_uts_topology,
                           set_3gpp_scenario_parameters)

CARRIER_FREQUENCY = 2.2e9
ELEVATION_ANGLE = 30.0
TOPOLOGY = ("ut_loc", "bs_loc", "ut_orientations", "bs_orientations", "ut_velocities",
            "in_state")


def _antenna(carrier_frequency=CARRIER_FREQUENCY, precision=None):
    return Antenna(polarization="single", polarization_type="V", antenna_pattern="omni",
                   carrier_frequency=carrier_frequency, precision=precision)


def _model(carrier_frequency=CARRIER_FREQUENCY, ut_array=None, bs_array=None,
           direction="downlink", elevation_angle=ELEVATION_ANGLE, precision=None):
    ut_array = _antenna(carrier_frequency) if ut_array is None else ut_array
    bs_array = _antenna(carrier_frequency) if bs_array is None else bs_array
    return DenseUrban(carrier_frequency=carrier_frequency, ut_array=ut_array,
                      bs_array=bs_array, direction=direction,
                      elevation_angle=elevation_angle, precision=precision)


def _set_topology_without(name):
    """A first set_topology that leaves out the topology tensor ``name``"""
    model = _model()
    topology = dict(zip(TOPOLOGY, gen_single_sector_topology(
        batch_size=2, num_ut=2, scenario="dur", elevation_angle=ELEVATION_ANGLE,
        bs_height=600e3)))
    topology[name] = None
    model.set_topology(**topology)


def _uts_topology(drop_area):
    return generate_uts_topology(
        batch_size=2, num_ut=2, drop_area=drop_area, cell_loc_xy=None, bs_height=600e3,
        elevation_angle=ELEVATION_ANGLE, isd=200.0, min_ut_height=1.5, max_ut_height=1.5,
        indoor_probability=0.0, min_ut_velocity=0.0, max_ut_velocity=0.0)


# Check -> (a call with an input that violates it, the message of the check)
CHECKS = {
    "element_pattern": (lambda: AntennaElement("isotropic"),
                        "The radiation_pattern must be one of"),
    "panel_polarization": (lambda: AntennaPanel(1, 1, "triple", 0.5, 0.5),
                           "polarization must be either 'single' or 'dual'"),
    "array_polarization": (
        lambda: PanelArray(1, 1, "triple", "V", "omni", CARRIER_FREQUENCY),
        "polarization must be either 'single' or 'dual'"),
    # The panel width and height are 0.5 wavelengths for two elements at 0.5
    "panel_horizontal_spacing": (
        lambda: PanelArray(2, 2, "single", "V", "omni", CARRIER_FREQUENCY, num_cols=2,
                           panel_horizontal_spacing=0.4),
        "Panel horizontal spacing must be larger than the panel width"),
    "panel_vertical_spacing": (
        lambda: PanelArray(2, 2, "single", "V", "omni", CARRIER_FREQUENCY, num_rows=2,
                           panel_vertical_spacing=0.4),
        "Panel vertical spacing must be larger than panel height"),
    "single_polarization_type": (
        lambda: PanelArray(1, 1, "single", "VH", "omni", CARRIER_FREQUENCY),
        "For single polarization, polarization_type must be 'V' or 'H'"),
    "dual_polarization_type": (
        lambda: PanelArray(1, 1, "dual", "V", "omni", CARRIER_FREQUENCY),
        "For dual polarization, polarization_type must be 'VH' or 'cross'"),
    "ant_pol2": (lambda: _antenna().ant_pol2,
                 "This property is not defined with single polarization"),
    "ant_ind_pol2": (lambda: _antenna().ant_ind_pol2,
                     "This property is not defined with single polarization"),
    "ant_pos_pol2": (lambda: _antenna().ant_pos_pol2,
                     "This property is not defined with single polarization"),
    "direction": (lambda: _model(direction="sideways"),
                  "'direction' must be 'uplink' or 'downlink'"),
    "carrier_frequency": (lambda: _model(carrier_frequency=6e9),
                          "Carrier frequency must be either in S Band (1.9GHz-4GHz) or "
                          "Ka Band (19GHz - 40GHz)"),
    # Arrays built for 2 GHz with a model at 2.2 GHz
    "wavelengths": (lambda: _model(ut_array=_antenna(2.0e9), bs_array=_antenna(2.0e9)),
                    "The carrier frequencies of ut antenna, bs antenna and scenario "
                    "must match"),
    "elevation_angle_below_10": (lambda: _model(elevation_angle=5.0),
                                 "elevation angle must be in range [10,90]"),
    "elevation_angle_above_90": (lambda: _model(elevation_angle=95.0),
                                 "elevation angle must be in range [10,90]"),
    **{f"set_topology_without_{name}": (lambda name=name: _set_topology_without(name),
                                        f"`{name}` is None and was not previously set")
       for name in TOPOLOGY},
    "scenario": (lambda: set_3gpp_scenario_parameters("rural"),
                 "`scenario` must be one of 'urb', 'dur', 'sur'"),
    "drop_area": (lambda: _uts_topology("disk"),
                  "Drop area must be either 'sector' or 'cell'"),
}


@pytest.mark.parametrize("check", sorted(CHECKS))
def test_invalid_input(check):
    call, message = CHECKS[check]
    with pytest.raises(ValueError, match=re.escape(message)):
        call()


# Check of a type -> (a call with an input of the wrong type, the message of the check)
TYPE_CHECKS = {
    "ut_array": (lambda: _model(ut_array="antenna"),
                 "'ut_array' must be an instance of PanelArray"),
    "bs_array": (lambda: _model(bs_array="antenna"),
                 "'bs_array' must be an instance of PanelArray"),
}


@pytest.mark.parametrize("check", sorted(TYPE_CHECKS))
def test_input_of_wrong_type(check):
    call, message = TYPE_CHECKS[check]
    with pytest.raises(TypeError, match=re.escape(message)):
        call()


@pytest.mark.parametrize("arrays,model", [("single", "double"), ("double", "single")])
@pytest.mark.parametrize("name", ["ut_array", "bs_array"])
def test_precision_of_the_arrays(name, arrays, model):
    # One array in another precision than the model: the error names both precisions,
    # not the carrier frequencies, which match
    antennas = {"ut_array": _antenna(precision=model),
                "bs_array": _antenna(precision=model), name: _antenna(precision=arrays)}
    message = (f"is in {model} precision, but its {name} (Antenna) is in {arrays} "
               f"precision")
    with pytest.raises(ValueError, match=re.escape(message)):
        _model(precision=model, **antennas)


# Run under python -O: prints the name of each check and the exception that it raised,
# or "no error"
_UNDER_O = """
from openntn import Antenna, AntennaElement, DenseUrban

def antenna(carrier_frequency):
    return Antenna(polarization="single", polarization_type="V", antenna_pattern="omni",
                   carrier_frequency=carrier_frequency)

checks = {
    "elevation angle 5 deg": lambda: DenseUrban(2.2e9, antenna(2.2e9), antenna(2.2e9),
                                                "downlink", 5.0),
    "carrier frequency 6 GHz": lambda: DenseUrban(6e9, antenna(6e9), antenna(6e9),
                                                  "downlink", 30.0),
    "element pattern": lambda: AntennaElement("isotropic"),
    "ant_pol2 of a single-polarized array": lambda: antenna(2.2e9).ant_pol2,
}
for name, check in checks.items():
    try:
        check()
        print(name + ": no error")
    except Exception as error:
        print(name + ": " + type(error).__name__)
"""


@pytest.mark.slow
def test_checks_under_python_o(tmp_path):
    # The subprocess imports the package that this process imported, from an empty
    # working directory
    package_parent = os.path.dirname(os.path.dirname(os.path.abspath(openntn.__file__)))
    path = [package_parent] + ([os.environ["PYTHONPATH"]] if os.environ.get("PYTHONPATH")
                               else [])
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(path), PYTHONDONTWRITEBYTECODE="1")
    result = subprocess.run([sys.executable, "-O", "-c", _UNDER_O], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=600, check=False)
    assert result.returncode == 0, result.stderr
    assert result.stdout.splitlines() == [
        "elevation angle 5 deg: ValueError", "carrier frequency 6 GHz: ValueError",
        "element pattern: ValueError", "ant_pol2 of a single-polarized array: ValueError"]
