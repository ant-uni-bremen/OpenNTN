# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# This file checks that the plotting methods run and create their figures. It does not
# check what the figures show. The non-interactive Agg backend is selected so that the
# tests need no display.
import matplotlib
import matplotlib.pyplot as plt
import pytest

from openntn import utils
from openntn import Antenna, AntennaArray, AntennaElement, AntennaPanel, DenseUrban

CARRIER_FREQUENCY = 2e9


@pytest.fixture(autouse=True)
def agg_backend():
    backend = matplotlib.get_backend()
    plt.switch_backend("Agg")
    yield
    plt.close("all")
    plt.switch_backend(backend)


@pytest.mark.parametrize("pattern", ["omni", "38.901", "aperture", "dlp"])
def test_antenna_element_show(pattern):
    AntennaElement(pattern).show()
    # Vertical cut, horizontal cut, 3D pattern
    assert len(plt.get_fignums()) == 3


@pytest.mark.parametrize("polarization", ["single", "dual"])
def test_antenna_panel_show(polarization):
    AntennaPanel(2, 2, polarization, 0.5, 0.5).show()
    assert len(plt.get_fignums()) == 1


@pytest.mark.parametrize("polarization,polarization_type",
                         [("single", "V"), ("single", "H"), ("dual", "cross"), ("dual", "VH")])
def test_panel_array_show(polarization, polarization_type):
    AntennaArray(2, 2, polarization, polarization_type, "38.901", CARRIER_FREQUENCY).show()
    assert len(plt.get_fignums()) == 1


def test_panel_array_show_element_radiation_pattern():
    array = AntennaArray(2, 2, "dual", "VH", "38.901", CARRIER_FREQUENCY)
    array.show_element_radiation_pattern()
    assert len(plt.get_fignums()) == 3


def test_show_topology():
    ut_array = Antenna(polarization="single",
                       polarization_type="V",
                       antenna_pattern="38.901",
                       carrier_frequency=CARRIER_FREQUENCY)
    bs_array = AntennaArray(num_rows=1,
                            num_cols=4,
                            polarization="dual",
                            polarization_type="VH",
                            antenna_pattern="38.901",
                            carrier_frequency=CARRIER_FREQUENCY)
    model = DenseUrban(carrier_frequency=CARRIER_FREQUENCY,
                       ut_array=ut_array,
                       bs_array=bs_array,
                       direction="uplink",
                       elevation_angle=30.0)
    topology = utils.gen_single_sector_topology(batch_size=2,
                                                num_ut=3,
                                                scenario="dur",
                                                elevation_angle=30.0,
                                                bs_height=600000.0)
    model.set_topology(*topology)
    model.show_topology()
    assert len(plt.get_fignums()) == 1
