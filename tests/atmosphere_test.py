# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the default atmosphere of the gas loss.
#
# TR 38.811 V15.4.0, clause 6.6.4: "For all UEs, dry air pressure, water-vapour density,
# water vapour partial pressure and temperature correspond to the mean annual global
# reference atmosphere given in Recommendation ITU-R P835. For all UEs, this corresponds
# to the following values: T = 288.15 K, p = 1013.25 hPa, rho = 7.5 g/m3".
#
# The gas loss with the default atmosphere is checked against Annex 2 of ITU-R P.676-10
# in gas_attenuation_test.py. The scintillation loss does not depend on the atmosphere
# (scintillation_test.py and known_issues_test.py).
from sionna.phy import config

from openntn import Antenna, Urban
from openntn.utils import gen_single_sector_topology

PRESSURE = 1013.25
WATER_VAPOUR_DENSITY = 7.5


def _scenario(carrier_frequency, elevation_angle):
    config.precision = "double"
    antenna = Antenna(polarization="single", polarization_type="V",
                      antenna_pattern="38.901", carrier_frequency=carrier_frequency)
    model = Urban(carrier_frequency=carrier_frequency, ut_array=antenna, bs_array=antenna,
                  direction="downlink", elevation_angle=elevation_angle)
    model.set_topology(*gen_single_sector_topology(batch_size=2, num_ut=2, scenario="urb",
                                                   elevation_angle=elevation_angle,
                                                   bs_height=600e3))
    return model._scenario


def test_default_atmosphere():
    sc = _scenario(2.2e9, 30.0)
    assert sc.temperature == 288.15
    assert sc.atmospheric_pressure == PRESSURE
    assert sc.water_vapor_density == WATER_VAPOUR_DENSITY
