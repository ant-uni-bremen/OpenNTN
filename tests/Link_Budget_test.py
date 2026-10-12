# SPDX-FileCopyrightText: Copyright (c) 2025 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the free space path loss, the atmospheric loss and the scintillation loss
# against the link budgets of 3GPP TR 38.821 V16.2.0, Table 6.1.3.3-1 ("Link budgets
# results"), for its 30 cases in both directions.
#
# The cases are the calibration cases of Table 6.1.1.1-9, which gives the satellite
# orbit of each case: GEO for SC1 to SC5 and SC16 to SC20, LEO at 600 km for SC6 to
# SC10 and SC21 to SC25, LEO at 1200 km for SC11 to SC15 and SC26 to SC30; the first
# three cases of each group of five are in the Ka band, the other two in the S band.
# Table 6.1.3.2-1 gives the carrier frequencies (S band: 2 GHz in both directions; Ka
# band: 20 GHz downlink, 30 GHz uplink), the target elevation angles (30 degrees for
# LEO, 12.5 degrees for GEO Set 1, the cases SC1 to SC5, and 20 degrees for GEO Set 2,
# the cases SC16 to SC20) and the losses: the free space path loss of TR 38.811 V15.4.0,
# eq. (6.6-2), the atmospheric loss of eq. (6.6-8) and the scintillation loss of clause
# 6.6.6, with the tropospheric loss of Table 6.6.6.2.1-1 and an ionospheric loss of
# 2.2 dB, "Based on P3 curve for 1% of time from Figure 6.6.6.1.4-1 of [2] after
# frequency scaling" (its note 1). TR 38.811 clause 6.6.6.1.4 gives that ionospheric
# loss for latitudes within +-20 degrees of the magnetic equator and 0 dB elsewhere, so
# the S-band cases are evaluated at latitude 0.
#
# Tolerance: 0.1 dB, inclusive. Table 6.1.3.3-1 prints the losses with one decimal, and
# a computation by the clauses it cites differs from the printed values within that
# resolution.
import pytest
from sionna.phy import config

from openntn import Antenna, DenseUrban
from openntn import utils

TOLERANCE = 0.1  # dB

# TR 38.821 V16.2.0, Table 6.1.1.1-9 (orbit of each case) and Table 6.1.3.2-1 (target
# elevation angle): satellite height [m] and elevation angle [deg] per group of cases
ORBITS = {
    range(1, 6): (35786e3, 12.5),       # GEO, Set 1
    range(6, 11): (600e3, 30.0),        # LEO-600, Set 1
    range(11, 16): (1200e3, 30.0),      # LEO-1200, Set 1
    range(16, 21): (35786e3, 20.0),     # GEO, Set 2
    range(21, 26): (600e3, 30.0),       # LEO-600, Set 2
    range(26, 31): (1200e3, 30.0),      # LEO-1200, Set 2
}

# TR 38.821 V16.2.0, Table 6.1.3.3-1, as printed: case, transmission mode, frequency
# [GHz], free space path loss [dB], atmospheric loss [dB], scintillation loss [dB]
TABLE = (
    ("SC1", "DL", 20.0, 210.6, 1.2, 1.1), ("SC1", "UL", 30.0, 214.1, 1.1, 1.1),
    ("SC2", "DL", 20.0, 210.6, 1.2, 1.1), ("SC2", "UL", 30.0, 214.1, 1.1, 1.1),
    ("SC3", "DL", 20.0, 210.6, 1.2, 1.1), ("SC3", "UL", 30.0, 214.1, 1.1, 1.1),
    ("SC4", "DL", 2.0, 190.6, 0.2, 2.2), ("SC4", "UL", 2.0, 190.6, 0.2, 2.2),
    ("SC5", "DL", 2.0, 190.6, 0.2, 2.2), ("SC5", "UL", 2.0, 190.6, 0.2, 2.2),
    ("SC6", "DL", 20.0, 179.1, 0.5, 0.3), ("SC6", "UL", 30.0, 182.6, 0.5, 0.3),
    ("SC7", "DL", 20.0, 179.1, 0.5, 0.3), ("SC7", "UL", 30.0, 182.6, 0.5, 0.3),
    ("SC8", "DL", 20.0, 179.1, 0.5, 0.3), ("SC8", "UL", 30.0, 182.6, 0.5, 0.3),
    ("SC9", "DL", 2.0, 159.1, 0.1, 2.2), ("SC9", "UL", 2.0, 159.1, 0.1, 2.2),
    ("SC10", "DL", 2.0, 159.1, 0.1, 2.2), ("SC10", "UL", 2.0, 159.1, 0.1, 2.2),
    ("SC11", "DL", 20.0, 184.5, 0.5, 0.3), ("SC11", "UL", 30.0, 188.0, 0.5, 0.3),
    ("SC12", "DL", 20.0, 184.5, 0.5, 0.3), ("SC12", "UL", 30.0, 188.0, 0.5, 0.3),
    ("SC13", "DL", 20.0, 184.5, 0.5, 0.3), ("SC13", "UL", 30.0, 188.0, 0.5, 0.3),
    ("SC14", "DL", 2.0, 164.5, 0.1, 2.2), ("SC14", "UL", 2.0, 164.5, 0.1, 2.2),
    ("SC15", "DL", 2.0, 164.5, 0.1, 2.2), ("SC15", "UL", 2.0, 164.5, 0.1, 2.2),
    ("SC16", "DL", 20.0, 210.4, 0.8, 0.5), ("SC16", "UL", 30.0, 213.9, 0.7, 0.5),
    ("SC17", "DL", 20.0, 210.4, 0.8, 0.5), ("SC17", "UL", 30.0, 213.9, 0.7, 0.5),
    ("SC18", "DL", 20.0, 210.4, 0.8, 0.5), ("SC18", "UL", 30.0, 213.9, 0.7, 0.5),
    ("SC19", "DL", 2.0, 190.4, 0.1, 2.2), ("SC19", "UL", 2.0, 190.4, 0.1, 2.2),
    ("SC20", "DL", 2.0, 190.4, 0.1, 2.2), ("SC20", "UL", 2.0, 190.4, 0.1, 2.2),
    ("SC21", "DL", 20.0, 179.1, 0.5, 0.3), ("SC21", "UL", 30.0, 182.6, 0.5, 0.3),
    ("SC22", "DL", 20.0, 179.1, 0.5, 0.3), ("SC22", "UL", 30.0, 182.6, 0.5, 0.3),
    ("SC23", "DL", 20.0, 179.1, 0.5, 0.3), ("SC23", "UL", 30.0, 182.6, 0.5, 0.3),
    ("SC24", "DL", 2.0, 159.1, 0.1, 2.2), ("SC24", "UL", 2.0, 159.1, 0.1, 2.2),
    ("SC25", "DL", 2.0, 159.1, 0.1, 2.2), ("SC25", "UL", 2.0, 159.1, 0.1, 2.2),
    ("SC26", "DL", 20.0, 184.5, 0.5, 0.3), ("SC26", "UL", 30.0, 188.0, 0.5, 0.3),
    ("SC27", "DL", 20.0, 184.5, 0.5, 0.3), ("SC27", "UL", 30.0, 188.0, 0.5, 0.3),
    ("SC28", "DL", 20.0, 184.5, 0.5, 0.3), ("SC28", "UL", 30.0, 188.0, 0.5, 0.3),
    ("SC29", "DL", 2.0, 164.5, 0.1, 2.2), ("SC29", "UL", 2.0, 164.5, 0.1, 2.2),
    ("SC30", "DL", 2.0, 164.5, 0.1, 2.2), ("SC30", "UL", 2.0, 164.5, 0.1, 2.2),
)
QUANTITIES = ("free_space_path_loss", "atmospheric_loss", "scintillation_loss")

def _orbit(case):
    number = int(case[2:])
    return next(value for cases, value in ORBITS.items() if number in cases)


def _params():
    return [pytest.param(case, direction, frequency, quantity, value,
                         id=f"{case}-{direction}-{quantity}")
            for case, direction, frequency, *expected in TABLE
            for quantity, value in zip(QUANTITIES, expected)]


def _losses(case, direction, frequency):
    """Free space path loss, atmospheric loss and scintillation loss [dB] of one link."""
    config.precision = "double"
    height, elevation_angle = _orbit(case)
    carrier_frequency = frequency * 1e9
    antenna = Antenna(polarization="single", polarization_type="V",
                      antenna_pattern="38.901", carrier_frequency=carrier_frequency)
    model = DenseUrban(carrier_frequency=carrier_frequency, ut_array=antenna,
                       bs_array=antenna,
                       direction="downlink" if direction == "DL" else "uplink",
                       elevation_angle=elevation_angle)
    topology = utils.gen_single_sector_topology(batch_size=1, num_ut=1, scenario="dur",
                                                elevation_angle=elevation_angle,
                                                bs_height=height)
    # S band: the ionospheric loss of the table applies within +-20 degrees of the
    # magnetic equator (TR 38.811 clause 6.6.6.1.4)
    latitude = 0.0 if frequency < 6.0 else None
    model.set_topology(*topology, latitude=latitude)
    sc = model._scenario
    return {"free_space_path_loss": float(sc.free_space_pathloss[0, 0, 0]),
            "atmospheric_loss": float(sc.gas_pathloss[0, 0, 0]),
            "scintillation_loss": float(sc.scintillation_pathloss[0, 0, 0])}


@pytest.mark.parametrize("case,direction,frequency,quantity,expected", _params())
def test_link_budget(case, direction, frequency, quantity, expected):
    actual = _losses(case, direction, frequency)[quantity]
    assert abs(actual - expected) <= TOLERANCE, (
        f"{case} {direction}: {quantity.replace('_', ' ')} {actual:.4f} dB, TR 38.821 "
        f"V16.2.0 Table 6.1.3.3-1 gives {expected} dB (tolerance {TOLERANCE} dB)")


def test_table_complete():
    # 30 cases in both directions; the orbits of Table 6.1.1.1-9 cover every case
    assert len(TABLE) == 60
    assert {row[0] for row in TABLE} == {f"SC{n}" for n in range(1, 31)}
    assert all(_orbit(f"SC{n}") for n in range(1, 31))
