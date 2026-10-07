# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of behaviour that the current implementation does not meet yet.
#
# Each test states the behaviour required by the specification or by the documented
# interface. It is marked as a strict expected failure whose reason states the observed
# and the expected behaviour. Strict means that the test fails as soon as the behaviour
# changes, so the marker has to be removed in the same change.
import math
import unittest

import pytest
import torch

from openntn import Antenna, AntennaArray, Urban
from openntn import utils

from statistical_checks import assert_std

CARRIER_FREQUENCY = 2.2e9
ELEVATION_ANGLE = 50.0
BS_HEIGHT = 600000.0


def _model(enable_shadow_fading=True, direction="downlink"):
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
    return Urban(carrier_frequency=CARRIER_FREQUENCY,
                 ut_array=ut_array,
                 bs_array=bs_array,
                 direction=direction,
                 elevation_angle=ELEVATION_ANGLE,
                 enable_pathloss=True,
                 enable_shadow_fading=enable_shadow_fading)


def _topology(batch_size=4, num_ut=3):
    return list(utils.gen_single_sector_topology(batch_size=batch_size,
                                                 num_ut=num_ut,
                                                 scenario="urb",
                                                 elevation_angle=ELEVATION_ANGLE,
                                                 bs_height=BS_HEIGHT))


def _applied_shadow_fading_db(model):
    """Shadow fading [dB] that the model applies to the channel coefficients.

    The gain of step 12 is applied to unit coefficients; the deterministic parts of the
    path loss (FSPL, clutter loss for NLoS, gas, scintillation and entry loss) are then
    removed, so what remains is the shadow fading that reaches the channel.
    """
    sc = model._scenario
    shape = list(sc.los.shape) + [1, 1, 1, 1]
    ones = torch.ones(shape, dtype=torch.complex64, device=sc.device)
    gain = model._step_12(ones, model._lsp.sf).reshape(sc.los.shape)
    loss_db = -20.0 * torch.log10(torch.abs(gain).to(torch.float64))
    angle_str = str(round(ELEVATION_ANGLE / 10.0) * 10)
    clutter_loss = sc._params_nlos["CL_" + angle_str]
    deterministic = (sc.free_space_pathloss + sc.gas_pathloss + sc.scintillation_pathloss
                     + sc.entry_pathloss).to(torch.float64)
    deterministic = deterministic + torch.where(sc.los, 0.0, clutter_loss)
    return loss_db - deterministic, sc.los


class ShadowFading(unittest.TestCase):
    """TR 38.811 V15.4.0, Section 6.6.2, eq. (6.6-4): the path loss contains one shadow
    fading term SF ~ N(0, sigma_SF^2); Table 6.6.2-2 gives sigma_SF = 4 dB (LoS) and
    6 dB (NLoS) for the urban scenario.

    Scenario of both tests: urban, S band (2.2 GHz), downlink, 50 degrees elevation,
    satellite at 600 km, 2000 links with one UT per batch example, so that the LSP draws
    of different links are independent. Measured quantity: the shadow fading in dB that
    step 12 applies to the channel coefficients, i.e. the loss of the step 12 gain on
    unit coefficients minus FSPL, gas, scintillation and entry loss and, for NLoS links,
    the clutter loss."""

    @pytest.mark.xfail(strict=True, raises=AssertionError,
                       reason="Under investigation: the shadow fading applied to the channel has "
                              "about sqrt(2) times sigma_SF, as the shadow fading of the basic path "
                              "loss (TR 38.811 eq. 6.6-4) and the shadow fading large-scale "
                              "parameter are both applied; expected sigma_SF of Table 6.6.2-2")
    def test_shadow_fading_applied_once(self):
        # Expected: standard deviation sigma_SF. Measured on the current code: 5.7 dB
        # (LoS) and 8.3 dB (NLoS), the sum of two independent draws.
        model = _model()
        model.set_topology(*_topology(batch_size=2000, num_ut=1))
        sf_db, los = _applied_shadow_fading_db(model)
        assert_std(sf_db[los], 4.0, "applied SF, LoS")
        assert_std(sf_db[~los], 6.0, "applied SF, NLoS")

    @pytest.mark.xfail(strict=True, raises=AssertionError,
                       reason="Under investigation: with enable_shadow_fading=False the shadow "
                              "fading of the basic path loss (TR 38.811 eq. 6.6-4) is still "
                              "applied; expected no shadow fading")
    def test_shadow_fading_disabled(self):
        # TR 38.811 has no switch for the shadow fading; enable_shadow_fading=False is a
        # model option ("If True, apply shadow fading. Otherwise doesn't."). With it, the
        # SF term of eq. (6.6-4) is absent and the applied shadow fading must be 0 dB.
        # Measured on the current code: the SF draw of the basic path loss remains,
        # standard deviation 4.0 dB (LoS) and 6.1 dB (NLoS).
        model = _model(enable_shadow_fading=False)
        model.set_topology(*_topology(batch_size=2000, num_ut=1))
        sf_db, los = _applied_shadow_fading_db(model)
        self.assertLess(float(sf_db.abs().max()), 1e-3,
                        f"applied SF remains: standard deviation {float(sf_db[los].std()):.2f} dB "
                        f"(LoS), {float(sf_db[~los].std()):.2f} dB (NLoS)")


class RayOffsets(unittest.TestCase):

    @pytest.mark.xfail(strict=True, raises=AssertionError,
                       reason="The offset of ray 16 is -0.1481; TR 38.901 Table 7.5-3 gives "
                              "+1.1481 and -1.1481 for rays 15 and 16")
    def test_ray_offsets(self):
        # TR 38.901 V16.1.0, Table 7.5-3: offsets +-a_m for the ray pairs (1,2) to (19,20).
        basis = [0.0447, 0.1413, 0.2492, 0.3715, 0.5129, 0.6797, 0.8844, 1.1481, 1.5195,
                 2.1551]
        expected = [v for a in basis for v in (a, -a)]
        model = _model()
        offsets = model._ray_sampler._ray_offsets.cpu().tolist()
        for m, (actual, exp) in enumerate(zip(offsets, expected), start=1):
            self.assertTrue(math.isclose(actual, exp, rel_tol=1e-6),
                            f"ray {m}: {actual}, expected {exp}")


class TopologyAliasing(unittest.TestCase):

    @pytest.mark.xfail(strict=True, raises=AssertionError,
                       reason="set_topology overwrites in place the tensors passed in an earlier "
                              "call; expected: its arguments are not modified")
    def test_set_topology_keeps_caller_tensors(self):
        model = _model()
        first = _topology()
        snapshot = [t.clone() for t in first]
        model.set_topology(*first)
        model.set_topology(*_topology())
        names = ("ut_loc", "bs_loc", "ut_orientations", "bs_orientations",
                 "ut_velocities", "in_state")
        for name, tensor, saved in zip(names, first, snapshot):
            self.assertTrue(torch.equal(tensor, saved), f"{name} was changed by set_topology")


class AtmosphericParameters(unittest.TestCase):
    """The set_topology docstring: "not specifying a parameter leads to the reuse of the
    previously given value"."""

    @pytest.mark.xfail(strict=True, raises=AssertionError,
                       reason="An atmospheric parameter passed to set_topology without topology "
                              "tensors does not change the gas loss; expected: it takes effect")
    def test_parameter_alone_updates_gas_loss(self):
        topology = _topology()
        reference = _model()._scenario
        reference.set_topology(*topology, temperature=300.0)

        sc = _model()._scenario
        sc.set_topology(*topology)
        sc.set_topology(temperature=300.0)
        torch.testing.assert_close(sc.gas_pathloss, reference.gas_pathloss)

    @pytest.mark.xfail(strict=True, raises=AssertionError,
                       reason="set_topology resets the atmospheric parameters that are not passed "
                              "to their defaults; expected: the previous values are reused")
    def test_parameter_persists(self):
        sc = _model()._scenario
        sc.set_topology(*_topology(), temperature=300.0)
        sc.set_topology(*_topology())
        self.assertEqual(sc.temperature, 300.0)

    @pytest.mark.xfail(strict=True, raises=TypeError,
                       reason="The set_topology method of the channel models does not accept "
                              "atmospheric parameters (TypeError); expected: accepted as by the "
                              "scenario")
    def test_channel_set_topology_accepts_parameters(self):
        model = _model()
        model.set_topology(*_topology(), temperature=300.0)
        self.assertEqual(model._scenario.temperature, 300.0)


class AtmosphericDefaults(unittest.TestCase):

    @pytest.mark.xfail(strict=True, raises=AssertionError,
                       reason="Under investigation: the default atmosphere is 273 K and 1020 hPa; "
                              "TR 38.811 clause 6.6.4 gives 288.15 K and 1013.25 hPa for all UEs")
    def test_gas_loss_defaults(self):
        # TR 38.811 V15.4.0, clause 6.6.4: T = 288.15 K, p = 1013.25 hPa,
        # rho = 7.5 g/m^3 for all UEs.
        topology = _topology()
        reference = _model()._scenario
        reference.set_topology(*topology, temperature=288.15, atmospheric_pressure=1013.25,
                               water_vapor_density=7.5)
        sc = _model()._scenario
        sc.set_topology(*topology)
        torch.testing.assert_close(sc.gas_pathloss, reference.gas_pathloss)
