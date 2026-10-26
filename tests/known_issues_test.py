# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of behaviour that the implementation did not meet, or does not meet yet.
#
# Each test states the behaviour required by the specification or by the documented
# interface. Where the implementation does not meet it yet, the test is marked as a
# strict expected failure whose reason states the observed and the expected behaviour.
# Strict means that the test fails as soon as the behaviour changes, so the marker has
# to be removed in the same change; the test then stays as a regression test.
import json
import math
import os
import unittest

import torch

from openntn import Antenna, AntennaArray, Urban
from openntn import utils

from statistical_checks import assert_correlation, assert_std

CARRIER_FREQUENCY = 2.2e9
ELEVATION_ANGLE = 50.0
BS_HEIGHT = 600000.0


def _model(enable_shadow_fading=True, direction="downlink",
           carrier_frequency=CARRIER_FREQUENCY):
    ut_array = Antenna(polarization="single",
                       polarization_type="V",
                       antenna_pattern="38.901",
                       carrier_frequency=carrier_frequency)
    bs_array = AntennaArray(num_rows=1,
                            num_cols=4,
                            polarization="dual",
                            polarization_type="VH",
                            antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency)
    return Urban(carrier_frequency=carrier_frequency,
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


with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "spec_tables",
                       "tr38811_v15.4.0.json"), encoding="ascii") as _f:
    TABLES = json.load(_f)["tables"]


def _limited_correlation(a):
    """Correlation of Z and min(Z, a) for a standard normal Z: Cov(Z, min(Z, a)) is
    Phi(a), and min(Z, a) has the mean -phi(a) + a (1 - Phi(a)) and the second moment
    Phi(a) - a phi(a) + a^2 (1 - Phi(a))."""
    pdf = math.exp(-0.5 * a * a) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + math.erf(a / math.sqrt(2.0)))
    mean = -pdf + a * (1.0 - cdf)
    second = cdf - a * pdf + a * a * (1.0 - cdf)
    return cdf / math.sqrt(second - mean * mean)


class ShadowFading(unittest.TestCase):
    """TR 38.811 V15.4.0, Section 6.6.2, eq. (6.6-4): the path loss contains one shadow
    fading term SF ~ N(0, sigma_SF^2); Table 6.6.2-2 gives sigma_SF = 4 dB (LoS) and
    6 dB (NLoS) for the urban scenario. Tables 6.7.2-3a (LoS) and 6.7.2-4a (NLoS) give
    the cross-correlations of this shadow fading with the other large scale parameters,
    which the large scale parameter SF of TR 38.901 V16.1.0, clause 7.5, step 4 carries;
    step 12 applies it to the channel coefficients.

    Scenario of the tests: urban, S band (2.2 GHz), downlink, 50 degrees elevation,
    satellite at 600 km, links with one UT per batch example, so that the LSP draws
    of different links are independent. Measured quantity: the shadow fading in dB that
    step 12 applies to the channel coefficients, i.e. the loss of the step 12 gain on
    unit coefficients minus FSPL, gas, scintillation and entry loss and, for NLoS links,
    the clutter loss."""

    def test_shadow_fading_applied_once(self):
        # Expected: standard deviation sigma_SF.
        model = _model()
        model.set_topology(*_topology(batch_size=2000, num_ut=1))
        sf_db, los = _applied_shadow_fading_db(model)
        assert_std(sf_db[los], 4.0, "applied SF, LoS")
        assert_std(sf_db[~los], 6.0, "applied SF, NLoS")

    def test_shadow_fading_disabled(self):
        # TR 38.811 has no switch for the shadow fading; enable_shadow_fading=False is a
        # model option ("If True, apply shadow fading. Otherwise doesn't."). With it, the
        # SF term of eq. (6.6-4) is absent and the applied shadow fading must be 0 dB.
        model = _model(enable_shadow_fading=False)
        model.set_topology(*_topology(batch_size=2000, num_ut=1))
        sf_db, los = _applied_shadow_fading_db(model)
        self.assertLess(float(sf_db.abs().max()), 1e-3,
                        f"applied SF remains: standard deviation {float(sf_db[los].std()):.2f} dB "
                        f"(LoS), {float(sf_db[~los].std()):.2f} dB (NLoS)")

    def test_shadow_fading_correlations(self):
        # Tables 6.7.2-3a (LoS) and 6.7.2-4a (NLoS), 50 degrees: the correlations of the
        # shadow fading with lg DS and lg ZSA (the sign of the shadow fading as NOTE 2 of
        # the tables: positive means more received power). Step 4 limits ZSA to 52
        # degrees ("ZSA = min(ZSA, 52°)"); lg ZSA and SF are jointly normal, so the
        # correlation of SF with the limited lg ZSA is that of the table times
        # corr(Z, min(Z, a)) = Phi(a) / sqrt(Var(min(Z, a))) for a standard normal Z,
        # with a = (lg 52 - mu_lgZSA) / sigma_lgZSA of the table. DS is not limited.
        # 4000 links per link state, LoS and NLoS forced.
        k = int(ELEVATION_ANGLE / 10) - 1
        for los, table in ((True, "6.7.2-3a"), (False, "6.7.2-4a")):
            rows = TABLES[table]["rows"]
            model = _model()
            model.set_topology(*_topology(batch_size=4000, num_ut=1), los=los)
            gain_db = -_applied_shadow_fading_db(model)[0]
            a = ((math.log10(52.0) - float(rows["mu_lgZSA"][k]))
                 / float(rows["sigma_lgZSA"][k]))
            label = f"{'LoS' if los else 'NLoS'}, Table {table}"
            expected = {
                "ds": float(rows["corr_DS_vs_SF"][k]),
                "zsa": float(rows["corr_ZSA_vs_SF"][k]) * _limited_correlation(a)}
            for name, rho in expected.items():
                lg = torch.log10(getattr(model._lsp, name).double())
                assert_correlation(gain_db, lg, rho, f"{label}: SF and lg {name.upper()}")


class RayOffsets(unittest.TestCase):

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

    def test_set_topology_keeps_gradients(self):
        # The stored topology is a copy, not a detached tensor: quantities derived from
        # it stay differentiable with respect to the tensors passed in.
        model = _model()
        topology = _topology()
        ut_loc = topology[0].clone().requires_grad_(True)
        topology[0] = ut_loc
        model.set_topology(*topology)
        model._scenario.distance_2d.sum().backward()
        self.assertIsNotNone(ut_loc.grad)
        self.assertTrue(bool(torch.isfinite(ut_loc.grad).all()))
        self.assertGreater(float(ut_loc.grad.abs().sum()), 0.0)


class AtmosphericParameters(unittest.TestCase):
    """The set_topology docstring: "not specifying a parameter leads to the reuse of the
    previously given value"."""

    def test_parameter_alone_updates_gas_loss(self):
        topology = _topology()
        reference = _model()._scenario
        reference.set_topology(*topology, temperature=300.0)

        sc = _model()._scenario
        sc.set_topology(*topology)
        sc.set_topology(temperature=300.0)
        torch.testing.assert_close(sc.gas_pathloss, reference.gas_pathloss)

    def test_temperature_leaves_ka_band_scintillation_loss_unchanged(self):
        # Ka band: the tropospheric scintillation loss is the value of TR 38.811
        # V15.4.0, Table 6.6.6.2.1-1 (clause 6.6.6.2.1), so the temperature does not
        # change it, whether it is passed alone or with the topology. The gas loss,
        # which the temperature changes, has its own test above.
        topology = _topology()
        reference = _model(carrier_frequency=20e9)._scenario
        reference.set_topology(*topology, temperature=300.0)

        sc = _model(carrier_frequency=20e9)._scenario
        sc.set_topology(*topology)
        before = sc.scintillation_pathloss.clone()
        sc.set_topology(temperature=300.0)
        self.assertTrue(torch.equal(sc.scintillation_pathloss, before))
        self.assertTrue(torch.equal(reference.scintillation_pathloss, before))

    def test_parameter_alone_draws_nothing(self):
        # Only the gas and scintillation losses are recomputed: the LoS states, the
        # basic path loss and the large scale parameters, among them the shadow fading,
        # stay as they are.
        model = _model()
        model.set_topology(*_topology())
        sc = model._scenario
        lsp_names = ("ds", "asd", "asa", "sf", "k_factor", "zsa", "zsd")

        def state():
            return ([sc.los.clone(), sc.basic_pathloss.clone(), sc.distance_2d_in.clone()]
                    + [getattr(model._lsp, n).clone() for n in lsp_names])

        before = state()
        model.set_topology(temperature=300.0)
        self.assertEqual(sc.temperature, 300.0)
        for b, a in zip(before, state()):
            self.assertTrue(torch.equal(b, a))

    def test_parameter_persists(self):
        sc = _model()._scenario
        sc.set_topology(*_topology(), temperature=300.0)
        sc.set_topology(*_topology())
        self.assertEqual(sc.temperature, 300.0)

    def test_channel_set_topology_accepts_parameters(self):
        model = _model()
        model.set_topology(*_topology(), temperature=300.0)
        self.assertEqual(model._scenario.temperature, 300.0)
        model.set_topology(*_topology())
        self.assertEqual(model._scenario.temperature, 300.0)


class AtmosphericDefaults(unittest.TestCase):

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
