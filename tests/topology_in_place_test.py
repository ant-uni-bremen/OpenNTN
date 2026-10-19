# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the tensors that set_topology derives from the topology.
#
# set_topology writes the tensors that it derives from the topology into the existing
# tensors when their shapes stay the same and no gradients are involved, as it writes
# the topology itself: the means and standard deviations of the large scale parameters,
# the ZOD offset, the path loss terms and the cluster mask, and the large scale
# parameters drawn with the topology. Their addresses then stay the same across
# topologies of the same shapes, which CUDA graphs (torch.compile with
# mode="reduce-overhead") need, since a graph records the addresses of its inputs. The
# values equal those of a new model given the same topology, bit for bit. A new batch
# size gives tensors of the new shape. A tensor that requires gradients is stored anew
# instead, because writing into a tensor that carries the graph of an earlier call would
# chain the graphs of the two calls.
import pytest
import torch
from sionna.phy import config

from openntn import Antenna, AntennaArray, DenseUrban, SubUrban, Urban
from openntn.utils import gen_single_sector_topology

SCENARIOS = ((DenseUrban, "dur"), (Urban, "urb"), (SubUrban, "sur"))
LINKS = (("downlink", 2.2e9), ("uplink", 30.0e9))
ELEVATION_ANGLE = 30.0
SATELLITE_HEIGHT = 600e3
SCENARIO_TENSORS = ("_lsp_log_mean", "_lsp_log_std", "_zod_offset", "_fspl", "_pl_b",
                    "_pl_g", "_pl_s", "_pl_e")
LSPS = ("ds", "asd", "asa", "sf", "k_factor", "zsa", "zsd")


def _model(scenario, direction, carrier_frequency, precision):
    ut_array = Antenna(polarization="single", polarization_type="V",
                       antenna_pattern="38.901", carrier_frequency=carrier_frequency,
                       precision=precision)
    bs_array = AntennaArray(num_rows=1, num_cols=4, polarization="dual",
                            polarization_type="VH", antenna_pattern="38.901",
                            carrier_frequency=carrier_frequency, precision=precision)
    return scenario(carrier_frequency=carrier_frequency, ut_array=ut_array,
                    bs_array=bs_array, direction=direction,
                    elevation_angle=ELEVATION_ANGLE, precision=precision)


def _topology(key, precision, seed, batch_size=4):
    config.seed = seed
    return gen_single_sector_topology(batch_size=batch_size, num_ut=3, scenario=key,
                                      elevation_angle=ELEVATION_ANGLE,
                                      bs_height=SATELLITE_HEIGHT, precision=precision)


def _derived(model):
    """The 16 tensors that set_topology derives from the topology, by name"""
    tensors = {name: getattr(model._scenario, name) for name in SCENARIO_TENSORS}
    tensors["_cluster_mask"] = model._ray_sampler._cluster_mask
    tensors.update({f"_lsp.{name}": getattr(model._lsp, name) for name in LSPS})
    return tensors


def _bits(tensor):
    """The bits of a real or complex floating point tensor, as integers."""
    if tensor.is_complex():
        tensor = torch.view_as_real(tensor)
    width = {torch.float64: torch.int64, torch.float32: torch.int32}[tensor.dtype]
    return tensor.contiguous().view(width)


def _assert_same(model, new, label):
    # The derived tensors and the outputs of a call of model equal those of new, a model
    # given the same topology with the same seed, bit for bit
    tensors, expected = _derived(model), _derived(new)
    differing = sorted(name for name in expected
                       if tensors[name].shape != expected[name].shape
                       or not torch.equal(_bits(tensors[name]), _bits(expected[name])))
    assert not differing, f"{label}: differ from a new model: {', '.join(differing)}"
    config.seed = 9
    a, tau = model(4, 30e3)
    config.seed = 9
    a_new, tau_new = new(4, 30e3)
    for name, value, value_new in (("a", a, a_new), ("tau", tau, tau_new)):
        assert value.shape == value_new.shape, name
        assert torch.equal(_bits(value), _bits(value_new)), f"{label}: {name} differs"


CASES = [pytest.param(scenario, key, direction, carrier_frequency, precision,
                      id=f"{scenario.__name__}-{direction}-{precision}")
         for scenario, key in SCENARIOS
         for direction, carrier_frequency in LINKS
         for precision in ("single", "double")]


@pytest.mark.parametrize("scenario,key,direction,carrier_frequency,precision", CASES)
def test_written_in_place(scenario, key, direction, carrier_frequency, precision):
    # Two topologies of the same shapes: after the second, the 16 tensors are at the
    # addresses of the first, with the values of a new model given the second topology
    model = _model(scenario, direction, carrier_frequency, precision)
    config.seed = 5
    model.set_topology(*_topology(key, precision, 1))
    first = {name: tensor.data_ptr() for name, tensor in _derived(model).items()}
    second_topology = _topology(key, precision, 2)
    config.seed = 6
    model.set_topology(*second_topology)
    moved = sorted(name for name, tensor in _derived(model).items()
                   if tensor.data_ptr() != first[name])
    assert not moved, f"{len(moved)} of {len(first)} at a new address: {', '.join(moved)}"
    new = _model(scenario, direction, carrier_frequency, precision)
    config.seed = 6
    new.set_topology(*second_topology)
    _assert_same(model, new, "second topology")


def test_new_batch_size():
    # A topology with another batch size gives tensors of the new shape, with the values
    # of a new model given that topology
    model = _model(DenseUrban, "downlink", 2.2e9, "double")
    config.seed = 5
    model.set_topology(*_topology("dur", "double", 1, batch_size=4))
    topology = _topology("dur", "double", 2, batch_size=8)
    config.seed = 6
    model.set_topology(*topology)
    shapes = {name: tensor.shape[0] for name, tensor in _derived(model).items()}
    assert set(shapes.values()) == {8}, shapes
    new = _model(DenseUrban, "downlink", 2.2e9, "double")
    config.seed = 6
    new.set_topology(*topology)
    _assert_same(model, new, "batch size 8")


def test_with_gradients():
    # With UT locations that require gradients the large scale parameters require
    # gradients as well, through their spatial correlation, and are stored anew in every
    # call; the other 9 tensors do not, and are written in place. The gradient of the UT
    # locations through the LOS angles and the large scale parameters is finite.
    config.precision = "double"
    model = _model(Urban, "downlink", 2.2e9, "double")
    lsp_names = {f"_lsp.{name}" for name in LSPS}
    for call in (1, 2):
        ut_loc, *rest = _topology("urb", "double", call)
        ut_loc = ut_loc.clone().requires_grad_(True)
        before = ({name: tensor.data_ptr() for name, tensor in _derived(model).items()}
                  if call == 2 else None)
        config.seed = 5
        model.set_topology(ut_loc, *rest)
        derived = _derived(model)
        requires_grad = {name for name, tensor in derived.items() if tensor.requires_grad}
        assert requires_grad == lsp_names, f"call {call}: {sorted(requires_grad)}"
        if before is not None:
            stayed = {name for name, tensor in derived.items()
                      if tensor.data_ptr() == before[name]}
            assert stayed == set(derived) - lsp_names, f"call {call}: {sorted(stayed)}"
        (model._scenario.los_aod.sum() + model._lsp.ds.sum()).backward()
        assert ut_loc.grad is not None and bool(torch.isfinite(ut_loc.grad).all()), \
            f"call {call}"
