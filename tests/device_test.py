# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Tests of the device of the channel models.
#
# A model is built on a device, as Sionna's models are: its argument device, by default
# sionna.phy.config.device at the construction, decides where every part of the model
# creates its tensors. The antenna arrays are built by the user and must be on the same
# device. The model stays on that device and in its precision. On a machine without a
# GPU only the device cpu is available; the tests marked gpu (tests/gpu_device_test.py)
# build the models on cuda:0 while config.device is cpu.
import warnings

import pytest
import torch
from sionna.phy import Object, config

from openntn import Antenna, AntennaArray, DenseUrban, SubUrban, Urban
from openntn.utils import gen_single_sector_topology

SCENARIOS = ((DenseUrban, "dur"), (Urban, "urb"), (SubUrban, "sur"))
CARRIER_FREQUENCY = 2.0e9
ELEVATION_ANGLE = 50.0


def _arrays(device, precision=None):
    ut_array = Antenna(polarization="single", polarization_type="V",
                       antenna_pattern="38.901", carrier_frequency=CARRIER_FREQUENCY,
                       precision=precision, device=device)
    bs_array = AntennaArray(num_rows=1, num_cols=4, polarization="dual",
                            polarization_type="VH", antenna_pattern="38.901",
                            carrier_frequency=CARRIER_FREQUENCY, precision=precision,
                            device=device)
    return {"ut_array": ut_array, "bs_array": bs_array}


@pytest.mark.parametrize("scenario,key", SCENARIOS)
def test_device_argument(scenario, key):
    # Built with device="cpu" while config.device is cpu: every part of the model is on
    # cpu, and so are its tensors and its outputs
    config.device = "cpu"
    model = scenario(carrier_frequency=CARRIER_FREQUENCY, direction="downlink",
                     elevation_angle=ELEVATION_ANGLE, device="cpu", **_arrays("cpu"))
    devices = {name or "model": module.device for name, module in model.named_modules()
               if isinstance(module, Object)}
    assert set(devices.values()) == {"cpu"}, devices
    model.set_topology(*gen_single_sector_topology(
        batch_size=4, num_ut=3, scenario=key, elevation_angle=ELEVATION_ANGLE,
        bs_height=600e3))
    assert {buffer.device.type for buffer in model.buffers()} == {"cpu"}
    a, tau = model(14, 30e3)
    assert a.device.type == tau.device.type == "cpu"


@pytest.mark.parametrize("array", ["ut_array", "bs_array"])
@pytest.mark.parametrize("scenario,key", SCENARIOS)
def test_array_on_another_device(scenario, key, array):
    # An antenna array on another device than the model raises ValueError when the model
    # is built. On the CPU, an array whose device is set to cuda:0 stands in for an array
    # built on a GPU.
    config.device = "cpu"
    arrays = _arrays("cpu")
    arrays[array]._device_str = "cuda:0"
    message = (rf"is on device 'cpu', but its {array} \({type(arrays[array]).__name__}\) "
               rf"is on device 'cuda:0'")
    with pytest.raises(ValueError, match=message):
        scenario(carrier_frequency=CARRIER_FREQUENCY, direction="downlink",
                 elevation_angle=ELEVATION_ANGLE, **arrays)


# A move to the device meta (it stands in for a GPU) or a cast, with the precision of the
# model and the dtype and the device of its tensors after the move or the cast
MOVES = (("single", "meta", r"torch\.float32 on meta"),
         ("single", torch.float64, r"torch\.float64 on cpu"),
         ("double", torch.float32, r"torch\.float32 on cpu"))


def _model_and_topology(scenario, key, precision):
    config.device = "cpu"
    model = scenario(carrier_frequency=CARRIER_FREQUENCY, direction="downlink",
                     elevation_angle=ELEVATION_ANGLE, precision=precision,
                     **_arrays("cpu", precision))
    topology = gen_single_sector_topology(
        batch_size=4, num_ut=3, scenario=key, elevation_angle=ELEVATION_ANGLE,
        bs_height=600e3, precision=precision)
    return model, topology


def _move(model, move):
    with warnings.catch_warnings():
        # A cast to another floating point dtype also casts the complex buffers to real,
        # which PyTorch warns about
        warnings.simplefilter("ignore")
        model.to(move)


@pytest.mark.parametrize("order", ["before", "between", "after"])
@pytest.mark.parametrize("precision,move,tensors", MOVES,
                         ids=["single-meta", "single-float64", "double-float32"])
@pytest.mark.parametrize("scenario,key", SCENARIOS)
def test_moved_model(scenario, key, precision, move, tensors, order):
    # A model stays on the device and in the precision of its construction: .to() moves
    # or casts its registered tensors, but not its device and its precision. Moved or
    # cast, it raises ValueError in every order: in set_topology when moved or cast
    # before its first set_topology ("before") or between two ("between"), and when
    # called when moved or cast after its last set_topology ("after")
    model, topology = _model_and_topology(scenario, key, precision)
    message = (rf"was built on device 'cpu' in {precision} precision, but its tensors "
               rf"are {tensors}")
    if order == "before":
        _move(model, move)
        with pytest.raises(ValueError, match=message):
            model.set_topology(*topology)
    elif order == "between":
        model.set_topology(*topology)
        _move(model, move)
        with pytest.raises(ValueError, match=message):
            model.set_topology(*topology)
    else:
        model.set_topology(*topology)
        _move(model, move)
        with pytest.raises(ValueError, match=message):
            model(14, 30e3)


@pytest.mark.parametrize("scenario,key", SCENARIOS)
def test_moved_scenario(scenario, key):
    # The scenario of a moved model raises ValueError as well when its own set_topology
    # is called
    model, topology = _model_and_topology(scenario, key, "single")
    model.set_topology(*topology)
    _move(model, "meta")
    with pytest.raises(ValueError, match=rf"{scenario.__name__}Scenario was built on "
                                         rf"device 'cpu' in single precision, but its "
                                         rf"tensors are torch\.float32 on meta"):
        model._scenario.set_topology(*topology)
