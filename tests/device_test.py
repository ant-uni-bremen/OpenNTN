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


def _arrays(device):
    ut_array = Antenna(polarization="single", polarization_type="V",
                       antenna_pattern="38.901", carrier_frequency=CARRIER_FREQUENCY,
                       device=device)
    bs_array = AntennaArray(num_rows=1, num_cols=4, polarization="dual",
                            polarization_type="VH", antenna_pattern="38.901",
                            carrier_frequency=CARRIER_FREQUENCY, device=device)
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


@pytest.mark.parametrize("move", ["meta", "float64"])
@pytest.mark.parametrize("scenario,key", SCENARIOS)
def test_moved_model(scenario, key, move):
    # A model stays on the device and in the precision of its construction. A
    # single-precision model moved to another device after set_topology (the device meta
    # stands in for a GPU) or cast to double precision raises ValueError when called:
    # .to() moves or casts its registered tensors, but not its device and its precision
    config.device = "cpu"
    model = scenario(carrier_frequency=CARRIER_FREQUENCY, direction="downlink",
                     elevation_angle=ELEVATION_ANGLE, precision="single", **_arrays("cpu"))
    model.set_topology(*gen_single_sector_topology(
        batch_size=4, num_ut=3, scenario=key, elevation_angle=ELEVATION_ANGLE,
        bs_height=600e3, precision="single"))
    with warnings.catch_warnings():
        # The cast to float64 also casts the complex buffers to real, which PyTorch warns
        # about
        warnings.simplefilter("ignore")
        model.to("meta" if move == "meta" else torch.float64)
    with pytest.raises(ValueError, match=r"was built on device 'cpu' in single precision, "
                                         r"but its tensors are torch\.float(32 on meta|64 "
                                         r"on cpu)"):
        model(14, 30e3)
