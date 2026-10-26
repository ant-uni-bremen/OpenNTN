# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# This file verifies that OpenNTN runs end-to-end on a CUDA GPU using the idiomatic
# Sionna 2.0 "construct-on-target" pattern (config.device set before construction), for
# every implemented TR38.811 scenario, that the models built with device="cuda:0" while
# config.device is cpu run on the GPU, that constant/topology tensors follow
# .to()/.cuda(), and that a model moved with .cuda() after its construction raises
# ValueError when it is called. It is skipped automatically when no CUDA GPU is visible.
import unittest

import pytest
import torch
from sionna.phy import Object, config
from openntn import Antenna, AntennaArray, DenseUrban, Urban, SubUrban
from openntn import utils

pytestmark = pytest.mark.gpu


@unittest.skipUnless(torch.cuda.is_available(), "No CUDA GPU available")
class GPUDevice(unittest.TestCase):
    r"""End-to-end GPU checks for the TR38.811 NTN channel models."""

    CARRIER_FREQUENCY = 2.0e9
    BATCH_SIZE = 64
    NUM_TIME_STEPS = 14
    SAMPLING_FREQUENCY = 15e3
    BS_HEIGHT = 600000.0
    SEED = 42

    def _build(self, ModelClass, device):
        # construct-on-target: device is chosen before the model is built
        config.device = device
        config.seed = GPUDevice.SEED
        fc = GPUDevice.CARRIER_FREQUENCY
        tx = Antenna(polarization="single", polarization_type="V",
                     antenna_pattern="38.901", carrier_frequency=fc)
        rx = Antenna(polarization="single", polarization_type="V",
                     antenna_pattern="38.901", carrier_frequency=fc)
        cm = ModelClass(carrier_frequency=fc, ut_array=rx, bs_array=tx,
                        direction="downlink", elevation_angle=30.0)
        return cm

    def _run(self, ModelClass, scenario, device):
        cm = self._build(ModelClass, device)
        cm.set_topology(*utils.gen_single_sector_topology(
            batch_size=GPUDevice.BATCH_SIZE, num_ut=1, scenario=scenario,
            bs_height=GPUDevice.BS_HEIGHT))
        a, tau = cm(GPUDevice.NUM_TIME_STEPS, GPUDevice.SAMPLING_FREQUENCY)
        return a, tau

    def _check_scenario(self, ModelClass, scenario):
        a_gpu, tau_gpu = self._run(ModelClass, scenario, "cuda:0")
        # The public call runs on GPU and returns finite, GPU-resident tensors
        self.assertTrue(a_gpu.is_cuda)
        self.assertTrue(tau_gpu.is_cuda)
        self.assertTrue(torch.isfinite(a_gpu).all())
        self.assertTrue(torch.isfinite(tau_gpu).all())
        self.assertGreater(torch.mean(torch.abs(a_gpu) ** 2).item(), 0.0)
        # Same shapes as a CPU build (channel structure is device-independent)
        a_cpu, tau_cpu = self._run(ModelClass, scenario, "cpu")
        self.assertEqual(a_gpu.shape, a_cpu.shape)
        self.assertEqual(tau_gpu.shape, tau_cpu.shape)

    def test_dense_urban_gpu(self):
        self._check_scenario(DenseUrban, "dur")

    def test_urban_gpu(self):
        self._check_scenario(Urban, "urb")

    def test_sub_urban_gpu(self):
        self._check_scenario(SubUrban, "sur")

    def _check_device_argument(self, ModelClass, scenario):
        # Built with device="cuda:0" while config.device is cpu: the topology is made on
        # the CPU, every part of the model is on cuda:0, and the call returns finite
        # tensors on cuda:0
        config.device = "cpu"
        config.seed = GPUDevice.SEED
        fc = GPUDevice.CARRIER_FREQUENCY
        tx = Antenna(polarization="single", polarization_type="V",
                     antenna_pattern="38.901", carrier_frequency=fc, device="cuda:0")
        rx = Antenna(polarization="single", polarization_type="V",
                     antenna_pattern="38.901", carrier_frequency=fc, device="cuda:0")
        cm = ModelClass(carrier_frequency=fc, ut_array=rx, bs_array=tx,
                        direction="downlink", elevation_angle=30.0, device="cuda:0")
        cm.set_topology(*utils.gen_single_sector_topology(
            batch_size=GPUDevice.BATCH_SIZE, num_ut=1, scenario=scenario,
            bs_height=GPUDevice.BS_HEIGHT))
        a, tau = cm(GPUDevice.NUM_TIME_STEPS, GPUDevice.SAMPLING_FREQUENCY)
        self.assertEqual(config.device, "cpu")
        self.assertEqual({m.device for m in cm.modules() if isinstance(m, Object)},
                         {"cuda:0"})
        self.assertEqual(a.device, torch.device("cuda:0"))
        self.assertEqual(tau.device, torch.device("cuda:0"))
        self.assertTrue(torch.isfinite(a).all())
        self.assertTrue(torch.isfinite(tau).all())

    def test_dense_urban_device_argument(self):
        self._check_device_argument(DenseUrban, "dur")

    def test_urban_device_argument(self):
        self._check_device_argument(Urban, "urb")

    def test_sub_urban_device_argument(self):
        self._check_device_argument(SubUrban, "sur")

    def test_buffers_move_with_to(self):
        # Scenario topology tensors and the LSP correlation-sqrt matrices are
        # registered buffers, so .cuda() must move them off the CPU.
        cm = self._build(DenseUrban, "cpu")
        cm.set_topology(*utils.gen_single_sector_topology(
            batch_size=8, num_ut=1, scenario="dur", bs_height=GPUDevice.BS_HEIGHT))
        sc = cm._scenario
        lsp_gen = cm._lsp_sampler
        self.assertFalse(sc._ut_loc.is_cuda)
        self.assertFalse(lsp_gen._cross_lsp_correlation_matrix_sqrt.is_cuda)
        cm.cuda()
        self.assertTrue(sc._ut_loc.is_cuda)
        self.assertTrue(sc._los.is_cuda)
        self.assertTrue(sc._distance_3d.is_cuda)
        self.assertTrue(lsp_gen._cross_lsp_correlation_matrix_sqrt.is_cuda)
        self.assertTrue(lsp_gen._spatial_lsp_correlation_matrix_sqrt.is_cuda)

    def _check_moved_model(self, ModelClass, scenario):
        # A model stays on the device and in the precision of its construction: a
        # single-precision model built on the CPU and moved with .cuda() after
        # set_topology raises ValueError when called, since .cuda() moves its registered
        # tensors but not its device (test_moved_model of tests/device_test.py, with a
        # GPU in place of the device meta)
        config.device = "cpu"
        fc = GPUDevice.CARRIER_FREQUENCY
        ut_array = Antenna(polarization="single", polarization_type="V",
                           antenna_pattern="38.901", carrier_frequency=fc)
        bs_array = AntennaArray(num_rows=1, num_cols=4, polarization="dual",
                                polarization_type="VH", antenna_pattern="38.901",
                                carrier_frequency=fc)
        cm = ModelClass(carrier_frequency=fc, ut_array=ut_array, bs_array=bs_array,
                        direction="downlink", elevation_angle=50.0, precision="single")
        cm.set_topology(*utils.gen_single_sector_topology(
            batch_size=4, num_ut=3, scenario=scenario, elevation_angle=50.0,
            bs_height=GPUDevice.BS_HEIGHT, precision="single"))
        cm.cuda()
        with self.assertRaisesRegex(ValueError, r"was built on device 'cpu' in single "
                                                r"precision, but its tensors are "
                                                r"torch\.float32 on cuda:0"):
            cm(14, 30e3)

    def test_dense_urban_moved_model(self):
        self._check_moved_model(DenseUrban, "dur")

    def test_urban_moved_model(self):
        self._check_moved_model(Urban, "urb")

    def test_sub_urban_moved_model(self):
        self._check_moved_model(SubUrban, "sur")


if __name__ == "__main__":
    unittest.main()
