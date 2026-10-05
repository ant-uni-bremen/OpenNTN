# This file verifies that OpenNTN runs end-to-end on a CUDA GPU using the idiomatic
# Sionna 2.0 "construct-on-target" pattern (config.device set before construction), for
# every implemented TR38.811 scenario, and that constant/topology tensors follow .to()/.cuda().
# It is skipped automatically when no CUDA GPU is visible.
import unittest
import torch
from sionna.phy import config
from openntn import Antenna, DenseUrban, Urban, SubUrban
from openntn import utils


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


if __name__ == "__main__":
    unittest.main()
