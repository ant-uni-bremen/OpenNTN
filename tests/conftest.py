# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# Shared test configuration.
#
# Every test starts from the same seed, so results do not depend on which tests ran
# before or in which order. The seed can be changed with the environment variable
# OPENNTN_TEST_SEED to check that statistical tolerances are not tuned to one seed.
# The global Sionna configuration (device, precision) is restored after each test, so
# a test that changes it cannot affect the next one.
import os
import random

import numpy as np
import pytest
import torch
from sionna.phy import config

SEED = int(os.environ.get("OPENNTN_TEST_SEED", "42"))


@pytest.fixture(autouse=True)
def seed_and_restore_config():
    device = config.device
    precision = config.precision
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    # Resets the Python, NumPy and per-device PyTorch generators used by Sionna and
    # OpenNTN.
    config.seed = SEED
    yield
    config.device = device
    config.precision = precision
