# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# This file runs the quickstart of README.md, its first python code block, so that the
# example on the PyPI page keeps working. README.md is the project description of every
# release and cannot be changed on PyPI after the upload. The example is run as
# published; its configuration is small (16 examples, 4 user terminals, one time step).
import pathlib
import re

import torch

README = pathlib.Path(__file__).resolve().parent.parent / "README.md"

BATCH_SIZE = 16
NUM_UT = 4
NUM_BS_ANT = 32  # 4 x 4 dual-polarized
NUM_SUBCARRIERS = 72


def first_python_block(text):
    match = re.search(r"^```python\n(.*?)^```\s*$", text, re.MULTILINE | re.DOTALL)
    assert match is not None, "README.md has no python code block"
    return match.group(1)


def test_readme_quickstart(capsys):
    code = first_python_block(README.read_text(encoding="utf-8"))
    namespace: dict = {"__name__": "readme_quickstart"}
    exec(compile(code, str(README), "exec"), namespace)

    a, tau, h_freq = namespace["a"], namespace["tau"], namespace["h_freq"]
    # Downlink: the user terminals receive, the satellite transmits.
    num_paths = a.shape[5]
    assert a.shape == (BATCH_SIZE, NUM_UT, 1, 1, NUM_BS_ANT, num_paths, 1)
    assert tau.shape == (BATCH_SIZE, NUM_UT, 1, num_paths)
    assert h_freq.shape == (BATCH_SIZE, NUM_UT, 1, 1, NUM_BS_ANT, 1, NUM_SUBCARRIERS)
    assert torch.isfinite(torch.view_as_real(a)).all()
    assert torch.isfinite(tau).all()
    assert torch.isfinite(torch.view_as_real(h_freq)).all()
    assert torch.count_nonzero(h_freq) > 0
    # The example prints the shape of the frequency response and nothing else.
    assert capsys.readouterr().out.strip() == str(h_freq.shape)
