# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# pytest puts this directory on sys.path, so its modules are imported by their bare
# names. A module named like a standard-library module is shadowed wherever the
# interpreter has that module built in, because built-in modules are found before
# sys.path is searched. Example: CPython for Windows compiles the accelerator module
# _statistics into the interpreter, so a local _statistics.py cannot be imported there,
# while on Linux and macOS it is a separate file that comes after this directory.
# sys.stdlib_module_names is the same on every platform, so the check fails everywhere.
import os
import sys

TEST_DIR = os.path.dirname(os.path.abspath(__file__))


def test_no_module_shadows_the_standard_library():
    modules = sorted(os.path.splitext(name)[0] for name in os.listdir(TEST_DIR)
                     if name.endswith(".py"))
    shadowing = [name for name in modules if name in sys.stdlib_module_names]
    assert not shadowing, f"modules named like standard-library modules: {shadowing}"
