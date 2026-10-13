# SPDX-FileCopyrightText: Copyright (c) 2026 Arbeitsbereich Nachrichtentechnik
# SPDX-License-Identifier: MIT
#
# The random numbers of the package come from Sionna's helpers rand, randint, normal and
# uniform (sionna.phy.utils), with the generator of the device of the model. In eager
# mode they call the PyTorch function with that generator; under torch.compile, which
# cannot capture a generator in the graph, they draw from PyTorch's default generator of
# the device, which sionna.phy.config.seed seeds with the same seed. A PyTorch random
# function called directly with a generator breaks the graph of torch.compile.
#
# The test parses every module of the package and finds no call of a PyTorch random
# function and no in-place random method of a tensor.
import ast
import os

import openntn

TORCH_FUNCTIONS = {"rand", "randn", "randint", "randperm", "normal", "bernoulli",
                   "multinomial", "poisson", "rand_like", "randn_like", "randint_like"}
TENSOR_METHODS = {"uniform_", "normal_", "random_", "bernoulli_", "exponential_",
                  "geometric_", "log_normal_", "cauchy_"}


def _random_calls(path):
    """The direct PyTorch random calls of one module, as (line, call)."""
    with open(path, encoding="utf-8") as f:
        tree = ast.parse(f.read(), path)
    # Names bound to the module torch, and PyTorch random functions imported by name
    torch_names = {"torch"}
    imported = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "torch":
                    torch_names.add(alias.asname or "torch")
        elif isinstance(node, ast.ImportFrom) and node.module in ("torch", "torch.random"):
            for alias in node.names:
                if alias.name in TORCH_FUNCTIONS:
                    imported[alias.asname or alias.name] = alias.name
    calls = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute):
            if (func.attr in TORCH_FUNCTIONS and isinstance(func.value, ast.Name)
                    and func.value.id in torch_names):
                calls.append((node.lineno, f"{func.value.id}.{func.attr}"))
            elif func.attr in TENSOR_METHODS:
                calls.append((node.lineno, f"Tensor.{func.attr}"))
        elif isinstance(func, ast.Name) and func.id in imported:
            calls.append((node.lineno, f"torch.{imported[func.id]}"))
    return calls


def test_random_numbers_through_sionna():
    package = os.path.dirname(os.path.abspath(openntn.__file__))
    found = []
    for directory, _, names in os.walk(package):
        for name in sorted(names):
            if name.endswith(".py"):
                path = os.path.join(directory, name)
                found += [f"{os.path.relpath(path, package)}:{line} {call}"
                          for line, call in _random_calls(path)]
    assert not found, "PyTorch random functions called directly: " + ", ".join(found)
