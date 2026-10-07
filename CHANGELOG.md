# Changelog

## 2.0.0a1

First pre-release of OpenNTN for Sionna 2.x, which is built on PyTorch.

- Import name `openntn` (`import openntn`), instead of the module `tr38811` that the
  installation scripts of the earlier versions linked into Sionna.
- Installation with pip from PyPI (`pip install openntn`) instead of `install.sh`.
- Sionna 2.x is required (`sionna>=2.0.1,<3.0`), with Python 3.11 or later.
- The licence is declared as `MIT AND Apache-2.0`: files derived from Sionna are under
  the Apache License 2.0, all other files under the MIT License. Both licence texts are
  included.
- The unused model data files were removed.
