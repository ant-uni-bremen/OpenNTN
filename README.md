<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/ant-uni-bremen/OpenNTN/v2.0.0rc1/docs/images/openntn-logo-dark.png"><img alt="OpenNTN" src="https://raw.githubusercontent.com/ant-uni-bremen/OpenNTN/v2.0.0a1/docs/images/openntn-logo.png" width="400"></picture></p>

<p align="center">3GPP TR 38.811 non-terrestrial network channel models for Sionna</p>

<p align="center">
<a href="https://pypi.org/project/openntn/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/openntn"></a>
<a href="https://pypi.org/project/openntn/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/openntn"></a>
<img alt="Licence: MIT AND Apache-2.0" src="https://img.shields.io/badge/licence-MIT%20AND%20Apache--2.0-blue">
<a href="https://doi.org/10.1109/WSA65299.2025.11202820"><img alt="Paper: WSA 2025" src="https://img.shields.io/badge/paper-WSA%202025-blue"></a>
</p>

## Overview

OpenNTN is an open-source extension of [Sionna](https://nvlabs.github.io/sionna/) that
implements the non-terrestrial network (NTN) channel models of 3GPP TR 38.811 for the
links between user terminals and a satellite or an aerial platform. Its channel models
follow the interface of Sionna's TR 38.901 models, so they can be used in Sionna's
link-level and system-level simulations. OpenNTN 2 is built on Sionna 2, which uses
PyTorch.

## Key features

- **Scenarios:** dense urban (`DenseUrban`), urban (`Urban`) and suburban (`SubUrban`),
  with the parameter tables of TR 38.811 clause 6.7.2 and the fast-fading process of
  TR 38.901 clause 7.5.
- **Platforms:** one satellite, from 160 km to 35 786 km, or one aerial platform, from
  8 km to 50 km, per call, at the same height in every batch example.
- **Bands and links:** S band (1.9 GHz to 4 GHz) and Ka band (19 GHz to 40 GHz), uplink
  and downlink, elevation angles from 10 to 90 degrees.
- **Propagation (TR 38.811 clause 6.6):** line-of-sight probability, free-space path
  loss, shadow fading, clutter loss, atmospheric gas absorption, ionospheric
  scintillation in the S band and tropospheric scintillation in the Ka band. The shadow
  fading is the large scale parameter SF of TR 38.901 clause 7.5, step 4, applied once
  in step 12: TR 38.811 eq. (6.6-4) writes it into the basic path loss, and its Tables
  6.7.2-1a to 6.7.2-8b give its cross-correlations and correlation distance, so the
  basic path loss of the models is the free-space path loss plus the clutter loss.
- **Satellite Doppler:** the Doppler shift of the satellite's orbital motion, in
  addition to the motion of the user terminals; it can be switched off.
- **Antennas:** Sionna-style antenna elements, panels and arrays with the TR 38.901
  element pattern, an omnidirectional pattern, a circular-aperture pattern with an
  adjustable aperture radius and the co-phased dual-linear-polarization pattern of
  3GPP R1-1802551.
- **Topology:** a generator for one satellite or aerial platform and the user terminals
  of a single sector at a given elevation angle and height.
- **Sionna integration:** the models are Sionna channel models. They return path
  coefficients and delays in Sionna's format and work with Sionna's channel functions
  and blocks, such as `cir_to_ofdm_channel` and `OFDMChannel`.

## Installation and requirements

OpenNTN 2 requires Python 3.11 or later and Sionna 2.0.1 or later within 2.x
(`sionna>=2.0.1,<3.0`). Further dependencies (PyTorch, NumPy, matplotlib) are installed
with it.

1. Install PyTorch for your platform, CPU or CUDA, as described at
   https://pytorch.org/get-started/locally/.
2. Install OpenNTN:

   ```sh
   python -m pip install openntn
   ```

On a machine with an NVIDIA GPU, the PyTorch build has to match the driver: the default
build of PyTorch 2.14.1 on PyPI, for example, uses CUDA 13.0 and needs an NVIDIA driver
of version 580 or later. With an older driver, first install a build for an older CUDA
version from the PyTorch index, then OpenNTN; the build for CUDA 12.6 runs with driver
535:

```sh
python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
```

The changes of this version are listed in the
[changelog](https://github.com/ant-uni-bremen/OpenNTN/blob/v2.0.0/CHANGELOG.md).

> **Note:** openntn 2.x requires Sionna 2.x. Installing it into an environment with
> Sionna 1.x upgrades Sionna to 2.x. For Sionna 1.x and Sionna 0.19, use OpenNTN
> 1.0.0 and 0.19.0, which their installation scripts install from the tags
> [v1.0.0](https://github.com/ant-uni-bremen/OpenNTN/tree/v1.0.0) and
> [v0.19.0](https://github.com/ant-uni-bremen/OpenNTN/tree/v0.19.0) of the repository.

Sionna is also published as `sionna-no-rt`, without the ray tracer, which OpenNTN does
not use. openntn depends on the distribution `sionna`; to use `sionna-no-rt` instead,
install it first and then OpenNTN without its dependencies, which `sionna-no-rt`
already brings (PyTorch, NumPy and matplotlib, with the same lower bounds):

```sh
python -m pip install "sionna-no-rt>=2.0.1,<3.0"
python -m pip install --no-deps openntn
```

An environment that holds an earlier version of OpenNTN, installed by its installation
script as the distribution `OpenNTN` (1.0.0 or 0.19.0 from the tags, 0.1.0 from the
branches before them), keeps it when `python -m pip install openntn` runs, as pip takes
it for the same project. Use a new environment, or uninstall the earlier version first:

```sh
python -m pip uninstall OpenNTN
```

## Versions for Sionna 2, 1.x and 0.19

| OpenNTN | Sionna | Framework | Installation | Import |
|---|---|---|---|---|
| 2.x, this version | 2.x, from 2.0.1 | PyTorch | `python -m pip install openntn`, from PyPI | `import openntn` |
| 1.0.0, for Sionna 1.x | 1.x | TensorFlow | its installation script at the tag [v1.0.0](https://github.com/ant-uni-bremen/OpenNTN/tree/v1.0.0) (Linux) | `sionna.phy.channel.tr38811` |
| 0.19.0, for Sionna 0.19 | 0.19 (0.19.2) | TensorFlow | its installation script at the tag [v0.19.0](https://github.com/ant-uni-bremen/OpenNTN/tree/v0.19.0) (Linux) | `sionna.channel.tr38811` |

The versions 1.0.0 and 0.19.0 are not on PyPI; their installation scripts install the
version of their tag. The TensorFlow versions of the example notebooks belong to them.
The
[migration guide](https://github.com/ant-uni-bremen/OpenNTN/blob/v2.0.0/docs/migration.md)
describes what changes for their users.

## Quickstart

```python
from sionna.phy.channel import cir_to_ofdm_channel, subcarrier_frequencies

from openntn import Antenna, AntennaArray, Urban
from openntn.utils import gen_single_sector_topology

carrier_frequency = 2.2e9   # S band downlink
elevation_angle = 50.0      # degrees

# User terminal: one antenna; satellite: 4 x 4 dual-polarized array
ut_array = Antenna(polarization="single", polarization_type="V",
                   antenna_pattern="omni", carrier_frequency=carrier_frequency)
bs_array = AntennaArray(num_rows=4, num_cols=4, polarization="dual",
                        polarization_type="cross", antenna_pattern="38.901",
                        carrier_frequency=carrier_frequency)

channel_model = Urban(carrier_frequency=carrier_frequency, ut_array=ut_array,
                      bs_array=bs_array, direction="downlink",
                      elevation_angle=elevation_angle)

# 16 examples of a single sector: one satellite at 600 km, 4 user terminals
topology = gen_single_sector_topology(batch_size=16, num_ut=4, scenario="urb",
                                      elevation_angle=elevation_angle, bs_height=600e3)
channel_model.set_topology(*topology)

# One channel realization: path coefficients a and path delays tau
a, tau = channel_model(num_time_samples=1, sampling_frequency=15e3)

# Frequency response on 72 subcarriers with Sionna's channel functions
frequencies = subcarrier_frequencies(num_subcarriers=72, subcarrier_spacing=30e3)
h_freq = cir_to_ofdm_channel(frequencies, a, tau, normalize=True)
print(h_freq.shape)  # [batch, rx, rx antennas, tx, tx antennas, time steps, subcarriers]
```

A channel model runs on the device given by its argument `device`, for example
`device="cuda:0"`, which defaults to `sionna.phy.config.device` at the construction of
the model, and in the precision given by its argument `precision`, `"single"` or
`"double"`, which defaults to `sionna.phy.config.precision`. Create the antennas on the
same device and in the same precision, with their arguments `device` and `precision`,
which have the same defaults; otherwise the model raises `ValueError`. A model stays on
its device and in its precision: build it with the device and precision it is to use
instead of moving it with `.to()` or casting it. Moved or cast, it raises `ValueError`
in its next `set_topology` or call.

## Example notebooks

Three Jupyter notebooks show the models at work with Sionna 2 and PyTorch:

- [channel_response.ipynb](https://github.com/ant-uni-bremen/OpenNTN/blob/v2.0.0/example/channel_response.ipynb):
  the topology, the path coefficients, the path delays and the frequency response of
  the channels, and their condition numbers in the three scenarios and for different
  heights, elevation angles, numbers of user terminals and carrier frequencies.
- [BER_sims.ipynb](https://github.com/ant-uni-bremen/OpenNTN/blob/v2.0.0/example/BER_sims.ipynb):
  an OFDM downlink from a satellite at 600 km in the S band, with LDPC coding and LMMSE
  equalization, and the uncoded bit error rate of the three scenarios over Eb/N0 with
  perfect channel knowledge.
- [ML_based_SIMO_receiver.ipynb](https://github.com/ant-uni-bremen/OpenNTN/blob/v2.0.0/example/ML_based_SIMO_receiver.ipynb):
  the neural receiver of Sionna's tutorial "Neural Receiver for OFDM SIMO Systems", in
  PyTorch, trained on the channels of the dense urban scenario and evaluated beside the
  receivers with perfect channel knowledge and with LS channel estimation.

By default, `BER_sims.ipynb` and `ML_based_SIMO_receiver.ipynb` load the results of
their full runs, which the repository stores beside them, and `channel_response.ipynb`
runs with its own batch sizes. The environment variable `OPENNTN_NOTEBOOK_MODE` set to
`full` computes the full runs (the training of the receiver needs a GPU), set to
`reduced` it runs them with small batches and writes no file. The TensorFlow notebooks of
the earlier versions belong to the versions for Sionna 1.x and 0.19.

## Scope and conventions of the models

- **Elevation angle and positions:** the elevation angle of a model sets the slant
  range, the LOS probability, the parameter tables and the losses of every link, while
  the LOS directions follow the positions; place the satellite or platform so that it
  appears at that elevation from the user terminals, as `gen_single_sector_topology`
  does.
- **Reference elevation angles:** every table is taken at the reference elevation angle
  of TR 38.811 nearest to the elevation angle (10 to 90 degrees in steps of 10 degrees),
  an angle midway between two taking the higher one; no table is interpolated.
- **Orientations:** the topology generators place the satellite or platform at
  (0, 0, h) facing the nadir and turn each user terminal toward it. A directional
  satellite element at the nadir sees the user terminals off its boresight, so set
  `bs_orientations` and `ut_orientations` for beam pointing or a defined polarization.
- **Delays:** paths without power, of clusters that a link does not have, carry
  placeholder delays of seconds, as in Sionna's models, so read the delays `tau`
  together with the path coefficients. The delays are relative to the first path; the
  propagation delay of the link is not included.
- **User terminals:** outdoors only; `set_topology` raises `ValueError` for indoor user
  terminals.

The [description of the models](https://github.com/ant-uni-bremen/OpenNTN/blob/v2.0.0/docs/model.md)
lists the scope, the conventions and the limits of the models in full, with their
calibration against 3GPP TR 38.821 and their speed on a GPU.

## Migrating from the versions for Sionna 1.x and 0.19

OpenNTN 2 keeps the names of the models, the antennas and the topology generators, but
it is a separate package, `openntn`, built on PyTorch: its inputs and outputs are
PyTorch tensors, it is installed with pip, and corrections of the models change
simulation results. The
[migration guide](https://github.com/ant-uni-bremen/OpenNTN/blob/v2.0.0/docs/migration.md)
lists the changes by topic, with the changes of the API that existing code has to
follow; the [changelog](https://github.com/ant-uni-bremen/OpenNTN/blob/v2.0.0/CHANGELOG.md)
describes every change and correction.

## Citation

If you use OpenNTN in your research, please cite the version of the software that you
used, as the
[citation file](https://github.com/ant-uni-bremen/OpenNTN/blob/v2.0.0/CITATION.cff)
of the repository describes it, and the paper that presents the framework:

T. Düe, M. Vakilifard, C. Bockelmann, D. Wübben and A. Dekorsy, "OpenNTN: An
Open-Source Framework for Non-Terrestrial Network Channel Simulations," 2025 28th
International Workshop on Smart Antennas (WSA), Erlangen, Germany, 2025,
https://doi.org/10.1109/WSA65299.2025.11202820

```bibtex
@inproceedings{OpenNTNPaper,
  author = {T. D\"{u}e and M. Vakilifard and C. Bockelmann and D. W\"{u}bben and A. Dekorsy},
  title = {OpenNTN: An Open-Source Framework for Non-Terrestrial Network Channel Simulations},
  booktitle = {2025 28th International Workshop on Smart Antennas (WSA)},
  address = {Erlangen, Germany},
  year = {2025},
  month = {Sep},
  pages = {1--7},
  doi = {10.1109/WSA65299.2025.11202820}
}
```

## Licence

OpenNTN is distributed under the licence expression `MIT AND Apache-2.0`:

- Files derived from NVIDIA Sionna are licensed under the Apache License 2.0
  ([LICENSE-APACHE](https://github.com/ant-uni-bremen/OpenNTN/blob/v2.0.0/LICENSE-APACHE)).
  Their headers keep NVIDIA's copyright notice and, for modified files, state that
  they were modified at the University of Bremen.
- All other files are licensed under the MIT License
  ([LICENSE](https://github.com/ant-uni-bremen/OpenNTN/blob/v2.0.0/LICENSE)).

Data files cannot carry a header. `TDL-A30.json`, `TDL-B100.json` and `TDL-C300.json`
in `openntn/models` are unchanged copies of Sionna's files and therefore under
Apache-2.0; the other TDL and CDL files hold the NTN-TDL and NTN-CDL parameters of
TR 38.811 (Tables 6.9.1-1 to 6.9.1-4 and 6.9.2-1 to 6.9.2-4) and are under MIT like all
other files.

The OpenNTN logo is not covered by these licences.

## Maintainer and contact

Louis Lagona <lagona@ant.uni-bremen.de>

Bug reports and questions: https://github.com/ant-uni-bremen/OpenNTN/issues

## Acknowledgements

OpenNTN was originally written by Tim Düe at the Department of Communications
Engineering (Arbeitsbereich Nachrichtentechnik) of the University of Bremen,
https://www.ant.uni-bremen.de/. It builds on NVIDIA Sionna.
