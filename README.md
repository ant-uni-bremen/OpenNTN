<p align="center"><img alt="OpenNTN" src="https://raw.githubusercontent.com/ant-uni-bremen/OpenNTN/v2.0.0a1/docs/images/openntn-logo.png" width="400"></p>

<p align="center">3GPP TR 38.811 non-terrestrial network channel models for Sionna</p>

<p align="center">
<a href="https://pypi.org/project/openntn/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/openntn"></a>
<a href="https://pypi.org/project/openntn/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/openntn"></a>
<img alt="Licence: MIT AND Apache-2.0" src="https://img.shields.io/badge/licence-MIT%20AND%20Apache--2.0-blue">
<a href="https://doi.org/10.1109/WSA65299.2025.11202820"><img alt="Paper: WSA 2025" src="https://img.shields.io/badge/paper-WSA%202025-blue"></a>
</p>

## Overview

OpenNTN is an open-source extension of [Sionna](https://nvlabs.github.io/sionna/) that
implements the non-terrestrial network (NTN) channel models of 3GPP TR 38.811 for
satellite links. Its channel models follow the interface of Sionna's TR 38.901 models,
so they can be used in Sionna's link-level and system-level simulations. OpenNTN 2 is
built on Sionna 2, which uses PyTorch.

## Key features

- **Scenarios:** dense urban (`DenseUrban`), urban (`Urban`) and suburban (`SubUrban`),
  with the parameter tables of TR 38.811 clause 6.7.2 and the fast-fading process of
  TR 38.901 clause 7.5.
- **Bands and links:** S band (1.9 GHz to 4 GHz) and Ka band (19 GHz to 40 GHz), uplink
  and downlink, elevation angles from 10 to 90 degrees.
- **Propagation (TR 38.811 clause 6.6):** line-of-sight probability, free-space path
  loss, shadow fading, clutter loss, atmospheric gas absorption, ionospheric
  scintillation in the S band and tropospheric scintillation in the Ka band.
- **Satellite Doppler:** the Doppler shift of the satellite's orbital motion, in
  addition to the motion of the user terminals; it can be switched off.
- **Antennas:** Sionna-style antenna elements, panels and arrays with the TR 38.901
  element pattern, an omnidirectional pattern, a circular-aperture pattern and the
  co-phased dual-linear-polarization pattern of 3GPP R1-1802551.
- **Topology:** a generator for one satellite and the user terminals of a single sector
  at a given elevation angle and satellite height.
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

OpenNTN 2 is a pre-release until 2.0.0, and its interfaces can still change. pip
installs a pre-release only if no final release of openntn is available; otherwise
`--pre` is needed (`python -m pip install --pre openntn`). The changes of this version
are listed in the
[changelog](https://github.com/ant-uni-bremen/OpenNTN/blob/v2.0.0a1/CHANGELOG.md).

> **Note:** openntn 2.x requires Sionna 2.x. Installing it into an environment with
> Sionna 1.x upgrades Sionna to 2.x. For Sionna 1.x and Sionna 0.19, use the
> corresponding versions of OpenNTN from the repository at
> https://github.com/ant-uni-bremen/OpenNTN.

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

The models run on the device in `sionna.phy.config.device`; set it before the antennas
and the channel model are created.

## Citation

If you use OpenNTN in your research, please cite:

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
  ([LICENSE-APACHE](https://github.com/ant-uni-bremen/OpenNTN/blob/v2.0.0a1/LICENSE-APACHE)).
  Their headers keep NVIDIA's copyright notice and, for modified files, state that
  they were modified at the University of Bremen.
- All other files are licensed under the MIT License
  ([LICENSE](https://github.com/ant-uni-bremen/OpenNTN/blob/v2.0.0a1/LICENSE)).

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
