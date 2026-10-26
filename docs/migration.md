# Migrating from the versions for Sionna 1.x and 0.19

This guide is for users of the earlier versions of OpenNTN, the version for Sionna 1.x
and the version for Sionna 0.19, which are built on TensorFlow. OpenNTN 2 keeps the
names of the channel models, the antennas and the topology generators, but it is a
separate package, `openntn`, built on Sionna 2 and PyTorch, and its corrections of the
models change simulation results. The guide lists what changes for existing code, by
topic; the [changelog](../CHANGELOG.md) describes every change and every correction,
with the earlier versions that each correction affects.

## 1. The package and its import

The earlier versions were linked into Sionna by their installation scripts and imported
from Sionna's channel package. OpenNTN 2 is a package of its own:

| Version | Import |
|---|---|
| for Sionna 1.x | `from sionna.phy.channel.tr38811 import DenseUrban` |
| for Sionna 0.19 | `from sionna.channel.tr38811 import DenseUrban` |
| OpenNTN 2 | `from openntn import DenseUrban` |

The utilities move in the same way, for example
`from sionna.phy.channel.tr38811.utils import gen_single_sector_topology` becomes
`from openntn.utils import gen_single_sector_topology`. The package exports the classes
of the earlier versions under the same names (`AntennaElement`, `AntennaPanel`,
`PanelArray`, `Antenna`, `AntennaArray`, `LSP`, `LSPGenerator`, `Rays`,
`RaysGenerator`, `SystemLevelScenario`, `Topology`, `ChannelCoefficientsGenerator`,
`SystemLevelChannel`, `DenseUrbanScenario`, `DenseUrban`, `UrbanScenario`, `Urban`,
`SubUrbanScenario` and `SubUrban`), but not `TDL` and `CDL` (section 5).
`openntn.__version__` gives the installed version.

## 2. The installation

- OpenNTN 2 is installed with pip from PyPI, `python -m pip install openntn`, on any
  operating system, instead of with the installation scripts, which ran on Linux only.
  It changes no file of Sionna.
- It requires Python 3.11 or later and Sionna 2.0.1 or later within 2.x; installing it
  into an environment with Sionna 1.x upgrades Sionna to 2.x.
- It depends on PyTorch, NumPy and matplotlib, besides Sionna; it needs neither
  TensorFlow nor `importlib_resources`.
- `python -m pip install "openntn[test]"` adds what the test suite of the source
  distribution needs.
- The earlier versions are installed as the distribution `OpenNTN`, 1.0.0 and 0.19.0
  from the tags `v1.0.0` and `v0.19.0` and 0.1.0 from the branches before these tags,
  which pip takes for the same project as `openntn`: in an environment that holds one
  of them, `python -m pip install openntn` keeps the earlier version. Use a new
  environment, or uninstall the earlier version first with
  `python -m pip uninstall OpenNTN`.
- The installation scripts of the earlier versions also linked their package into
  Sionna, as `sionna/phy/channel/tr38811` (Sionna 1.x) or `sionna/channel/tr38811`
  (Sionna 0.19), and added the line `from . import tr38811` to the `__init__.py` of
  Sionna's channel package. OpenNTN 2 neither needs nor removes them: the line goes
  when Sionna is reinstalled, the link stays until it is deleted. A new environment
  avoids both.

## 3. PyTorch tensors

- The classes and functions take and return PyTorch tensors (`torch.Tensor`) instead
  of TensorFlow tensors, with the same shapes: the topology tensors of `set_topology`,
  the path coefficients and delays of a call of `DenseUrban`, `Urban` and `SubUrban`,
  and the inputs and outputs of the utilities. Code around the models uses PyTorch and
  Sionna 2 as well.
- `set_topology` converts the topology tensors to the precision and the device of the
  model and also accepts NumPy arrays.
- Counts are Python ints instead of TensorFlow integer tensors, for example
  `num_ant` of the antenna arrays and `batch_size` and `num_ut` of the scenarios; the
  antenna indices `ant_ind_pol1` and `ant_ind_pol2` are `torch.int64`.
- `AntennaElement.field` takes PyTorch tensors of any floating precision and device,
  but no Python numbers or NumPy arrays.
- `torch.compile` takes the place of TensorFlow's `tf.function`. `set_topology` runs
  eagerly, outside the compiled function, and the call of a model can be compiled. Code
  that ran `set_topology` inside a function traced by `tf.function` calls it before the
  compiled function.
- The edge cases of some utilities differ from those of the TensorFlow versions, for
  example the normalization of `cir_to_ofdm_channel` and `cir_to_time_channel` for a
  link whose energy is NaN; the changelog lists them.

## 4. Precision, device and random numbers

- The precision is set with `precision="single"` or `precision="double"`, by default
  `sionna.phy.config.precision`, which sets the precision of every object at once. The
  version for Sionna 0.19 took a TensorFlow `dtype` instead; an argument `dtype` now
  raises `TypeError`.
- The objects are objects of Sionna 2, PyTorch modules: the real data type is the
  attribute `dtype` (`torch.float32` or `torch.float64`), the complex one `cdtype`, and
  the attribute `device` is new.
- Double precision works throughout. The version for Sionna 1.x raised an error of
  TensorFlow in double precision.
- The models, their scenarios, the antenna classes, `ChannelCoefficientsGenerator` and
  several utilities take the argument `device`, by default `sionna.phy.config.device`.
  The antenna arrays of a model are built on the device and in the precision of the
  model; otherwise the model raises `ValueError`.
- A model stays on the device and in the precision of its construction: moved with
  `.to()` or cast, it raises `ValueError` in its next `set_topology` or call. Build the
  model with the device and the precision it is to use.
- Every random number comes from the generators of Sionna 2, which
  `sionna.phy.config.seed` seeds, so the seed alone fixes every random number of a
  simulation on one device. The earlier versions drew most of their random numbers from
  TensorFlow's global generator, which the seed of Sionna does not seed; a run of an
  earlier version repeats its random numbers only when `tf.random.set_seed` is set as
  well.
- The same seed gives other realizations than in the earlier versions, other
  realizations on the CPU than on a GPU, and, unless
  `torch._inductor.config.fallback_random` is set, other ones in a call compiled with
  `torch.compile` than in an eager call. The statistics of the results do not depend on
  this (section 7).

## 5. Removed and renamed names

| Earlier versions | OpenNTN 2 |
|---|---|
| `TDL` and `CDL`, with their modules `tdl` and `cdl` | removed; the NTN-TDL and NTN-CDL models of TR 38.811 are not implemented, and their parameter files stay in `openntn/models` |
| `utils.compute_stallite_doppler` | removed; the channel applies the Doppler shift of the satellite |
| `utils.gen_single_sector_topology_interferers` | removed; one satellite or aerial platform per call, `utils.gen_single_sector_topology` |
| `utils.compute_pathloss_additional` and the scenario property `additional_pathloss` | removed; the models consider clear sky, as the baseline of TR 38.811 clause 6.6.5 |
| `utils.compute_pathloss_basic`, `compute_pathloss_entry`, `compute_pathloss_gas` and `compute_pathloss_scintilation` | private; the scenario properties `basic_pathloss`, `gas_pathloss`, `scintillation_pathloss`, `entry_pathloss` and `free_space_pathloss` give the path loss terms |
| the parameters `lwc`, `rain_rate`, `relative_humidity`, `diameter_earth_antenna` and `antenna_efficiency` of `set_topology`, and the scenario properties of the same names | removed; they had no effect, and passing one raises `TypeError` |
| the arguments `average_street_width` and `average_building_height` of the models and the scenarios, and the scenario properties of the same names | removed; no computation used them |
| `AntennaPanel.porlarization` | `AntennaPanel.polarization` |
| the argument `dtype` (the version for Sionna 0.19) | `precision` (section 4) |
| `basic_pathloss` with a shadow fading term | the free space path loss plus the clutter loss of NLOS links; the shadow fading is the large scale parameter SF |

Code that builds the intermediate objects itself is also affected: `Topology` takes the
LOS propagation phase `los_phase`, which the scenarios provide as their property
`los_phase`; `Rays` takes the optional `strongest_cluster_indices`; `Topology`, `LSP`
and `Rays` are plain classes; `SystemLevelScenario` no longer enforces its abstract
methods, so a subclass that lacks one is built and the missing method returns `None`,
which fails where it is used (for the two parameter file paths already in the
constructor); a star import of `openntn.channel_coefficients`, `openntn.lsp` or
`openntn.rays` binds only their two classes.

## 6. Arguments by keyword, and the checks of the inputs

- The parameters of `set_topology` after `latitude` are taken by keyword only:
  `atmospheric_pressure`, `temperature` and `water_vapor_density`, and
  `doppler_enabled` of the scenarios. The `set_topology` method of the scenarios of the
  earlier versions took `lwc` and `rain_rate` by position after `latitude`; such a call
  now raises `TypeError`:

  ```python
  # Earlier versions: lwc and rain_rate by position after latitude
  scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                        ut_velocities, in_state, los, 47.0, lwc, rain_rate)
  # OpenNTN 2: lwc and rain_rate removed, the atmosphere by keyword
  scenario.set_topology(ut_loc, bs_loc, ut_orientations, bs_orientations,
                        ut_velocities, in_state, los, 47.0, temperature=300.0)
  ```

- The `set_topology` method of `DenseUrban`, `Urban` and `SubUrban` takes `latitude`,
  `atmospheric_pressure`, `temperature` and `water_vapor_density` and passes them to the
  scenario. A value persists until it is passed again, and a value passed without new
  topology tensors recomputes the gas and scintillation losses without drawing anything
  else again. The defaults are the reference atmosphere of TR 38.811 clause 6.6.4
  (288.15 K, 1013.25 hPa, 7.5 g/m³); `latitude`, by default 47°, is the latitude of the
  user terminals relative to the magnetic equator.
- `always_generate_lsp`, `doppler_enabled`, `enable_pathloss` and `enable_shadow_fading`
  must be `True` or `False`; other values raise `TypeError`. A call of the models or
  their scenarios that passes the average street width and building height of the
  earlier versions by position raises `TypeError` as well.
- Invalid inputs raise `ValueError` instead of `AssertionError`, also under
  `python -O`: an elevation angle outside 10° to 90°, a carrier frequency outside
  1.9 GHz to 4 GHz and 19 GHz to 40 GHz, a `direction` other than `"uplink"` and
  `"downlink"`, antenna arrays whose carrier frequency differs from that of the model,
  and the other checks that the changelog lists. A `ut_array` or `bs_array` that is not
  an antenna array of the package raises `TypeError`.
- `set_topology` raises `ValueError` unless `bs_loc` holds one satellite, from 160 km to
  35 786 km, or one aerial platform, from 8 km to 50 km, at the same height in every
  batch example; for indoor user terminals; and for a `latitude` outside -90° to 90°.

## 7. Results that change, and why

Results of OpenNTN 2 differ from those of the earlier versions for two reasons.

- Corrections of the models. The sections "Changes that affect simulation results" of
  the changelog describe each correction with what was computed before and now, the
  clause of TR 38.811, TR 38.901 or the ITU-R recommendation that it follows, who is
  affected, its effect and the earlier versions that it affects. They concern the
  generation of the channel (the ray offsets, the elimination of weak clusters, the LOS
  arrival direction, the sign and the precision of the LOS phase, the zenith angle
  offset of departure, the random signs and the ray coupling per link, the ray coupling
  within the sub-clusters, the correlation of the large scale parameters, user
  terminals at the same location, the departure spreads and the cluster azimuth spread
  at the satellite, the angular scaling factors and values of the parameter files that
  differed from the tables), the losses (the atmospheric parameters of `set_topology`
  and their defaults, the gas attenuation, the tropospheric scintillation, the
  ionospheric scintillation and its absence for aerial platforms, the shadow fading
  applied once, the elevation angles midway between two reference angles), the
  satellite Doppler shift, and the antennas and the polarization (the gain of the
  element `"dlp"`, the slant angles of `"VH"` arrays, the orientation of the user
  terminals of the topology generators, the Faraday rotation in the uplink).
- Other random numbers. Every random number is drawn with PyTorch from the generators
  of Sionna 2 (section 4), so the same seed gives other realizations of every random
  quantity. Within rounding, single precision also computes some quantities otherwise.
  These differences do not change the statistics of the results.

A result of an earlier version therefore matches a result of OpenNTN 2 in its
statistics at most, and only where no correction concerns it.

## 8. The example notebooks

The three notebooks of `example/` are ported to Sionna 2, PyTorch and `openntn`. The
environment variable `OPENNTN_NOTEBOOK_MODE` sets what they compute, and the results of
the full runs of `BER_sims.ipynb` and `ML_based_SIMO_receiver.ipynb` are stored as
NumPy files beside them, which replace the pickled results of the earlier
`BER_sims.ipynb`. Their setups changed in several points, among them the noise
variance of `BER_sims.ipynb`, whose curves lie 3.01 dB to the left of those of the
earlier notebook, and the evaluation range of the receiver; the changelog lists every
change. The TensorFlow notebooks belong to the versions for Sionna 1.x and 0.19.
