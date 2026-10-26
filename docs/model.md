# The channel models: scope, conventions and limits

This page states what the channel models `DenseUrban`, `Urban` and `SubUrban` of OpenNTN
implement of 3GPP TR 38.811 V15.4.0 and TR 38.901 V16.1.0, which TR 38.811 uses for the
fast fading, where they take a side, and what they leave out. The
[changelog](../CHANGELOG.md) describes the corrections of the models and the earlier
versions that each correction affects.

## Platforms, heights and elevation angles

- **Platforms.** A model simulates one satellite, from 160 km to 35 786 km, or one
  aerial platform, from 8 km to 50 km, per call, at the same height in every batch
  example; `set_topology` raises `ValueError` otherwise. TR 38.811 Table 4.5-1 gives the
  typical satellite altitudes of its evaluations (LEO 600 km to 1500 km, MEO 7000 km to
  20 000 km, GEO 35 786 km); satellites from 160 km to below 600 km lie below these
  altitudes and are accepted as well.
- **Satellites and aerial platforms.** The departure angle spreads at a satellite are
  zero (NOTE 8 of TR 38.811 Tables 6.7.2-1a to 6.7.2-8b), in both directions, and the
  motion of a satellite that is not geostationary adds a Doppler shift. Below 6 GHz,
  the ionospheric scintillation loss applies to every satellite whose user terminals
  lie within 20° of the magnetic equator, and to no aerial platform, whose links stay
  below the ionosphere. The Faraday rotation of TR 38.811 clause 6.8.2, given "for
  spaceborne BS above the ionosphere", applies from 600 km, the lowest satellite
  altitude of TR 38.811: a satellite below 600 km has the ionospheric scintillation
  loss, but no Faraday rotation.
- **Elevation angles.** Elevation angles from 10° to 90° are accepted, the range of the
  tables of TR 38.811. Every table is taken at the reference elevation angle nearest to
  the elevation angle, an angle midway between two taking the higher one, as TR 38.811
  specifies for its LOS probability, shadow fading and clutter loss; no table is
  interpolated.
- **Elevation angle and positions.** The elevation angle of a model sets the slant
  range, the LOS probability, the parameter tables and the losses of every link, while
  the LOS directions follow the positions; place the satellite or platform so that it
  appears at that elevation from the user terminals, as `gen_single_sector_topology`
  does.
- **Carrier frequencies and environments.** Carrier frequencies from 1.9 GHz to 4 GHz
  use the S-band tables of TR 38.811 and from 19 GHz to 40 GHz the Ka-band tables; other
  frequencies are rejected. The models cover the dense urban, urban and sub-urban
  environments; the rural and open environments are not implemented.
- **User terminals.** User terminals are outdoors; user terminals in buildings or
  vehicles (TR 38.901 clauses 7.4.3.1 and 7.4.3.2) are not modelled, and
  `set_topology` raises `ValueError` for indoor user terminals.

## Topology and antennas

- **Orientations of the topology generators.** The topology generators place the
  satellite or platform at (0, 0, h) facing the nadir and turn each user terminal toward
  it, with a slant of the elevation angle plus 90°; the power of single elements of
  dual-polarized arrays and the coupling of single-polarized antennas depend on these
  orientations, so set `bs_orientations` and `ut_orientations` for beam pointing or a
  defined polarization. A directional satellite element at the nadir sees the user
  terminals off its boresight: the main lobe of the element `"aperture"` with its
  default radius of 10 wavelengths reaches only user terminals above 86.5°, and toward
  user terminals at 80°, 60°, 30° and 10° its gain lies 30.54 dB, 43.97 dB, 60.91 dB
  and 49.80 dB below the gain at its boresight; the element `"38.901"` gives a user
  terminal at 10° (azimuth 0) 18.18 dB less than pointed at it.
- **Antenna elements and arrays.** The antenna elements are linearly polarized, with
  the fields of polarization model 2 of TR 38.901 clause 7.3.2; circular polarization
  and the VSAT antenna of TR 38.811 clause 6.4.2 are not implemented. The element
  `"omni"` is isotropic in all directions, a simplification of the quasi-isotropic UE
  antenna of TR 38.811 clause 6.4.2, a dipole that is omnidirectional in one plane. An
  array has one orientation, so the two back-to-back panels of the UE of R1-1802551 for
  the higher band cannot be built as one array. The antenna arrays of a model are built
  on the device and in the precision of the model; otherwise the model raises
  `ValueError`.

## Propagation and losses

- **Gas loss.** The gas loss is the Earth-space attenuation of TR 38.811 eq. (6.6-8) for
  every platform; for an aerial platform below 10 km, Annex 2 of ITU-R P.676-10 gives a
  lower inclined-path value (at 8 km up to 0.34 dB less in the supported bands). As
  TR 38.811 clause 6.6.4 states, the method does not apply to user terminals above
  10 km and to frequencies within 0.5 GHz of the centres of resonance lines; the models
  check neither. The oxygen absorption of TR 38.901 clause 7.6.1 is not applied; it is
  0 dB/km below 52 GHz, and the gas loss of TR 38.811 covers the whole path.
- **Rain and clouds.** The models consider clear sky, the baseline of TR 38.811
  clause 6.6.5 for system-level simulations.
- **Shadowing.** Shadowing is the log-normal shadow fading of TR 38.811 clause 6.6.2,
  applied once as the large scale parameter SF. The "LMS" shadowing that Table 6.10.1-1
  names belongs to the two-state flat-fading model of clause 6.7.1 (ITU-R P.681), which
  the models do not implement.
- **Ionospheric scintillation.** Below 6 GHz the ionospheric scintillation loss is the
  one value of TR 38.811 clause 6.6.6.1.4 at every elevation (2.2 dB at 2 GHz within
  20° of the magnetic equator), the level of ITU-R P.531-13 Figure 12, which is given
  for about 20° elevation.
- **Faraday rotation.** The Faraday rotation is 108°/f² (f in GHz) on every link to a
  satellite from 600 km, the estimated maximum of ITU-R P.531-13, Table 3, for about 30°
  elevation, as TR 38.811 eq. (6.8-2) gives it.
- **Inconsistencies of TR 38.811.** Where TR 38.811 is inconsistent the models use the
  Earth radius of 6371 km of clause 6.3 and the sign of NOTE 2 of Tables 6.7.2 (a
  positive SF raises the received power).

## Fast fading

- **Model.** The models implement the frequency-selective fast fading of TR 38.811
  clause 6.7.2; the two-state flat-fading model of clause 6.7.1 is not implemented. The
  calibration of TR 38.811 Annex A is not reproduced as specified (one elevation per
  call); the models are compared with the calibration results of TR 38.821 (below).
- **K-factor.** The Rician K-factor follows the tables of TR 38.811 clause 6.7.2 as
  printed, without clipping; they give large means at low elevation (urban 31.83 dB in
  the S band and 40.18 dB in the Ka band at 10°) and standard deviations up to
  18.96 dB.
- **Spatial consistency and other components.** Only the large scale parameters are
  correlated in space; delays, angles, powers, phases and the LOS state are drawn
  independently per link and per call (drop-based simulation, without the spatial
  consistency of TR 38.901 clause 7.6.3). Blockage, multi-frequency correlation, UT
  rotation, ground reflection and embedded devices (TR 38.901 clause 7.6) are not
  modelled; two models at different carrier frequencies draw independent parameters.
- **Delays.** Delays are relative to the first path; the propagation delay d_3D/c
  (2.0 ms at 600 km and 90°, 6.4 ms at 10°) is not included. Every link has the number
  of clusters of the link state with the most clusters; the clusters that the state of a
  link does not have get zero power and a placeholder delay of 1 s, divided on LOS links
  by the factor C_τ of TR 38.901 clause 7.5, step 5, as the other delays are, as in
  Sionna's models (up to 12.7 s in the reference outputs of the test suite). Read the
  delays `tau` together with the path coefficients: paths of zero power carry
  placeholder delays of seconds.
- **Doppler shift.** Within a call the Doppler phase of each path is linear in time;
  the UT velocity gives each path its Doppler shift, and the motion of a
  non-geostationary satellite adds one shift per batch example, common to all user
  terminals and paths (TR 38.811 clauses 6.7.1 and 6.9.2); geostationary satellites,
  aerial platforms and scatterers do not move.

## Calibration against TR 38.821

The models were compared with the system-level calibration of 3GPP TR 38.821 V16.2.0,
clause 6.1.1: the coupling loss, the geometry SIR and the geometry SINR at the 5th, 50th
and 95th percentiles of its Tables 6.1.1.2-1 (downlink) and 6.1.1.2-2 (uplink), in all
30 cases of its Table 6.1.1.1-9 (GEO, LEO at 1200 km and at 600 km; the two sets of
satellite parameters; handheld terminals in the S band and VSATs in the Ka band; the
frequency reuse factors 1 and 3 and the reuse factor 2 with polarization reuse). The
rural baseline is reproduced with the sub-urban model, to which TR 38.811 gives the same
LOS probability, shadow fading and clutter loss (Tables 6.6.1-1 and 6.6.2-3), every UE
in line of sight. The beam layout, the drop and the attachment of the UEs, the antenna
gains and the link budget follow TR 38.821 and are computed outside the models, with the
satellite pattern of the models (TR 38.811 clause 6.4.1, the aperture radius from the
equivalent aperture) and the maximum gains of TR 38.821. A model has one elevation
angle, so the UEs are grouped by their elevation in steps of 0.1°. The polarization
enters as TR 38.821 states it: no depolarization loss, and the beams of the other
polarization do not interfere. The calibration therefore does not exercise the
polarization model, and, as a large scale calibration, not the fast fading either.
TR 38.821 cites TR 38.811 V15.2.0; the models follow V15.4.0, which later corrected the
shadow fading parameters of the rural scenario; the tables that the calibration uses
are the same in both versions.

With 200 drops of 10 UEs per beam in each case and direction, 538 of the 540 percentiles
lie within 0.3 dB (the coupling loss, the downlink SIR and SINR) or 0.5 dB (the uplink
SIR and SINR) of the average of the results that the companies contributed to TR 38.821
(3GPP R1-1913404 and R1-1913405), among them all 144 of the eight cases of the first
priority, within 0.16 dB. The two others: in one case (LEO at 600 km, set 2, Ka band,
reuse factor 2 with polarization reuse) the downlink coupling loss at the 5th
percentile, 0.33 dB below the average and within the range of the companies; in one
case (GEO, set 2, S band, reuse factor 3) the downlink geometry SIR at the 95th
percentile, 0.65 dB above the average and 0.05 dB above the highest company. That case
discards the 24 beams whose 3 dB beam reaches beyond the limb of the Earth, as TR 38.821
discards a beam with coverage outside the Earth; without that discard the value lies
0.39 dB below the average, within the range of the companies. In the uplink no UE is
placed below the elevation angle of 10° that the models accept; with the discarded beams
no cell of a beam reaches below it.

## Precision, devices and speed

- **Device and precision.** A model runs on the device and in the precision of its
  construction, its arguments `device` and `precision`, which default to
  `sionna.phy.config.device` and `sionna.phy.config.precision`. It stays there: moved
  with `.to()` or cast, it raises `ValueError` in its next `set_topology` or call.
- **Gradients.** The path coefficients and delays are detached from the autograd graph,
  as in Sionna, so gradients with respect to the topology exist only for the quantities
  of the scenario.
- **Tensors of a model.** `set_topology` writes the tensors that it derives from the
  topology into the existing tensors when their shapes stay the same and no gradients
  are involved. A tensor taken from a model after `set_topology`, such as the path loss
  terms or the means of the large scale parameters, is overwritten by the next
  `set_topology` of the same shapes; clone it to keep its values.
- **Random numbers.** Every random number comes from the generators of Sionna, which
  `sionna.phy.config.seed` seeds. The same seed gives other realizations on the CPU
  than on a GPU, and a call compiled with `torch.compile` draws other random numbers
  than an eager call, unless `torch._inductor.config.fallback_random` is set.
- **Compilation and speed.** `set_topology` runs eagerly, outside a compiled function;
  the call of a model can be compiled with `torch.compile`, without graph breaks, and
  with `torch.compile(mode="reduce-overhead")`, which records CUDA graphs, across new
  topologies of the same shapes. On an NVIDIA GeForce RTX 4090 (PyTorch 2.14.1, Sionna
  2.0.1 and 2.2.0), an eager call of `DenseUrban` (a satellite at 600 km; 4 batch
  examples of 3 user terminals or 128 of 10) took a median of 8.2 ms to 8.7 ms, the time
  of launching its GPU kernels from the host; compiled it was 3.2 to 4.7 times faster,
  and with CUDA graphs 4.5 to 19.4 times faster. A topology with another batch size
  records new CUDA graphs, and PyTorch keeps every recorded graph, so alternating
  between two batch sizes records the graphs again at every change.
