# The channel models: scope, conventions and limits

This page states what the channel models `DenseUrban`, `Urban` and `SubUrban` and the
link-level models `TDL` and `CDL` of OpenNTN implement of 3GPP TR 38.811 V15.4.0 and
TR 38.901 V16.1.0, which TR 38.811 uses for the fast fading, where they take a side, and
what they leave out. The [changelog](../CHANGELOG.md) describes the corrections of the
models and the earlier versions that each correction affects.

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

## The link-level models NTN-TDL and NTN-CDL

- **Models.** `TDL` implements the tapped delay line models NTN-TDL-A to NTN-TDL-D of
  TR 38.811 clause 6.9.2, with the normalized delays and powers of Tables 6.9.2-1 to
  6.9.2-4, which are given at the reference elevation angle of 50°: NTN-TDL-A and
  NTN-TDL-B for NLOS links, NTN-TDL-C and NTN-TDL-D for LOS links, whose first tap adds
  a specular path to its Rayleigh fading. The power delay profile is normalized to a
  total power of one, the specular path included.
- **NTN-CDL models.** `CDL` implements the clustered delay line models NTN-CDL-A to
  NTN-CDL-D of clause 6.9.1, from which clause 6.9.2 filters the NTN-TDL models, with
  the same delays and powers and with the angles and the angular spreads of the clusters
  and the cross polarization power ratios of Tables 6.9.1-1 to 6.9.1-4, also given at
  50°. The angular spreads of departure at the satellite are zero, and all clusters
  leave the satellite at the azimuth 0. The model takes the antenna arrays of the
  package, `ut_array` and `bs_array`, and the direction of the link.
- **Procedures of TR 38.901.** As clause 6.9.2 asks, `TDL` applies the procedures of
  TR 38.901 clause 7.7, with the sum of sinusoids and the order of the random draws of
  Sionna's TR 38.901 TDL: the classical Doppler spectrum of each tap for a user terminal
  at a speed from `min_speed` to `max_speed`, with the peak of the specular path at 0.7
  times the maximum Doppler frequency (clause 7.7.2); the scaling of the delays to the
  RMS delay spread `delay_spread` (clause 7.7.3); a desired K-factor (clause 7.7.6); the
  spatial correlation of several antennas given by correlation matrices (clause
  7.7.5.2).
- **K-factor.** `k_factor` replaces the K-factor of TR 38.901 eq. (7.7.6-2), the power
  of the specular path over the sum of the powers of all Rayleigh taps, which the tables
  give as 10.000 dB for NTN-TDL-C and 7.000 dB for NTN-TDL-D; the delays are then
  renormalized to an RMS delay spread of one, the specular path included. The NOTEs of
  Tables 6.9.2-3 and 6.9.2-4 give another K-factor, K1 of the first tap, the power of
  the specular path over that of the Rayleigh part of the first tap, 10.224 dB and
  11.707 dB, which Sionna's TR 38.901 TDL reports as its `k_factor`. The `k_factor` of
  `CDL` replaces the same K-factor of NTN-CDL-C and NTN-CDL-D, the power of the specular
  path over that of all Laplacian clusters, and does not change its angles.
- **Angles of the CDL.** The zenith angle of departure of every ray and of the specular
  path is 90° plus the elevation angle of the link (eq. (6.9-1)); at 50° that is the
  140° of the tables, and the 190° that a sentence of clause 6.9.1 names for the tables
  is read as 140°, as a zenith angle lies between 0° and 180°. The zenith angles of
  arrival follow eq. (6.9-2) with the desired mean zenith angle of arrival equal to that
  of the model: with the angular spreads of the model they move by 50° minus the
  elevation angle, so that the specular path arrives at 90° minus the elevation angle,
  from the direction of the satellite, as Figure 6.9.1-1 shows it. Zenith angles outside
  [0°, 180°] are then reflected by the rule of TR 38.901 clause 7.5, step 7, which
  clause 6.9.1 names: wrapped to [0°, 360°), an angle of 180° or more becomes 360° minus
  the angle, at the same azimuth. The rule applies at every elevation angle, also at
  50°, where four rays of the last cluster of NTN-CDL-B lie above 180° in the table.
- **Angular spreads of the CDL.** `angular_spreads`, a pair of a desired ASA and ZSA in
  degrees, scales the azimuth angles of arrival by eq. (7.7-5) of TR 38.901
  clause 7.7.5.1 and the zenith angles of arrival by eq. (6.9-2), on every ray and on
  the specular path, with the desired means equal to those of the model. The differences
  of the azimuth angles to their mean are wrapped to [-180°, 180°] before the scaling,
  as TR 38.901 V18.0.0 adds to the clause: the azimuth angles of arrival of the tables
  lie near 180°, and without the wrap the first cluster of NTN-CDL-A, at 178.8° with the
  mean of the model at -176.4°, would move to 1.2° instead of -178.8° at half the
  spread. The spreads and the means of the model are those of Annex A of TR 38.901 over
  the rays of the table with their offsets and over the specular path. The procedure
  scales the angles of the rays, not the spreads: a scaled model need not reach the
  desired spreads; NTN-CDL-C at 50° with the desired ASA 30° and ZSA 10°, for example,
  reaches 14.1° and 8.6°, as its specular path carries 0.91 of the power. As the
  specular path is scaled with the rays, it arrives near, not at, the direction of the
  satellite: NTN-CDL-C at 70° with the desired ASA 30° and ZSA 10° puts it at the zenith
  angle 19.68° and the azimuth -180.84°, against 20° and -180° with the angular spreads
  of the model.
- **Rays, Doppler and polarization of the CDL.** The rays of each cluster have the
  offset angles of TR 38.901 Table 7.5-3 and are coupled at random within the cluster in
  every batch example (clause 7.7.1); the path coefficients follow steps 10 and 11 of
  clause 7.5 without sub-clusters, with the cross polarization power ratio of the table
  for every ray. A moving user terminal turns every ray by the Doppler of its direction
  at the terminal (clause 6.8.1), with the velocity `ut_velocity` or a speed from
  `min_speed` to `max_speed` in a random direction. The polarization of every ray turns
  by the Faraday rotation of clause 6.8.2 for a satellite from 600 km, as in the
  system-level models. With `normalize_delays=False` the delays are the normalized
  delays of the table, which `delay_spread` does not scale.
- **Orientations and uplink of the CDL.** The angles are those of the coordinate system
  of the tables, in which the clusters leave the satellite at the azimuth 0 and, with
  the angular spreads of the model, the specular path arrives at the user terminal from
  the azimuth 180°. By default both antennas point along the line of sight: the
  satellite with the bearing 0 and a downtilt of the elevation angle, toward the zenith
  angle of departure, and the user terminal with the bearing 180° and an uptilt of the
  elevation angle, toward the satellite; `bs_orientation` and `ut_orientation` [rad]
  replace them. In the uplink the departure and the arrival angles of the tables are
  swapped, the velocity of the user terminal turns the rays by their angles of
  departure, which are those at the terminal, and the Faraday rotation acts between the
  scatterers and the satellite, as in the system-level models.
- **Doppler shift of the satellite.** The shift of the formula of clause 6.9.2, with the
  speed of a circular orbit at the height of the satellite and the elevation angle of
  the link, multiplies all taps of `TDL` and all clusters, rays and antennas of `CDL`:
  all rays of the CDL leave the satellite in one direction, so that the term of the
  satellite of clause 6.8.1 is the same for all of them. The clause names the angle of
  the formula as that of the reference model but defines it as the satellite elevation
  angle; the tables are a reference at 50°, while the link is simulated at its own
  elevation angle. The formula gives the largest shift, of an approaching satellite seen
  from a user terminal in its orbital plane: 29.63 kHz at 600 km, 50° and 2 GHz,
  45.40 kHz at 10°. It is 0 for a geostationary satellite and for an aerial platform.
  `doppler_shift` replaces it: 0 for a link whose terminal pre-compensates the shift, as
  TS 38.101-5 sets the Doppler shift of the satellite motion to zero for the performance
  requirements of its Table 8.2.1.2.2.1.1-3 (clause A.4.4), a negative value for a
  receding satellite, or the shift of a moving aerial platform.
- **Sampling of the shift.** The shift is contained in the path coefficients at every
  sample. A channel in the frequency domain that is sampled once per OFDM symbol, such
  as that of `cir_to_ofdm_channel` or `OFDMChannel`, carries its phase from symbol to
  symbol, but not the inter-carrier interference that it causes within a symbol, which
  only the channel in the time domain at the sampling rate of the signal shows, such as
  that of `TimeChannel`. At 600 km, 50° and 2 GHz the shift is 1.98 subcarrier spacings
  of 15 kHz.
- **Antennas.** Several antennas of `TDL` get the same power delay profile, with the
  spatial correlation of the given matrices. The NTN-CDL models have zero angular
  spreads of departure at the satellite, so a TDL that is consistent with them has a
  correlation of rank one across the antennas of the satellite.
- **Duration, bands and parameters.** The speeds of the user terminal and of the
  satellite and the elevation angle are constant during a call, which suits simulations
  of a few TTIs, as clauses 6.9.1 and 6.9.2 allow; a satellite pass with a changing
  elevation angle is not modelled. The carrier frequencies, the elevation angles and the
  heights are those of the system-level models. Suitable delay spreads, angular spreads
  and K-factors for a band, an environment and an elevation angle follow from the
  parameters of clause 6.7.2, as clause 6.9.1 states.
  `openntn.utils.link_level_parameters` returns them for the scenario "dur", "urb" or
  "sur", the carrier frequency, the elevation angle and the link state, ready for
  `delay_spread`, `angular_spreads` and `k_factor`: the medians of the delay spread, the
  ASA and the ZSA and the mean K-factor of Tables 6.7.2-1a to 6.7.2-6b at the reference
  elevation angle nearest to the elevation angle of the link, an angle midway between
  two taking the higher one, as in the system-level models. The median is one choice of
  a suitable value: TR 38.901 clause 7.7.3 gives the medians of its system-level delay
  spreads as examples, and their 90th percentiles for NLOS links. For a desired
  K-factor, TR 38.901 clause 7.7.6 refers to the mean and the standard deviation of the
  K-factor of its system-level parameters in Table 7.5-6 and allows other values, which
  is how the function takes the mean K-factor of the tables. The NLOS tables give no
  K-factor, and the function returns `None` for it, the default of `TDL` and `CDL`.
- **Test profiles of TS 38.101-5.** The delay profiles of TS 38.101-5 V19.6.0, Annex
  B.2.1, follow from the model: NTN-TDL-A with the RMS delay spread 100 ns, its delays
  rounded to 5 ns and its powers to 0.1 dB, is NTN-TDLA100; NTN-TDL-C with the RMS delay
  spread 5 ns and the K-factor 7.95 dB, a K-factor that TS 38.101-5 does not state,
  gives NTN-TDLC5 within 0.08 dB, its delays rounded to 5 ns.
- **CDL and Sionna's TR 38.901 CDL.** Sionna's CDL reading the same tables gives the
  same delays and, in distribution, the same powers in the downlink at 50° below 600 km
  with the same orientations. It applies none of the steps of clause 6.9.1 for the
  elevation angle of the link and has no desired angular spreads, no desired K-factor
  and no Doppler shift of the satellite; it does not reflect the four rays of NTN-CDL-B
  above 180°, has no Faraday rotation, and points both antennas horizontally by default.
  In Sionna 2.0.1 the velocity of a moving user terminal turns the rays in the uplink by
  their angles of arrival, which are those of the satellite, so that its uplink has no
  Doppler spread; Sionna 2.2.0 uses the angles of departure, as `CDL` does.
- **Device, precision and compilation.** Like the system-level models, `TDL` and `CDL`
  run on the device and in the precision of their construction and raise `ValueError` in
  their next call when they are moved with `.to()` or cast; the antenna arrays of `CDL`
  must be on its device and in its precision. Their calls can be compiled with
  `torch.compile` without graph breaks.

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
