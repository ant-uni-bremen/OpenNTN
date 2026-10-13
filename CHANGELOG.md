# Changelog

The format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

## [2.0.0a3] - 2026-10-13

Third pre-release of OpenNTN for Sionna 2.x: the models take the argument `device`, and
`torch.compile` captures their channel generation without graph breaks; the simulation
results are unchanged.

### Added

- The argument `device` of `DenseUrban`, `Urban`, `SubUrban`, their scenarios and
  `SystemLevelScenario`: the device on which the model creates its tensors and
  computes, for example `"cuda:0"`. It defaults to `sionna.phy.config.device` at the
  construction, the device that the models took before. The antenna arrays `ut_array`
  and `bs_array` must be built on the same device (the argument `device` of the
  antenna classes); otherwise the construction raises `ValueError`. Before, Sionna
  2.0.1 accepted such an array, and Sionna 2.2.0 raised a `ValueError` of its own with
  the same text. `set_topology` now draws the random LOS states on the device of the
  model; before, it drew them on the device in `sionna.phy.config.device` at the time
  of the call, which failed when that was not the device of the model (2.0.0a1 and
  2.0.0a2).

### Changed

- A model that was moved to another device or cast to another precision after its last
  `set_topology`, for example with `.to()`, `.cuda()` or `.double()`, raises
  `ValueError` when it is called: the message names the device and the precision of the
  construction and those of the tensors, and advises to build the model with the device
  and precision it is to use. The models stay on the device and in the precision of
  their construction, as Sionna's models do: `.to()` moves or casts the registered
  tensors, but not the device and the precision of the model and its parts. Before, the
  call failed inside the channel generation with a `RuntimeError` such as "Tensor on
  device cpu is not on the expected device meta!" after a move, or "expected scalar
  type Double but found ComplexFloat" after a cast to double precision. A model moved
  or cast at another point, before `set_topology` or before a new topology, still fails
  with an error from PyTorch, in `set_topology` or in the call.

### Fixed

- `torch.compile` captures the channel generation of `DenseUrban`, `Urban` and
  `SubUrban` without graph breaks, also with `fullgraph=True`, and the results are
  unchanged. The Faraday rotation of TR 38.811 V15.4.0, clause 6.8.2, converted its
  angle ψ = 108°/f², with f in GHz (eq. (6.8-2)), from a tensor to a Python number in
  every call, and `DenseUrban` held its number of rays per cluster, 20, as a tensor,
  which the generation of the rays converted to a Python number to use it as a slice
  bound. With the code of 2.0.0a2, `torch._dynamo.explain` found 11 graph breaks in a
  call of `Urban` or `SubUrban` and 19 in a call of `DenseUrban` (20 with
  `always_generate_lsp=True`), in both link directions and both precisions, and
  `fullgraph=True` failed. The angle is now computed once, when the model is built, and
  the number of rays per cluster is the integer 20 in the three models. Affected earlier
  versions: 2.0.0a1 and 2.0.0a2; the versions for Sionna 0.19 and 1.x are built on
  TensorFlow and are not affected.
- With 2.0.0a2, on a machine with a CUDA GPU, 7 cases of
  `tests/differentiable_topology_test.py` and `tests/lsp_spatial_correlation_test.py`
  failed, because the tests built tensors on the CPU: defects of the tests, not of the
  package. The tests now build these tensors on the device of the topology.
  `tests/lsp_spatial_correlation_test.py` now reads the parameter files through the
  installed package, so that it also runs against an installed package. No result of the
  package changes. Affected earlier version: 2.0.0a2.

## [2.0.0a2] - 2026-10-12

Second pre-release of OpenNTN for Sionna 2.x. Its corrections change simulation results;
they are listed first, each with the earlier versions it affects.

### Changes that affect simulation results

Elevation angles in these entries are the reference angles of TR 38.811 (10° to 90° in
steps of 10°); the models use the parameters of the reference angle nearest to the
elevation angle.

- Ray offset angles: ray 16 of every cluster is now placed at -1.1481 times the
  intra-cluster angle spread, as in TR 38.901 V16.1.0, Table 7.5-3, which TR 38.811
  V15.4.0, clause 6.7.2, uses; it was placed at -0.1481 times that spread. This
  concerns the azimuth and zenith angles of arrival and departure of every cluster, in
  all scenarios, link states, bands and directions. With the wrong value the 20 offsets
  had a mean of 0.050 instead of 0 and a standard deviation of 0.966 instead of 1.000,
  so the rays of each cluster were shifted by 0.05 times the intra-cluster spread and
  their spread was 3.4 % too small (computed from the table). Affected earlier
  versions: the versions for Sionna 0.19 and 1.x, and 2.0.0a1.
- Sub-urban NLOS, Ka band, uplink, 80°: the mean of lg ZSD is now -3.20, as in TR
  38.811 V15.4.0, Table 6.7.2-6b; the parameter file held -3.30. The median ZSD (the
  zenith spread on the satellite side) and the zenith spread of the rays within each
  cluster on that side (TR 38.901 V16.1.0, eq. (7.5-20)) were 21 % too small:
  10^-3.30 instead of 10^-3.20 degrees, that is 0.00050° instead of 0.00063°. Affected
  earlier versions: the versions for Sionna 0.19 and 1.x, and 2.0.0a1.
- Urban NLOS, Ka band: the cluster ASA (c_ASA) at 30°, 40° and 50° is now 16.4°, 17.86°
  and 19.74°, as in TR 38.811 V15.4.0, Table 6.7.2-4b; the parameter files of both
  directions held 18.14°, 16.04° and 17.86°. The azimuth spread of the rays within each
  cluster on the UT side was 11 % too large at 30° and 10 % too small at 40° and 50°.
  Affected earlier versions: the versions for Sionna 0.19 and 1.x, and 2.0.0a1.
- Angular scaling factors: the factors C_phi^NLOS and C_theta^NLOS now follow the
  number of clusters of the link state at the elevation angle, as in TR 38.811 V15.4.0,
  Tables 6.7.2-1aa and 6.7.2-1ab, with the numbers of clusters of Tables 6.7.2-3a to
  6.7.2-6b (TR 38.901 V16.1.0, clause 7.5, step 7: "a scaling factor related to the
  total number of clusters"). Each parameter file held one pair of factors for all
  elevation angles, so the factors of another number of clusters were used for urban
  LOS links at 10° (4 clusters, the factors of 3 used), urban NLOS and sub-urban LOS
  links at 70° to 90° (2 clusters, the factors of 3 used) and sub-urban NLOS links at
  60° to 90° (3 clusters, the factors of 4 used), in both bands and both directions;
  dense urban is not affected. The cluster angles of TR 38.901 eqs. (7.5-9) and
  (7.5-14) are divided by these factors: they were 0.74 times (azimuth) and 0.72 times
  (zenith) their specified size where 2 clusters apply, 0.87 and 0.85 times for
  sub-urban NLOS links at 60° to 90°, and 1.15 and 1.17 times for urban LOS links at
  10°. The factors are now tables in the code; the keys `CPhiNLoS` and `CThetaNLoS`
  were removed from the parameter files. Affected earlier versions: the versions for
  Sionna 0.19 and 1.x, and 2.0.0a1.
- Atmospheric parameters of `set_topology` (`latitude`, `atmospheric_pressure`,
  `temperature`, `water_vapor_density`): a value now persists until it is passed again,
  as the `set_topology` documentation states, and a value passed without new topology
  tensors recomputes the gas and scintillation losses (TR 38.811 V15.4.0, clauses 6.6.4
  and 6.6.6) without drawing anything else again. Before, every call reset the
  parameters it did not pass to their defaults, and the losses were recomputed only
  together with a new topology. The `set_topology` method of `DenseUrban`, `Urban` and
  `SubUrban` now accepts these parameters and forwards them to the scenario; it raised a
  `TypeError` before. The default temperature and pressure are now those of the
  reference atmosphere of clause 6.6.4 (see the entry on the default atmosphere).
  `latitude` decides the ionospheric scintillation loss (see the entry on the
  ionospheric scintillation). The parameters `lwc`, `rain_rate`, `relative_humidity`,
  `diameter_earth_antenna` and `antenna_efficiency`, which had no effect and which the
  scenario reset in the same way, are removed (see "Removed"). This concerns every
  simulation that sets atmospheric parameters. For example, at 30° in the urban
  scenario, a temperature of 300 K instead of the default changes the gas loss from
  0.0728 dB to 0.0654 dB at 2.2 GHz and from 0.5156 dB to 0.4939 dB at 20 GHz (computed
  with this version); where earlier versions had reset a value or not applied it, they
  used the defaults instead. Affected earlier versions: the versions for Sionna 0.19 and
  1.x, and 2.0.0a1.
- Cluster elimination: clusters with less than -25 dB power compared with the
  strongest cluster of their link are now removed, as in TR 38.901 V16.1.0, clause
  7.5, step 6, which TR 38.811 V15.4.0, clause 6.7.2, uses; before, all clusters were
  kept with their own delays and angles. The threshold applies to the cluster powers of
  TR 38.901 eq. (7.5-6) on LOS and NLOS links, before the LOS specular component is
  added. A removed cluster gets zero power, so the shapes of the outputs do not
  change, and the remaining powers are not renormalized. This concerns all scenarios,
  link states, bands and directions. In 10,000 links per scenario, link state and
  elevation angle (S band, downlink, computed with 2.0.0a1), 0.6 % to 21 % of the links
  had at least one such cluster. A removed cluster has less than 10^-2.5 (0.32 %) of
  the power of the strongest cluster, so the cluster power of a link is now lower by
  less than 0.32 %, 0.63 % or 0.94 % (0.014 dB, 0.028 dB or 0.042 dB) with 2, 3 or 4
  clusters; on LOS links the specular component is not affected, so the power of the
  link falls by less. Affected earlier versions: the versions for Sionna 0.19 and 1.x,
  and 2.0.0a1.
- LOS arrival direction: the zenith angle of arrival of the LOS ray is now 180° minus
  the zenith angle of departure, so that the LOS arrival direction is the reversed
  departure direction, as in TR 38.901 V16.1.0, clause 7.5, step 1c, and clause 7.1,
  which TR 38.811 V15.4.0, clause 6.7.2 and eq. (6.8-1b), uses. It was the zenith angle
  of departure plus 180°, which together with the azimuth angle of arrival points at
  the elevation of the satellite but in the opposite azimuth: the arrival direction was
  off by 180° minus twice the elevation angle, that is by 160°, 120° and 60° at 10°,
  30° and 60°, and correct only at 90°. This concerns the LOS ray of every LOS link,
  in all scenarios, bands and directions: the antenna pattern at the UT, the phases
  across a UT antenna array, and the Doppler shift of the LOS ray due to the motion of
  the UT. With one 38.901 element at the UT and a 1 x 4 dual-polarized 38.901 panel at
  the satellite, the mean power of the first path (the LOS ray and the first cluster)
  was 10.0 dB, 12.7 dB and 9.0 dB too low at 10°, 30° and 60°, in both directions
  (dense urban, S band, 1000 forced LOS links, path loss and shadow fading off,
  computed); with one element with an isotropic pattern at the UT the power does not
  change. The zenith angles of arrival of the clusters keep their distribution,
  because TR 38.901 eq. (7.5-18) reflects them into [0°, 180°]; their realizations
  change. Affected earlier versions: the versions for Sionna 0.19 and 1.x, and 2.0.0a1.
- Sign of the LOS propagation phase: the LOS ray now contains exp(-j2π d3D/λ0), as in
  TR 38.901 V16.1.0, eq. (7.5-29), and TR 38.811 V15.4.0, eq. (6.8-1b); it contained
  exp(+j2π d3D/λ0). The minus sign is the one consistent with the Doppler term of the
  same equation: when a UT moves towards the satellite, d3D shrinks and -2π d3D/λ0
  grows at the rate of the Doppler phase. This concerns every
  LOS link, in all scenarios, bands and directions. The statistics do not change,
  because the phases of the rays of the first cluster, to which the LOS ray is added,
  are uniformly distributed; single realizations of the channel coefficients of LOS
  links change. Affected earlier versions: the versions for Sionna 0.19 and 1.x, and
  2.0.0a1.
- LOS propagation phase in single precision: the phase 2π d3D/λ0 of the LOS ray is now
  computed in double precision and reduced modulo 2π before the cast to the model
  precision, with d3D of TR 38.811 V15.4.0, eq. (6.6-3), evaluated in double precision
  in a form without cancellation, and with the carrier frequency as given (in single
  precision, 30 GHz, for example, is not representable). Before, d3D and the phase were
  computed in the model precision. In single precision one unit in the last place of
  d3D at 600 km (6.25 cm) is 0.4 to 6 wavelengths between 2 and 30 GHz, and d3D was off
  by up to 0.77 m at 600 km (0.5 m at 90°), so the phase was off by up to 3.1 rad, that
  is random. The LOS ray is added coherently to the first cluster, so the first-path
  power and the total power of single LOS links were affected; their distributions
  were not. The phase is now within 1e-6 rad of the specified value in single and
  double precision; for elevation angles from 10° to 90° in steps of 0.5°, 600 km and
  carrier frequencies of 2.0, 2.2, 2.5, 3.0, 3.5, 4.0, 20, 22.5, 25, 27.5 and 30 GHz,
  the largest errors are 4.3e-7 rad in single and 2.3e-7 rad in double precision
  (computed). `distance_3d` is within half a unit in the last place in single
  precision and within 2 units in double precision (600 km, steps of 0.25°,
  computed). In double precision the results change only within rounding.
  This concerns every LOS link in single precision, in all scenarios, bands and
  directions. Affected earlier versions: the versions for Sionna 0.19 and 1.x, and
  2.0.0a1, in single precision, their default precision.
- Zenith angle of departure offset: the offset mu_offset,ZOD of TR 38.901 V16.1.0,
  eq. (7.5-19), is now 0 for every link. TR 38.811 V15.4.0, clause 6.7.2, replaces the
  parameter tables of TR 38.901 with its own tables, which have no offset row. Before,
  NLOS links used the expression of TR 38.901 Table 7.5-9 (rural macro cell,
  arctan((35 - 3.5)/d_2D) - arctan((35 - 1.5)/d_2D)) in all three scenarios, computed
  in radians and added to angles in degrees. This concerns the zenith angles of the
  clusters of NLOS links at the satellite, in all scenarios, bands and directions; LOS
  links had no offset. Because the zenith spread of the clusters at the satellite is
  zero, the offset was the whole deviation of these clusters from the LOS zenith
  direction: less than 6e-6° at elevation angles of 10° to 60° (horizontal UT
  distances of 346 km and more) and at most 0.031° at 90°, for a UT 32 m from the point
  below the satellite (computed from the expression). The clusters of NLOS links are
  now centred on the LOS zenith direction at the satellite. The effect on the channel
  is small: of the reference outputs of the test suite (600 km), only the per-antenna
  powers of urban NLOS links at 90° changed by more than a relative 1e-5. Affected
  earlier versions: the versions for Sionna 0.19 and 1.x, and 2.0.0a1.
- Random signs and ray coupling per link: the random sign X_n of the cluster angles
  (TR 38.901 V16.1.0, clause 7.5, step 7, eqs. (7.5-11), (7.5-16) and (7.5-19)) and the
  random coupling of the rays within a cluster (step 8) are now drawn for each link
  between a satellite and a UT, as TR 38.901 defines them; TR 38.811 V15.4.0, clause
  6.7.2, uses this procedure unchanged. Before, both were drawn once per satellite and
  shared by all its UTs, so the clusters of all UTs of a satellite had the same signs of
  their angle offsets and the same pairing of ray angles. This concerns simulations
  with more than one UT per satellite, in all scenarios, link states, bands and
  directions: the angles of different UTs were correlated, which matters for
  multi-user and interference studies. For example, for two UTs in the dense urban
  scenario (S band, 30°, 2000 drops), the sign of the AOA offset of a cluster (other
  than the strongest) agreed in 69 % (NLOS) to 76 % (LOS) of the clusters instead of
  50 %, and on average 14 to 16 of the 20 rays of a cluster had the same position in
  the AOA order instead of 1 (computed with the earlier version). The statistics of a
  single link do not change. Because more random numbers are drawn, the realizations
  of the cluster angles, of the ray coupling and of the random variables drawn after
  them (cross-polarization ratios, initial phases) change for every link. Affected
  earlier versions: the versions for Sionna 0.19 and 1.x, and 2.0.0a1.
- Ray coupling in the two strongest clusters: the rays of the two strongest clusters
  of a link are now coupled within the sub-clusters of TR 38.901 V16.1.0, Table 7.5-5,
  as clause 7.5, step 8, requires ("within a cluster n, or within a sub-cluster in the
  case of two strongest clusters"), and step 11 splits the same two clusters into
  sub-clusters (eq. (7.5-26)); TR 38.811 V15.4.0, clause 6.7.2, uses these steps
  unchanged. The two strongest clusters are those with the largest powers of eq.
  (7.5-6) after the cluster elimination. Before, the rays of every cluster were coupled
  across the whole cluster, so a sub-cluster, at its own delay, carried angle offsets
  drawn from all 20 rays of its cluster instead of those that Table 7.5-5 assigns to
  it, and step 11 sorted the clusters again, which could select a different second
  cluster when fewer than two clusters of a link have non-zero power (34 of 2100
  sampled links in the sub-urban scenario, LOS, S band, 30°). This concerns all
  scenarios, link states, bands and directions. The angular spread of each sub-cluster
  now follows Table 7.5-5: the rms ray offsets of the sub-clusters R1, R2 and R3 are
  0.987, 1.006 and 1.025 times the intra-cluster spread (computed from Tables 7.5-3
  and 7.5-5), where before they were random around 1; the total angular spread of a
  cluster, the cluster powers and delays and the mean power of a link do not
  change. Realizations of the channel coefficients change for every link; no
  additional random numbers are drawn. Affected earlier versions: the versions for
  Sionna 0.19 and 1.x, and 2.0.0a1.
- Correlation of the large scale parameters: the large scale parameters (delay
  spread, angular spreads, Ricean K-factor, shadow fading) are now generated as TR
  38.901 V16.1.0, clause 7.5, step 4, prescribes, which TR 38.811 V15.4.0, clause
  6.7.2, uses with its own tables: with the procedure of clause 3.3.1 of WINNER II
  D1.1.2 (TR 38.901 reference [14]), which first correlates each parameter spatially
  across the UTs, with the correlation exp(-d/d_corr) of its correlation distance,
  and then applies the square root of the cross-correlation matrix of the parameters
  to each link, with the Cholesky decomposition in the order [SF, K, DS, ASD, ASA,
  ZSD, ZSA] that step 4 specifies. Before, the cross-correlation was applied first and
  the spatial correlation second, so the order of the Cholesky decomposition had no
  effect, and the parameters of one link did not have the cross-correlations of the
  tables when other UTs of the same link state were within the correlation distances
  (12 m to 50 m in the TR 38.811 tables of the three scenarios). This concerns all
  scenarios, link states, bands and directions. For example, for two UTs 20 m apart
  (dense urban, LOS, S band, downlink, 30°, 20,000 drops, computed), the correlation
  between the logarithms of the zenith spreads of arrival of the two UTs is now 0.507
  instead of 0.265, of the azimuth spreads of arrival 0.440 instead of 0.268 and of the
  delay spreads 0.473 instead of 0.512, against 0.505, 0.435 and 0.473 computed from the
  tables; the shadow fading and the K-factor keep their correlations exp(-d/d_corr),
  0.582 and 0.189 (measured 0.587 and 0.191). In the same setting, for the second of two
  UTs 5 m apart, the cross-correlation between its zenith spread of arrival and its
  shadow fading was -0.776 instead of -0.800 of the table (the first had -0.805). For
  UTs that are far apart compared with the correlation distances the statistics do not
  change. The realizations of all large scale parameters change for every link, and with
  them the cluster delays and powers; the path losses and the link states do not change.
  Affected earlier versions: the versions for Sionna 0.19 and 1.x, and 2.0.0a1.
- Satellites from 160 km to below 600 km: the Doppler shift due to the motion of the
  satellite now applies to every satellite, that is to every platform from 160 km to
  35 786 km, as the zero departure spreads at the satellite (NOTE 8 of TR 38.811
  V15.4.0, Tables 6.7.2-1a to 6.7.2-8b) did before; it applied from 600 km on. The
  Faraday rotation of clause 6.8.2 still applies from 600 km on, as in the earlier
  versions: the clause gives it "In case of propagation for spaceborne BS above the
  ionosphere", and 600 km is the lowest satellite altitude of TR 38.811 (Table 4.5-1,
  LEO; Table 5.1-1, non-GEO down to 600 km), while a lower satellite flies within the F
  region of the ionosphere, below much of its electron content. This concerns satellites
  from 160 km to below 600 km, in all scenarios, link states, bands and directions:
  their channel coefficients now contain the phase drift of the satellite Doppler shift.
  TR 38.811, clause 3.1, places LEO satellites from 500 km on; the models accept
  satellites from 160 km, very low orbits included. Results at 600 km and above and of
  aerial platforms (8 km to 50 km) do not change. Affected earlier versions: the
  versions for Sionna 0.19 and 1.x, and 2.0.0a1.
- Satellite Doppler shift: the Doppler shift due to the motion of the satellite is now
  f_c (v_sat/c) (R/(R+h)) cos(α) cos(2πU), with the speed v_sat of a circular orbit at
  the height h, the radius R of the Earth, the elevation angle α and U uniform per batch
  example (the direction of the orbit relative to the UTs, as before); its largest value
  is the shift of TR 38.811 V15.4.0, clauses 6.7.1 and 6.9.2, f_d = (v_sat/c) (R/(R+h))
  cos(α) f_c. The factor R/(R+h) was missing, so the shift was (R+h)/R times too large:
  by 9.4 % at 600 km, 18.8 % at 1200 km and 157 % at 10 000 km. At 10° the largest shift
  was 24.84 ppm of the carrier frequency at 600 km and 23.84 ppm at 1200 km and is now
  22.70 ppm and 20.06 ppm (TR 38.821 V16.2.0, Table 6.1.1.1-8, gives 24 ppm and 20 ppm
  as the maximum Doppler shifts for elevation angles of at least 10°). A geostationary
  satellite (35 786 km) now induces no Doppler shift, as TR 38.811, clause 5.3.2.3,
  states ("the Geostationary satellite is fixed and therefore no Doppler shift is
  induced except that due to possible UE motion"); it had up to 10.1 ppm, 202 kHz at
  20 GHz. As in clause 5.3.4.3, the rotation of the Earth is neglected. This concerns
  the phase of the channel coefficients over time on every satellite link with the
  Doppler shift enabled, in all scenarios, link states, bands and directions; powers,
  delays and angles do not change, and of the reference outputs of the test suite only
  the time correlations changed. Affected earlier versions: the versions for Sionna 0.19
  and 1.x, and 2.0.0a1.
- Departure spreads at the satellite in the uplink: NOTE 8 of TR 38.811 V15.4.0, Tables
  6.7.2-1a to 6.7.2-8b ("For satellite (e.g.GEO/LEO), the departure angle spreads are
  zeros, i.e. µlgASD and µlgZSD are –∞, and corresponding standard deviations are
  zeros"), now applies to satellites in both directions, with zero cross-correlations of
  ASD and ZSD with the other large scale parameters, as the downlink parameter files had
  it. The satellite is the departure end of the rays as the generator draws them in both
  directions, so the ASD and ZSD of the uplink are spreads at the satellite as well.
  Before, only the downlink parameter files applied the note: in the uplink the large
  scale parameters ASD and ZSD took the values of the tables (medians of up to 0.78° and
  0.09°), the rays of each cluster kept the intra-cluster zenith spread (3/8)·10^µlgZSD
  of TR 38.901 V16.1.0, eq. (7.5-20), at the satellite (rays up to 0.07° from their
  cluster), and the cross-correlations with ASD and ZSD entered the generation of the
  other large scale parameters, whose realizations therefore change (their distribution
  per link does not). The parameter files now hold the tables as printed, and the models
  apply the note for satellites (160 km to 35 786 km). Aerial platforms (8 km to 50 km)
  now take the spreads of the tables in both directions; in the downlink the parameter
  files had given them zero spreads. This concerns the uplink of satellites and both
  directions of aerial platforms, in all scenarios, link states and bands. Affected
  earlier versions: the versions for Sionna 0.19 and 1.x, and 2.0.0a1.
- Cluster azimuth spread at the satellite: every ray now leaves a satellite in the
  azimuth of its cluster, that is in the LOS direction. TR 38.811 V15.4.0, NOTE 8 of
  Tables 6.7.2-1a to 6.7.2-8b, makes the departure angle spreads of a satellite zero,
  and clause 6.9.1 states for the link-level models derived from this model that "the
  azimuth and elevation angular spreads of departure, ASD and ZSD, can be considered
  zero, and all cluster AOD and ZOD angles the same". Before, the rays of the urban
  scenario left the satellite at the AOD of their cluster plus c_ASD α_m (TR 38.901
  V16.1.0, eq. (7.5-13), applied to the departure), with the cluster ASD c_ASD of the
  urban Tables 6.7.2-3a to 6.7.2-4b (0.08° to 6.63°) and the offsets α_m of Table 7.5-3:
  up to 14.3° in azimuth at 90°, and up to 0.74° from the LOS direction (urban NLOS, S
  band, 70°; computed). Seen from a satellite at 600 km, a scatterer 1 km from the UT is
  at most 0.03° (10°) to 0.1° (90°) off the direction of the UT. The dense urban and
  sub-urban tables print c_ASD = 0, so only urban links of satellites (160 km to
  35 786 km) are affected, in both directions, link states and bands; aerial platforms
  keep the c_ASD of the tables. In the reference outputs of the test suite (600 km), the
  power of single urban links changed by up to 0.79 dB and the power per antenna pair by
  up to 0.07 times the largest of its configuration (NLOS, 90°); the total power of a
  configuration of six links changed by less than 0.1 dB. Affected earlier versions: the
  versions for Sionna 0.19 and 1.x, and 2.0.0a1.
- Default atmosphere: the defaults of `temperature` and `atmospheric_pressure` are now
  the mean annual global reference atmosphere of TR 38.811 V15.4.0, clause 6.6.4 ("For
  all UEs, [...] T = 288.15 K, p = 1013.25 hPa, ρ = 7.5 g/m³"). The defaults were 273 K
  and 1020 hPa, and the formulas of ITU-R P.676 and P.618-13 that took the temperature
  in degrees Celsius used T - 273 instead of T - 273.15. The gas loss of this version,
  by ITU-R P.676-10, takes t = T - 273.15 (see "Gas attenuation"), and the scintillation
  loss no longer uses the temperature (see "Tropospheric scintillation"). With the gas
  formulas of the earlier versions, the defaults give a lower gas loss, for example
  0.2056 dB instead of 0.2405 dB at 2.2 GHz and 10°, 1.4779 dB instead of 1.5711 dB at
  20 GHz and 10°, 1.4019 dB instead of 1.5941 dB at 30 GHz and 10°, and 0.2434 dB
  instead of 0.2768 dB at 30 GHz and 90° (urban, 600 km, computed). Where the
  temperature was set, the conversion alone changes the gas loss of these formulas at
  288.15 K by at most 0.0031 dB in the bands that the models accept (at 40 GHz and 10°,
  1013.25 hPa, 7.5 g/m³). This concerns the gas loss in all scenarios, link states,
  bands and directions; the scintillation loss no longer depends on the temperature.
  Affected earlier versions: the versions for Sionna 0.19 and 1.x, and 2.0.0a1.
- Gas attenuation: the gas loss now follows Annex 2 of ITU-R P.676-10 (09/2013) as
  printed: the specific attenuation of dry air of eqs. (22a) to (22u), every branch from
  1 GHz to 350 GHz, that of water vapour of eqs. (23a) to (23d), the equivalent heights
  of eqs. (25a) to (25e), (26a) and (26b), and the slant path of eqs. (27) and (28)
  with TR 38.811 V15.4.0, eq. (6.6-8). The atmospheric pressure is the dry air pressure
  p, as clause 6.6.4 calls its 1013.25 hPa; Annex 2 takes the total pressure p + e, with
  the water vapour partial pressure e = ρT/216.7 (Annex 1, eq. (4)), and the temperature
  t = T - 273.15 in degrees Celsius. Clause 6.6.4 takes "The method of Annex 2 in ITU-R
  P.676" without a version, and clause 2 makes the latest version apply to such a
  reference ("For a non-specific reference, the latest version applies."). P.676-10 is
  the last revision whose Annex 2 gives the specific attenuations and the equivalent
  heights in closed form, the form of the earlier versions; with the reference
  atmosphere of clause 6.6.4 the revision in force, P.676-13 (08/2022), gives at 10° in
  the bands that the models accept between 0.2738 dB more (22.41 GHz; 0.2663 dB more at
  22.74 GHz, outside the 0.5 GHz around the water vapour line at 22.235 GHz that clause
  6.6.4 excludes from the method) and 0.0676 dB less (40 GHz) (computed). Before, the
  loss used curve fits in the form of Annex 2 of P.676-6 to P.676-10 up to 54 GHz, with
  two coefficients of the water vapour equivalent height that no revision prints (-8.1
  and 1.5, where P.676-6 to P.676-11 print -8.6 and 1.58), and followed no revision
  above 54 GHz: from 54 GHz to 66 GHz it used expressions of P.676-5, one of them with
  the factor of another equation, and from 66 GHz to 120 GHz the dry-air term of eq.
  (22e) lacked the factor f² rp² 10^-3 and had 1.15 ξ5 in the exponent of (f - 66)
  instead of added to its power. The second made the zenith loss of dry air jump from
  10.8 dB at 65.9 GHz to 985 dB at 66.1 GHz, and together they made the attenuation
  1.6 % to 7.4 % of the printed value from 69 GHz to 120 GHz (dry air, zenith, 288.15 K,
  1013.25 hPa, computed with the earlier formulas). Above 120 GHz dry air did not
  attenuate, the limit of eq. (25e) on the equivalent height of dry air was missing, and
  the atmospheric pressure was taken as the total pressure. With the reference
  atmosphere of clause 6.6.4 and at 10°, the loss changes against the earlier formulas
  with t = T - 273.15 by +0.0040 dB at 2 GHz, +0.0041 dB at 2.2 GHz, +0.0068 dB at
  20 GHz (1.4846 dB instead of 1.4779 dB) and +0.0186 dB at 30 GHz (1.4205 dB instead of
  1.4019 dB); in the bands that the models accept (1.9 GHz to 4 GHz and 19 GHz to
  40 GHz) the change lies between -0.0112 dB (22.29 GHz; -0.0089 dB at 22.74 GHz,
  outside the excluded range) and +0.0384 dB (40 GHz), relative between -0.37 %
  (22.25 GHz) and +2.04 % (3.75 GHz), mostly from the dry reading of the pressure
  (computed). In dB the change scales with 1/sin of the elevation angle. This concerns
  all scenarios, link states, bands and directions. `utils.compute_pathloss_gas` raises
  `ValueError` for carrier frequencies outside 1 GHz to 350 GHz, the range of Annex 2;
  the models accept only the S and Ka bands. As clause 6.6.4 states, the method does not
  apply to UE altitudes above 10 km and to frequencies within 0.5 GHz of the centres of
  resonance lines; the models do not check either. Affected earlier versions: the
  versions for Sionna 0.19 and 1.x, and 2.0.0a1.
- Tropospheric scintillation: in the Ka band the scintillation loss is now the value of
  TR 38.811 V15.4.0, Table 6.6.6.2.1-1, at the reference elevation angle nearest to the
  elevation angle: 1.08 dB at 10°, 0.48 dB at 20°, 0.30 dB at 30°, 0.22 dB at 40°,
  0.17 dB at 50°, 0.13 dB at 60° and 0.12 dB from 70° to 90°, as clause 6.6.6.2.1
  specifies ("As baseline for system-level simulations above 6 GHz, the fading due to
  scintillation PL_S is equal to the tropospheric attenuation level at 99% of the time
  derived from Figure 6.6.6.2.1-1 and summarized in Table 6.6.6.2.1-1"). The table is
  given for 20 GHz in Toulouse; it is used at every carrier frequency from 6 GHz on, as
  TR 38.821 V16.2.0 uses it at 20 GHz and 30 GHz (Tables 6.1.3.2-1 and 6.1.3.3-1). An
  elevation angle midway between two reference angles takes the higher one. Before, the
  loss was the fade depth of ITU-R P.618-13, section 2.4.1, exceeded 0.01 % of the time,
  where the table gives the depth exceeded 1 % of the time; by that method the depth
  exceeded 0.01 % of the time is 2.40 times the depth exceeded 1 %. It was computed at
  the elevation angle itself, from the temperature, the relative humidity and the
  diameter and efficiency of the Earth-stationed antenna. That computation also
  evaluated the effective path length of eq. (41) of P.618-13 as
  2h_L/sqrt(sin²θ + 2.35·10^-4 + sin θ) instead of
  2h_L/(sqrt(sin²θ + 2.35·10^-4) + sin θ), which changed the loss by -1.85 % at 10° and
  by +6.68 % at 90° (20 GHz, the default antenna, computed). With the defaults of the
  earlier versions the loss was 1.5507 dB, 0.4090 dB and 0.1660 dB at 20 GHz and 10°,
  30° and 90°, and 1.8943 dB at 30 GHz and 10° (urban, 600 km, computed); it is now
  1.08 dB, 0.30 dB, 0.12 dB and 1.08 dB. This concerns the scintillation loss of every
  link in the Ka band, in all scenarios, link states and directions; it no longer
  depends on the atmospheric parameters (see "Removed"). Affected earlier versions: the
  versions for Sionna 0.19 and 1.x, and 2.0.0a1.
- Ionospheric scintillation: in the S band the scintillation loss is now the ionospheric
  scintillation A_IS of TR 38.811 V15.4.0, clause 6.6.6.1.4, only for UTs within 20° of
  the magnetic equator, and 0 dB at higher latitudes, as the clause specifies ("For
  system-level simulations below 6 GHz, PL_S is equal to A_IS from Equation (6.6-13) for
  latitudes of maximum ±20°. For latitudes between ±20° and ±60° of latitude, PL_S = 0.
  [...] the choice of PL_S = 0 is also applied for latitude above ±60°."). The latitude
  is the `latitude` of `set_topology`, the latitude of the UTs relative to the magnetic
  equator, around which the clause places the region of the model; its default of 47°
  gives 0 dB, as the calibration of TR 38.821 V16.2.0 assumes (Table 6.1.1.1-5, NOTE 2:
  "the ionospheric scintillation loss shall be considered equal to zero (i.e., the UEs
  are located between 20 and 60 degrees of latitude)"). Before,
  A_IS = 1.1 (f/4 GHz)^-1.5/√2 dB applied at every latitude, and `latitude` had no
  effect. This concerns the scintillation loss of every link in the S band, in all
  scenarios, link states and directions: with the default latitude it is now 0 dB
  instead of 2.2 dB at 2 GHz and 1.9069 dB at 2.2 GHz; within 20° of the magnetic
  equator it is unchanged. Affected earlier versions: the versions for Sionna 0.19 and
  1.x, and 2.0.0a1.
- Shadow fading applied once: the shadow fading now reaches the channel once, as the
  large scale parameter SF of TR 38.901 V16.1.0, clause 7.5, step 4, which step 12
  applies to the channel coefficients. TR 38.811 V15.4.0, clause 6.6.2, eq. (6.6-4),
  writes one shadow fading term SF ~ N(0, σ_SF²) into the basic path loss, with σ_SF of
  Tables 6.6.2-1 to 6.6.2-3, and Tables 6.7.2-1a to 6.7.2-8b give the cross-correlations
  of this shadow fading with the other large scale parameters and its correlation
  distance, which only the large scale parameter carries. Before, the basic path loss
  contained a second shadow fading draw per link, independent of the first and without
  these correlations, so the shadow fading applied to the channel had about √2 times
  σ_SF and weaker correlations, and with `enable_shadow_fading=False` the draw in the
  basic path loss remained. In the urban scenario (S band, downlink, 50°, 600 km, 4000
  links with one UT each, LOS and NLOS forced, computed), the standard deviation of the
  applied shadow fading is now 4.05 dB (LOS) and 6.08 dB (NLOS) instead of 5.60 dB and
  8.57 dB (Table 6.6.2-2: 4 dB and 6 dB); its correlation with lg DS is now -0.411 and
  -0.214 instead of -0.273 and -0.126 (Tables 6.7.2-3a and 6.7.2-4a: -0.4 and -0.21),
  and with lg ZSA -0.776 and -0.356 instead of -0.528 and -0.271 (the tables give -0.8
  and -0.38; with the limit of ZSA to 52° of step 4 the expected values are -0.771 and
  -0.348); with `enable_shadow_fading=False` it is now 0 dB instead of a standard
  deviation of 4.07 dB and 6.09 dB. Because two draws per link fewer are made, the
  realizations of the large scale parameters, and with them of the cluster delays and
  powers, change for every link; their statistics do not change, and neither do the link
  states. This concerns all scenarios, link states, bands and directions. Affected
  earlier versions: the versions for Sionna 0.19 and 1.x, and 2.0.0a1.
- Elevation angles midway between two reference angles: an elevation angle of 25°, 45°,
  65° or 85° now takes the values of the higher reference angle, 30°, 50°, 70° or 90°,
  in every table: the LOS probability, the shadow fading and the clutter loss, the large
  scale parameters, the number of clusters and the cluster parameters. TR 38.811
  V15.4.0, clauses 6.6.1 and 6.6.2, takes the reference angle nearest to the elevation
  angle and gives no rule for an angle midway between two; the models now take the
  higher one in every lookup. Before, Python's rounding of ties to the even number took
  the lower angle, 20°, 40°, 60° or 80°, at these four angles (the tropospheric
  scintillation table of this version already took the higher one); 15°, 35°, 55° and
  75° took the higher angle and still do. For example, at 25° the LOS probability of the
  urban scenario is now 0.493 instead of 0.386, and the clutter loss of the urban NLOS
  links in the S band 29.0 dB instead of 30.9 dB (the values of Tables 6.6.1-1 and
  6.6.2-2 at 30° and 20°). This concerns simulations at exactly 25°, 45°, 65° or 85°, in
  all scenarios, link states, bands and directions; other elevation angles do not
  change. Affected earlier versions: the versions for Sionna 0.19 and 1.x, and 2.0.0a1.
- UTs at the same location: two or more UTs at the same location in the horizontal plane
  and in the same link state, in one batch example, made the spatial correlation of the
  large scale parameters (delay spread, angular spreads, Ricean K-factor, shadow fading)
  of that batch example singular, because the correlation exp(-d/d_corr) of TR 38.901
  V16.1.0, clause 7.5, step 4, which TR 38.811 V15.4.0, clause 6.7.2, uses, is 1 at the
  distance d = 0. Its Cholesky factorization then stopped without an error at the second
  of these UTs and left unfactored entries in the rows of the UTs after them, so the UTs
  listed after them in that batch example got large scale parameters with a log-domain
  variance above the specified one: 1 + 2c^2 times it in the case measured, a UT after
  two co-located UTs with the correlation c to them; for the shadow fading of a UT 7.6 m
  away (dense urban, LOS, S band, downlink, 30°, double precision, computed) 2.33 times,
  a standard deviation 1.53 times the specified one. Where rounding left a pivot just
  above 0 instead, mostly in single precision, the factorization continued past these
  UTs or stopped at a later one. The gradients of the large scale parameters of that
  batch example with respect to the UT locations were NaN. Now UTs at one location in
  one state share their large scale parameters, as the correlation exp(-d/d_corr) = 1 at
  d = 0 requires and as in the procedure of clause 3.3.1 of WINNER II D1.1.2, which step
  4 uses, where such UTs read the same grid node; every UT has the specified
  distribution, and the gradients are finite. This concerns topologies with such UTs,
  whatever their heights (the distance is measured in the horizontal plane), in all
  scenarios, link states, bands and directions; random drops practically never place
  two UTs at one location. Topologies without them are unchanged bit for bit. Affected
  earlier versions: 2.0.0a1. The versions for Sionna 0.19 and 1.x build the same
  singular correlation for such UTs. The version for Sionna 1.x factors it with the
  Cholesky decomposition of TensorFlow, which returns NaN for it without an error
  (computed with TensorFlow 2.18.1 and 2.19.0 on the CPU), so that every UT of such a
  batch example got NaN large scale parameters, also with only two UTs at one location;
  the version for Sionna 0.19 takes a symmetric matrix square root, whose result for
  such UTs depends on TensorFlow.

### Added

- `Rays` takes the optional argument `strongest_cluster_indices`, the indices of the two
  strongest clusters of each link, whose rays are coupled within the sub-clusters and
  which the channel coefficients split into sub-clusters; `RaysGenerator` sets it. Code
  that constructs `Rays` directly can omit it; the two clusters with the largest powers
  are then split, as before.

### Changed

- `Topology` takes the new argument `los_phase`, the LOS propagation phase per link,
  which `ChannelCoefficientsGenerator` uses for the LOS ray; the scenarios provide it
  as `los_phase`, and the channel models pass it. Code that constructs a `Topology`
  directly has to pass it.
- `set_topology` now checks the platform and raises `ValueError` unless `bs_loc` holds
  one BS at the same height in every batch example, either a satellite from 160 km to
  35 786 km or an aerial platform from 8 km to 50 km. TR 38.811 V15.4.0, clause 3.1,
  places aerial platforms at 8 km to 50 km and LEO satellites from 500 km on; the models
  accept satellites from 160 km, very low orbits included. Table 4.5-1 gives typical
  altitudes (LEO 600 km to 1500 km, MEO 7000 km to 20 000 km, GEO 35 786 km),
  Table 5.1-1 the reference deployments (GEO, non-GEO down to 600 km, UAS 8 km to
  50 km). The models simulate one platform with one nominal elevation angle per call.
  Before, `set_topology` accepted every input: with a second BS the channel call failed,
  with heights that differed across the batch the 3D distance, the path loss and the
  switches of the satellite took the height of the first batch example for every
  example, and any height was used as given. The versions for Sionna 0.19 and 1.x and
  2.0.0a1 use the height of the first batch example in the same way.
- `utils.compute_pathloss_gas` raises `ValueError` for carrier frequencies outside 1 GHz
  to 350 GHz, the range of Annex 2 of ITU-R P.676-10 (see "Gas attenuation").
- `set_topology` raises `ValueError` for a `latitude` outside -90° to 90°, and a
  `latitude` passed without new topology tensors now recomputes the scintillation loss.
  `latitude` is documented as the latitude of the UTs relative to the magnetic equator
  (see "Ionospheric scintillation"); before, any value was accepted, and it had no
  effect.
- `basic_pathloss` of the scenarios is now the free space path loss plus, for NLOS
  links, the clutter loss, without a shadow fading term; the shadow fading is the large
  scale parameter SF, which step 12 applies (see "Shadow fading applied once"). Before,
  `basic_pathloss` contained a shadow fading draw of its own, as in the versions for
  Sionna 0.19 and 1.x and 2.0.0a1.
- `set_topology` raises `ValueError` if a UT is indoor (`in_state` `True`), before
  anything is stored; the models accept outdoor UTs only, for satellites and aerial
  platforms. TR 38.811 V15.4.0 considers only outdoor UTs for satellites (clause 6.1.2:
  "Only outdoor conditions are considered for satellite operations"; Table 5.1-1: "100%
  Outdoors" for the satellite deployments) and indoor UTs as well for aerial platforms
  (clause 6.1.2: "additional indoor conditions are also considered for HAPS";
  Table 5.1-1: "Indoor and Outdoor" for UAS including HAPS), but the models have neither
  the building entry loss of clause 6.6.3 nor clusters for indoor UTs. Before, indoor
  UTs were accepted, but all their clusters were masked, so their channel coefficients
  were NaN (dense urban, 2 GHz, downlink, 30°, 600 km, LOS and NLOS forced, computed
  with the code before this change: every coefficient of an indoor UT), and no building
  entry loss applied. The indoor probability of the topology generators was already
  forced to 0; its documentation now says that the argument is ignored. Affected earlier
  versions: 2.0.0a1, and the versions for Sionna 0.19 and 1.x, which have the same code
  for indoor UTs.

### Removed

- `utils.compute_stallite_doppler`, which nothing used: it returned the carrier
  frequency plus the Doppler shift of TR 38.811 V15.4.0, clauses 6.7.1 and 6.9.2, which
  the channel now applies (see "Changes that affect simulation results").
- `TDL` and `CDL`, with their modules `openntn.tdl` and `openntn.cdl`: placeholders that
  raised `NotImplementedError` when constructed (in the versions for Sionna 0.19 and 1.x
  they were copies of the TR 38.901 models marked as not working). The link-level models
  NTN-TDL and NTN-CDL of TR 38.811 V15.4.0, clause 6.9, are not implemented in this
  version; their parameter files stay in `openntn/models`, and Sionna's `TDL` and `CDL`
  of TR 38.901 are unchanged.
- The rain and cloud attenuation of TR 38.811 V15.4.0, clause 6.6.5: the scenario
  property `additional_pathloss` (it raised `AttributeError`, because no model computed
  the attenuation) and `utils.compute_pathloss_additional`, which failed when called.
  The models follow the baseline of clause 6.6.5, "For system-level simulations, the
  baseline is to consider clear sky conditions only", as before.
- The parameters `lwc` and `rain_rate` of `set_topology` and the scenario properties of
  the same names: only the rain and cloud attenuation read them, which no model applied,
  so they had no effect.
- The parameters `relative_humidity`, `diameter_earth_antenna` and `antenna_efficiency`
  of `set_topology` and the scenario properties of the same names: only the computation
  of the tropospheric scintillation by ITU-R P.618-13 read them, and the scintillation
  loss now follows TR 38.811 V15.4.0, Table 6.6.6.2.1-1 (see "Tropospheric
  scintillation"), so they had no effect. A call that passes one of the five removed
  parameters raises `TypeError`; the scenario of 2.0.0a1 and of the versions for Sionna
  0.19 and 1.x accepted them.
- `utils.gen_single_sector_topology_interferers`, which failed in every scenario. The
  models simulate one satellite or aerial platform per call (see "Changed");
  `utils.gen_single_sector_topology` generates the topology of one sector.

### Fixed

- `set_topology` no longer modifies the tensors passed to it. The scenario kept the
  topology tensors (UT and BS locations, orientations, UT velocities, indoor state) as
  its own storage when they already had the model's data type and device, and a later
  call wrote its values into them, so a tensor passed in an earlier call changed. The
  scenario now keeps copies, through which gradients with respect to the tensors
  passed still flow. The outputs for given inputs are unchanged; results change only
  for scripts that used such a tensor again after a later `set_topology` call.
  Affected earlier version: 2.0.0a1; the versions for Sionna 0.19 and 1.x are not
  affected.
- Repeated `set_topology` calls with topology tensors that require gradients: every call
  now works, and the quantities of the scenario, for example the 2D distances and the
  LOS angles, stay differentiable with respect to the tensors of that call. Before, the
  scenario wrote the values of a later call into the tensors it had stored in the first
  call, which carried the graph of that call, so the backward pass of the second
  differentiable call failed with "Trying to backward through the graph a second time",
  and after a differentiable call the stored tensors kept requiring gradients in calls
  without gradients. Calls without gradients still write in place, so that the stored
  tensors keep their addresses. The path coefficients and delays of the channel models
  are detached, as before and as in Sionna; their documentation now says so. Affected
  earlier version: 2.0.0a1; the versions for Sionna 0.19 and 1.x are not affected.
- Gradients with respect to the UT locations of the scenario property
  `matrix_ut_distance_2d`, the distances between the UTs in the horizontal plane, and of
  the large scale parameters, whose spatial correlation is computed from these
  distances, were NaN in the x and y coordinate of every UT: the distance of a UT to
  itself, which is always 0, was taken as a square root, whose derivative at 0 is
  infinite. They are now finite: the distance of a UT to itself has the derivative 0,
  and the distances are unchanged bit for bit. For example, with the dense urban
  scenario at 2.2 GHz, downlink, 30°, 600 km and 2 batch examples with 3 UTs each, the
  gradient of the sum of the distances was NaN in 12 of the 18 entries of the UT
  locations, and so were the gradients of the delay spread, the angular spreads, the
  Ricean K-factor and the shadow fading. This concerns users who differentiate with
  respect to the UT locations through these quantities; the path coefficients and delays
  are detached and not affected, and no simulation result changes. Affected earlier
  versions: 2.0.0a1, and the versions for Sionna 0.19 and 1.x, which compute the
  distances between the UTs in the same way.
- The test suite now checks the data files of the NTN-CDL and NTN-TDL models in
  `openntn/models` against TR 38.811 V15.4.0, Tables 6.9.1-1 to 6.9.1-4 and 6.9.2-1 to
  6.9.2-4, value by value; it called them TR 38.901 profiles without TR 38.811 values.
  The files equal the tables; no model of the package reads them.
- Documentation of the scenario properties: `doppler_enabled` had the description of the
  antenna efficiency; it now states that the flag switches the Doppler shift due to the
  motion of the satellite, while the Doppler shift due to the motion of the UTs is
  always applied. `gas_pathloss`, `scintillation_pathloss` and `entry_pathloss` called
  themselves the basic path loss of TR 38.901, clause 7.4.1; they now name the gas
  attenuation (TR 38.811 V15.4.0, clause 6.6.4, eq. (6.6-8)), the scintillation loss
  (clause 6.6.6) and the building entry loss (clause 6.6.3). `free_space_pathloss` now
  cites eq. (6.6-2) of TR 38.811 instead of TR 38.901, clause 7.4.1. No result changes.
  Affected earlier versions (the same texts): the versions for Sionna 0.19 and 1.x, and
  2.0.0a1.

## [2.0.0a1] - 2026-10-07

First pre-release of OpenNTN for Sionna 2.x, which is built on PyTorch.

- Import name `openntn` (`import openntn`), instead of the module `tr38811` that the
  installation scripts of the earlier versions linked into Sionna.
- Installation with pip from PyPI (`pip install openntn`) instead of `install.sh`.
- Sionna 2.x is required (`sionna>=2.0.1,<3.0`), with Python 3.11 or later.
- The licence is declared as `MIT AND Apache-2.0`: files derived from Sionna are under
  the Apache License 2.0, all other files under the MIT License. Both licence texts are
  included.
- The unused model data files were removed.
