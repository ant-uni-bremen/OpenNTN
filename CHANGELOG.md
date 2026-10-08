# Changelog

The format follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

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
- Atmospheric parameters of `set_topology` (`latitude`, `lwc`, `rain_rate`,
  `atmospheric_pressure`, `temperature`, `water_vapor_density`, `relative_humidity`,
  `diameter_earth_antenna`, `antenna_efficiency`): a value now persists until it is
  passed again, as the `set_topology` documentation states, and a value passed without
  new topology tensors recomputes the gas and scintillation losses (TR 38.811 V15.4.0,
  clauses 6.6.4 and 6.6.6) without drawing anything else again. Before, every call
  reset the parameters it did not pass to their defaults, and the losses were
  recomputed only together with a new topology. The `set_topology` method of
  `DenseUrban`, `Urban` and `SubUrban` now accepts these parameters and forwards them
  to the scenario; it raised a `TypeError` before. The default values are unchanged.
  `latitude`, `lwc` and `rain_rate` have no effect, because the rain and cloud
  attenuation is not active; their documentation now says so. This concerns every
  simulation that sets atmospheric parameters. For example, at 30° in the urban
  scenario, a temperature of 300 K and a relative humidity of 80 % instead of the
  defaults change the gas loss from 0.0835 dB to 0.0648 dB at 2.2 GHz and from
  0.5456 dB to 0.4927 dB at 20 GHz, and the scintillation loss at 20 GHz from 0.409 dB
  to 1.230 dB (computed with this version); where earlier versions had reset a value or
  not applied it, they used the defaults instead. Affected earlier versions: the
  versions for Sionna 0.19 and 1.x, and 2.0.0a1.
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

### Changed

- `Topology` takes the new argument `los_phase`, the LOS propagation phase per link,
  which `ChannelCoefficientsGenerator` uses for the LOS ray; the scenarios provide it
  as `los_phase`, and the channel models pass it. Code that constructs a `Topology`
  directly has to pass it.

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
