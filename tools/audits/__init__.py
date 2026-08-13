"""Acquisition and spatial-processing audits for the Raw2Event testbed.

These read-only diagnostics test whether the observed simulator rankings are
artifacts of the capture rig rather than properties of the simulators:

* ``cadence_audit``                 -- Pi hardware-timestamp cadence from the
                                       per-recording metadata sidecars.
* ``rate_spectrum_audit``           -- event-rate power spectra for matched
                                       real / raw-simulated / RGB-simulated streams.
* ``notch_ranking_robustness``      -- removes the measured frame cadence and its
                                       harmonics, then recomputes the temporal
                                       upstream rankings.
* ``roi_retention_audit``           -- what the AprilTag-driven ROI extraction and
                                       coordinate normalization preserve.
* ``resolution_sensitivity_audit``  -- whether spatial rankings survive resampling
                                       the unified representation to lower grids.

All modules are runnable as ``python -m tools.audits.<name> --help``.
"""
