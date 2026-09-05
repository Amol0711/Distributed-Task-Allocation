# Reference simulation results

The top-level tables summarize the application campaigns, exploration outcomes,
coordination checks, predictable covariance, and selected-set fingerprints.
The subdirectories group application evaluation, finite controller certificates,
projected-estimator diagnostics, paired statistics, and exact-oracle enumeration.

Run `python scripts/verify_reference_results.py` to check every recorded file
against `MANIFEST.json`. The numerical reproduction commands are listed in the
root README. A hash check establishes file integrity; the corresponding
regeneration command additionally recomputes the numerical experiment.
