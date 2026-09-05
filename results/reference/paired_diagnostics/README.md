# Paired-retention diagnostics

For each evaluation seed, retention is the cumulative true value of the
shielded policy divided by the cumulative true value of the exploitation-only
policy. The reported uncertainty is the sample standard deviation divided by
the square root of the number of seeds. Ratios using policy-conditioned greedy
denominators cannot be divided to recover this statistic.

Run `python scripts/build_statistical_diagnostics.py` to reconstruct this table
and the trajectory diagnostics without changing the execution records.
