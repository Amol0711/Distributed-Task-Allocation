# Distributed Task Allocation Simulations

This repository implements reproducible numerical experiments for distributed
allocation with aggregate feedback, resource constraints, and switched tracking
dynamics. It provides simulation engines, synthetic benchmark tables, fixed seed
registries, controller and reference configurations, numerical results, and
executable checks of allocation, communication, estimation, and physical bounds.

## Experimental systems

| System | Numerical setting | Evaluation population |
|---|---|---|
| `SAT-COV-V1` | Synthetic micro-satellite sensing and coverage with residual resources and local tracking models | 32 seeds, 480 episodes per seed |
| `UAV-AG-V1` | Synthetic aerial-vehicle allocation over agricultural field cells with residual resources and local tracking models | 32 seeds, 500 episodes per seed |
| `TRAJ-MICRO-V1` | Three-agent matching allocation with one two-dimensional physical block, four controller modes, and trajectory-generated returns | 12 seeds, 240 episodes per seed |

All input tables are synthetic. The configurations and seed registries specify
the numerical systems completely; no external dataset download is required.
Four design seeds per application are separate from the 32 evaluation seeds.
The evaluation population does not select model parameters or exploration
coefficients.

The application campaigns use a structured set value plus predeclared aggregate
return noise. Their tracking simulations provide physical diagnostics but do not
generate the learning return. The trajectory system instead generates its return
from controlled execution and evaluates a shared-reset deviation bound before
learning. Its resource screen is nonbinding. These systems test different
interfaces and are not interchangeable.

## Environment and installation

Reference reproduction uses Python 3.13.5, NumPy 2.3.5, SciPy 1.17.0, and
pytest 9.0.2. `requirements-lock.txt` records the tested package versions;
`requirements.txt` records compatible version ranges. `environment.yml` provides
the corresponding Conda environment specification.

From the repository root, create and activate a virtual environment, then install
the tested dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-lock.txt
```

On Windows PowerShell, the activation command is
`.venv\Scripts\Activate.ps1`. The shell runners require Bash; the individual
Python commands can also be executed directly.

```bash
python scripts/check_environment.py
```

Byte-identical replay is verified in the tested numerical environment. Changes in
numerical libraries, operating system, or floating-point implementation can alter
serialized floating-point results even when numerical agreement is retained.
The shell runners set single-threaded numerical kernels and use process-level
parallelism for independent trials.

## Quick verification

```bash
bash run_quick.sh
```

The command checks package and reference integrity, regenerates controller and
reference-reset evidence, verifies the archived estimator audit, replays the
trajectory execution, checks the exact-oracle records, runs the complete test
suite, and generates short baseline and exploration campaigns. Each short run
uses one evaluation seed per application and 12 episodes. Results are written to
`results/quick_generated/` and validated independently.

The reference-file check and the archived estimator check establish consistency
of stored records. Fresh estimator execution and exhaustive geometry regeneration
are separate commands documented below.

```bash
python -m pytest -q tests
```

Tests cover estimator constraints, resource and agreement invariants, return-route
scope, reference resets, terminal execution, diagnostic arithmetic, serialization,
and rejection of inconsistent records.

## Complete application evaluation

```bash
WORKERS=4 bash run_all.sh
```

The complete run evaluates instantaneous-value myopic selection and two
horizon-score policies using exploration coefficients `0` and `0.25`. It uses all
32 evaluation seeds and the full configured horizon for each application. The
standard evaluation contains 192 policy-seed records and 94,080 episode rows.

Outputs are written to `results/generated/`. The final aggregation produces
certificate summaries, paired statistics, bootstrap intervals, covariance
diagnostics, and plotting tables under `results/generated/plot_data/`. Every
output path is internal to this repository by default.

A normal run replaces `results/generated/`. To retain completed trials and resume
an interrupted evaluation, use

```bash
RESUME=1 WORKERS=4 bash run_all.sh
```

The runners validate existing trace identifiers, lengths, source hashes, and
numerical checks before resuming. `run_quick.sh` replaces only its own output
directory and does not remove complete application results.

## Component-level reproduction

### Controller and reference-reset certificates

```bash
python scripts/verify_control_evidence.py
```

This command regenerates the evidence in a temporary directory and compares it
with `results/reference/control_certificates/`. The two applications contain
four local controller templates each. Eight within-mode tests, 32 ordered
controller-pair tests, and 32 ordered reference-pair tests check the declared
quadratic bounds and reset envelopes. The certified reference diameters are
`0.030` and `0.035`, respectively.

### Projected-estimator constraints

```bash
python scripts/verify_projected_estimator_audit.py --reproduce --workers 4
```

This command executes 64 application-seed audits and 94,080 estimator updates
across distributed upper-confidence selection, projected-mean selection, and
upper-confidence selection without the resource filter. It compares the
nonnegative quadratic minimizer with the minimizer that also satisfies the
Euclidean prior bound and total-value cap. All 66 reference audit files are
compared after regeneration. CSV columns and line endings are deterministic
for fresh and resumed runs.

### Exact-oracle geometry

```bash
python scripts/verify_exact_oracle_audit.py --rerun
```

The first design seed of each application fixes the context stream for this
experiment. The audit enumerates 320 small feasible families containing 773,888
feasible sets in total. It checks exhaustive optima against an independent
branch-and-bound implementation, greedy maximality, contextual curvature, and
zero and nonuniform marginal-shortfall certificate arithmetic. It does not
perform a learning update or simulate a physical trajectory.

### Trajectory-generated execution

```bash
python scripts/verify_trajectory_microcase.py
```

The verifier replays all 12 seeds and 2,880 episodes in a temporary directory.
The model contains 136 hereditary matching states and 60 maximal terminal
matchings. A fixed public prefix contains 18 exploration episodes per seed.
Each exploitation allocation uses tagged maximum agreement on a three-agent
path graph with diameter two.

The 2,664 executed exploitation allocations have corresponding zero-scale
recomputations at the same estimator states. These 2,664 recomputations are
allocation diagnostics, not independently executed physical trials. Together,
the two branches produce 5,328 allocation records and 42,624 agreement-round
instances. The 2,880 return floods belong to the executed policy.

```bash
python scripts/build_statistical_diagnostics.py
python scripts/build_trajectory_figure_data.py
```

These commands recompute paired-retention statistics, clipping diagnostics,
confidence-scale contributions, fixed-estimator score crossings, and cumulative
plotting tables from the reference records.

## Interpretation of the numerical outputs

The application value ratio compares cumulative true value with exact-score
greedy value evaluated along the same policy's resource trajectory. It is not a
ratio to the unrestricted exact optimum. Paired retention first divides the
shielded and exploitation-only cumulative values separately for each seed and
then averages those ratios. Dividing the two reported greedy-normalized means
does not recover this statistic. Observed-return bootstrap intervals use the
recorded noisy returns rather than latent true values.

The trajectory retention ratio uses the exact episodewise optimum. Its plotted
mean is the mean of seedwise cumulative ratios. Entries reported with one
standard error use the sample standard deviation divided by the square root of
the number of independent seeds. A pointwise standard-error band is not a
simultaneous confidence statement.

The certificate ledger charges each exploration episode by `F_max/q` and each
exploitation episode by the smaller of its raw finite-channel charge and
`F_max/q`. The normalized clipped utilization is `q B_K/(K F_max)` and lies in
`[0,1]`. Raw charges and clipping excess remain available. Support-derived and
enlarged-prior calculations on an application trace are fixed-trace
recertifications, not executions of alternative learning policies. The recorded
application horizons have a zero certified covariance floor; they do not
establish an empirical asymptotic regret rate.

## Repository structure

```text
configs/                         Numerical systems and return-route settings
datasets/                        Synthetic application inputs
seeds/                           Design and evaluation seeds and public codebooks
src/                             Simulation engines and certificate arithmetic
scripts/                         Execution, aggregation, and verification commands
tests/                           Unit, regression, and consistency tests
results/reference/               Application summaries and component audits
results/trajectory_microcase/     Trajectory records and diagnostic outputs
.github/workflows/               Automated validation workflow
```

The reference subdirectories are `application_evaluation`, `control_certificates`,
`projected_estimator`, `paired_diagnostics`, and `exact_oracle`.

## Integrity and numerical provenance

```bash
python scripts/verify_reference_results.py
python scripts/verify_package_integrity.py
```

`PACKAGE_MANIFEST.json` and `SHA256SUMS.txt` enumerate distributable files and
checksums. Component manifests record source and input hashes. Generated
application outputs and local environment caches are excluded from the release
inventory. An integrity check does not substitute for numerical regeneration.

After an intentional change has been numerically verified, the inventory can be
rewritten with `python scripts/refresh_integrity_metadata.py`. That command
records existing bytes and does not assert unexecuted test results. Do not use it
to bypass a failed integrity check on an unmodified download.

Reuse conditions are stated in `LICENSE_NOTICE.txt`.
