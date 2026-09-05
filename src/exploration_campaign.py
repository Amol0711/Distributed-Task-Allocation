"""Exploration simulation engine.

The module reuses the campaign allocation, estimator, resource, and tracking
primitives while adding a public nonnegative exploration codebook. Common random
number streams are materialized before branching so that exploration settings
can be compared on matched synthetic trials.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import csv
import hashlib
import json
import math
import time

import numpy as np

from campaign_engine import (
    Allocation,
    ContextGenerator,
    Estimator,
    GraphInfo,
    MethodState,
    PhysicalState,
    argmax_tie,
    assignment_violations,
    coverage_feature,
    coverage_value,
    curvature,
    feasible_mask,
    graph_info,
    load_assets,
    physical_epoch,
    random_ball,
    raw_value,
    sha256_file,
    stable_hash,
    write_csv_atomic,
    allocate,
    beta_radius,
)
from exploration import public_nonnegative_codebook
from certificate_arithmetic import (
    CertificateLedger,
    certify_episode,
    clipped_transfer_increment,
    universal_normalization,
)

SOFTWARE_VERSION = "1.1.0"
EXPERIMENT_ID = "public-codebook-exploration"
RAW_SCHEMA = "exploration-campaign-v2"
TOL = 1.0e-10

C_EXP_GRID = (0.0, 0.25, 0.5, 1.0)

RAW_FIELDS = [
    "schema_version", "software_version", "experiment_id", "experiment_config_hash",
    "campaign_id", "partition", "trial_index",
    "trial_seed", "scenario_seed", "feedback_seed", "physical_seed",
    "algorithm_seed", "exploration_seed", "codebook_seed", "seed_registry_hash",
    "c_exp", "variant_id", "epoch", "source_config_hash",
    "engine_hash", "certificate_module_hash", "codebook_hash", "codebook_file_sha256", "codebook_size", "codebook_index",
    "exploration_uniform", "exploration_probability", "exploration_indicator",
    "expected_exploration_count", "realized_exploration_count", "branch",
    "graph_family", "graph_edges", "graph_diameter", "active_q",
    "selected_count", "selected_elements", "selection_hash", "assignment_valid",
    "true_value", "realized_return", "oracle_greedy_value", "oracle_value_ratio",
    "cumulative_true_value", "cumulative_oracle_value", "cumulative_oracle_ratio",
    "empirical_curvature", "comparison_factor", "fixed_comparator_factor",
    "beta", "selected_width_sum", "score_quantization_epsilon",
    "certificate_cap", "raw_exploitation_charge", "raw_observable_bound_increment",
    "observable_bound_increment", "certificate_clip_indicator", "certificate_clip_excess",
    "cumulative_exploration_bound", "cumulative_ucb_bound", "cumulative_raw_ucb_bound",
    "cumulative_observable_bound", "cumulative_clip_excess", "cumulative_clipped_episodes",
    "universal_normalized_observable_bound",
    "support_sigma", "support_beta", "support_calibration_mismatch",
    "support_raw_exploitation_charge", "support_raw_observable_bound_increment",
    "support_observable_bound_increment", "support_certificate_clip_indicator",
    "support_certificate_clip_excess", "support_cumulative_exploration_bound",
    "support_cumulative_ucb_bound", "support_cumulative_raw_ucb_bound",
    "support_cumulative_observable_bound", "support_cumulative_clip_excess",
    "support_cumulative_clipped_episodes", "support_universal_normalized_observable_bound",
    "support_transfer_enlargement", "support_transfer_increment", "support_transfer_identity_holds",
    "prior_btheta", "prior_beta", "prior_calibration_mismatch",
    "prior_raw_exploitation_charge", "prior_raw_observable_bound_increment",
    "prior_observable_bound_increment", "prior_certificate_clip_indicator",
    "prior_certificate_clip_excess", "prior_cumulative_exploration_bound",
    "prior_cumulative_ucb_bound", "prior_cumulative_raw_ucb_bound",
    "prior_cumulative_observable_bound", "prior_cumulative_clip_excess",
    "prior_cumulative_clipped_episodes", "prior_universal_normalized_observable_bound",
    "prior_transfer_enlargement", "prior_transfer_increment", "prior_transfer_identity_holds",
    "bound_arithmetic_holds",
    "max_normalized_marginal_error", "parameter_confidence_holds",
    "marginal_confidence_holds", "design_min_eigenvalue_regularized",
    "design_min_eigenvalue_unregularized", "design_rank_unregularized",
    "audit_checkpoint", "audit_codebook_covariance_min_eigenvalue",
    "audit_codebook_covariance_rank", "audit_codebook_covariance_max_eigenvalue",
    "audit_codebook_distributed_mismatches", "audit_codebook_feasibility_failures",
    "central_distributed_mismatch", "allocation_rounds", "round_law_expected",
    "directed_transmissions", "resource_violations", "complete_family_violations",
    "minimum_residual_resource", "resource_consumption_sum", "fallback_episode",
    "tracking_rms", "tracking_peak", "minimum_mode_inequality_slack",
    "minimum_jump_inequality_slack", "finite_time_envelope_slack", "uub_radius_ratio",
    "control_limit_margin", "model_validity_margin", "f_max",
]


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def domain_seed(base_seed: int, domain: str) -> int:
    """Return a deterministic uint64 seed from a registered uint32 seed and domain."""
    payload = f"{int(base_seed)}|{domain}|{EXPERIMENT_ID}".encode("ascii")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def c_tag(c_exp: float) -> str:
    if abs(c_exp) < 1e-15:
        return "C0"
    text = format(float(c_exp), ".12g").replace("-", "m").replace(".", "p")
    return f"C{text}"


def exploration_probabilities(K: int, c_exp: float) -> np.ndarray:
    if K < 0 or not math.isfinite(c_exp) or c_exp < 0.0:
        raise ValueError("require K>=0 and finite c_exp>=0")
    if K == 0:
        return np.empty(0, dtype=float)
    if c_exp == 0.0:
        return np.zeros(K, dtype=float)
    k = np.arange(1, K + 1, dtype=float)
    return np.minimum(1.0, c_exp * k ** (-1.0 / 3.0))


def public_codebook(d: int, cfg: Mapping[str, Any]) -> np.ndarray:
    return public_nonnegative_codebook(
        d,
        seed=int(cfg["seed"]),
        directions_per_dimension=int(cfg["directions_per_dimension"]),
        alpha=float(cfg["dirichlet_alpha"]),
    )


def codebook_text(codebook: np.ndarray) -> str:
    rows = ["code_index," + ",".join(f"h{j+1}" for j in range(codebook.shape[1]))]
    for i, row in enumerate(codebook):
        rows.append(str(i) + "," + ",".join(format(float(v), ".17g") for v in row))
    return "\n".join(rows) + "\n"


def codebook_digest(codebook: np.ndarray) -> str:
    return hashlib.sha256(codebook_text(codebook).encode("ascii")).hexdigest()


def load_registered_codebook(
    sim_root: Path, experiment_config: Mapping[str, Any], campaign_id: str, dimension: int
) -> tuple[np.ndarray, str, str]:
    """Load and verify the public codebook registered for one application.

    Returns the matrix, its canonical matrix digest, and the SHA-256 digest of
    the serialized public CSV.  The serialized matrix must agree with the
    deterministic construction encoded in the included experiment configuration.
    """
    applications = {str(item["campaign_id"]): item for item in experiment_config["applications"]}
    if campaign_id not in applications:
        raise ValueError(f"campaign {campaign_id!r} is absent from the experiment configuration")
    app = applications[campaign_id]
    if int(app["dimension"]) != int(dimension):
        raise ValueError("registered codebook dimension disagrees with the application")
    path = sim_root / str(app["codebook_file"])
    if not path.is_file():
        raise FileNotFoundError(f"included public codebook not found: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
    expected_fields = ["campaign_id", "codebook_index", "probability"] + [
        f"h_{j+1}" for j in range(dimension)
    ]
    if reader.fieldnames != expected_fields:
        raise ValueError(f"invalid public-codebook schema in {path.name}")
    expected_size = 5 * dimension
    if len(rows) != expected_size:
        raise ValueError(f"public codebook has {len(rows)} rows; expected {expected_size}")
    values: list[list[float]] = []
    probabilities: list[float] = []
    for index, row in enumerate(rows):
        if row["campaign_id"] != campaign_id or int(row["codebook_index"]) != index:
            raise ValueError("public-codebook campaign/index mismatch")
        probabilities.append(float(row["probability"]))
        values.append([float(row[f"h_{j+1}"]) for j in range(dimension)])
    matrix = np.asarray(values, dtype=float)
    probability = np.asarray(probabilities, dtype=float)
    if np.any(matrix < -1.0e-15) or not np.allclose(matrix.sum(axis=1), 1.0, rtol=0.0, atol=2.0e-15):
        raise ValueError("public codebook must contain nonnegative simplex directions")
    if not np.allclose(probability, np.full(expected_size, 1.0 / expected_size), rtol=0.0, atol=2.0e-17):
        raise ValueError("public codebook distribution is not the configured uniform law")
    generated = public_codebook(dimension, experiment_config["codebook"])
    if not np.allclose(matrix, generated, rtol=0.0, atol=2.0e-16):
        raise ValueError("serialized public codebook differs from the configured construction")
    return matrix, codebook_digest(matrix), sha256_file(path)


@dataclass
class VariantState:
    c_exp: float
    method: MethodState
    cumulative_true: float = 0.0
    cumulative_oracle: float = 0.0
    expected_exploration_count: float = 0.0
    realized_exploration_count: int = 0
    direct_certificate: CertificateLedger = None  # type: ignore[assignment]
    support_certificate: CertificateLedger = None  # type: ignore[assignment]
    prior_certificate: CertificateLedger = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self.direct_certificate is None:
            self.direct_certificate = CertificateLedger()
        if self.support_certificate is None:
            self.support_certificate = CertificateLedger()
        if self.prior_certificate is None:
            self.prior_certificate = CertificateLedger()


@dataclass(frozen=True)
class CodebookAllocation:
    allocation: Allocation
    selected_feature: np.ndarray


def make_variant_state(
    *, assets: Any, initial: np.ndarray, n_agents: int, sigma: float, c_exp: float
) -> VariantState:
    p = assets.cfg["primary"]
    theta = np.asarray(p["theta_star"], dtype=float)
    lam = float(p["ridge_regularization"])
    delta = float(p["confidence_delta"])
    estimator = Estimator(len(theta), lam, sigma, delta, 1.05 * float(np.linalg.norm(theta)), float(theta.sum()))
    resources = np.tile(np.asarray(p["resource_initial"], dtype=float), (n_agents, 1))
    idle = list(assets.tracking.modes)[0]
    P = assets.tracking.modes[idle].p
    bounds = np.asarray([float(x @ P @ x) for x in initial])
    physical = PhysicalState(initial.copy(), [idle] * n_agents, bounds.copy())
    return VariantState(c_exp, MethodState(resources, estimator, physical))


def allocate_codebook_quantized(
    *,
    batch: Any,
    base: np.ndarray,
    direction: np.ndarray,
    bits: int,
    graph: GraphInfo,
    max_additions: int,
) -> CodebookAllocation:
    """Public-codebook score greedy through the distributed consensus interface."""
    h = np.asarray(direction, dtype=float)
    if h.ndim != 1 or h.shape[0] != batch.p_h.shape[1] or np.any(h < -1e-14):
        raise ValueError("direction must be a nonnegative vector of the basis dimension")
    selected: list[int] = []
    selected_mask = np.zeros(batch.size, dtype=bool)
    n_owner = int(batch.owner.max()) + 1
    n_task = int(batch.task.max()) + 1
    used_owner = np.zeros(n_owner, dtype=bool)
    used_task = np.zeros(n_task, dtype=bool)
    quota_counts = np.zeros(len(batch.quota_caps), dtype=int)
    failure = np.ones(batch.p_h.shape[1], dtype=float)
    mismatch = 0
    quant_eps = 0.5 / (2 ** bits)

    while len(selected) < max_additions:
        candidate = feasible_mask(
            batch, base, selected_mask, used_owner, used_task, quota_counts
        )
        idx = np.flatnonzero(candidate)
        if idx.size == 0:
            break
        psi = batch.p_h[idx] * failure
        scores = psi @ h
        quantized = np.rint(scores * (2 ** bits)) / (2 ** bits)
        central_pos = argmax_tie(quantized, batch.element_id[idx])
        central = int(idx[central_pos])

        local: list[int] = []
        for owner in range(n_owner):
            pos = np.flatnonzero(batch.owner[idx] == owner)
            if pos.size:
                winner_pos = pos[argmax_tie(quantized[pos], batch.element_id[idx[pos]])]
                local.append(int(idx[winner_pos]))
        local_arr = np.asarray(local, dtype=int)
        local_scores = np.asarray(
            [quantized[int(np.flatnonzero(idx == e)[0])] for e in local_arr],
            dtype=float,
        )
        distributed = int(
            local_arr[argmax_tie(local_scores, batch.element_id[local_arr])]
        )
        mismatch += int(central != distributed)
        choice = distributed
        selected.append(choice)
        selected_mask[choice] = True
        used_owner[batch.owner[choice]] = True
        used_task[batch.task[choice]] = True
        quota_counts[batch.quota[choice]] += 1
        failure *= 1.0 - batch.p_h[choice]

    feature = 1.0 - failure
    return CodebookAllocation(
        Allocation(
            selected=selected,
            mismatch=mismatch,
            width_sum=0.0,
            max_norm_error=None,
            marginal_holds=None,
            quant_eps=quant_eps,
            partial_violations=0,
        ),
        feature,
    )


def fixed_trace_calibration_mismatch(
    *,
    batch: Any,
    base: np.ndarray,
    selected: Sequence[int],
    estimator: Estimator,
    target_beta: float,
    bits: int,
) -> float:
    """Replay a baseline winner sequence under enlarged encoded UCB scores.

    The returned sum is the deterministic calibration channel
    ``sum_l [max_e Ubar_target(e)-Ubar_target(e_l)]_+`` on the immutable
    baseline partial sets.  It is zero when the baseline winner remains a
    target-score maximizer and is never used to relabel the trace as a run of
    the enlarged-score policy.
    """
    selected_mask = np.zeros(batch.size, dtype=bool)
    n_owner = int(batch.owner.max()) + 1
    n_task = int(batch.task.max()) + 1
    used_owner = np.zeros(n_owner, dtype=bool)
    used_task = np.zeros(n_task, dtype=bool)
    quota_counts = np.zeros(len(batch.quota_caps), dtype=int)
    failure = np.ones(batch.p_h.shape[1], dtype=float)
    mismatch = 0.0
    scale = float(2 ** bits)

    for stage, choice_value in enumerate(selected, 1):
        choice = int(choice_value)
        candidate = feasible_mask(
            batch, base, selected_mask, used_owner, used_task, quota_counts
        )
        idx = np.flatnonzero(candidate)
        if idx.size == 0 or not bool(candidate[choice]):
            raise RuntimeError(
                f"baseline winner is infeasible during target replay at stage {stage}"
            )
        psi = batch.p_h[idx] * failure
        means = psi @ estimator.theta
        widths = np.sqrt(
            np.maximum(
                0.0,
                np.einsum(
                    "ij,ji->i", psi, np.linalg.solve(estimator.V, psi.T)
                ),
            )
        )
        target_scores = means + float(target_beta) * widths
        encoded = np.rint(target_scores * scale) / scale
        choice_pos = int(np.flatnonzero(idx == choice)[0])
        mismatch += max(0.0, float(np.max(encoded) - encoded[choice_pos]))

        selected_mask[choice] = True
        used_owner[batch.owner[choice]] = True
        used_task[batch.task[choice]] = True
        quota_counts[batch.quota[choice]] += 1
        failure *= 1.0 - batch.p_h[choice]
    return mismatch


def audit_codebook_covariance(
    *, batch: Any, base: np.ndarray, codebook: np.ndarray, bits: int,
    graph: GraphInfo, max_additions: int,
) -> dict[str, Any]:
    features: list[np.ndarray] = []
    mismatches = 0
    feasibility_failures = 0
    for h in codebook:
        result = allocate_codebook_quantized(
            batch=batch,
            base=base,
            direction=h,
            bits=bits,
            graph=graph,
            max_additions=max_additions,
        )
        al = result.allocation
        features.append(result.selected_feature)
        mismatches += al.mismatch
        feasibility_failures += int(assignment_violations(batch, al.selected) != 0)
    X = np.asarray(features, dtype=float)
    covariance = (X.T @ X) / float(len(X))
    eig = np.linalg.eigvalsh((covariance + covariance.T) / 2.0)
    return {
        "minimum_eigenvalue": float(eig[0]),
        "maximum_eigenvalue": float(eig[-1]),
        "rank": int(np.count_nonzero(eig > 1.0e-10)),
        "mismatches": int(mismatches),
        "feasibility_failures": int(feasibility_failures),
    }


def checkpoint_epochs(K: int, fractions: Sequence[float]) -> set[int]:
    out: set[int] = set()
    for fraction in fractions:
        f = float(fraction)
        if f <= 0.0:
            out.add(1)
        else:
            out.add(min(K, max(1, int(round(f * K)))))
    out.add(K)
    return out


def raw_name(campaign_id: str, partition: str, index: int, seed: int, c_exp: float) -> str:
    return (
        f"exploration_{campaign_id}_{partition}_T{index:02d}_S{seed}_"
        f"{c_tag(c_exp)}_EXPLORATION_UCB.csv"
    )


def _finite_min_eig(matrix: np.ndarray) -> float:
    return float(np.linalg.eigvalsh((matrix + matrix.T) / 2.0)[0])


def run_trial(
    *,
    sim_root: Path,
    campaign_path: Path,
    experiment_config: Mapping[str, Any],
    seed_row: Mapping[str, Any],
    raw_dir: Path,
    engine_hash: str,
    experiment_config_hash: str,
    seed_registry_hash: str,
    c_values: Sequence[float] = C_EXP_GRID,
    write_raw: bool = True,
    epoch_limit: int | None = None,
) -> dict[str, Any]:
    """Run all exploration variants for one application trial using common random numbers."""
    start = time.perf_counter()
    assets = load_assets(sim_root, campaign_path)
    cfg = assets.cfg
    p = cfg["primary"]
    n_agents = int(p["agents"])
    n_tasks = int(p["tasks"])
    K = min(int(p["epochs"]), epoch_limit or int(p["epochs"]))
    theta = np.asarray(p["theta_star"], dtype=float)
    d = len(theta)
    sigma = float(p["feedback_noise_sigma"])
    bits = int(p["score_quantization_bits"])
    f_max = float(theta.sum())
    q = int(p["q"])
    lam = float(p["ridge_regularization"])
    delta = float(p["confidence_delta"])
    direct_btheta = 1.05 * float(np.linalg.norm(theta))
    # Application support specialization used by the fixed-trace route.
    # The configured reference quantities are L_Y=F_max/2 and
    # Y_max^{cl,app}=F_max+L_Y R_all.
    support_sigma = 0.5 * (
        f_max + 0.5 * f_max * float(assets.tracking.all_step_radius)
    )
    prior_btheta = 3.0 * float(np.linalg.norm(theta))
    certificate_module_hash = sha256_file(sim_root / "src" / "certificate_arithmetic.py")

    codebook, cb_digest, codebook_file_sha256 = load_registered_codebook(
        sim_root, experiment_config, str(cfg["campaign_id"]), d
    )
    c_values = tuple(float(c) for c in c_values)
    if sorted(c_values) != list(c_values) or any(c < 0.0 for c in c_values):
        raise ValueError("c_values must be sorted, unique, and nonnegative")
    if len(set(c_values)) != len(c_values):
        raise ValueError("c_values must be unique")

    graph_rng = np.random.default_rng(int(seed_row["algorithm_seed"]) ^ 0xA13F09)
    graph = graph_info(n_agents, str(p["graph"]), float(p["graph_radius"]), graph_rng)
    context = ContextGenerator(
        assets, int(seed_row["scenario_seed"]), n_agents, n_tasks, True
    )
    feedback_rng = np.random.default_rng(int(seed_row["feedback_seed"]))
    physical_rng = np.random.default_rng(int(seed_row["physical_seed"]))

    exploration_seed = int(
        seed_row.get(
            "exploration_seed",
            domain_seed(
                int(seed_row["algorithm_seed"]),
                experiment_config["common_random_numbers"]["exploration_uniform_domain_tag"],
            ),
        )
    )
    codebook_seed = int(
        seed_row.get(
            "codebook_seed",
            domain_seed(
                int(seed_row["algorithm_seed"]),
                experiment_config["common_random_numbers"]["codebook_index_domain_tag"],
            ),
        )
    )
    explore_rng = np.random.default_rng(exploration_seed)
    code_rng = np.random.default_rng(codebook_seed)
    explore_uniform = explore_rng.random(K)
    code_indices = code_rng.integers(0, len(codebook), size=K, dtype=np.int64)
    probabilities = {c: exploration_probabilities(K, c) for c in c_values}
    expected = {c: np.cumsum(probabilities[c]) for c in c_values}

    state_dim = next(iter(assets.tracking.modes.values())).a_cl.shape[0]
    input_dim = next(iter(assets.tracking.modes.values())).e_w.shape[1]
    initial = random_ball(
        physical_rng, (n_agents,), state_dim, assets.tracking.initial_error_bound
    )
    variants = {
        c: make_variant_state(
            assets=assets, initial=initial, n_agents=n_agents, sigma=sigma, c_exp=c
        )
        for c in c_values
    }
    rows: dict[float, list[dict[str, Any]]] = {c: [] for c in c_values}
    checkpoints = checkpoint_epochs(K, experiment_config["covariance_audit_fractions"])
    burn = max(1, math.ceil(0.2 * K))
    config_hash = sha256_file(campaign_path)

    for k in range(1, K + 1):
        batch = context.next(k)
        sign = 1.0 if feedback_rng.random() < 0.5 else -1.0
        disturbances = random_ball(
            physical_rng,
            (assets.tracking.horizon, n_agents),
            input_dim,
            assets.tracking.w_bound,
        )
        jumps = random_ball(
            physical_rng, (n_agents,), state_dim, assets.tracking.jump_bound
        )
        uniforms = physical_rng.random(n_agents)
        code_index = int(code_indices[k - 1])

        for c in c_values:
            variant = variants[c]
            method = variant.method
            estimator = method.estimator
            if estimator is None:
                raise AssertionError("exploration variant requires a projected estimator")
            affordable = np.all(
                batch.robust_cost <= method.resources[batch.owner] + 1.0e-12,
                axis=1,
            )
            base = batch.kinematic & affordable
            kappa = curvature(theta, batch.p_h[base])
            oracle = allocate(
                batch,
                base,
                "CENTRAL_GREEDY_ORACLE",
                theta,
                None,
                bits,
                graph,
                None,
                max_add=n_agents,
            )
            p_k = float(probabilities[c][k - 1])
            explore = bool(explore_uniform[k - 1] < p_k)
            beta = estimator.beta()
            if explore:
                code_result = allocate_codebook_quantized(
                    batch=batch,
                    base=base,
                    direction=codebook[code_index],
                    bits=bits,
                    graph=graph,
                    max_additions=n_agents,
                )
                allocation = code_result.allocation
                branch = "EXPLORATION"
            else:
                allocation = allocate(
                    batch,
                    base,
                    "DISTRIBUTED_UCB",
                    theta,
                    estimator,
                    bits,
                    graph,
                    None,
                    max_add=n_agents,
                )
                branch = "UCB"

            family_violations = assignment_violations(batch, allocation.selected)
            valid = family_violations == 0
            selected = allocation.selected if valid else []
            true_value = coverage_value(theta, batch.p_h, selected)
            oracle_value = coverage_value(theta, batch.p_h, oracle.selected)
            oracle_ratio = true_value / oracle_value if oracle_value > 1.0e-15 else 1.0
            feature = coverage_feature(batch.p_h, selected)
            noise = sign * min(sigma, 0.45 * true_value) if true_value > 0.0 else 0.0
            realized_return = max(0.0, true_value + noise)

            physical = physical_epoch(
                assets,
                method.physical,
                batch,
                selected,
                disturbances,
                jumps,
                uniforms,
                1.0,
                1.0,
                k >= burn,
            )
            if physical["fallback"]:
                realized_return = 0.0

            consumption = np.zeros_like(method.resources)
            for e in selected:
                consumption[batch.owner[e]] += batch.actual_cost[e]
            replenishment = np.asarray(p["resource_replenishment_per_epoch"], dtype=float)
            initial_resource = np.asarray(p["resource_initial"], dtype=float)
            method.resources = np.minimum(
                initial_resource, method.resources - consumption + replenishment
            )
            resource_violations = int(np.sum(method.resources < -1.0e-10))
            minimum_residual = float(method.resources.min())

            parameter_holds = bool(
                float((estimator.theta - theta) @ estimator.V @ (estimator.theta - theta))
                <= beta * beta + 1.0e-8
            )

            # All certificate quantities below are computed before the estimator
            # update.  The support and prior routes recertify the immutable
            # direct-score trace; they do not alter the selected sequence.
            support_beta = beta_radius(
                estimator.V, lam, support_sigma, delta, direct_btheta
            )
            prior_beta = beta_radius(
                estimator.V, lam, sigma, delta, prior_btheta
            )
            if support_beta + 1.0e-12 < beta or prior_beta + 1.0e-12 < beta:
                raise RuntimeError("fixed-trace target radius is not an enlargement")
            if explore:
                support_mismatch = 0.0
                prior_mismatch = 0.0
            else:
                support_mismatch = fixed_trace_calibration_mismatch(
                    batch=batch,
                    base=base,
                    selected=allocation.selected,
                    estimator=estimator,
                    target_beta=support_beta,
                    bits=bits,
                )
                prior_mismatch = fixed_trace_calibration_mismatch(
                    batch=batch,
                    base=base,
                    selected=allocation.selected,
                    estimator=estimator,
                    target_beta=prior_beta,
                    bits=bits,
                )

            L = len(allocation.selected)
            quantization_charge = 2.0 * L * float(allocation.quant_eps)
            raw_direct = 2.0 * beta * float(allocation.width_sum) + quantization_charge
            raw_support = (
                2.0 * support_beta * float(allocation.width_sum)
                + quantization_charge
                + support_mismatch
            )
            raw_prior = (
                2.0 * prior_beta * float(allocation.width_sum)
                + quantization_charge
                + prior_mismatch
            )
            direct_episode = certify_episode(
                exploration=explore,
                raw_exploitation_charge=raw_direct,
                f_max=f_max,
                q=q,
            )
            support_episode = certify_episode(
                exploration=explore,
                raw_exploitation_charge=raw_support,
                f_max=f_max,
                q=q,
            )
            prior_episode = certify_episode(
                exploration=explore,
                raw_exploitation_charge=raw_prior,
                f_max=f_max,
                q=q,
            )
            support_enlargement = max(0.0, raw_support - raw_direct)
            prior_enlargement = max(0.0, raw_prior - raw_direct)
            support_transfer = 0.0 if explore else clipped_transfer_increment(
                baseline_raw_charge=raw_direct,
                enlargement=support_enlargement,
                cap=direct_episode.cap,
            )
            prior_transfer = 0.0 if explore else clipped_transfer_increment(
                baseline_raw_charge=raw_direct,
                enlargement=prior_enlargement,
                cap=direct_episode.cap,
            )
            support_identity_holds = bool(
                abs(
                    support_episode.certified_charge
                    - direct_episode.certified_charge
                    - support_transfer
                ) <= 2.0e-10
            )
            prior_identity_holds = bool(
                abs(
                    prior_episode.certified_charge
                    - direct_episode.certified_charge
                    - prior_transfer
                ) <= 2.0e-10
            )
            if not support_identity_holds or not prior_identity_holds:
                raise RuntimeError("clipped fixed-trace transfer identity failed")

            if explore:
                variant.realized_exploration_count += 1
            variant.direct_certificate.add(direct_episode, exploration=explore)
            variant.support_certificate.add(support_episode, exploration=explore)
            variant.prior_certificate.add(prior_episode, exploration=explore)

            estimator.update(feature, realized_return)
            unregularized = estimator.V - estimator.lam * np.eye(d)
            design_min_reg = _finite_min_eig(estimator.V)
            design_min_unreg = max(0.0, _finite_min_eig(unregularized))
            design_rank = int(np.linalg.matrix_rank(unregularized, tol=1.0e-10))

            arithmetic_holds = bool(
                direct_episode.certified_charge <= direct_episode.cap + 1.0e-10
                and support_episode.certified_charge <= direct_episode.cap + 1.0e-10
                and prior_episode.certified_charge <= direct_episode.cap + 1.0e-10
                and support_identity_holds
                and prior_identity_holds
            )
            variant.cumulative_true += true_value
            variant.cumulative_oracle += oracle_value
            variant.expected_exploration_count = float(expected[c][k - 1])
            cumulative_ratio = (
                variant.cumulative_true / variant.cumulative_oracle
                if variant.cumulative_oracle > 1.0e-15
                else 1.0
            )
            direct_normalized = universal_normalization(
                cumulative_charge=variant.direct_certificate.total,
                epoch=k,
                f_max=f_max,
                q=q,
            )
            support_normalized = universal_normalization(
                cumulative_charge=variant.support_certificate.total,
                epoch=k,
                f_max=f_max,
                q=q,
            )
            prior_normalized = universal_normalization(
                cumulative_charge=variant.prior_certificate.total,
                epoch=k,
                f_max=f_max,
                q=q,
            )

            audit: dict[str, Any] | None = None
            if k in checkpoints:
                audit = audit_codebook_covariance(
                    batch=batch,
                    base=base,
                    codebook=codebook,
                    bits=bits,
                    graph=graph,
                    max_additions=n_agents,
                )

            L = len(allocation.selected)
            rounds = (L + 1) * graph.diameter
            transmissions = 2 * graph.edges * rounds
            selected_ids = [int(batch.element_id[e]) for e in selected]
            row = {
                "schema_version": RAW_SCHEMA,
                "software_version": SOFTWARE_VERSION,
                "experiment_id": EXPERIMENT_ID,
                "experiment_config_hash": experiment_config_hash,
                "campaign_id": cfg["campaign_id"],
                "partition": str(seed_row["partition"]),
                "trial_index": int(seed_row["trial_index"]),
                "trial_seed": int(seed_row["trial_seed"]),
                "scenario_seed": int(seed_row["scenario_seed"]),
                "feedback_seed": int(seed_row["feedback_seed"]),
                "physical_seed": int(seed_row["physical_seed"]),
                "algorithm_seed": int(seed_row["algorithm_seed"]),
                "exploration_seed": exploration_seed,
                "codebook_seed": codebook_seed,
                "seed_registry_hash": seed_registry_hash,
                "c_exp": c,
                "variant_id": c_tag(c),
                "epoch": k,
                "source_config_hash": config_hash,
                "engine_hash": engine_hash,
                "certificate_module_hash": certificate_module_hash,
                "codebook_hash": cb_digest,
                "codebook_file_sha256": codebook_file_sha256,
                "codebook_size": len(codebook),
                "codebook_index": code_index,
                "exploration_uniform": float(explore_uniform[k - 1]),
                "exploration_probability": p_k,
                "exploration_indicator": explore,
                "expected_exploration_count": variant.expected_exploration_count,
                "realized_exploration_count": variant.realized_exploration_count,
                "branch": branch,
                "graph_family": graph.family,
                "graph_edges": graph.edges,
                "graph_diameter": graph.diameter,
                "active_q": q,
                "selected_count": L,
                "selected_elements": "|".join(map(str, selected_ids)),
                "selection_hash": stable_hash("|".join(map(str, selected_ids))),
                "assignment_valid": valid,
                "true_value": true_value,
                "realized_return": realized_return,
                "oracle_greedy_value": oracle_value,
                "oracle_value_ratio": oracle_ratio,
                "cumulative_true_value": variant.cumulative_true,
                "cumulative_oracle_value": variant.cumulative_oracle,
                "cumulative_oracle_ratio": cumulative_ratio,
                "empirical_curvature": kappa,
                "comparison_factor": 1.0 / (q + kappa),
                "fixed_comparator_factor": 1.0 / (q + 1.0),
                "beta": beta,
                "selected_width_sum": allocation.width_sum,
                "score_quantization_epsilon": allocation.quant_eps,
                "certificate_cap": direct_episode.cap,
                "raw_exploitation_charge": direct_episode.raw_exploitation_charge,
                "raw_observable_bound_increment": direct_episode.raw_charge,
                "observable_bound_increment": direct_episode.certified_charge,
                "certificate_clip_indicator": int(direct_episode.clipped),
                "certificate_clip_excess": direct_episode.clip_excess,
                "cumulative_exploration_bound": variant.direct_certificate.exploration,
                "cumulative_ucb_bound": variant.direct_certificate.exploitation,
                "cumulative_raw_ucb_bound": variant.direct_certificate.raw_exploitation,
                "cumulative_observable_bound": variant.direct_certificate.total,
                "cumulative_clip_excess": variant.direct_certificate.clip_excess,
                "cumulative_clipped_episodes": variant.direct_certificate.clipped_episodes,
                "universal_normalized_observable_bound": direct_normalized,
                "support_sigma": support_sigma,
                "support_beta": support_beta,
                "support_calibration_mismatch": support_mismatch,
                "support_raw_exploitation_charge": support_episode.raw_exploitation_charge,
                "support_raw_observable_bound_increment": support_episode.raw_charge,
                "support_observable_bound_increment": support_episode.certified_charge,
                "support_certificate_clip_indicator": int(support_episode.clipped),
                "support_certificate_clip_excess": support_episode.clip_excess,
                "support_cumulative_exploration_bound": variant.support_certificate.exploration,
                "support_cumulative_ucb_bound": variant.support_certificate.exploitation,
                "support_cumulative_raw_ucb_bound": variant.support_certificate.raw_exploitation,
                "support_cumulative_observable_bound": variant.support_certificate.total,
                "support_cumulative_clip_excess": variant.support_certificate.clip_excess,
                "support_cumulative_clipped_episodes": variant.support_certificate.clipped_episodes,
                "support_universal_normalized_observable_bound": support_normalized,
                "support_transfer_enlargement": support_enlargement,
                "support_transfer_increment": support_transfer,
                "support_transfer_identity_holds": support_identity_holds,
                "prior_btheta": prior_btheta,
                "prior_beta": prior_beta,
                "prior_calibration_mismatch": prior_mismatch,
                "prior_raw_exploitation_charge": prior_episode.raw_exploitation_charge,
                "prior_raw_observable_bound_increment": prior_episode.raw_charge,
                "prior_observable_bound_increment": prior_episode.certified_charge,
                "prior_certificate_clip_indicator": int(prior_episode.clipped),
                "prior_certificate_clip_excess": prior_episode.clip_excess,
                "prior_cumulative_exploration_bound": variant.prior_certificate.exploration,
                "prior_cumulative_ucb_bound": variant.prior_certificate.exploitation,
                "prior_cumulative_raw_ucb_bound": variant.prior_certificate.raw_exploitation,
                "prior_cumulative_observable_bound": variant.prior_certificate.total,
                "prior_cumulative_clip_excess": variant.prior_certificate.clip_excess,
                "prior_cumulative_clipped_episodes": variant.prior_certificate.clipped_episodes,
                "prior_universal_normalized_observable_bound": prior_normalized,
                "prior_transfer_enlargement": prior_enlargement,
                "prior_transfer_increment": prior_transfer,
                "prior_transfer_identity_holds": prior_identity_holds,
                "bound_arithmetic_holds": arithmetic_holds,
                "max_normalized_marginal_error": allocation.max_norm_error,
                "parameter_confidence_holds": parameter_holds,
                "marginal_confidence_holds": allocation.marginal_holds,
                "design_min_eigenvalue_regularized": design_min_reg,
                "design_min_eigenvalue_unregularized": design_min_unreg,
                "design_rank_unregularized": design_rank,
                "audit_checkpoint": k in checkpoints,
                "audit_codebook_covariance_min_eigenvalue": None if audit is None else audit["minimum_eigenvalue"],
                "audit_codebook_covariance_rank": None if audit is None else audit["rank"],
                "audit_codebook_covariance_max_eigenvalue": None if audit is None else audit["maximum_eigenvalue"],
                "audit_codebook_distributed_mismatches": None if audit is None else audit["mismatches"],
                "audit_codebook_feasibility_failures": None if audit is None else audit["feasibility_failures"],
                "central_distributed_mismatch": allocation.mismatch,
                "allocation_rounds": rounds,
                "round_law_expected": rounds,
                "directed_transmissions": transmissions,
                "resource_violations": resource_violations,
                "complete_family_violations": family_violations,
                "minimum_residual_resource": minimum_residual,
                "resource_consumption_sum": float(consumption.sum()),
                "fallback_episode": physical["fallback"],
                "tracking_rms": physical["rms"],
                "tracking_peak": physical["peak"],
                "minimum_mode_inequality_slack": physical["mode_slack"],
                "minimum_jump_inequality_slack": physical["jump_slack"],
                "finite_time_envelope_slack": physical["envelope_slack"],
                "uub_radius_ratio": physical["uub_ratio"],
                "control_limit_margin": physical["control_margin"],
                "model_validity_margin": physical["validity_margin"],
                "f_max": f_max,
            }
            rows[c].append(row)

    files: list[dict[str, Any]] = []
    if write_raw:
        for c in c_values:
            path = raw_dir / raw_name(
                cfg["campaign_id"],
                str(seed_row["partition"]),
                int(seed_row["trial_index"]),
                int(seed_row["trial_seed"]),
                c,
            )
            digest, count = write_csv_atomic(path, rows[c], RAW_FIELDS)
            try:
                recorded_path = path.relative_to(sim_root).as_posix()
            except ValueError:
                recorded_path = path.as_posix()
            files.append(
                {
                    "relative_path": recorded_path,
                    "sha256": digest,
                    "rows": count,
                    "c_exp": c,
                }
            )

    aggregates: dict[str, dict[str, Any]] = {}
    for c in c_values:
        rr = rows[c]
        last = rr[-1]
        aggregates[c_tag(c)] = {
            "c_exp": c,
            "epochs": K,
            "true_value": float(last["cumulative_true_value"]),
            "oracle_value": float(last["cumulative_oracle_value"]),
            "cumulative_oracle_ratio": float(last["cumulative_oracle_ratio"]),
            "mean_epoch_oracle_ratio": float(np.mean([r["oracle_value_ratio"] for r in rr])),
            "realized_explorations": int(last["realized_exploration_count"]),
            "expected_explorations": float(last["expected_exploration_count"]),
            "exploration_fraction": float(last["realized_exploration_count"]) / K,
            "observable_bound": float(last["cumulative_observable_bound"]),
            "universal_normalized_observable_bound": float(last["universal_normalized_observable_bound"]),
            "support_observable_bound": float(last["support_cumulative_observable_bound"]),
            "support_universal_normalized_observable_bound": float(last["support_universal_normalized_observable_bound"]),
            "prior_observable_bound": float(last["prior_cumulative_observable_bound"]),
            "prior_universal_normalized_observable_bound": float(last["prior_universal_normalized_observable_bound"]),
            "direct_clipped_episodes": int(last["cumulative_clipped_episodes"]),
            "support_clipped_episodes": int(last["support_cumulative_clipped_episodes"]),
            "prior_clipped_episodes": int(last["prior_cumulative_clipped_episodes"]),
            "terminal_design_min_eigenvalue_unregularized": float(last["design_min_eigenvalue_unregularized"]),
            "terminal_design_rank_unregularized": int(last["design_rank_unregularized"]),
            "resource_violations": int(sum(int(r["resource_violations"]) for r in rr)),
            "family_violations": int(sum(int(r["complete_family_violations"]) for r in rr)),
            "mismatches": int(sum(int(r["central_distributed_mismatch"]) for r in rr)),
            "fallback_episodes": int(sum(int(bool(r["fallback_episode"])) for r in rr)),
            "bound_failures": int(sum(not bool(r["bound_arithmetic_holds"]) for r in rr)),
            "minimum_envelope_slack": float(min(float(r["finite_time_envelope_slack"]) for r in rr)),
            "minimum_control_margin": float(min(float(r["control_limit_margin"]) for r in rr)),
            "minimum_model_validity_margin": float(min(float(r["model_validity_margin"]) for r in rr)),
            "maximum_uub_ratio": float(max(float(r["uub_radius_ratio"]) for r in rr if r["uub_radius_ratio"] is not None and math.isfinite(float(r["uub_radius_ratio"])))),
            "audit_minimum_covariance_eigenvalue": float(min(float(r["audit_codebook_covariance_min_eigenvalue"]) for r in rr if r["audit_codebook_covariance_min_eigenvalue"] is not None)),
            "audit_minimum_covariance_rank": int(min(int(r["audit_codebook_covariance_rank"]) for r in rr if r["audit_codebook_covariance_rank"] is not None)),
            "audit_mismatches": int(sum(int(r["audit_codebook_distributed_mismatches"] or 0) for r in rr)),
            "audit_feasibility_failures": int(sum(int(r["audit_codebook_feasibility_failures"] or 0) for r in rr)),
        }

    return {
        "status": "PASS",
        "software_version": SOFTWARE_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "campaign_id": cfg["campaign_id"],
        "partition": str(seed_row["partition"]),
        "trial_index": int(seed_row["trial_index"]),
        "trial_seed": int(seed_row["trial_seed"]),
        "epochs": K,
        "codebook_hash": cb_digest,
        "codebook_size": len(codebook),
        "files": files,
        "aggregates": aggregates,
        "elapsed_seconds": time.perf_counter() - start,
        "graph": {"family": graph.family, "edges": graph.edges, "diameter": graph.diameter},
    }


def load_raw(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))
