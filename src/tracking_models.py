"""Numerical construction and simulation of the application tracking models.

The routines build the configured linear feedback modes, evaluate the associated
quadratic bounds, and simulate deterministic or seeded disturbance trajectories.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
import json
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy.linalg import block_diag, eigh, eigvalsh, solve_discrete_are, solve_discrete_lyapunov


TOL = 1.0e-10


def sym(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def max_generalized_eigenvalue(a: np.ndarray, b: np.ndarray) -> float:
    """Return max lambda for a v = lambda b v, with symmetric a and b>0."""
    return float(eigh(sym(a), sym(b), eigvals_only=True).max())


def min_eigenvalue(matrix: np.ndarray) -> float:
    return float(eigvalsh(sym(matrix)).min())


def max_eigenvalue(matrix: np.ndarray) -> float:
    return float(eigvalsh(sym(matrix)).max())


def dlqr(a: np.ndarray, b: np.ndarray, q: np.ndarray, r: np.ndarray) -> np.ndarray:
    """Discrete-time infinite-horizon LQR gain u=-Kx."""
    riccati = solve_discrete_are(a, b, q, r)
    return np.linalg.solve(b.T @ riccati @ b + r, b.T @ riccati @ a)


@dataclass(frozen=True)
class ControllerMode:
    name: str
    a_cl: np.ndarray
    e_w: np.ndarray
    p: np.ndarray
    k: np.ndarray
    spectral_radius: float
    contraction_floor: float


@dataclass(frozen=True)
class TrackingBounds:
    application: str
    modes: dict[str, ControllerMode]
    horizon: int
    v_lower: float
    v_upper: float
    lambda_floor: float
    lambda_c: float
    mu_floor: float
    mu: float
    c_w: float
    c_jump: float
    rho_h: float
    minimum_horizon: int
    w_bound: float
    jump_bound: float
    initial_error_bound: float
    bounded_epoch_input: float
    all_step_radius: float
    tracking_radius_bound: float
    g_w: float
    g_jump: float
    control_bound: float
    control_limit: float
    validity_radius: float
    minimum_within_mode_block_margin: float
    minimum_jump_block_margin: float
    numerical_padding_absolute: float
    numerical_padding_relative: float


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _axis_model(config: dict[str, Any]) -> tuple[np.ndarray, list[np.ndarray]]:
    dt = float(config["sample_time"])
    a_axis = np.array([[1.0, dt], [0.0, 1.0]], dtype=float)
    b_axes = [
        float(gain) * np.array([[0.5 * dt * dt], [dt]], dtype=float)
        for gain in config["axis_input_gains"]
    ]
    return a_axis, b_axes


def build_controller_modes(config: dict[str, Any]) -> dict[str, ControllerMode]:
    """Build normalized tracking templates and their quadratic functions."""
    a_axis, b_axes = _axis_model(config)
    q_cert_axis = np.diag(np.asarray(config["state_weight_axis"], dtype=float))
    modes: dict[str, ControllerMode] = {}
    for mode_name, mode_cfg in config["controller_templates"].items():
        q_lqr_axis = np.diag(np.asarray(mode_cfg["lqr_state_weight_axis"], dtype=float))
        r_axis = np.array([[float(mode_cfg["lqr_input_weight"])]], dtype=float)
        a_blocks: list[np.ndarray] = []
        e_blocks: list[np.ndarray] = []
        p_blocks: list[np.ndarray] = []
        k_blocks: list[np.ndarray] = []
        for b_axis in b_axes:
            k_axis = dlqr(a_axis, b_axis, q_lqr_axis, r_axis)
            a_cl_axis = a_axis - b_axis @ k_axis
            p_axis = solve_discrete_lyapunov(a_cl_axis.T, q_cert_axis)
            if min_eigenvalue(p_axis) <= 0.0:
                raise ValueError(f"nonpositive Lyapunov matrix for {mode_name}")
            a_blocks.append(a_cl_axis)
            e_blocks.append(b_axis)
            p_blocks.append(p_axis)
            k_blocks.append(k_axis)
        a_cl = block_diag(*a_blocks)
        e_w = block_diag(*e_blocks)
        p = block_diag(*p_blocks)
        k = block_diag(*k_blocks)
        spectral_radius = float(max(abs(np.linalg.eigvals(a_cl))))
        contraction_floor = max_generalized_eigenvalue(a_cl.T @ p @ a_cl, p)
        if not spectral_radius < 1.0 - 1.0e-12:
            raise ValueError(f"unstable controller template {mode_name}")
        modes[mode_name] = ControllerMode(
            name=mode_name,
            a_cl=a_cl,
            e_w=e_w,
            p=p,
            k=k,
            spectral_radius=spectral_radius,
            contraction_floor=contraction_floor,
        )
    return modes


def _within_mode_cw(
    mode: ControllerMode,
    lambda_c: float,
    *,
    padding_absolute: float,
    padding_relative: float,
) -> tuple[float, float]:
    margin = sym(lambda_c * mode.p - mode.a_cl.T @ mode.p @ mode.a_cl)
    if min_eigenvalue(margin) <= 0.0:
        raise ValueError(f"lambda_c does not dominate mode {mode.name}")
    gain = sym(
        mode.e_w.T @ mode.p @ mode.e_w
        + mode.e_w.T
        @ mode.p
        @ mode.a_cl
        @ np.linalg.solve(margin, mode.a_cl.T @ mode.p @ mode.e_w)
    )
    c_w_exact = max_eigenvalue(gain)
    c_w = c_w_exact + max(
        padding_absolute, padding_relative * max(1.0, abs(c_w_exact))
    )
    block = np.block(
        [
            [margin, -mode.a_cl.T @ mode.p @ mode.e_w],
            [-mode.e_w.T @ mode.p @ mode.a_cl, c_w * np.eye(mode.e_w.shape[1]) - mode.e_w.T @ mode.p @ mode.e_w],
        ]
    )
    return c_w, min_eigenvalue(block)


def _jump_cj(
    source: ControllerMode,
    target: ControllerMode,
    mu: float,
    *,
    padding_absolute: float,
    padding_relative: float,
) -> tuple[float, float]:
    margin = sym(mu * source.p - target.p)
    if min_eigenvalue(margin) <= 0.0:
        raise ValueError(f"mu does not dominate jump {source.name}->{target.name}")
    gain = sym(target.p + target.p @ np.linalg.solve(margin, target.p))
    c_jump_exact = max_eigenvalue(gain)
    c_jump = c_jump_exact + max(
        padding_absolute, padding_relative * max(1.0, abs(c_jump_exact))
    )
    n = target.p.shape[0]
    block = np.block(
        [
            [margin, -target.p],
            [-target.p, c_jump * np.eye(n) - target.p],
        ]
    )
    return c_jump, min_eigenvalue(block)


def phase_radii(
    *,
    v_lower: float,
    lambda_c: float,
    mu: float,
    c_w: float,
    c_jump: float,
    horizon: int,
    w_bound: float,
    jump_bound: float,
) -> list[float]:
    rho_h = mu * lambda_c**horizon
    if not rho_h < 1.0:
        raise ValueError("rho_h must be below one")
    d_bar = (
        mu * c_w * (1.0 - lambda_c**horizon) / (1.0 - lambda_c) * w_bound**2
        + c_jump * jump_bound**2
    )
    values: list[float] = []
    for tau in range(horizon + 1):
        radius_sq = (
            lambda_c**tau * d_bar / (v_lower * (1.0 - rho_h))
            + c_w * (1.0 - lambda_c**tau) / (v_lower * (1.0 - lambda_c)) * w_bound**2
        )
        values.append(math.sqrt(max(radius_sq, 0.0)))
    return values


def build_tracking_bounds(config: dict[str, Any]) -> tuple[TrackingBounds, list[dict[str, Any]], list[dict[str, Any]]]:
    modes = build_controller_modes(config)
    padding_cfg = config.get(
        "numerical_padding", {"absolute": 1.0e-8, "relative": 1.0e-6}
    )
    padding_absolute = float(padding_cfg["absolute"])
    padding_relative = float(padding_cfg["relative"])
    if padding_absolute <= 0.0 or padding_relative <= 0.0:
        raise ValueError("numerical padding must be strictly positive")
    lambda_floor = max(mode.contraction_floor for mode in modes.values())
    lambda_fraction = float(config["lambda_slack_fraction"])
    lambda_c = lambda_floor + lambda_fraction * (1.0 - lambda_floor)
    if not lambda_floor < lambda_c < 1.0:
        raise ValueError("invalid lambda_c slack rule")

    pairwise_ratios: dict[tuple[str, str], float] = {}
    for source_name, target_name in itertools.product(modes, modes):
        pairwise_ratios[(source_name, target_name)] = max_generalized_eigenvalue(
            modes[target_name].p, modes[source_name].p
        )
    mu_floor = max(1.0, max(pairwise_ratios.values()))
    mu = mu_floor * (1.0 + float(config["mu_slack_fraction"]))

    mode_rows: list[dict[str, Any]] = []
    c_w_values: dict[str, float] = {}
    mode_block_margins: dict[str, float] = {}
    for mode_name, mode in modes.items():
        c_w_mode, block_margin = _within_mode_cw(
            mode,
            lambda_c,
            padding_absolute=padding_absolute,
            padding_relative=padding_relative,
        )
        c_w_values[mode_name] = c_w_mode
        mode_block_margins[mode_name] = block_margin
        mode_rows.append(
            {
                "application": config["application"],
                "mode": mode_name,
                "state_dimension": mode.a_cl.shape[0],
                "input_dimension": mode.e_w.shape[1],
                "spectral_radius": mode.spectral_radius,
                "contraction_floor": mode.contraction_floor,
                "lambda_c": lambda_c,
                "c_w_mode": c_w_mode,
                "within_mode_block_min_eigenvalue": block_margin,
                "p_min_eigenvalue": min_eigenvalue(mode.p),
                "p_max_eigenvalue": max_eigenvalue(mode.p),
                "feedback_spectral_norm": float(np.linalg.norm(mode.k, ord=2)),
                "fallback_policy": config["fallback_policy"],
            }
        )
    c_w = max(c_w_values.values())

    jump_rows: list[dict[str, Any]] = []
    c_jump_values: dict[tuple[str, str], float] = {}
    jump_block_margins: dict[tuple[str, str], float] = {}
    for source_name, target_name in itertools.product(modes, modes):
        c_jump_pair, block_margin = _jump_cj(
            modes[source_name],
            modes[target_name],
            mu,
            padding_absolute=padding_absolute,
            padding_relative=padding_relative,
        )
        c_jump_values[(source_name, target_name)] = c_jump_pair
        jump_block_margins[(source_name, target_name)] = block_margin
        jump_rows.append(
            {
                "application": config["application"],
                "source_mode": source_name,
                "target_mode": target_name,
                "comparison_floor": pairwise_ratios[(source_name, target_name)],
                "mu": mu,
                "c_jump_pair": c_jump_pair,
                "jump_block_min_eigenvalue": block_margin,
            }
        )
    c_jump = max(c_jump_values.values())

    v_lower = min(min_eigenvalue(mode.p) for mode in modes.values())
    v_upper = max(max_eigenvalue(mode.p) for mode in modes.values())
    horizon = int(config["horizon"])
    rho_h = mu * lambda_c**horizon
    minimum_horizon = 1 + math.floor(math.log(mu) / math.log(1.0 / lambda_c))
    if not horizon >= minimum_horizon or not rho_h < 1.0:
        raise ValueError("configured horizon does not satisfy the configured dwell condition")

    w_bound = float(config["disturbance_bound"])
    jump_bound = float(config["jump_bound"])
    initial_error_bound = float(config["initial_error_bound"])
    d_bar = (
        mu * c_w * (1.0 - lambda_c**horizon) / (1.0 - lambda_c) * w_bound**2
        + c_jump * jump_bound**2
    )
    radii = phase_radii(
        v_lower=v_lower,
        lambda_c=lambda_c,
        mu=mu,
        c_w=c_w,
        c_jump=c_jump,
        horizon=horizon,
        w_bound=w_bound,
        jump_bound=jump_bound,
    )
    all_step_radius = max(radii)
    epoch_start_v = max(v_upper * initial_error_bound**2, d_bar / (1.0 - rho_h))
    transient_radii = [
        math.sqrt(
            (
                lambda_c**tau * epoch_start_v
                + c_w * (1.0 - lambda_c**tau) / (1.0 - lambda_c) * w_bound**2
            )
            / v_lower
        )
        for tau in range(horizon + 1)
    ]
    tracking_radius_bound = max(transient_radii)
    g_w = math.sqrt(
        c_w
        / (v_lower * (1.0 - lambda_c))
        * (1.0 + mu * (1.0 - lambda_c**horizon) / (1.0 - rho_h))
    )
    g_jump = math.sqrt(c_jump / (v_lower * (1.0 - rho_h)))
    max_k_norm = max(float(np.linalg.norm(mode.k, ord=2)) for mode in modes.values())
    control_bound = max_k_norm * tracking_radius_bound
    control_limit = float(config["normalized_control_limit"])
    validity_radius = float(config["normalized_validity_radius"])
    if not control_bound < control_limit:
        raise ValueError("tracking bound does not exclude actuator saturation")
    if not tracking_radius_bound < validity_radius:
        raise ValueError("tracking bound exceeds the normalized model-validity radius")

    cert = TrackingBounds(
        application=str(config["application"]),
        modes=modes,
        horizon=horizon,
        v_lower=v_lower,
        v_upper=v_upper,
        lambda_floor=lambda_floor,
        lambda_c=lambda_c,
        mu_floor=mu_floor,
        mu=mu,
        c_w=c_w,
        c_jump=c_jump,
        rho_h=rho_h,
        minimum_horizon=minimum_horizon,
        w_bound=w_bound,
        jump_bound=jump_bound,
        initial_error_bound=initial_error_bound,
        bounded_epoch_input=d_bar,
        all_step_radius=all_step_radius,
        tracking_radius_bound=tracking_radius_bound,
        g_w=g_w,
        g_jump=g_jump,
        control_bound=control_bound,
        control_limit=control_limit,
        validity_radius=validity_radius,
        minimum_within_mode_block_margin=min(mode_block_margins.values()),
        minimum_jump_block_margin=min(jump_block_margins.values()),
        numerical_padding_absolute=padding_absolute,
        numerical_padding_relative=padding_relative,
    )
    return cert, mode_rows, jump_rows


def random_vector_in_ball(rng: np.random.Generator, dimension: int, radius: float) -> np.ndarray:
    if radius <= 0.0:
        return np.zeros(dimension)
    direction = rng.normal(size=dimension)
    norm = float(np.linalg.norm(direction))
    if norm <= np.finfo(float).eps:
        direction[0] = 1.0
        norm = 1.0
    return direction / norm * radius * float(rng.random()) ** (1.0 / dimension)


def simulate_tracking_library(
    bounds: TrackingBounds,
    *,
    seed: int,
    epochs: int,
    fallback_probability: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Simulate the actual linear modes and evaluate the configured trajectory checks."""
    rng = np.random.default_rng(seed)
    mode_names = list(bounds.modes)
    state_dim = next(iter(bounds.modes.values())).a_cl.shape[0]
    input_dim = next(iter(bounds.modes.values())).e_w.shape[1]
    initial_mode = mode_names[int(rng.integers(len(mode_names)))]
    state = random_vector_in_ball(rng, state_dim, bounds.initial_error_bound)
    current_mode = initial_mode
    initial_v = float(state @ bounds.modes[current_mode].p @ state)
    epoch_start_bound = initial_v
    rows: list[dict[str, Any]] = []
    minimum_mode_slack = math.inf
    minimum_jump_slack = math.inf
    minimum_envelope_slack = math.inf
    fallback_episodes = 0

    for epoch in range(1, epochs + 1):
        mode = bounds.modes[current_mode]
        fallback_step: int | None = None
        if float(rng.random()) < fallback_probability:
            fallback_step = int(rng.integers(0, bounds.horizon))
            fallback_episodes += 1
        start_v = float(state @ mode.p @ state)
        if start_v > epoch_start_bound + 5.0e-9:
            raise AssertionError("epoch-start convolution bound violated")
        within_bound = start_v
        for tau in range(bounds.horizon + 1):
            v_now = float(state @ mode.p @ state)
            error_norm = float(np.linalg.norm(state))
            finite_bound = (
                bounds.lambda_c**tau * epoch_start_bound
                + bounds.c_w
                * (1.0 - bounds.lambda_c**tau)
                / (1.0 - bounds.lambda_c)
                * bounds.w_bound**2
            )
            envelope_slack = finite_bound - v_now
            minimum_envelope_slack = min(minimum_envelope_slack, envelope_slack)
            if envelope_slack < -5.0e-9:
                raise AssertionError("finite-time envelope violated")
            rows.append(
                {
                    "application": bounds.application,
                    "epoch": epoch,
                    "tau": tau,
                    "mode": current_mode,
                    "fallback_active": int(fallback_step is not None and tau >= fallback_step),
                    "error_norm": error_norm,
                    "lyapunov_value": v_now,
                    "finite_time_bound": finite_bound,
                    "envelope_slack": envelope_slack,
                }
            )
            if tau == bounds.horizon:
                break
            disturbance = random_vector_in_ball(rng, input_dim, bounds.w_bound)
            next_state = mode.a_cl @ state + mode.e_w @ disturbance
            next_v = float(next_state @ mode.p @ next_state)
            mode_rhs = bounds.lambda_c * v_now + bounds.c_w * float(disturbance @ disturbance)
            mode_slack = mode_rhs - next_v
            minimum_mode_slack = min(minimum_mode_slack, mode_slack)
            if mode_slack < -5.0e-9:
                raise AssertionError("within-mode inequality violated")
            state = next_state
            within_bound = bounds.lambda_c * within_bound + bounds.c_w * bounds.w_bound**2

        if epoch == epochs:
            break
        source_mode = mode
        next_mode_name = mode_names[int(rng.integers(len(mode_names)))]
        target_mode = bounds.modes[next_mode_name]
        jump = random_vector_in_ball(rng, state_dim, bounds.jump_bound)
        source_v = float(state @ source_mode.p @ state)
        state = state + jump
        target_v = float(state @ target_mode.p @ state)
        jump_rhs = bounds.mu * source_v + bounds.c_jump * float(jump @ jump)
        jump_slack = jump_rhs - target_v
        minimum_jump_slack = min(minimum_jump_slack, jump_slack)
        if jump_slack < -5.0e-9:
            raise AssertionError("cross-mode jump inequality violated")
        epoch_input = (
            bounds.mu
            * bounds.c_w
            * (1.0 - bounds.lambda_c**bounds.horizon)
            / (1.0 - bounds.lambda_c)
            * bounds.w_bound**2
            + bounds.c_jump * bounds.jump_bound**2
        )
        epoch_start_bound = bounds.rho_h * epoch_start_bound + epoch_input
        current_mode = next_mode_name

    summary = {
        "status": "PASS",
        "application": bounds.application,
        "seed": seed,
        "epochs": epochs,
        "state_checks": len(rows),
        "within_mode_checks": epochs * bounds.horizon,
        "jump_checks": max(0, epochs - 1),
        "fallback_episodes": fallback_episodes,
        "minimum_mode_slack": minimum_mode_slack,
        "minimum_jump_slack": minimum_jump_slack,
        "minimum_envelope_slack": minimum_envelope_slack,
        "maximum_observed_error_norm": max(float(row["error_norm"]) for row in rows),
        "tracking_radius_bound": bounds.tracking_radius_bound,
    }
    return rows, summary


def tracking_bounds_to_dict(bounds: TrackingBounds, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": "PASS",
        "application": bounds.application,
        "normalization": config["normalization"],
        "sample_time": config["sample_time"],
        "mode_count": len(bounds.modes),
        "mode_names": list(bounds.modes),
        "horizon": bounds.horizon,
        "bounds": {
            "v_lower": bounds.v_lower,
            "v_upper": bounds.v_upper,
            "lambda_floor": bounds.lambda_floor,
            "lambda_c": bounds.lambda_c,
            "mu_floor": bounds.mu_floor,
            "mu": bounds.mu,
            "c_w": bounds.c_w,
            "c_jump": bounds.c_jump,
            "rho_h": bounds.rho_h,
            "minimum_horizon": bounds.minimum_horizon,
            "g_w": bounds.g_w,
            "g_jump": bounds.g_jump,
            "bounded_epoch_input": bounds.bounded_epoch_input,
            "all_step_radius": bounds.all_step_radius,
            "tracking_radius_bound": bounds.tracking_radius_bound,
            "minimum_within_mode_block_margin": bounds.minimum_within_mode_block_margin,
            "minimum_jump_block_margin": bounds.minimum_jump_block_margin,
            "numerical_padding_absolute": bounds.numerical_padding_absolute,
            "numerical_padding_relative": bounds.numerical_padding_relative,
        },
        "bounds_and_gates": {
            "disturbance_bound": bounds.w_bound,
            "jump_bound": bounds.jump_bound,
            "initial_error_bound": bounds.initial_error_bound,
            "normalized_control_bound": bounds.control_bound,
            "normalized_control_limit": bounds.control_limit,
            "normalized_validity_radius": bounds.validity_radius,
            "actuator_saturation_excluded": bounds.control_bound < bounds.control_limit,
            "model_validity_tube_pass": bounds.tracking_radius_bound < bounds.validity_radius,
        },
        "fallback_semantics": config["fallback_policy"],
        "joint_mode_lift": (
            "Each assignment mode is a product of configured local templates. Under the "
            "block-maximum norm and max-composed Lyapunov function, the local generalized-"
            "eigenvalue and Schur-complement inequalities lift to every multi-agent "
            "assignment without enumerating the assignment family."
        ),
    }


def dwell_sweep(bounds: TrackingBounds, max_horizon: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for horizon in range(1, max_horizon + 1):
        rho_h = bounds.mu * bounds.lambda_c**horizon
        rows.append(
            {
                "application": bounds.application,
                "horizon": horizon,
                "rho_h": rho_h,
                "strictly_stable": int(rho_h < 1.0),
                "minimum_horizon": bounds.minimum_horizon,
            }
        )
    return rows
