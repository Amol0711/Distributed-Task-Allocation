"""Trajectory-generated allocation with bounded disturbances and shared resets.

The simulator uses a capped contextual feature map, terminal matchings, four
controller modes, and one two-dimensional physical block. Owner-partitioned
maximum agreement runs on a three-agent path graph. Configured-scale decisions
are executed physically; zero-scale decisions are fixed-estimator diagnostics.
The public exploration codebook and all random streams are specified in the
configuration and seed registries. Independent audits check geometry, physical
bounds, distributed agreement, and deterministic execution records.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
from typing import Mapping, Sequence

# The target execution is byte-replayed in validation.  Fix the BLAS thread
# count before importing NumPy so small reduction-order differences cannot alter
# serialized floating-point fields across machines or shell environments.
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np

from campaign_engine import beta_radius, constrained_quadratic_minimizer


LOCKED_EPISODE_RECORDS_SHA256 = (
    "fd953e541ae54c428096ffbbfd6a71d32bcaf724ee93459418cc0a0394be7b16"
)
LOCKED_TRIAL_SUMMARY_SHA256 = (
    "6188e4a675d89c3f30c6df28a20f9313689ad29398d97951c420177cadfdb8df"
)
LOCKED_POLICY_FINGERPRINT = (
    "4ef980f72c051e887ac85783f350c7255cf73e7b287080b8075c4bae9c2c2622"
)
LOCKED_SELECTION_FINGERPRINT = (
    "3aea33fe57fb2858bf797bc8a3d2b7749f4c1c3ff6d9b40e6d120cf40f14c55c"
)
LOCKED_PRIMARY_EVIDENCE_FINGERPRINT = (
    "81672022ca28a206e5128d8660a29db717d563f658e529e0ce4e473f118fb3d0"
)

# Exact coordinate bounds of the predeclared edge-feature map.  They provide
# an analytic model-closure proof independent of the finite 240-episode audit.
EDGE_FEATURE_LOWER = 0.12
EDGE_FEATURE_SPAN = 0.16
EDGE_FEATURE_UPPER = EDGE_FEATURE_LOWER + EDGE_FEATURE_SPAN


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_column_hash(rows: Sequence[dict], columns: Sequence[str]) -> str:
    """Hash selected fields in the frozen row order used by the evidence lock."""
    digest = hashlib.sha256()
    for row in rows:
        digest.update(
            ("\x1f".join(str(row[column]) for column in columns) + "\n").encode()
        )
    return digest.hexdigest()


def load_config(path: Path) -> dict:
    cfg = json.loads(path.read_text(encoding="utf-8"))
    cfg["theta_star"] = np.asarray(cfg["theta_star"], dtype=float)
    cfg["trajectory_return_vector"] = np.asarray(cfg["trajectory_return_vector"], dtype=float)
    cfg["initial_tracking_error"] = np.asarray(cfg["initial_tracking_error"], dtype=float)
    cfg["controller_templates"] = tuple(
        np.asarray(a, dtype=float) for a in cfg["controller_templates"]
    )
    raw_e = cfg.get("disturbance_input_templates")
    if raw_e is None:
        # Load configurations with identity disturbance-input templates.
        n = cfg["controller_templates"][0].shape[0]
        raw_e = [np.eye(n).tolist() for _ in cfg["controller_templates"]]
    cfg["disturbance_input_templates"] = tuple(
        np.asarray(e, dtype=float) for e in raw_e
    )
    if len(cfg["disturbance_input_templates"]) != len(cfg["controller_templates"]):
        raise ValueError("controller and disturbance-input template counts differ")
    cfg.setdefault("feature_saturation_cap", 1.0)
    cfg.setdefault("terminal_assignment_cardinality", cfg["agents"])
    cfg.setdefault("fallback_value", 0.0)
    cfg.setdefault("fallback_mode", 0)
    cfg.setdefault(
        "fallback_reference",
        [0.0] * cfg["controller_templates"][0].shape[0],
    )
    cfg["fallback_reference"] = np.asarray(
        cfg["fallback_reference"], dtype=float
    )
    cfg.setdefault("physical_blocks", 1)
    cfg.setdefault(
        "physical_state_dimension", cfg["controller_templates"][0].shape[0]
    )
    cfg.setdefault(
        "allocation_network",
        {
            "nodes": cfg["agents"],
            "undirected_edges": [[i, i + 1] for i in range(cfg["agents"] - 1)],
            "diameter": max(0, cfg["agents"] - 1),
            "diameter_upper_bound": max(0, cfg["agents"] - 1),
            "rounds_per_stage": max(0, cfg["agents"] - 1),
            "feedback_source_agent": 0,
            "ownership_rule": "edge_owned_by_first_coordinate",
            "identifier_order": "owner_major_task_minor",
            "score_encoding": "identity",
            "encoding_error": 0.0,
            "required_message_tags": [
                "episode",
                "stage",
                "partial_set",
                "score_scale",
            ],
        },
    )
    cfg.setdefault(
        "exploration_codebook",
        {
            "family": "lexicographic_terminal_matchings",
            "size": math.perm(cfg["tasks"], cfg["agents"]),
            "schedule_multiplier": 7,
            "schedule_offset": 3,
            "schedule_index_base": 0,
            "public_assignment_dispatch": True,
        },
    )
    cfg.setdefault(
        "resource_screen",
        {
            "mode": "vacuous_owner_supported_zero_cost",
            "resource_dimension_per_owner": 1,
            "initial_balance_per_owner": 1.0,
            "robust_edge_cost": 0.0,
            "episode_replenishment_per_owner": 0.0,
        },
    )

    if cfg["agents"] <= 0 or cfg["tasks"] < cfg["agents"]:
        raise ValueError("the microcase requires tasks >= agents > 0")
    if cfg["feature_dimension"] <= 0:
        raise ValueError("feature_dimension must be positive")
    if cfg["theta_star"].shape != (cfg["feature_dimension"],):
        raise ValueError("theta_star does not match feature_dimension")
    # The structured model uses coordinate maps into [0,1].  The locked
    # microcase therefore fixes the cap at one rather than accepting an
    # arbitrary positive rescaling that would change the model class.
    if float(cfg["feature_saturation_cap"]) != 1.0:
        raise ValueError("the locked saturating basis requires cap exactly one")
    if cfg["terminal_assignment_cardinality"] != cfg["agents"]:
        raise ValueError("the microcase terminal cardinality must equal agents")
    if cfg["fallback_value"] != 0.0:
        raise ValueError("the absorbing fallback must have zero benefit")
    if not isinstance(cfg["fallback_mode"], int) or not (
        0 <= cfg["fallback_mode"] < len(cfg["controller_templates"])
    ):
        raise ValueError("fallback_mode must index the controller library")
    if cfg["physical_blocks"] != 1:
        raise ValueError("the locked microcase has exactly one physical block")

    network = cfg["allocation_network"]
    if int(network.get("nodes", -1)) != cfg["agents"]:
        raise ValueError("allocation network node count must equal agents")
    network["undirected_edges"] = tuple(
        tuple(int(node) for node in edge) for edge in network["undirected_edges"]
    )
    if any(len(edge) != 2 for edge in network["undirected_edges"]):
        raise ValueError("allocation network edges must contain two endpoints")
    if any(
        u == v or not (0 <= u < cfg["agents"] and 0 <= v < cfg["agents"])
        for u, v in network["undirected_edges"]
    ):
        raise ValueError("allocation network edge endpoints are invalid")
    if len({tuple(sorted(edge)) for edge in network["undirected_edges"]}) != len(
        network["undirected_edges"]
    ):
        raise ValueError("allocation network contains duplicate undirected edges")
    if network.get("ownership_rule") != "edge_owned_by_first_coordinate":
        raise ValueError("the locked ownership rule is the edge first coordinate")
    if network.get("identifier_order") != "owner_major_task_minor":
        raise ValueError("the locked identifier order is owner-major/task-minor")
    if network.get("score_encoding") != "identity":
        raise ValueError("the locked score encoder is identity")
    if float(network.get("encoding_error", math.inf)) != 0.0:
        raise ValueError("the identity score encoder must have zero error")
    required_tags = tuple(network.get("required_message_tags", ()))
    if required_tags != ("episode", "stage", "partial_set", "score_scale"):
        raise ValueError("the locked proposal tag fields are incomplete")
    diameter = allocation_graph_diameter(cfg)
    if int(network.get("diameter", -1)) != diameter:
        raise ValueError("declared allocation-network diameter is incorrect")
    if int(network.get("diameter_upper_bound", -1)) < diameter:
        raise ValueError("diameter upper bound is smaller than the true diameter")
    if int(network.get("rounds_per_stage", -1)) != int(
        network["diameter_upper_bound"]
    ):
        raise ValueError("consensus rounds must equal the diameter upper bound")
    if not (0 <= int(network.get("feedback_source_agent", -1)) < cfg["agents"]):
        raise ValueError("feedback source agent is invalid")

    codebook = cfg["exploration_codebook"]
    if codebook.get("family") != "lexicographic_terminal_matchings":
        raise ValueError("the public codebook must be the terminal matching family")
    if int(codebook.get("size", -1)) != math.perm(cfg["tasks"], cfg["agents"]):
        raise ValueError("the public codebook size is incorrect")
    if int(codebook.get("schedule_index_base", -1)) != 0:
        raise ValueError("the locked public codebook uses zero-based indices")
    if not bool(codebook.get("public_assignment_dispatch", False)):
        raise ValueError("the exploration prefix must use public assignment dispatch")
    multiplier = int(codebook.get("schedule_multiplier", 0))
    if math.gcd(multiplier, int(codebook["size"])) != 1:
        raise ValueError("the exploration schedule multiplier must be coprime to codebook size")

    resource = cfg["resource_screen"]
    if resource.get("mode") != "vacuous_owner_supported_zero_cost":
        raise ValueError("the target microcase uses the vacuous zero-cost resource screen")
    if int(resource.get("resource_dimension_per_owner", 0)) != 1:
        raise ValueError("resource dimension per owner must equal one")
    if float(resource.get("initial_balance_per_owner", -1.0)) < 0.0:
        raise ValueError("initial owner resource balance must be nonnegative")
    if float(resource.get("robust_edge_cost", math.inf)) != 0.0:
        raise ValueError("the vacuous target resource screen requires zero edge cost")
    if float(resource.get("episode_replenishment_per_owner", math.inf)) != 0.0:
        raise ValueError("the vacuous target resource screen requires zero replenishment")

    state_dimension = cfg["controller_templates"][0].shape[0]
    if cfg["physical_state_dimension"] != state_dimension:
        raise ValueError("physical_state_dimension does not match template size")
    if cfg["initial_tracking_error"].shape != (state_dimension,):
        raise ValueError("initial_tracking_error has incompatible dimension")
    if cfg["trajectory_return_vector"].shape != (state_dimension,):
        raise ValueError("trajectory_return_vector has incompatible dimension")
    if cfg["fallback_reference"].shape != (state_dimension,):
        raise ValueError("fallback_reference has incompatible dimension")
    if not np.all(np.isfinite(cfg["fallback_reference"])):
        raise ValueError("fallback_reference must be finite")

    disturbance_dimension = cfg["disturbance_input_templates"][0].shape[1]
    if disturbance_dimension <= 0:
        raise ValueError("disturbance-input dimension must be positive")
    for a, e in zip(
        cfg["controller_templates"], cfg["disturbance_input_templates"], strict=True
    ):
        if a.shape != (state_dimension, state_dimension):
            raise ValueError("controller templates must be square and dimension matched")
        if e.shape != (state_dimension, disturbance_dimension):
            raise ValueError("disturbance-input templates must have common dimensions")
    cfg["disturbance_dimension"] = disturbance_dimension
    return cfg


def edges(cfg: dict) -> tuple[tuple[int, int], ...]:
    return tuple((i, j) for i in range(cfg["agents"]) for j in range(cfg["tasks"]))


def edge_identifier(cfg: dict, edge: tuple[int, int]) -> int:
    """Return the public owner-major/task-minor identifier."""
    owner, task = edge
    return owner * cfg["tasks"] + task


def allocation_graph_neighbors(cfg: dict) -> dict[int, tuple[int, ...]]:
    """Return the deterministic adjacency map of the allocation network."""
    neighbors = {node: set() for node in range(cfg["agents"])}
    for u, v in cfg["allocation_network"]["undirected_edges"]:
        neighbors[u].add(v)
        neighbors[v].add(u)
    return {node: tuple(sorted(items)) for node, items in neighbors.items()}


def allocation_graph_diameter(cfg: dict) -> int:
    """Compute the exact diameter of the configured undirected graph."""
    node_count = int(cfg["allocation_network"].get("nodes", cfg["agents"]))
    if node_count <= 0:
        raise ValueError("allocation network must contain at least one node")
    neighbors: dict[int, set[int]] = {node: set() for node in range(node_count)}
    for raw_edge in cfg["allocation_network"]["undirected_edges"]:
        if len(raw_edge) != 2:
            raise ValueError("allocation network edges must contain two endpoints")
        u, v = (int(raw_edge[0]), int(raw_edge[1]))
        if not (0 <= u < node_count and 0 <= v < node_count) or u == v:
            raise ValueError("allocation network edge endpoints are invalid")
        neighbors[u].add(v)
        neighbors[v].add(u)

    diameter = 0
    for source in range(node_count):
        distances = {source: 0}
        frontier = [source]
        while frontier:
            current = frontier.pop(0)
            for neighbor in sorted(neighbors[current]):
                if neighbor in distances:
                    continue
                distances[neighbor] = distances[current] + 1
                frontier.append(neighbor)
        if len(distances) != node_count:
            raise ValueError("allocation network must be connected")
        diameter = max(diameter, max(distances.values(), default=0))
    return diameter


def resource_screened_edges(cfg: dict, k: int) -> tuple[tuple[int, int], ...]:
    """Return the episode-boundary screened ground set.

    The target microcase intentionally activates no consumable resource channel:
    each owner-supported robust edge cost is zero, the residual balance is
    nonnegative, and the screen therefore retains all fifteen raw edges at every
    episode.  Keeping this interface explicit lets the distributed contract be
    audited without introducing a new policy-dependent resource model.
    """
    if not (1 <= k <= cfg["episodes"]):
        raise ValueError("episode index is outside the configured horizon")
    resource = cfg["resource_screen"]
    if resource["mode"] != "vacuous_owner_supported_zero_cost":
        raise ValueError("unsupported resource-screen mode")
    if float(resource["robust_edge_cost"]) != 0.0:
        raise ValueError("vacuous resource screen has nonzero robust cost")
    return edges(cfg)


def public_exploration_codebook(
    cfg: dict,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return the predeclared lexicographic terminal-assignment codebook."""
    codebook = terminal_matchings(cfg)
    if len(codebook) != int(cfg["exploration_codebook"]["size"]):
        raise AssertionError("public exploration codebook size mismatch")
    return codebook


def exploration_schedule_indices(cfg: dict) -> tuple[int, ...]:
    """Return the zero-based deterministic public-code schedule."""
    spec = cfg["exploration_codebook"]
    size = int(spec["size"])
    multiplier = int(spec["schedule_multiplier"])
    offset = int(spec["schedule_offset"])
    return tuple(
        (multiplier * t + offset) % size
        for t in range(cfg["exploration_episodes"])
    )


def exploration_schedule(
    cfg: dict,
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return the public terminal assignments dispatched in the prefix."""
    codebook = public_exploration_codebook(cfg)
    return tuple(codebook[index] for index in exploration_schedule_indices(cfg))


def exploration_codebook_rows(cfg: dict) -> list[dict]:
    """Return the canonical human-readable public-codebook rows."""
    rows: list[dict] = []
    for index, matching in enumerate(public_exploration_codebook(cfg)):
        rows.append(
            {
                "zero_based_index": index,
                "one_based_index": index + 1,
                "matching_id": sid(matching),
                "edge_1": f"{matching[0][0]}-{matching[0][1]}",
                "edge_2": f"{matching[1][0]}-{matching[1][1]}",
                "edge_3": f"{matching[2][0]}-{matching[2][1]}",
            }
        )
    return rows


def exploration_schedule_rows(cfg: dict) -> list[dict]:
    """Return the canonical 18-episode public-dispatch schedule rows."""
    codebook = public_exploration_codebook(cfg)
    rows: list[dict] = []
    for episode, index in enumerate(exploration_schedule_indices(cfg), start=1):
        rows.append(
            {
                "episode": episode,
                "zero_based_index": index,
                "one_based_index": index + 1,
                "matching_id": sid(codebook[index]),
            }
        )
    return rows


def terminal_matchings(cfg: dict) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return the maximal matching family that can receive a physical mode."""
    return tuple(
        tuple((i, task_tuple[i]) for i in range(cfg["agents"]))
        for task_tuple in itertools.permutations(range(cfg["tasks"]), cfg["agents"])
    )


def hereditary_matchings(cfg: dict) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return every partial matching used by the greedy construction.

    The family includes the empty set and all cardinalities up to the number of
    agents.  In the locked three-agent/five-task instance it has 136 members;
    its 60 maximal members are exactly :func:`terminal_matchings`.
    """
    family: list[tuple[tuple[int, int], ...]] = []
    for size in range(cfg["agents"] + 1):
        for agent_subset in itertools.combinations(range(cfg["agents"]), size):
            for task_tuple in itertools.permutations(range(cfg["tasks"]), size):
                family.append(tuple(sorted(zip(agent_subset, task_tuple))))
    return tuple(family)


def matchings(cfg: dict) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Backward-compatible alias for the terminal executable family."""
    return terminal_matchings(cfg)


def is_partial_matching(cfg: dict, S: Sequence[tuple[int, int]]) -> bool:
    if len(S) > cfg["agents"]:
        return False
    owners = [i for i, _ in S]
    tasks = [j for _, j in S]
    return (
        len(set(S)) == len(S)
        and len(set(owners)) == len(owners)
        and len(set(tasks)) == len(tasks)
        and all(0 <= i < cfg["agents"] for i in owners)
        and all(0 <= j < cfg["tasks"] for j in tasks)
    )


def is_terminal_matching(cfg: dict, S: Sequence[tuple[int, int]]) -> bool:
    return (
        is_partial_matching(cfg, S)
        and len(S) == cfg["terminal_assignment_cardinality"]
    )


def context(k: int) -> np.ndarray:
    t = k - 1
    return np.asarray([
        0.5 + 0.5 * math.sin(2 * math.pi * t / 37),
        0.5 + 0.5 * math.cos(2 * math.pi * t / 53),
        0.5 + 0.5 * math.sin(2 * math.pi * t / 29 + 0.7),
    ])


def edge_feature(cfg: dict, e: tuple[int, int], k: int) -> np.ndarray:
    i, j = e
    c1, c2, c3 = context(k)
    ai, tj = (i + 1) / cfg["agents"], (j + 1) / cfg["tasks"]
    raw = np.asarray([
        1 - abs(ai - tj),
        1 - abs(tj - c1),
        1 - abs(ai - c2),
        0.5 + 0.5 * math.sin(2 * math.pi * c3 + 0.71 * i + 0.43 * j),
        0.5 + 0.5 * math.cos(2 * math.pi * c1 + 0.37 * i - 0.29 * j),
        0.35 + 0.65 * ((i + j + k) % 3) / 2,
    ])
    return EDGE_FEATURE_LOWER + EDGE_FEATURE_SPAN * np.clip(raw, 0, 1)


def raw_set_feature(cfg: dict, S: Sequence[tuple[int, int]], k: int) -> np.ndarray:
    """Return the uncapped additive coordinate load A(S,z_k)."""
    x = np.zeros(cfg["feature_dimension"])
    for e in S:
        x += edge_feature(cfg, e, k)
    return x


def set_feature(cfg: dict, S: Sequence[tuple[int, int]], k: int) -> np.ndarray:
    """Return the globally valid capped contextual basis phi(S,z_k).

    The coordinate map is ``g_m(x)=min{x, cap}`` with cap one in the locked
    configuration.  The static model audit proves that the cap is inactive on
    every member of the hereditary matching family, so this implementation is
    numerically identical to the archived additive basis on every candidate,
    selected, and comparator assignment.
    """
    return np.minimum(
        raw_set_feature(cfg, S, k), float(cfg["feature_saturation_cap"])
    )


def marginal_feature(
    cfg: dict,
    e: tuple[int, int],
    S: Sequence[tuple[int, int]],
    k: int,
) -> np.ndarray:
    """Return the capped marginal basis for adding ``e`` to partial set ``S``.

    The fast path returns the original edge feature exactly whenever the cap is
    inactive.  This preserves the locked floating-point score arithmetic while
    still implementing the correct global marginal map if a caller supplies a
    set outside the feasible matching family.
    """
    if e in S:
        raise ValueError("marginal element must not already belong to S")
    base_raw = raw_set_feature(cfg, S, k)
    edge = edge_feature(cfg, e, k)
    cap = float(cfg["feature_saturation_cap"])
    if np.all(base_raw + edge <= cap):
        return edge.copy()
    return np.minimum(base_raw + edge, cap) - np.minimum(base_raw, cap)


def value(cfg: dict, S: Sequence[tuple[int, int]], k: int) -> float:
    return float(cfg["theta_star"] @ set_feature(cfg, S, k))


def contextual_total_curvature(cfg: dict, k: int) -> float:
    """Evaluate total curvature on the complete fifteen-edge ground set."""
    ground = edges(cfg)
    full_value = value(cfg, ground, k)
    ratios: list[float] = []
    for e in ground:
        singleton = value(cfg, (e,), k)
        if singleton <= 0:
            continue
        without = tuple(u for u in ground if u != e)
        full_ground_marginal = full_value - value(cfg, without, k)
        ratios.append(full_ground_marginal / singleton)
    if not ratios:
        return 0.0
    curvature = 1.0 - min(ratios)
    return float(min(1.0, max(0.0, curvature)))


def feature_geometry_audit(cfg: dict) -> dict:
    """Certify capped-basis validity, feasible inactivity, and unit curvature."""
    hereditary = hereditary_matchings(cfg)
    terminal = terminal_matchings(cfg)
    ground = edges(cfg)
    cap = float(cfg["feature_saturation_cap"])

    maximum_feasible_coordinate = -math.inf
    maximum_feasible_saturation_residual = 0.0
    maximum_feasible_marginal_residual = 0.0
    maximum_feasible_formula_marginal_residual = 0.0
    minimum_ground_coordinate = math.inf
    maximum_ground_coordinate = -math.inf
    minimum_ground_without_edge_coordinate = math.inf
    minimum_singleton_value = math.inf
    curvatures: list[float] = []

    for k in range(1, cfg["episodes"] + 1):
        edge_features = {e: edge_feature(cfg, e, k) for e in ground}
        for S in hereditary:
            raw = np.zeros(cfg["feature_dimension"])
            for e in S:
                raw += edge_features[e]
            capped = np.minimum(raw, cap)
            maximum_feasible_coordinate = max(
                maximum_feasible_coordinate, float(raw.max(initial=0.0))
            )
            maximum_feasible_saturation_residual = max(
                maximum_feasible_saturation_residual,
                float(np.max(np.abs(capped - raw), initial=0.0)),
            )
            for e in ground:
                if e in S or not is_partial_matching(cfg, (*S, e)):
                    continue
                formula_marginal = np.minimum(raw + edge_features[e], cap) - capped
                implemented_marginal = (
                    edge_features[e]
                    if np.all(raw + edge_features[e] <= cap)
                    else formula_marginal
                )
                maximum_feasible_marginal_residual = max(
                    maximum_feasible_marginal_residual,
                    float(np.max(np.abs(implemented_marginal - edge_features[e]))),
                )
                maximum_feasible_formula_marginal_residual = max(
                    maximum_feasible_formula_marginal_residual,
                    float(np.max(np.abs(formula_marginal - edge_features[e]))),
                )

        ground_raw = np.zeros(cfg["feature_dimension"])
        for e in ground:
            ground_raw += edge_features[e]
        minimum_ground_coordinate = min(
            minimum_ground_coordinate, float(ground_raw.min())
        )
        maximum_ground_coordinate = max(
            maximum_ground_coordinate, float(ground_raw.max())
        )
        for e in ground:
            without = ground_raw - edge_features[e]
            minimum_ground_without_edge_coordinate = min(
                minimum_ground_without_edge_coordinate, float(without.min())
            )
            minimum_singleton_value = min(
                minimum_singleton_value,
                float(cfg["theta_star"] @ edge_features[e]),
            )
        curvatures.append(contextual_total_curvature(cfg, k))

    terminal_set = set(terminal)
    maximal_from_hereditary = {
        S
        for S in hereditary
        if not any(
            e not in S and is_partial_matching(cfg, (*S, e))
            for e in ground
        )
    }
    analytical_feasible_upper = (
        cfg["terminal_assignment_cardinality"] * EDGE_FEATURE_UPPER
    )
    analytical_ground_without_edge_lower = (
        (len(ground) - 1) * EDGE_FEATURE_LOWER
    )
    return {
        "edge_feature_coordinate_lower_bound": EDGE_FEATURE_LOWER,
        "edge_feature_coordinate_upper_bound": EDGE_FEATURE_UPPER,
        "analytical_feasible_raw_coordinate_upper_bound": analytical_feasible_upper,
        "analytical_ground_without_edge_coordinate_lower_bound": (
            analytical_ground_without_edge_lower
        ),
        "hereditary_assignment_count": len(hereditary),
        "hereditary_cardinality_counts": {
            str(size): sum(len(S) == size for S in hereditary)
            for size in range(cfg["agents"] + 1)
        },
        "terminal_assignment_count": len(terminal),
        "terminal_assignment_cardinality": cfg["terminal_assignment_cardinality"],
        "terminal_family_equals_maximal_family": maximal_from_hereditary
        == terminal_set,
        "maximum_feasible_raw_coordinate": maximum_feasible_coordinate,
        "feasible_saturation_cap_margin": cap - maximum_feasible_coordinate,
        "maximum_feasible_saturation_residual": maximum_feasible_saturation_residual,
        "maximum_feasible_marginal_residual": maximum_feasible_marginal_residual,
        "maximum_feasible_formula_marginal_residual": (
            maximum_feasible_formula_marginal_residual
        ),
        "minimum_ground_raw_coordinate": minimum_ground_coordinate,
        "maximum_ground_raw_coordinate": maximum_ground_coordinate,
        "minimum_ground_without_edge_raw_coordinate": (
            minimum_ground_without_edge_coordinate
        ),
        "ground_without_edge_saturation_margin": (
            minimum_ground_without_edge_coordinate - cap
        ),
        "minimum_singleton_value": minimum_singleton_value,
        "minimum_contextual_curvature": min(curvatures),
        "maximum_contextual_curvature": max(curvatures),
        "curvature_equals_one_all_episodes": all(
            value_ == 1.0 for value_ in curvatures
        ),
        "contextual_greedy_factor": 1.0
        / (cfg["q_extendibility"] + max(curvatures)),
        "fixed_greedy_factor": 1.0 / (cfg["q_extendibility"] + 1.0),
    }


def sid(S: Sequence[tuple[int, int]]) -> str:
    return ";".join(f"{i}-{j}" for i, j in sorted(S))


def parse_sid(value_: str) -> tuple[tuple[int, int], ...]:
    """Decode the canonical matching identifier used in locked CSV evidence."""
    if not value_:
        return tuple()
    return tuple(
        sorted(
            tuple(int(item) for item in token.split("-"))
            for token in value_.split(";")
        )
    )


@dataclass(frozen=True)
class TaggedProposal:
    """One score-greedy proposal carried by the synchronous agreement protocol."""

    raw_score: float
    encoded_score: float
    edge: tuple[int, int] | None
    owner: int
    identifier: int
    width: float
    episode: int
    stage: int
    partial_set_tag: str
    score_scale_tag: str


def null_proposal(
    cfg: dict,
    episode: int,
    stage: int,
    partial_set_tag: str,
    score_scale_tag: str,
) -> TaggedProposal:
    """Return the least proposal under the public total order."""
    return TaggedProposal(
        raw_score=-math.inf,
        encoded_score=-math.inf,
        edge=None,
        owner=-1,
        identifier=cfg["agents"] * cfg["tasks"],
        width=0.0,
        episode=episode,
        stage=stage,
        partial_set_tag=partial_set_tag,
        score_scale_tag=score_scale_tag,
    )


def proposal_order_key(proposal: TaggedProposal) -> tuple[int, float, int]:
    """Order proposals by encoded score and then the public identifier.

    The null proposal is strictly below every genuine proposal.  For equal
    encoded scores, the smaller owner-major/task-minor identifier wins, matching
    the archived centralized sorting rule ``(-score, identifier)`` exactly.
    """
    if proposal.edge is None:
        return (0, -math.inf, -proposal.identifier)
    return (1, proposal.encoded_score, -proposal.identifier)


def proposal_has_expected_tags(
    proposal: TaggedProposal,
    episode: int,
    stage: int,
    partial_set_tag: str,
    score_scale_tag: str,
) -> bool:
    return (
        proposal.episode == episode
        and proposal.stage == stage
        and proposal.partial_set_tag == partial_set_tag
        and proposal.score_scale_tag == score_scale_tag
    )


def encode_score(cfg: dict, score: float) -> float:
    """Apply the fixed score encoder used by the target execution."""
    if cfg["allocation_network"]["score_encoding"] != "identity":
        raise ValueError("unsupported score encoder")
    return float(score)


def candidate_score(
    cfg: dict,
    k: int,
    edge: tuple[int, int],
    partial: Sequence[tuple[int, int]],
    theta_hat: np.ndarray,
    V_inverse: np.ndarray,
    beta: float,
) -> tuple[float, float, float]:
    """Return raw score, encoded score, and confidence width for one edge."""
    phi = marginal_feature(cfg, edge, partial, k)
    width = math.sqrt(max(0.0, float(phi @ V_inverse @ phi)))
    raw_score = float(theta_hat @ phi) + beta * width
    encoded_score = encode_score(cfg, raw_score)
    return raw_score, encoded_score, width


def candidate_is_feasible(
    cfg: dict,
    k: int,
    edge: tuple[int, int],
    partial: Sequence[tuple[int, int]],
) -> bool:
    return edge in resource_screened_edges(cfg, k) and is_partial_matching(
        cfg, (*partial, edge)
    )


def best_local_proposal(
    cfg: dict,
    k: int,
    stage: int,
    owner: int,
    partial: Sequence[tuple[int, int]],
    theta_hat: np.ndarray,
    V_inverse: np.ndarray,
    beta: float,
    score_scale_tag: str,
) -> TaggedProposal:
    """Return one owner's best feasible genuine proposal or the null proposal."""
    partial_tag = sid(partial)
    candidates: list[TaggedProposal] = []
    for edge in resource_screened_edges(cfg, k):
        if edge[0] != owner or not candidate_is_feasible(cfg, k, edge, partial):
            continue
        raw_score, encoded_score, width = candidate_score(
            cfg, k, edge, partial, theta_hat, V_inverse, beta
        )
        candidates.append(
            TaggedProposal(
                raw_score=raw_score,
                encoded_score=encoded_score,
                edge=edge,
                owner=owner,
                identifier=edge_identifier(cfg, edge),
                width=width,
                episode=k,
                stage=stage,
                partial_set_tag=partial_tag,
                score_scale_tag=score_scale_tag,
            )
        )
    if not candidates:
        return null_proposal(cfg, k, stage, partial_tag, score_scale_tag)
    return max(candidates, key=proposal_order_key)


def tagged_max_consensus(
    cfg: dict,
    initial: Mapping[int, TaggedProposal],
    episode: int,
    stage: int,
    partial_set_tag: str,
    score_scale_tag: str,
) -> dict:
    """Run the exact synchronous tagged maximum recursion for ``Dbar`` rounds."""
    nodes = tuple(range(cfg["agents"]))
    if set(initial) != set(nodes):
        raise ValueError("initial proposal map must contain every allocation agent")
    neighbors = allocation_graph_neighbors(cfg)
    rounds = int(cfg["allocation_network"]["rounds_per_stage"])
    null = null_proposal(cfg, episode, stage, partial_set_tag, score_scale_tag)
    states = dict(initial)
    history: list[dict[int, TaggedProposal]] = [dict(states)]
    discarded_invalid_tags = 0
    for _ in range(rounds):
        next_states: dict[int, TaggedProposal] = {}
        for node in nodes:
            received = [states[node], *(states[neighbor] for neighbor in neighbors[node])]
            valid: list[TaggedProposal] = []
            for proposal in received:
                if proposal_has_expected_tags(
                    proposal, episode, stage, partial_set_tag, score_scale_tag
                ):
                    valid.append(proposal)
                else:
                    discarded_invalid_tags += 1
                    valid.append(null)
            next_states[node] = max(valid, key=proposal_order_key)
        states = next_states
        history.append(dict(states))
    winners = tuple(states[node] for node in nodes)
    return {
        "winner_by_agent": states,
        "history": tuple(history),
        "all_agents_agree": len(set(winners)) == 1,
        "winner": winners[0],
        "discarded_invalid_tags": discarded_invalid_tags,
        "rounds": rounds,
    }


def centralized_greedy_trace(
    cfg: dict,
    k: int,
    theta_hat: np.ndarray,
    V: np.ndarray,
    beta: float,
    score_scale_tag: str,
) -> dict:
    """Return the complete centralized stage sequence, including null termination."""
    chosen: list[tuple[int, int]] = []
    widths: list[float] = []
    winners: list[TaggedProposal] = []
    V_inverse = np.linalg.inv(V)
    for stage in range(1, cfg["agents"] + 2):
        partial_tag = sid(chosen)
        candidates: list[TaggedProposal] = []
        for edge in resource_screened_edges(cfg, k):
            if not candidate_is_feasible(cfg, k, edge, chosen):
                continue
            raw_score, encoded_score, width = candidate_score(
                cfg, k, edge, chosen, theta_hat, V_inverse, beta
            )
            candidates.append(
                TaggedProposal(
                    raw_score=raw_score,
                    encoded_score=encoded_score,
                    edge=edge,
                    owner=edge[0],
                    identifier=edge_identifier(cfg, edge),
                    width=width,
                    episode=k,
                    stage=stage,
                    partial_set_tag=partial_tag,
                    score_scale_tag=score_scale_tag,
                )
            )
        winner = (
            max(candidates, key=proposal_order_key)
            if candidates
            else null_proposal(cfg, k, stage, partial_tag, score_scale_tag)
        )
        winners.append(winner)
        if winner.edge is None:
            break
        chosen.append(winner.edge)
        widths.append(winner.width)
    if winners[-1].edge is not None:
        raise AssertionError("centralized greedy trace did not terminate with null")
    return {
        "matching": tuple(sorted(chosen)),
        "widths": tuple(widths),
        "stage_winners": tuple(winners),
        "accepted_stage_count": len(chosen),
        "null_stage": len(winners),
        "score_scale_tag": score_scale_tag,
    }


def distributed_greedy_trace(
    cfg: dict,
    k: int,
    theta_hat: np.ndarray,
    V: np.ndarray,
    beta: float,
    score_scale_tag: str,
) -> dict:
    """Run the owner-partitioned two-round tagged max-consensus allocation."""
    chosen: list[tuple[int, int]] = []
    widths: list[float] = []
    winners: list[TaggedProposal] = []
    stage_consensus: list[dict] = []
    V_inverse = np.linalg.inv(V)
    for stage in range(1, cfg["agents"] + 2):
        partial_tag = sid(chosen)
        initial = {
            owner: best_local_proposal(
                cfg,
                k,
                stage,
                owner,
                chosen,
                theta_hat,
                V_inverse,
                beta,
                score_scale_tag,
            )
            for owner in range(cfg["agents"])
        }
        consensus = tagged_max_consensus(
            cfg,
            initial,
            k,
            stage,
            partial_tag,
            score_scale_tag,
        )
        if not consensus["all_agents_agree"]:
            raise AssertionError("tagged max-consensus failed to reach common agreement")
        winner = consensus["winner"]
        winners.append(winner)
        stage_consensus.append(consensus)
        if winner.edge is None:
            break
        chosen.append(winner.edge)
        widths.append(winner.width)
    if winners[-1].edge is not None:
        raise AssertionError("distributed greedy trace did not terminate with null")
    return {
        "matching": tuple(sorted(chosen)),
        "widths": tuple(widths),
        "stage_winners": tuple(winners),
        "stage_consensus": tuple(stage_consensus),
        "accepted_stage_count": len(chosen),
        "null_stage": len(winners),
        "score_scale_tag": score_scale_tag,
    }


def trace_sequence(trace: Mapping[str, object]) -> tuple[str, ...]:
    """Serialize the accepted winners and terminal null in stage order."""
    return tuple(
        "NULL" if proposal.edge is None else f"{proposal.edge[0]}-{proposal.edge[1]}"
        for proposal in trace["stage_winners"]  # type: ignore[index]
    )


def assert_centralized_distributed_equivalence(
    cfg: dict,
    centralized: Mapping[str, object],
    distributed: Mapping[str, object],
) -> None:
    """Raise unless both realizations are identical at every tagged stage."""
    if centralized["matching"] != distributed["matching"]:
        raise AssertionError("centralized/distributed final matchings differ")
    if centralized["widths"] != distributed["widths"]:
        raise AssertionError("centralized/distributed winner widths differ")
    if trace_sequence(centralized) != trace_sequence(distributed):
        raise AssertionError("centralized/distributed stage sequences differ")
    for central, dist in zip(
        centralized["stage_winners"],  # type: ignore[index]
        distributed["stage_winners"],  # type: ignore[index]
        strict=True,
    ):
        if central.edge != dist.edge or central.identifier != dist.identifier:
            raise AssertionError("centralized/distributed stage winners differ")
        if central.raw_score != dist.raw_score or central.encoded_score != dist.encoded_score:
            raise AssertionError("centralized/distributed score arithmetic differs")
        if abs(central.encoded_score - central.raw_score) > float(
            cfg["allocation_network"]["encoding_error"]
        ):
            raise AssertionError("score encoder exceeds its declared error")


def tagged_feedback_flood(cfg: dict, value_: float, episode: int) -> dict:
    """Flood the scalar aggregate return from the predeclared source in ``Dbar`` rounds."""
    source = int(cfg["allocation_network"]["feedback_source_agent"])
    neighbors = allocation_graph_neighbors(cfg)
    rounds = int(cfg["allocation_network"]["diameter_upper_bound"])
    states: dict[int, tuple[float, int] | None] = {
        node: ((float(value_), episode) if node == source else None)
        for node in range(cfg["agents"])
    }
    history = [dict(states)]
    for _ in range(rounds):
        next_states = dict(states)
        for node in range(cfg["agents"]):
            candidates = [states[node], *(states[neighbor] for neighbor in neighbors[node])]
            valid = [item for item in candidates if item is not None and item[1] == episode]
            if valid:
                first = valid[0]
                if any(item != first for item in valid[1:]):
                    raise AssertionError("inconsistent tagged feedback values received")
                next_states[node] = first
        states = next_states
        history.append(dict(states))
    values = tuple(states[node] for node in range(cfg["agents"]))
    first_full_agreement_round = None
    for round_index, round_state in enumerate(history):
        round_values = tuple(round_state[node] for node in range(cfg["agents"]))
        if all(item is not None for item in round_values) and len(set(round_values)) == 1:
            first_full_agreement_round = round_index
            break
    return {
        "source_agent": source,
        "rounds": rounds,
        "history": tuple(history),
        "all_agents_received": all(item is not None for item in values),
        "all_agents_equal": len(set(values)) == 1,
        "first_full_agreement_round": first_full_agreement_round,
        "directed_transmissions": (
            rounds * 2 * len(cfg["allocation_network"]["undirected_edges"])
        ),
        "value": float(value_),
    }


def mode(cfg: dict, S: Sequence[tuple[int, int]]) -> int:
    if not is_terminal_matching(cfg, S):
        raise ValueError("physical modes are defined only for terminal matchings")
    return sum((i + 1) * (j + 1) for i, j in S) % len(cfg["controller_templates"])


def reference(S: Sequence[tuple[int, int]], cfg: dict) -> np.ndarray:
    if not is_terminal_matching(cfg, S):
        raise ValueError("physical references are defined only for terminal matchings")
    ts = np.asarray([j for _, j in S], dtype=float)
    ps = np.asarray([(i + 1) * (j + 1) for i, j in S], dtype=float)
    return np.asarray([0.008 * (ts.mean() - 2), 0.003 * (ps.mean() - 6)])


def fallback_mode(cfg: dict) -> int:
    """Return the predeclared certified mode used by an empty terminal output."""
    return int(cfg["fallback_mode"])


def fallback_reference(cfg: dict) -> np.ndarray:
    """Return a defensive copy of the predeclared absorbing reference."""
    return np.asarray(cfg["fallback_reference"], dtype=float).copy()


def execution_mode(cfg: dict, S: Sequence[tuple[int, int]]) -> int:
    """Resolve a terminal allocation to its ordinary or fallback mode.

    Nonempty construction-only sets are deliberately rejected.  The empty set
    is accepted only when it is the terminal output and is then routed to the
    separately configured fallback; in the locked matching family the empty
    set is not maximal, so this branch is an explicit contract interface rather
    than an observed target-trace event.
    """
    if len(S) == 0:
        return fallback_mode(cfg)
    return mode(cfg, S)


def execution_reference(cfg: dict, S: Sequence[tuple[int, int]]) -> np.ndarray:
    """Resolve a terminal allocation to its ordinary or fallback reference."""
    if len(S) == 0:
        return fallback_reference(cfg)
    return reference(S, cfg)


def shared_reset_deviation_radii(
    controller_templates: Sequence[np.ndarray],
    disturbance_input_templates: Sequence[np.ndarray],
    disturbance_bound: float,
    horizon: int,
) -> tuple[float, float, tuple[float, ...]]:
    """Return the uniform shared-reset radius recursion.

    For d_{tau+1}=A_a d_tau+E_a w_tau, d_0=0, and ||w_tau||_2 <= wbar,
    the exported radius is R_{tau+1}=a_max R_tau+e_max wbar.  The maxima
    are taken over the complete predeclared template libraries, so the result
    is valid before the assignment sequence is known.
    """
    if horizon < 0:
        raise ValueError("horizon must be nonnegative")
    if disturbance_bound < 0:
        raise ValueError("disturbance_bound must be nonnegative")
    if not controller_templates or not disturbance_input_templates:
        raise ValueError("template libraries must be nonempty")
    if len(controller_templates) != len(disturbance_input_templates):
        raise ValueError("controller and disturbance-input template counts differ")

    a_max = max(float(np.linalg.norm(a, 2)) for a in controller_templates)
    e_max = max(float(np.linalg.norm(e, 2)) for e in disturbance_input_templates)
    radii = [0.0]
    for _ in range(horizon):
        radii.append(a_max * radii[-1] + e_max * disturbance_bound)
    return a_max, e_max, tuple(radii)


def shared_reset_deviation_scale(
    phase_radii: Sequence[float],
    discount_factor: float,
    running_lipschitz: float,
    terminal_lipschitz: float,
    trajectory_comparison: float = 1.0,
) -> float:
    """Evaluate the general shared-reset deviation branch scale.

    ``phase_radii`` contains R_0,...,R_H.  The function implements
    c_dev[sum_{tau=0}^{H-1} gamma^tau L_r R_tau + gamma^H L_T R_H].
    """
    if not phase_radii:
        raise ValueError("phase_radii must contain at least R_0")
    if not (0 < discount_factor <= 1):
        raise ValueError("discount_factor must lie in (0,1]")
    if min(running_lipschitz, terminal_lipschitz, trajectory_comparison) < 0:
        raise ValueError("Lipschitz and comparison constants must be nonnegative")
    horizon = len(phase_radii) - 1
    running = sum(
        discount_factor**tau * running_lipschitz * phase_radii[tau]
        for tau in range(horizon)
    )
    terminal = discount_factor**horizon * terminal_lipschitz * phase_radii[horizon]
    return float(trajectory_comparison * (running + terminal))


def trajectory_linear_deviation_scale(
    phase_radii: Sequence[float],
    discount_factor: float,
    return_vector: np.ndarray,
) -> float:
    """Evaluate the locked linear-deviation return scale."""
    if not phase_radii:
        raise ValueError("phase_radii must contain at least R_0")
    return float(
        np.linalg.norm(return_vector, 2)
        * sum(
            discount_factor ** (tau - 1) * phase_radii[tau]
            for tau in range(1, len(phase_radii))
        )
    )


def physical_certificate(cfg: dict, family: Sequence[Sequence[tuple[int, int]]]) -> dict:
    mats = cfg["controller_templates"]
    emats = cfg["disturbance_input_templates"]
    amax, emax, r_dev = shared_reset_deviation_radii(
        mats,
        emats,
        cfg["disturbance_bound"],
        cfg["physical_horizon"],
    )
    lamc = max(float(np.linalg.norm(a, 2) ** 2) for a in mats) + cfg["lambda_c_margin"]
    if not (amax < 1 and amax * amax < lamc < 1):
        raise AssertionError("invalid contraction configuration")
    state_dimension = cfg["physical_state_dimension"]
    disturbance_dimension = cfg["disturbance_dimension"]
    cw = max(
        float(
            np.linalg.eigvalsh(
                e.T @ e
                + e.T
                @ a
                @ np.linalg.inv(lamc * np.eye(state_dimension) - a.T @ a)
                @ a.T
                @ e
            ).max()
        )
        for a, e in zip(mats, emats, strict=True)
    )
    mu = cfg["mu"]
    cj = 1 + 1 / (mu - 1)
    if not all(is_terminal_matching(cfg, S) for S in family):
        raise ValueError("physical certificate family must be terminal executable")
    # The certificate is uniform over ordinary terminal references and the
    # separately declared fallback reference.  In the locked configuration the
    # fallback lies inside the ordinary reference envelope, so including it
    # leaves the historical jump bound and all locked trajectory evidence
    # unchanged.
    refs = [reference(S, cfg) for S in family]
    refs_with_fallback = [*refs, fallback_reference(cfg)]
    jbar = max(
        float(np.linalg.norm(a - b))
        for a in refs_with_fallback
        for b in refs_with_fallback
    )
    H, wbar = cfg["physical_horizon"], cfg["disturbance_bound"]
    rho = mu * lamc**H
    dbar = mu * cw * (1 - lamc**H) / (1 - lamc) * wbar**2 + cj * jbar**2
    v0 = float(cfg["initial_tracking_error"] @ cfg["initial_tracking_error"])
    vbar = max(v0, dbar / (1 - rho))
    r_all_tau = [
        math.sqrt(
            lamc**t * vbar
            + cw * (1 - lamc**t) / (1 - lamc) * wbar**2
        )
        for t in range(H + 1)
    ]

    bnorm = float(np.linalg.norm(cfg["trajectory_return_vector"], 2))
    legacy_sigma = trajectory_linear_deviation_scale(
        r_dev, cfg["discount_factor"], cfg["trajectory_return_vector"]
    )
    general_sigma = shared_reset_deviation_scale(
        r_dev,
        cfg["discount_factor"],
        running_lipschitz=bnorm / cfg["discount_factor"],
        terminal_lipschitz=bnorm / cfg["discount_factor"],
        trajectory_comparison=1.0,
    )
    identity_residual = abs(general_sigma - legacy_sigma)
    if identity_residual > 5e-15:
        raise AssertionError("general and trajectory deviation scales do not agree")
    # Use the locked trajectory value as the canonical floating-point
    # representation after proving the algebraically identical general formula.
    general_sigma = legacy_sigma

    rall = max(r_all_tau)
    return {
        "a_max": amax,
        "disturbance_input_norm_max": emax,
        "lambda_c": lamc,
        "mu": mu,
        "c_w": cw,
        "c_J": cj,
        "reference_jump_bound": jbar,
        "fallback_mode": fallback_mode(cfg),
        "fallback_reference": fallback_reference(cfg).tolist(),
        "fallback_included_in_reference_library": True,
        "rho_H": rho,
        "disturbance_block_bound": dbar,
        "V_bar": vbar,
        "all_time_radius": rall,
        "deviation_phase_radii": list(r_dev),
        # Legacy names are retained so archived evidence readers remain valid.
        "fluctuation_phase_radii": list(r_dev),
        "shared_reset_deviation_scale": general_sigma,
        "trajectory_sigma": legacy_sigma,
        "deviation_scale_identity_residual": identity_residual,
        "actuator_limit": cfg["actuator_gain"] * rall * cfg["actuator_margin_factor"],
        "validity_radius": rall * cfg["validity_margin_factor"],
    }


def greedy(
    cfg: dict,
    k: int,
    theta_hat: np.ndarray,
    V: np.ndarray,
    beta: float,
) -> tuple[tuple[tuple[int, int], ...], tuple[float, ...]]:
    trace = centralized_greedy_trace(
        cfg,
        k,
        theta_hat,
        V,
        beta,
        score_scale_tag="configured",
    )
    return trace["matching"], trace["widths"]


def exact_optimum(cfg: dict, family: Sequence[Sequence[tuple[int, int]]], k: int):
    vals = [value(cfg, S, k) for S in family]
    ix = int(np.argmax(vals))
    return tuple(family[ix]), float(vals[ix])


def mean_se(vals: Sequence[float]) -> tuple[float, float]:
    a = np.asarray(vals, dtype=float)
    return float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else 0.0


def static_audit(cfg: dict) -> dict:
    family = terminal_matchings(cfg)
    geometry = feature_geometry_audit(cfg)
    cert = physical_certificate(cfg, family)
    codebook = public_exploration_codebook(cfg)
    schedule_indices = exploration_schedule_indices(cfg)
    schedule = exploration_schedule(cfg)
    graph_diameter = allocation_graph_diameter(cfg)
    graph_neighbors = allocation_graph_neighbors(cfg)
    assert len(family) == math.perm(cfg["tasks"], cfg["agents"]) == 60
    assert geometry["hereditary_assignment_count"] == 136
    assert geometry["terminal_assignment_count"] == 60
    assert geometry["terminal_family_equals_maximal_family"]
    assert geometry["edge_feature_coordinate_lower_bound"] == EDGE_FEATURE_LOWER
    assert geometry["edge_feature_coordinate_upper_bound"] == EDGE_FEATURE_UPPER
    assert geometry["analytical_feasible_raw_coordinate_upper_bound"] < cfg[
        "feature_saturation_cap"
    ]
    assert geometry["analytical_ground_without_edge_coordinate_lower_bound"] > cfg[
        "feature_saturation_cap"
    ]
    assert geometry["maximum_feasible_raw_coordinate"] < cfg["feature_saturation_cap"]
    assert geometry["maximum_feasible_saturation_residual"] == 0.0
    assert geometry["maximum_feasible_marginal_residual"] == 0.0
    assert (
        geometry["minimum_ground_without_edge_raw_coordinate"]
        > cfg["feature_saturation_cap"]
    )
    assert geometry["minimum_singleton_value"] > 0.0
    assert geometry["curvature_equals_one_all_episodes"]
    assert geometry["minimum_contextual_curvature"] == 1.0
    assert geometry["maximum_contextual_curvature"] == 1.0
    assert geometry["contextual_greedy_factor"] == 1 / 3
    assert geometry["fixed_greedy_factor"] == 1 / 3
    assert cfg["q_extendibility"] == 2
    assert cfg["physical_blocks"] == 1
    assert cfg["physical_state_dimension"] == 2
    assert cfg["fallback_value"] == 0.0
    assert len(codebook) == 60
    assert codebook == family
    assert len(schedule_indices) == cfg["exploration_episodes"] == 18
    assert len(set(schedule_indices)) == len(schedule_indices)
    assert schedule == tuple(codebook[index] for index in schedule_indices)
    assert graph_diameter == 2
    assert int(cfg["allocation_network"]["diameter"]) == graph_diameter
    assert int(cfg["allocation_network"]["diameter_upper_bound"]) == graph_diameter
    assert int(cfg["allocation_network"]["rounds_per_stage"]) == graph_diameter
    assert graph_neighbors == {0: (1,), 1: (0, 2), 2: (1,)}
    assert cfg["allocation_network"]["score_encoding"] == "identity"
    assert float(cfg["allocation_network"]["encoding_error"]) == 0.0
    assert all(
        resource_screened_edges(cfg, k) == edges(cfg)
        for k in range(1, cfg["episodes"] + 1)
    )
    assert np.all(cfg["theta_star"] >= 0)
    assert np.linalg.norm(cfg["theta_star"]) <= cfg["btheta"] + 1e-12
    assert cfg["theta_star"].sum() <= cfg["fmax"] + 1e-12
    edge_vals, full_vals = [], []
    for k in range(1, cfg["episodes"] + 1):
        edge_vals.extend(float(cfg["theta_star"] @ edge_feature(cfg, e, k)) for e in edges(cfg))
        full_vals.extend(value(cfg, S, k) for S in family)
    assert min(edge_vals) > cert["shared_reset_deviation_scale"] / cfg["agents"]
    assert min(full_vals) > cert["shared_reset_deviation_scale"]
    assert max(full_vals) <= cfg["fmax"] + 1e-12
    reward_weight = sum(
        cfg["discount_factor"]**tau for tau in range(cfg["physical_horizon"])
    ) + cfg["discount_factor"] ** cfg["physical_horizon"]
    minimum_stage_baseline = min(full_vals) / reward_weight
    maximum_stage_deviation_bound = (
        float(np.linalg.norm(cfg["trajectory_return_vector"], 2))
        / cfg["discount_factor"]
        * max(cert["deviation_phase_radii"])
    )
    minimum_stage_reward_lower_bound = (
        minimum_stage_baseline - maximum_stage_deviation_bound
    )
    assert minimum_stage_reward_lower_bound > 0
    for a, e in zip(
        cfg["controller_templates"], cfg["disturbance_input_templates"], strict=True
    ):
        M = cert["lambda_c"] * np.eye(cfg["physical_state_dimension"]) - a.T @ a
        assert np.linalg.eigvalsh(M).min() > 0
        block = np.block([
            [M, -a.T @ e],
            [
                -e.T @ a,
                cert["c_w"] * np.eye(cfg["disturbance_dimension"]) - e.T @ e,
            ],
        ])
        assert np.linalg.eigvalsh(block).min() >= -3e-10
    identity = np.eye(cfg["physical_state_dimension"])
    rblock = np.block(
        [
            [(cfg["mu"] - 1) * identity, -identity],
            [-identity, (cert["c_J"] - 1) * identity],
        ]
    )
    assert np.linalg.eigvalsh(rblock).min() >= -3e-10
    excluded = {"fluctuation_phase_radii", "deviation_phase_radii"}
    return {
        **{k: v for k, v in cert.items() if k not in excluded},
        **geometry,
        "matching_count": len(family),
        "feature_saturation_cap": cfg["feature_saturation_cap"],
        "terminal_assignment_cardinality": cfg["terminal_assignment_cardinality"],
        "theta_l1": float(cfg["theta_star"].sum()),
        "physical_blocks": cfg["physical_blocks"],
        "physical_state_dimension": cfg["physical_state_dimension"],
        "disturbance_dimension": cfg["disturbance_dimension"],
        "joint_physical_state_dimension": (
            cfg["physical_blocks"] * cfg["physical_state_dimension"]
        ),
        "fallback_value": cfg["fallback_value"],
        "fallback_mode": fallback_mode(cfg),
        "fallback_reference": fallback_reference(cfg).tolist(),
        "initial_quadratic_storage": float(
            cfg["initial_tracking_error"] @ cfg["initial_tracking_error"]
        ),
        "storage_eigenvalue_lower": 1.0,
        "storage_eigenvalue_upper": 1.0,
        "allocation_network_nodes": cfg["agents"],
        "allocation_network_edges": [
            list(edge) for edge in cfg["allocation_network"]["undirected_edges"]
        ],
        "allocation_network_diameter": graph_diameter,
        "allocation_network_diameter_upper_bound": int(
            cfg["allocation_network"]["diameter_upper_bound"]
        ),
        "consensus_rounds_per_stage": int(
            cfg["allocation_network"]["rounds_per_stage"]
        ),
        "feedback_source_agent": int(
            cfg["allocation_network"]["feedback_source_agent"]
        ),
        "score_encoding": cfg["allocation_network"]["score_encoding"],
        "encoding_error": float(cfg["allocation_network"]["encoding_error"]),
        "resource_screen_mode": cfg["resource_screen"]["mode"],
        "resource_screen_retained_edge_count": len(edges(cfg)),
        "exploration_codebook_size": len(codebook),
        "exploration_schedule_indices_zero_based": list(schedule_indices),
        "exploration_schedule_indices_one_based": [
            index + 1 for index in schedule_indices
        ],
        "theta_l2": float(np.linalg.norm(cfg["theta_star"])),
        "minimum_edge_value": min(edge_vals),
        "minimum_full_matching_value": min(full_vals),
        "maximum_full_matching_value": max(full_vals),
        "discounted_reward_weight": reward_weight,
        "minimum_stage_baseline": minimum_stage_baseline,
        "maximum_stage_deviation_bound": maximum_stage_deviation_bound,
        "minimum_stage_reward_lower_bound": minimum_stage_reward_lower_bound,
    }


def protocol_trace_record(
    cfg: dict,
    seed: int,
    episode: int,
    branch: str,
    centralized: Mapping[str, object],
    distributed: Mapping[str, object],
) -> dict:
    """Serialize one centralized/distributed exploitation comparison."""
    assert_centralized_distributed_equivalence(cfg, centralized, distributed)
    consensus_stages = distributed["stage_consensus"]  # type: ignore[index]
    encoded_residuals = [
        abs(proposal.encoded_score - proposal.raw_score)
        for proposal in distributed["stage_winners"]  # type: ignore[index]
        if proposal.edge is not None
    ]
    stage_count = len(consensus_stages)
    rounds_per_stage = int(cfg["allocation_network"]["rounds_per_stage"])
    directed_edges_per_round = 2 * len(
        cfg["allocation_network"]["undirected_edges"]
    )
    return {
        "campaign_id": cfg["campaign_id"],
        "seed": seed,
        "episode": episode,
        "protocol_branch": branch,
        "public_code_index_zero_based": "",
        "public_code_index_one_based": "",
        "selected_matching": sid(centralized["matching"]),  # type: ignore[arg-type]
        "centralized_matching": sid(centralized["matching"]),  # type: ignore[arg-type]
        "distributed_matching": sid(distributed["matching"]),  # type: ignore[arg-type]
        "centralized_stage_sequence": "|".join(trace_sequence(centralized)),
        "distributed_stage_sequence": "|".join(trace_sequence(distributed)),
        "accepted_stage_count": int(centralized["accepted_stage_count"]),
        "null_stage": int(centralized["null_stage"]),
        "stage_count": stage_count,
        "consensus_rounds_per_stage": rounds_per_stage,
        "consensus_round_instances": stage_count * rounds_per_stage,
        "directed_proposal_transmissions": (
            stage_count * rounds_per_stage * directed_edges_per_round
        ),
        "centralized_distributed_equal": 1,
        "all_agents_agree": int(
            all(stage["all_agents_agree"] for stage in consensus_stages)
        ),
        "all_tags_valid": int(
            all(stage["discarded_invalid_tags"] == 0 for stage in consensus_stages)
        ),
        "discarded_invalid_tags": sum(
            int(stage["discarded_invalid_tags"]) for stage in consensus_stages
        ),
        "max_encoding_residual": max(encoded_residuals, default=0.0),
        "resource_screen_retained_edges": len(resource_screened_edges(cfg, episode)),
        "resource_screen_nonbinding": int(
            resource_screened_edges(cfg, episode) == edges(cfg)
        ),
    }


def exploration_dispatch_record(
    cfg: dict,
    seed: int,
    episode: int,
    code_index: int,
    matching: Sequence[tuple[int, int]],
) -> dict:
    """Serialize one common public-assignment codeword dispatch."""
    return {
        "campaign_id": cfg["campaign_id"],
        "seed": seed,
        "episode": episode,
        "protocol_branch": "exploration_public_assignment",
        "public_code_index_zero_based": code_index,
        "public_code_index_one_based": code_index + 1,
        "selected_matching": sid(matching),
        "centralized_matching": sid(matching),
        "distributed_matching": sid(matching),
        "centralized_stage_sequence": "PUBLIC:" + sid(matching),
        "distributed_stage_sequence": "PUBLIC:" + sid(matching),
        "accepted_stage_count": len(matching),
        "null_stage": 0,
        "stage_count": 0,
        "consensus_rounds_per_stage": 0,
        "consensus_round_instances": 0,
        "directed_proposal_transmissions": 0,
        "centralized_distributed_equal": 1,
        "all_agents_agree": 1,
        "all_tags_valid": 1,
        "discarded_invalid_tags": 0,
        "max_encoding_residual": 0.0,
        "resource_screen_retained_edges": len(resource_screened_edges(cfg, episode)),
        "resource_screen_nonbinding": int(
            resource_screened_edges(cfg, episode) == edges(cfg)
        ),
    }


def run_trial(
    cfg: dict,
    seed: int,
    protocol_records: list[dict] | None = None,
    feedback_records: list[dict] | None = None,
) -> tuple[list[dict], dict]:
    family = terminal_matchings(cfg)
    cert = physical_certificate(cfg, family)
    codebook = public_exploration_codebook(cfg)
    explore = exploration_schedule_indices(cfg)
    rng = np.random.default_rng(seed)
    V = cfg["ridge_lambda"] * np.eye(cfg["feature_dimension"])
    rhs = np.zeros(cfg["feature_dimension"])
    prev_end = cfg["initial_tracking_error"].copy()
    prev_S = None
    ctrue = copt = ccert = craw = 0.0
    diff = exploit = 0
    rows = []
    for k in range(1, cfg["episodes"] + 1):
        th = constrained_quadratic_minimizer(V, rhs, cfg["btheta"], cfg["fmax"])
        beta = beta_radius(
            V,
            cfg["ridge_lambda"],
            cert["shared_reset_deviation_scale"],
            cfg["confidence_delta"],
            cfg["btheta"],
        )
        beta0 = beta_radius(V, cfg["ridge_lambda"], 0.0, cfg["confidence_delta"], cfg["btheta"])
        is_exp = k <= cfg["exploration_episodes"]
        if is_exp:
            code_index = explore[k - 1]
            S = tuple(codebook[code_index]); widths = (); S0 = S
            if protocol_records is not None:
                protocol_records.append(
                    exploration_dispatch_record(cfg, seed, k, code_index, S)
                )
        else:
            central = centralized_greedy_trace(
                cfg, k, th, V, beta, score_scale_tag="configured"
            )
            distributed = distributed_greedy_trace(
                cfg, k, th, V, beta, score_scale_tag="configured"
            )
            central0 = centralized_greedy_trace(
                cfg, k, th, V, beta0, score_scale_tag="zero_scale"
            )
            distributed0 = distributed_greedy_trace(
                cfg, k, th, V, beta0, score_scale_tag="zero_scale"
            )
            assert_centralized_distributed_equivalence(cfg, central, distributed)
            assert_centralized_distributed_equivalence(cfg, central0, distributed0)
            S = central["matching"]
            widths = central["widths"]
            S0 = central0["matching"]
            if protocol_records is not None:
                protocol_records.append(
                    protocol_trace_record(
                        cfg, seed, k, "configured_scale", central, distributed
                    )
                )
                protocol_records.append(
                    protocol_trace_record(
                        cfg, seed, k, "zero_scale", central0, distributed0
                    )
                )
            exploit += 1
            diff += int(S != S0)
        Sstar, fstar = exact_optimum(cfg, family, k)
        f = value(cfg, S, k)
        if prev_S is None:
            jump = np.zeros(cfg["physical_state_dimension"])
            e0 = prev_end.copy()
        else:
            jump = reference(prev_S, cfg) - reference(S, cfg); e0 = prev_end + jump
        mode_id = mode(cfg, S)
        A = cfg["controller_templates"][mode_id]
        E = cfg["disturbance_input_templates"][mode_id]
        actual = e0.copy(); nominal = e0.copy(); eta = 0.0
        max_track = float(np.linalg.norm(actual)); max_phase = 0.0
        for tau in range(1, cfg["physical_horizon"] + 1):
            w = (
                cfg["disturbance_bound"]
                * rng.choice(
                    np.asarray([-1.0, 1.0]), size=cfg["disturbance_dimension"]
                )
                / math.sqrt(cfg["disturbance_dimension"])
            )
            actual = A @ actual + E @ w; nominal = A @ nominal; dev = actual - nominal
            eta += cfg["discount_factor"] ** (tau - 1) * float(cfg["trajectory_return_vector"] @ dev)
            max_track = max(max_track, float(np.linalg.norm(actual)))
            max_phase = max(
                max_phase,
                float(np.linalg.norm(dev)) / cert["deviation_phase_radii"][tau],
            )
        y = f + eta
        feedback = tagged_feedback_flood(cfg, y, k)
        if not feedback["all_agents_received"] or not feedback["all_agents_equal"]:
            raise AssertionError("tagged return flood failed")
        if feedback_records is not None:
            feedback_records.append(
                {
                    "campaign_id": cfg["campaign_id"],
                    "seed": seed,
                    "episode": k,
                    "source_agent": feedback["source_agent"],
                    "flood_rounds": feedback["rounds"],
                    "first_full_agreement_round": feedback[
                        "first_full_agreement_round"
                    ],
                    "directed_return_transmissions": feedback[
                        "directed_transmissions"
                    ],
                    "all_agents_received": int(feedback["all_agents_received"]),
                    "all_agents_equal": int(feedback["all_agents_equal"]),
                    "observed_return": y,
                }
            )
        x = set_feature(cfg, S, k)
        Vinv = np.linalg.inv(V)
        err = th - cfg["theta_star"]
        par_ratio = math.sqrt(max(0, float(err @ V @ err))) / beta
        edge_ratio = max(abs(float(err @ edge_feature(cfg, e, k))) /
                         (beta * math.sqrt(float(edge_feature(cfg, e, k) @ Vinv @ edge_feature(cfg, e, k)))) for e in edges(cfg))
        ceiling = cfg["fmax"] / cfg["q_extendibility"]
        raw = ceiling if is_exp else float(sum(2 * beta * w for w in widths)); inc = min(raw, ceiling)
        # Retain the archived stronger 1/q comparator as a regression gate.
        # Since kappa=1, closing it also closes the curvature-dependent 1/(q+kappa)
        # comparator without altering any locked observable or policy output.
        locked_comparator_shortfall = fstar / cfg["q_extendibility"] - f
        slack = inc - locked_comparator_shortfall
        ctrue += f
        copt += fstar
        ccert += inc
        craw += raw
        rows.append({
            "campaign_id": cfg["campaign_id"], "seed": seed, "episode": k,
            "exploration": int(is_exp), "selected_matching": sid(S), "zero_scale_matching": sid(S0),
            "zero_scale_difference": int(S != S0), "optimal_matching": sid(Sstar), "mode": mode(cfg, S),
            "true_value": f, "optimal_value": fstar, "value_ratio": f / fstar,
            "observed_return": y,
            "trajectory_fluctuation": eta,
            "trajectory_deviation": eta,
            "trajectory_sigma": cert["trajectory_sigma"],
            "shared_reset_deviation_scale": cert["shared_reset_deviation_scale"],
            "beta": beta, "beta_zero_scale": beta0, "parameter_confidence_ratio": par_ratio,
            "max_edge_confidence_ratio": edge_ratio, "raw_certificate_increment": raw,
            "clipped_certificate_increment": inc, "certificate_slack": slack,
            "cumulative_universal_utilization": cfg["q_extendibility"] * ccert / (k * cfg["fmax"]),
            "tracking_tube_utilization": max_track / cert["all_time_radius"],
            "trajectory_fluctuation_utilization": abs(eta) / cert["trajectory_sigma"],
            "trajectory_deviation_utilization": abs(eta) / cert["shared_reset_deviation_scale"],
            "phase_fluctuation_tube_utilization": max_phase,
            "phase_deviation_tube_utilization": max_phase,
            "reset_utilization": float(np.linalg.norm(jump)) / cert["reference_jump_bound"] if cert["reference_jump_bound"] else 0.0,
            "actuator_utilization": cfg["actuator_gain"] * max_track / cert["actuator_limit"],
            "validity_utilization": max_track / cert["validity_radius"],
            "theta_hat_l1": float(th.sum()), "theta_hat_l2": float(np.linalg.norm(th)),
        })
        V += np.outer(x, x)
        rhs += x * y
        prev_end = actual
        prev_S = S
    trial = {
        "campaign_id": cfg["campaign_id"], "seed": seed, "episodes": cfg["episodes"],
        "cumulative_true_value": ctrue, "cumulative_optimum_value": copt,
        "cumulative_value_retention": ctrue / copt,
        "final_universal_certificate_utilization": cfg["q_extendibility"] * ccert / (cfg["episodes"] * cfg["fmax"]),
        "final_raw_certificate_utilization": cfg["q_extendibility"] * craw / (cfg["episodes"] * cfg["fmax"]),
        "parameter_confidence_covered": int(max(r["parameter_confidence_ratio"] for r in rows) <= 1 + 1e-10),
        "edge_confidence_covered": int(max(r["max_edge_confidence_ratio"] for r in rows) <= 1 + 1e-10),
        "certificate_covered": int(min(r["certificate_slack"] for r in rows) >= -1e-10),
        "return_nonnegative": int(min(r["observed_return"] for r in rows) >= -1e-12),
        "max_parameter_confidence_ratio": max(r["parameter_confidence_ratio"] for r in rows),
        "max_edge_confidence_ratio": max(r["max_edge_confidence_ratio"] for r in rows),
        "max_trajectory_fluctuation_utilization": max(r["trajectory_fluctuation_utilization"] for r in rows),
        "max_trajectory_deviation_utilization": max(r["trajectory_deviation_utilization"] for r in rows),
        "max_phase_fluctuation_tube_utilization": max(r["phase_fluctuation_tube_utilization"] for r in rows),
        "max_phase_deviation_tube_utilization": max(r["phase_deviation_tube_utilization"] for r in rows),
        "max_tracking_tube_utilization": max(r["tracking_tube_utilization"] for r in rows),
        "max_reset_utilization": max(r["reset_utilization"] for r in rows),
        "max_actuator_utilization": max(r["actuator_utilization"] for r in rows),
        "max_validity_utilization": max(r["validity_utilization"] for r in rows),
        "zero_scale_counterfactual_difference_fraction": diff / exploit,
        "minimum_return": min(r["observed_return"] for r in rows), "maximum_return": max(r["observed_return"] for r in rows),
    }
    return rows, trial


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def model_closure_audit(
    cfg: dict,
    rows: Sequence[dict],
    episode_path: Path,
    trial_path: Path,
    static_certificate: dict | None = None,
) -> dict:
    """Check terminal-assignment domains and deterministic execution records.

    The audit evaluates capped features, curvature, fallback behavior, and
    physical dimensions. A precomputed static certificate avoids repeating
    exhaustive feature enumeration.
    """
    geometry = static_certificate or static_audit(cfg)
    selected_terminal = all(
        is_terminal_matching(cfg, parse_sid(str(row["selected_matching"])))
        for row in rows
    )
    zero_scale_terminal = all(
        is_terminal_matching(cfg, parse_sid(str(row["zero_scale_matching"])))
        for row in rows
    )
    optimum_terminal = all(
        is_terminal_matching(cfg, parse_sid(str(row["optimal_matching"])))
        for row in rows
    )

    policy_hash = canonical_column_hash(
        rows, ("campaign_id", "seed", "episode", "selected_matching")
    )
    selection_hash = canonical_column_hash(
        rows,
        (
            "campaign_id",
            "seed",
            "episode",
            "selected_matching",
            "zero_scale_matching",
            "optimal_matching",
        ),
    )
    primary_hash = canonical_column_hash(
        rows,
        (
            "campaign_id",
            "seed",
            "episode",
            "true_value",
            "optimal_value",
            "observed_return",
            "beta",
            "clipped_certificate_increment",
        ),
    )
    episode_sha = sha256(episode_path)
    trial_sha = sha256(trial_path)
    expected_hereditary_count = sum(
        math.comb(cfg["agents"], size) * math.perm(cfg["tasks"], size)
        for size in range(cfg["agents"] + 1)
    )
    expected_terminal_count = math.perm(cfg["tasks"], cfg["agents"])
    contextual_factor = 1.0 / (
        cfg["q_extendibility"] + geometry["maximum_contextual_curvature"]
    )
    fixed_factor = 1.0 / (cfg["q_extendibility"] + 1.0)
    gates = {
        "hereditary_family_complete": (
            len(hereditary_matchings(cfg)) == expected_hereditary_count
        ),
        "terminal_family_exactly_maximal": bool(
            geometry["terminal_family_equals_maximal_family"]
        ),
        "terminal_family_count_correct": (
            len(terminal_matchings(cfg)) == expected_terminal_count
        ),
        "analytical_saturation_bounds_close": (
            geometry["analytical_feasible_raw_coordinate_upper_bound"]
            < cfg["feature_saturation_cap"]
            and geometry[
                "analytical_ground_without_edge_coordinate_lower_bound"
            ]
            > cfg["feature_saturation_cap"]
        ),
        "saturation_inactive_on_hereditary_family": (
            geometry["maximum_feasible_raw_coordinate"]
            < cfg["feature_saturation_cap"]
            and geometry["maximum_feasible_saturation_residual"] == 0.0
            and geometry["maximum_feasible_marginal_residual"] == 0.0
        ),
        "global_extension_unit_curvature": bool(
            geometry["curvature_equals_one_all_episodes"]
        ),
        "contextual_and_fixed_factors_equal_one_third": (
            contextual_factor == fixed_factor == 1 / 3
        ),
        "single_two_dimensional_physical_block": (
            cfg["physical_blocks"] == 1
            and cfg["physical_state_dimension"] == 2
            and len(cfg["controller_templates"]) == 4
        ),
        "zero_benefit_fallback_configured": cfg["fallback_value"] == 0.0,
        "fallback_mode_and_reference_configured": (
            execution_mode(cfg, tuple()) == fallback_mode(cfg)
            and np.array_equal(
                execution_reference(cfg, tuple()), fallback_reference(cfg)
            )
        ),
        "all_selected_assignments_terminal": selected_terminal,
        "all_zero_scale_assignments_terminal": zero_scale_terminal,
        "all_exact_optima_terminal": optimum_terminal,
        "episode_records_byte_locked": episode_sha == LOCKED_EPISODE_RECORDS_SHA256,
        "trial_summary_byte_locked": trial_sha == LOCKED_TRIAL_SUMMARY_SHA256,
        "policy_fingerprint_locked": policy_hash == LOCKED_POLICY_FINGERPRINT,
        "selection_fingerprint_locked": (
            selection_hash == LOCKED_SELECTION_FINGERPRINT
        ),
        "primary_evidence_fingerprint_locked": (
            primary_hash == LOCKED_PRIMARY_EVIDENCE_FINGERPRINT
        ),
    }
    return {
        "audit_type": "terminal-model-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "model_interface_version": "terminal-model-v1",
        "episode_rows": len(rows),
        "fallback_invocations": sum(
            not str(row["selected_matching"]) for row in rows
        ),
        "family": {
            "ground_edge_count": len(edges(cfg)),
            "hereditary_assignment_count": len(hereditary_matchings(cfg)),
            "hereditary_cardinality_counts": geometry[
                "hereditary_cardinality_counts"
            ],
            "terminal_assignment_count": len(terminal_matchings(cfg)),
            "terminal_cardinality": cfg["terminal_assignment_cardinality"],
            "terminal_family_equals_maximal_family": geometry[
                "terminal_family_equals_maximal_family"
            ],
        },
        "structured_value": {
            "basis": "coordinatewise_min_of_additive_load_and_one",
            "feature_saturation_cap": cfg["feature_saturation_cap"],
            "edge_feature_coordinate_lower_bound": geometry[
                "edge_feature_coordinate_lower_bound"
            ],
            "edge_feature_coordinate_upper_bound": geometry[
                "edge_feature_coordinate_upper_bound"
            ],
            "analytical_feasible_raw_coordinate_upper_bound": geometry[
                "analytical_feasible_raw_coordinate_upper_bound"
            ],
            "analytical_ground_without_edge_coordinate_lower_bound": geometry[
                "analytical_ground_without_edge_coordinate_lower_bound"
            ],
            "maximum_feasible_raw_coordinate": geometry[
                "maximum_feasible_raw_coordinate"
            ],
            "feasible_saturation_cap_margin": geometry[
                "feasible_saturation_cap_margin"
            ],
            "maximum_feasible_saturation_residual": geometry[
                "maximum_feasible_saturation_residual"
            ],
            "maximum_feasible_marginal_residual": geometry[
                "maximum_feasible_marginal_residual"
            ],
            "minimum_ground_raw_coordinate": geometry[
                "minimum_ground_raw_coordinate"
            ],
            "maximum_ground_raw_coordinate": geometry[
                "maximum_ground_raw_coordinate"
            ],
            "minimum_ground_without_edge_raw_coordinate": geometry[
                "minimum_ground_without_edge_raw_coordinate"
            ],
            "minimum_singleton_value": geometry["minimum_singleton_value"],
            "minimum_contextual_curvature": geometry[
                "minimum_contextual_curvature"
            ],
            "maximum_contextual_curvature": geometry[
                "maximum_contextual_curvature"
            ],
            "q_extendibility": cfg["q_extendibility"],
            "contextual_greedy_factor": contextual_factor,
            "fixed_greedy_factor": fixed_factor,
        },
        "physical": {
            "physical_blocks": cfg["physical_blocks"],
            "state_dimension_per_block": cfg["physical_state_dimension"],
            "joint_state_dimension": (
                cfg["physical_blocks"] * cfg["physical_state_dimension"]
            ),
            "controller_template_count": len(cfg["controller_templates"]),
            "joint_mode_count_exercised": len(cfg["controller_templates"]),
            "per_agent_product_redesign_exercised": False,
            "fallback_mode": fallback_mode(cfg),
            "fallback_reference": fallback_reference(cfg).tolist(),
            "fallback_included_in_uniform_reference_certificate": True,
        },
        "file_sha256": {
            "episode_records.csv": episode_sha,
            "trial_summary.csv": trial_sha,
        },
        "fingerprints": {
            "policy": policy_hash,
            "selection_zero_scale_optimum": selection_hash,
            "primary_evidence": primary_hash,
        },
        "gates": gates,
    }


def distributed_reproducibility_audit(
    cfg: dict,
    rows: Sequence[dict],
    protocol_records: Sequence[dict],
    feedback_records: Sequence[dict],
    episode_path: Path,
    trial_path: Path,
    protocol_path: Path,
    feedback_path: Path,
    static_certificate: dict | None = None,
) -> dict:
    """Check distributed execution against centralized score selection.

    The audit verifies ownership, message tags, agreement rounds, exploration
    dispatch, resource screening, and return floods. Configured and zero-scale
    selections are compared at the same estimator state.
    """
    static = static_certificate or static_audit(cfg)
    seeds = tuple(int(seed) for seed in cfg["evaluation_seeds"])
    episodes = int(cfg["episodes"])
    exploration_episodes = int(cfg["exploration_episodes"])
    exploitation_episodes_per_seed = episodes - exploration_episodes
    expected_episode_rows = len(seeds) * episodes
    expected_exploration_dispatches = len(seeds) * exploration_episodes
    expected_exploitation_episodes = len(seeds) * exploitation_episodes_per_seed
    expected_exploitation_traces = 2 * expected_exploitation_episodes
    expected_protocol_rows = (
        expected_exploration_dispatches + expected_exploitation_traces
    )
    expected_stages_per_trace = cfg["agents"] + 1
    expected_stage_records = expected_exploitation_traces * expected_stages_per_trace
    rounds_per_stage = int(cfg["allocation_network"]["rounds_per_stage"])
    expected_consensus_round_instances = expected_stage_records * rounds_per_stage
    directed_network_edges = 2 * len(
        cfg["allocation_network"]["undirected_edges"]
    )
    expected_proposal_transmissions = (
        expected_consensus_round_instances * directed_network_edges
    )
    expected_feedback_records = expected_episode_rows
    feedback_rounds = int(cfg["allocation_network"]["diameter_upper_bound"])
    expected_feedback_round_instances = expected_feedback_records * feedback_rounds
    expected_return_transmissions = (
        expected_feedback_round_instances * directed_network_edges
    )

    codebook = public_exploration_codebook(cfg)
    schedule_indices = exploration_schedule_indices(cfg)
    codebook_rows = exploration_codebook_rows(cfg)
    schedule_rows = exploration_schedule_rows(cfg)
    codebook_fingerprint = canonical_column_hash(
        codebook_rows,
        (
            "zero_based_index",
            "one_based_index",
            "matching_id",
            "edge_1",
            "edge_2",
            "edge_3",
        ),
    )
    schedule_fingerprint = canonical_column_hash(
        schedule_rows,
        ("episode", "zero_based_index", "one_based_index", "matching_id"),
    )

    exploration_records = [
        record
        for record in protocol_records
        if record["protocol_branch"] == "exploration_public_assignment"
    ]
    configured_records = [
        record
        for record in protocol_records
        if record["protocol_branch"] == "configured_scale"
    ]
    zero_scale_records = [
        record
        for record in protocol_records
        if record["protocol_branch"] == "zero_scale"
    ]
    exploitation_records = configured_records + zero_scale_records
    allowed_branches = {
        "exploration_public_assignment",
        "configured_scale",
        "zero_scale",
    }

    episode_lookup = {
        (int(row["seed"]), int(row["episode"])): row for row in rows
    }
    exploration_expected_keys = {
        (seed, episode)
        for seed in seeds
        for episode in range(1, exploration_episodes + 1)
    }
    configured_expected_keys = {
        (seed, episode)
        for seed in seeds
        for episode in range(exploration_episodes + 1, episodes + 1)
    }
    zero_expected_keys = configured_expected_keys.copy()
    feedback_expected_keys = {
        (seed, episode)
        for seed in seeds
        for episode in range(1, episodes + 1)
    }

    exploration_actual_keys = [
        (int(record["seed"]), int(record["episode"]))
        for record in exploration_records
    ]
    configured_actual_keys = [
        (int(record["seed"]), int(record["episode"]))
        for record in configured_records
    ]
    zero_actual_keys = [
        (int(record["seed"]), int(record["episode"]))
        for record in zero_scale_records
    ]
    feedback_actual_keys = [
        (int(record["seed"]), int(record["episode"]))
        for record in feedback_records
    ]

    exploration_dispatch_exact = True
    for record in exploration_records:
        episode = int(record["episode"])
        index = schedule_indices[episode - 1]
        expected_matching = sid(codebook[index])
        exploration_dispatch_exact = exploration_dispatch_exact and (
            int(record["public_code_index_zero_based"]) == index
            and int(record["public_code_index_one_based"]) == index + 1
            and record["selected_matching"] == expected_matching
            and record["centralized_matching"] == expected_matching
            and record["distributed_matching"] == expected_matching
            and record["selected_matching"]
            == episode_lookup[(int(record["seed"]), episode)]["selected_matching"]
        )

    exploitation_policy_exact = True
    for record in configured_records:
        key = (int(record["seed"]), int(record["episode"]))
        exploitation_policy_exact = exploitation_policy_exact and (
            record["selected_matching"] == episode_lookup[key]["selected_matching"]
        )
    for record in zero_scale_records:
        key = (int(record["seed"]), int(record["episode"]))
        exploitation_policy_exact = exploitation_policy_exact and (
            record["selected_matching"]
            == episode_lookup[key]["zero_scale_matching"]
        )

    feedback_exact = True
    for record in feedback_records:
        key = (int(record["seed"]), int(record["episode"]))
        feedback_exact = feedback_exact and (
            float(record["observed_return"])
            == float(episode_lookup[key]["observed_return"])
        )

    owner_counts = {
        owner: sum(edge[0] == owner for edge in edges(cfg))
        for owner in range(cfg["agents"])
    }
    expected_schedule_zero_based = (
        3,
        10,
        17,
        24,
        31,
        38,
        45,
        52,
        59,
        6,
        13,
        20,
        27,
        34,
        41,
        48,
        55,
        2,
    )

    episode_sha = sha256(episode_path)
    trial_sha = sha256(trial_path)
    policy_hash = canonical_column_hash(
        rows, ("campaign_id", "seed", "episode", "selected_matching")
    )
    selection_hash = canonical_column_hash(
        rows,
        (
            "campaign_id",
            "seed",
            "episode",
            "selected_matching",
            "zero_scale_matching",
            "optimal_matching",
        ),
    )
    primary_hash = canonical_column_hash(
        rows,
        (
            "campaign_id",
            "seed",
            "episode",
            "true_value",
            "optimal_value",
            "observed_return",
            "beta",
            "clipped_certificate_increment",
        ),
    )

    gates = {
        "path_graph_exact": (
            allocation_graph_neighbors(cfg) == {0: (1,), 1: (0, 2), 2: (1,)}
            and allocation_graph_diameter(cfg) == 2
            and int(cfg["allocation_network"]["diameter"]) == 2
            and int(cfg["allocation_network"]["diameter_upper_bound"]) == 2
            and rounds_per_stage == 2
        ),
        "unique_owner_partition_exact": (
            owner_counts == {0: 5, 1: 5, 2: 5}
            and len({edge_identifier(cfg, edge) for edge in edges(cfg)})
            == len(edges(cfg))
        ),
        "required_message_tags_declared": tuple(
            cfg["allocation_network"]["required_message_tags"]
        )
        == ("episode", "stage", "partial_set", "score_scale"),
        "identity_encoder_zero_error": (
            cfg["allocation_network"]["score_encoding"] == "identity"
            and float(cfg["allocation_network"]["encoding_error"]) == 0.0
            and max(
                (float(record["max_encoding_residual"]) for record in exploitation_records),
                default=0.0,
            )
            == 0.0
        ),
        "public_codebook_complete_terminal_family": (
            len(codebook) == math.perm(cfg["tasks"], cfg["agents"]) == 60
            and codebook == terminal_matchings(cfg)
            and all(is_terminal_matching(cfg, matching) for matching in codebook)
        ),
        "public_schedule_fixed": (
            schedule_indices == expected_schedule_zero_based
            and len(set(schedule_indices)) == exploration_episodes == 18
        ),
        "protocol_branch_set_exact": (
            {record["protocol_branch"] for record in protocol_records}
            == allowed_branches
        ),
        "protocol_record_count_exact": len(protocol_records) == expected_protocol_rows,
        "exploration_dispatch_count_and_keys_exact": (
            len(exploration_records) == expected_exploration_dispatches
            and len(exploration_actual_keys) == len(set(exploration_actual_keys))
            and set(exploration_actual_keys) == exploration_expected_keys
        ),
        "configured_trace_count_and_keys_exact": (
            len(configured_records) == expected_exploitation_episodes
            and len(configured_actual_keys) == len(set(configured_actual_keys))
            and set(configured_actual_keys) == configured_expected_keys
        ),
        "zero_scale_trace_count_and_keys_exact": (
            len(zero_scale_records) == expected_exploitation_episodes
            and len(zero_actual_keys) == len(set(zero_actual_keys))
            and set(zero_actual_keys) == zero_expected_keys
        ),
        "exploration_dispatch_matches_public_schedule": exploration_dispatch_exact,
        "centralized_distributed_matching_equality_all_traces": all(
            int(record["centralized_distributed_equal"]) == 1
            and record["centralized_matching"] == record["distributed_matching"]
            for record in exploitation_records
        ),
        "centralized_distributed_stage_sequence_equality_all_traces": all(
            record["centralized_stage_sequence"]
            == record["distributed_stage_sequence"]
            for record in exploitation_records
        ),
        "three_acceptances_and_null_stage_all_traces": all(
            int(record["accepted_stage_count"]) == cfg["agents"] == 3
            and int(record["stage_count"]) == expected_stages_per_trace == 4
            and int(record["null_stage"]) == expected_stages_per_trace == 4
            and str(record["centralized_stage_sequence"]).endswith("|NULL")
            for record in exploitation_records
        ),
        "all_agents_agree_and_all_tags_valid": all(
            int(record["all_agents_agree"]) == 1
            and int(record["all_tags_valid"]) == 1
            and int(record["discarded_invalid_tags"]) == 0
            for record in exploitation_records
        ),
        "consensus_accounting_exact": (
            sum(int(record["stage_count"]) for record in exploitation_records)
            == expected_stage_records
            and sum(
                int(record["consensus_round_instances"])
                for record in exploitation_records
            )
            == expected_consensus_round_instances
            and sum(
                int(record["directed_proposal_transmissions"])
                for record in exploitation_records
            )
            == expected_proposal_transmissions
        ),
        "resource_screen_nonbinding_all_records": (
            all(
                int(record["resource_screen_retained_edges"]) == len(edges(cfg))
                and int(record["resource_screen_nonbinding"]) == 1
                for record in protocol_records
            )
            and all(
                resource_screened_edges(cfg, episode) == edges(cfg)
                for episode in range(1, episodes + 1)
            )
        ),
        "protocol_outputs_match_locked_policy_columns": exploitation_policy_exact,
        "feedback_count_and_keys_exact": (
            len(feedback_records) == expected_feedback_records
            and len(feedback_actual_keys) == len(set(feedback_actual_keys))
            and set(feedback_actual_keys) == feedback_expected_keys
        ),
        "feedback_flood_exact_all_episodes": (
            feedback_exact
            and all(
                int(record["source_agent"])
                == int(cfg["allocation_network"]["feedback_source_agent"])
                and int(record["flood_rounds"]) == feedback_rounds == 2
                and int(record["first_full_agreement_round"]) == feedback_rounds
                and int(record["all_agents_received"]) == 1
                and int(record["all_agents_equal"]) == 1
                and int(record["directed_return_transmissions"])
                == feedback_rounds * directed_network_edges
                for record in feedback_records
            )
        ),
        "feedback_accounting_exact": (
            len(feedback_records) * feedback_rounds
            == expected_feedback_round_instances
            and sum(
                int(record["directed_return_transmissions"])
                for record in feedback_records
            )
            == expected_return_transmissions
        ),
        "episode_records_byte_locked": episode_sha == LOCKED_EPISODE_RECORDS_SHA256,
        "trial_summary_byte_locked": trial_sha == LOCKED_TRIAL_SUMMARY_SHA256,
        "policy_fingerprint_locked": policy_hash == LOCKED_POLICY_FINGERPRINT,
        "selection_fingerprint_locked": (
            selection_hash == LOCKED_SELECTION_FINGERPRINT
        ),
        "primary_evidence_fingerprint_locked": (
            primary_hash == LOCKED_PRIMARY_EVIDENCE_FINGERPRINT
        ),
    }

    return {
        "audit_type": "distributed-protocol-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "protocol_interface_version": "distributed-protocol-v1",
        "network": {
            "nodes": cfg["agents"],
            "undirected_edges": [
                list(edge) for edge in cfg["allocation_network"]["undirected_edges"]
            ],
            "neighbors": {
                str(node): list(neighbors)
                for node, neighbors in allocation_graph_neighbors(cfg).items()
            },
            "diameter": allocation_graph_diameter(cfg),
            "diameter_upper_bound": int(
                cfg["allocation_network"]["diameter_upper_bound"]
            ),
            "rounds_per_stage": rounds_per_stage,
            "directed_edges_per_round": directed_network_edges,
            "feedback_source_agent": int(
                cfg["allocation_network"]["feedback_source_agent"]
            ),
        },
        "ownership": {
            "rule": cfg["allocation_network"]["ownership_rule"],
            "identifier_order": cfg["allocation_network"]["identifier_order"],
            "candidate_count_by_owner": owner_counts,
        },
        "encoding": {
            "scheme": cfg["allocation_network"]["score_encoding"],
            "declared_error": float(cfg["allocation_network"]["encoding_error"]),
            "maximum_observed_residual": max(
                (float(record["max_encoding_residual"]) for record in exploitation_records),
                default=0.0,
            ),
            "required_message_tags": list(
                cfg["allocation_network"]["required_message_tags"]
            ),
        },
        "resource_screen": {
            "mode": cfg["resource_screen"]["mode"],
            "robust_edge_cost": float(cfg["resource_screen"]["robust_edge_cost"]),
            "retained_edges_each_episode": len(edges(cfg)),
            "nonbinding_records": sum(
                int(record["resource_screen_nonbinding"])
                for record in protocol_records
            ),
            "protocol_records": len(protocol_records),
            "consumable_resource_channel_activated": False,
        },
        "exploration": {
            "codebook_family": cfg["exploration_codebook"]["family"],
            "codebook_size": len(codebook),
            "codebook_fingerprint": codebook_fingerprint,
            "schedule_multiplier": int(
                cfg["exploration_codebook"]["schedule_multiplier"]
            ),
            "schedule_offset": int(cfg["exploration_codebook"]["schedule_offset"]),
            "schedule_indices_zero_based": list(schedule_indices),
            "schedule_indices_one_based": [index + 1 for index in schedule_indices],
            "schedule_fingerprint": schedule_fingerprint,
            "dispatch_mode": "common_public_terminal_assignment",
        },
        "counts": {
            "evaluation_seeds": len(seeds),
            "episode_rows": len(rows),
            "exploration_dispatch_records": len(exploration_records),
            "exploitation_episodes": expected_exploitation_episodes,
            "configured_scale_traces": len(configured_records),
            "zero_scale_traces": len(zero_scale_records),
            "exploitation_traces": len(exploitation_records),
            "exploitation_stage_records": sum(
                int(record["stage_count"]) for record in exploitation_records
            ),
            "consensus_round_instances": sum(
                int(record["consensus_round_instances"])
                for record in exploitation_records
            ),
            "directed_proposal_transmissions": sum(
                int(record["directed_proposal_transmissions"])
                for record in exploitation_records
            ),
            "feedback_floods": len(feedback_records),
            "feedback_round_instances": len(feedback_records) * feedback_rounds,
            "directed_return_transmissions": sum(
                int(record["directed_return_transmissions"])
                for record in feedback_records
            ),
        },
        "static_certificate_excerpt": {
            "allocation_network_diameter": static["allocation_network_diameter"],
            "consensus_rounds_per_stage": static["consensus_rounds_per_stage"],
            "exploration_codebook_size": static["exploration_codebook_size"],
            "resource_screen_retained_edge_count": static[
                "resource_screen_retained_edge_count"
            ],
            "encoding_error": static["encoding_error"],
        },
        "file_sha256": {
            "episode_records.csv": episode_sha,
            "trial_summary.csv": trial_sha,
            "distributed_protocol_records.csv": sha256(protocol_path),
            "feedback_flood_records.csv": sha256(feedback_path),
        },
        "fingerprints": {
            "policy": policy_hash,
            "selection_zero_scale_optimum": selection_hash,
            "primary_evidence": primary_hash,
            "exploration_codebook": codebook_fingerprint,
            "exploration_schedule": schedule_fingerprint,
        },
        "gates": gates,
    }


def execute(config_path: Path, out: Path) -> dict:
    cfg = load_config(config_path)
    static = static_audit(cfg)
    rows: list[dict] = []
    trials: list[dict] = []
    protocol_records: list[dict] = []
    feedback_records: list[dict] = []
    for seed in cfg["evaluation_seeds"]:
        trial_rows, trial_summary = run_trial(
            cfg,
            int(seed),
            protocol_records=protocol_records,
            feedback_records=feedback_records,
        )
        rows.extend(trial_rows)
        trials.append(trial_summary)

    out.mkdir(parents=True, exist_ok=True)
    episode_path = out / "episode_records.csv"
    trial_path = out / "trial_summary.csv"
    protocol_path = out / "distributed_protocol_records.csv"
    feedback_path = out / "feedback_flood_records.csv"
    write_csv(episode_path, rows)
    write_csv(trial_path, trials)
    write_csv(protocol_path, protocol_records)
    write_csv(feedback_path, feedback_records)

    closure = model_closure_audit(cfg, rows, episode_path, trial_path, static)
    closure_path = out / "model_closure_audit.json"
    closure_path.write_text(
        json.dumps(closure, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if closure["status"] != "PASS":
        failed = [name for name, passed in closure["gates"].items() if not passed]
        raise AssertionError(f"terminal-model-v1 model-closure gate failed: {failed}")

    protocol_audit = distributed_reproducibility_audit(
        cfg,
        rows,
        protocol_records,
        feedback_records,
        episode_path,
        trial_path,
        protocol_path,
        feedback_path,
        static,
    )
    protocol_audit_path = out / "distributed_reproducibility_audit.json"
    protocol_audit_path.write_text(
        json.dumps(protocol_audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if protocol_audit["status"] != "PASS":
        failed = [
            name for name, passed in protocol_audit["gates"].items() if not passed
        ]
        raise AssertionError(
            f"distributed-protocol-v1 distributed-reproducibility gate failed: {failed}"
        )

    retention = mean_se([t["cumulative_value_retention"] for t in trials])
    clipped = mean_se(
        [t["final_universal_certificate_utilization"] for t in trials]
    )
    raw = mean_se([t["final_raw_certificate_utilization"] for t in trials])
    zero = mean_se(
        [t["zero_scale_counterfactual_difference_fraction"] for t in trials]
    )
    results = {
        "cumulative_value_retention_mean": retention[0],
        "cumulative_value_retention_se": retention[1],
        "final_universal_certificate_utilization_mean": clipped[0],
        "final_universal_certificate_utilization_se": clipped[1],
        "final_raw_certificate_utilization_mean": raw[0],
        "final_raw_certificate_utilization_se": raw[1],
        "parameter_confidence_trial_coverage": float(
            np.mean([t["parameter_confidence_covered"] for t in trials])
        ),
        "edge_confidence_trial_coverage": float(
            np.mean([t["edge_confidence_covered"] for t in trials])
        ),
        "certificate_trial_coverage": float(
            np.mean([t["certificate_covered"] for t in trials])
        ),
        "nonnegative_return_trial_coverage": float(
            np.mean([t["return_nonnegative"] for t in trials])
        ),
        "max_parameter_confidence_ratio": max(
            t["max_parameter_confidence_ratio"] for t in trials
        ),
        "max_edge_confidence_ratio": max(
            t["max_edge_confidence_ratio"] for t in trials
        ),
        "max_trajectory_fluctuation_utilization": max(
            t["max_trajectory_fluctuation_utilization"] for t in trials
        ),
        "max_trajectory_deviation_utilization": max(
            t["max_trajectory_deviation_utilization"] for t in trials
        ),
        "max_phase_fluctuation_tube_utilization": max(
            t["max_phase_fluctuation_tube_utilization"] for t in trials
        ),
        "max_phase_deviation_tube_utilization": max(
            t["max_phase_deviation_tube_utilization"] for t in trials
        ),
        "max_tracking_tube_utilization": max(
            t["max_tracking_tube_utilization"] for t in trials
        ),
        "max_reset_utilization": max(t["max_reset_utilization"] for t in trials),
        "max_actuator_utilization": max(
            t["max_actuator_utilization"] for t in trials
        ),
        "max_validity_utilization": max(
            t["max_validity_utilization"] for t in trials
        ),
        "zero_scale_counterfactual_difference_fraction_mean": zero[0],
        "zero_scale_counterfactual_difference_fraction_se": zero[1],
        "minimum_observed_return": min(t["minimum_return"] for t in trials),
        "maximum_observed_return": max(t["maximum_return"] for t in trials),
    }

    seed_dir = config_path.parents[1] / "seeds"
    codebook_asset = seed_dir / "trajectory_microcase_codebook.csv"
    schedule_asset = seed_dir / "trajectory_microcase_exploration_schedule.csv"
    source_hashes = {
        "configuration_sha256": sha256(config_path),
        "trajectory_source_sha256": sha256(Path(__file__)),
        "seed_registry_sha256": sha256(
            seed_dir / "trajectory_microcase_seeds.csv"
        ),
    }
    if codebook_asset.exists():
        source_hashes["exploration_codebook_sha256"] = sha256(codebook_asset)
    if schedule_asset.exists():
        source_hashes["exploration_schedule_sha256"] = sha256(schedule_asset)

    summary = {
        "campaign_id": cfg["campaign_id"],
        "status": (
            "VERIFIED_DISTRIBUTED_"
            "EXECUTION"
        ),
        "calibration_interface_version": "shared-reset-calibration-v1",
        "model_interface_version": "terminal-model-v1",
        "protocol_interface_version": "distributed-protocol-v1",
        "configuration": {
            **{
                key: cfg[key]
                for key in (
                    "agents",
                    "tasks",
                    "feature_dimension",
                    "feature_saturation_cap",
                    "q_extendibility",
                    "terminal_assignment_cardinality",
                    "fallback_value",
                    "fallback_mode",
                    "physical_blocks",
                    "physical_state_dimension",
                    "episodes",
                    "physical_horizon",
                    "exploration_episodes",
                    "fmax",
                    "btheta",
                    "ridge_lambda",
                    "confidence_delta",
                    "disturbance_bound",
                    "discount_factor",
                    "mu",
                )
            },
            "fallback_reference": fallback_reference(cfg).tolist(),
            "allocation_network": cfg["allocation_network"],
            "exploration_codebook": cfg["exploration_codebook"],
            "resource_screen": cfg["resource_screen"],
        },
        "evaluation_seed_count": len(cfg["evaluation_seeds"]),
        "static_certificate": static,
        "results": results,
        "protocol_audit_excerpt": {
            "status": protocol_audit["status"],
            "counts": protocol_audit["counts"],
            "network": protocol_audit["network"],
            "exploration": protocol_audit["exploration"],
        },
        "scope": {
            "target_execution": True,
            "return_generated_from_physical_trajectory": True,
            "shared_reset_deviation_scale_fixed_before_episode_one": True,
            "trajectory_scale_fixed_before_episode_one": True,
            "general_and_microcase_deviation_scales_identical": True,
            "globally_saturating_basis": True,
            "saturation_inactive_on_complete_hereditary_family": True,
            "unit_contextual_curvature_all_episodes": True,
            "physical_modes_and_returns_terminal_assignments_only": True,
            "absorbing_fallback_zero_benefit": True,
            "fallback_mode_and_reference_predeclared": True,
            "single_physical_block_base_case": True,
            "tagged_max_consensus_executed": True,
            "centralized_distributed_sequence_equality_all_exploitation_traces": True,
            "configured_and_zero_scale_protocols_checked": True,
            "identity_encoding_zero_error": True,
            "deterministic_public_assignment_codebook": True,
            "resource_screen_nonbinding_all_episodes": True,
            "tagged_feedback_flood_all_episodes": True,
            "zero_scale_recomputation_is_fixed_estimator_diagnostic_only": True,
            "large_campaigns_remain_aggregate_return_executions": True,
        },
        "source_hashes": source_hashes,
    }
    summary_path = out / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    manifest_files = (
        episode_path,
        trial_path,
        closure_path,
        protocol_path,
        feedback_path,
        protocol_audit_path,
        summary_path,
    )
    manifest = {path.name: sha256(path) for path in manifest_files}
    (out / "MANIFEST.json").write_text(
        json.dumps(
            {"campaign_id": cfg["campaign_id"], "files": manifest},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return summary
