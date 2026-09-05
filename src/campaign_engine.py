"""Simulation engine for the synthetic task-allocation campaigns.

The engine implements the allocation baselines, full-bandit estimator, pathwise
resource accounting, distributed winner reduction, and assignment-dependent
tracking-loop diagnostics used in the included experiments.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence
import csv
import hashlib
import json
import math
import time

import numpy as np
from scipy.linalg import eigvalsh
from scipy.optimize import minimize

from tracking_models import TrackingBounds, ControllerMode, build_tracking_bounds, load_json
from reference_reset import mode_from_assignment_map, validate_assignment_template_map

SOFTWARE_VERSION = "1.0.0"
RAW_SCHEMA = "baseline-campaign-v1"
TOL = 1.0e-10

PRIMARY_METHODS = (
    "DISTRIBUTED_UCB",
    "CENTRAL_GREEDY_ORACLE",
    "DISTRIBUTED_KNOWN_VALUE",
    "INSTANTANEOUS_MYOPIC",
    "PROJECTED_MEAN",
    "UCB_WITHOUT_RESOURCE_FILTER",
    "RANDOM_FEASIBLE",
)
SMALL_METHODS = (
    "DISTRIBUTED_UCB",
    "CENTRAL_GREEDY_ORACLE",
    "CENTRAL_EXACT_OPTIMUM",
)
LEARNING_METHODS = {
    "DISTRIBUTED_UCB", "PROJECTED_MEAN", "UCB_WITHOUT_RESOURCE_FILTER"
}
DISTRIBUTED_METHODS = {
    "DISTRIBUTED_UCB", "DISTRIBUTED_KNOWN_VALUE", "INSTANTANEOUS_MYOPIC",
    "PROJECTED_MEAN", "UCB_WITHOUT_RESOURCE_FILTER", "RANDOM_FEASIBLE",
}

RAW_FIELDS = [
    "schema_version", "software_version",
    "campaign_id", "campaign_scale", "partition", "trial_index", "trial_seed",
    "method_id", "epoch", "source_config_hash", "engine_hash", "graph_family",
    "graph_edges", "graph_diameter", "active_q", "selected_count", "selected_elements",
    "selection_hash", "assignment_valid", "true_value", "realized_return",
    "oracle_greedy_value", "oracle_value_ratio", "exact_optimum_value",
    "exact_optimum_ratio", "comparison_factor", "empirical_curvature",
    "scaled_regret_increment", "cumulative_scaled_regret",
    "oracle_gap_upper_bound_increment", "cumulative_oracle_gap_upper_bound",
    "allocation_lower_bound", "allocation_bound_slack", "beta", "selected_width_sum",
    "max_normalized_marginal_error", "parameter_confidence_holds",
    "marginal_confidence_holds", "block_gram_min_eigenvalue",
    "central_distributed_mismatch", "allocation_rounds", "round_law_expected",
    "directed_transmissions", "resource_violations", "complete_family_violations",
    "minimum_residual_resource", "resource_consumption_sum", "fallback_episode",
    "tracking_rms", "tracking_peak", "minimum_mode_inequality_slack",
    "minimum_jump_inequality_slack", "finite_time_envelope_slack", "uub_radius_ratio",
    "control_limit_margin", "model_validity_margin", "enumerated_feasible_sets"
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def raw_value(x: Any) -> Any:
    if x is None:
        return ""
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating, float)):
        if not math.isfinite(float(x)):
            return ""
        return format(float(x), ".17g")
    return x


def write_csv_atomic(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str]) -> tuple[str, int]:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fields), lineterminator="\n")
        w.writeheader()
        for row in rows:
            w.writerow({k: raw_value(row.get(k)) for k in fields})
    digest = sha256_file(tmp)
    if path.exists():
        if sha256_file(path) != digest:
            tmp.unlink()
            raise RuntimeError(f"existing output differs from deterministic rerun: {path}")
        tmp.unlink()
    else:
        tmp.replace(path)
    return digest, len(rows)


def random_ball(rng: np.random.Generator, prefix: tuple[int, ...], d: int, radius: float) -> np.ndarray:
    if radius <= 0:
        return np.zeros((*prefix, d))
    z = rng.normal(size=(*prefix, d))
    norm = np.linalg.norm(z, axis=-1, keepdims=True)
    norm = np.maximum(norm, np.finfo(float).eps)
    r = rng.random(size=(*prefix, 1)) ** (1.0 / d)
    return radius * r * z / norm


def coverage_feature(p: np.ndarray, selected: Sequence[int]) -> np.ndarray:
    if not selected:
        return np.zeros(p.shape[1])
    return 1.0 - np.prod(1.0 - p[np.asarray(selected, dtype=int)], axis=0)


def coverage_value(theta: np.ndarray, p: np.ndarray, selected: Sequence[int]) -> float:
    return float(theta @ coverage_feature(p, selected))


def curvature(theta: np.ndarray, p: np.ndarray) -> float:
    if p.size == 0:
        return 0.0
    singleton = p @ theta
    active = singleton > 1e-14
    if not np.any(active):
        return 0.0
    logf = np.log1p(-np.clip(p, 0.0, 1.0 - 1e-14))
    total = logf.sum(axis=0)
    ratios = []
    for i in np.flatnonzero(active):
        others = np.exp(np.clip(total - logf[i], -745, 0))
        ratios.append(float(theta @ (p[i] * others)) / float(singleton[i]))
    return float(np.clip(1.0 - min(ratios), 0.0, 1.0))


def beta_radius(V: np.ndarray, lam: float, sigma: float, delta: float, btheta: float) -> float:
    sign, ld = np.linalg.slogdet(V)
    if sign <= 0:
        raise ValueError("design matrix is not positive definite")
    d = V.shape[0]
    return sigma * math.sqrt(max(0.0, ld - d * math.log(lam) + 2 * math.log(1 / delta))) + math.sqrt(lam) * btheta


def nonnegative_quadratic_minimizer(V: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Solve the strictly convex nonnegative quadratic program by an active set.

    The dimension is at most eight.  Each iteration solves one principal
    system and enforces the KKT signs; cycling falls back to convergent
    coordinate minimization and is rejected unless the KKT residual is small.
    """
    d = len(rhs)
    active = set(range(d))
    seen: set[tuple[int, ...]] = set()
    tol = 2.0e-11
    for _ in range(8 * d + 8):
        key = tuple(sorted(active))
        if key in seen:
            break
        seen.add(key)
        x = np.zeros(d)
        if active:
            idx = np.asarray(sorted(active), dtype=int)
            sub = np.linalg.solve(V[np.ix_(idx, idx)], rhs[idx])
            if np.min(sub) < -tol:
                active.remove(int(idx[int(np.argmin(sub))]))
                continue
            x[idx] = np.maximum(sub, 0.0)
        grad = V @ x - rhs
        inactive = [j for j in range(d) if j not in active]
        if inactive:
            g = grad[inactive]
            if np.min(g) < -tol:
                active.add(int(inactive[int(np.argmin(g))]))
                continue
        active_residual = float(np.max(np.abs(grad[list(active)]))) if active else 0.0
        if active_residual <= 2.0e-9:
            return x
    # Deterministic coordinate descent fallback.
    x = np.maximum(np.linalg.solve(V, rhs), 0.0)
    for _ in range(2000):
        change = 0.0
        for j in range(d):
            new = max(0.0, (rhs[j] - float(V[j] @ x) + V[j, j] * x[j]) / V[j, j])
            change = max(change, abs(new - x[j])); x[j] = new
        if change < 1.0e-13:
            break
    grad = V @ x - rhs
    kkt = max(float(np.max(np.abs(grad[x > 1.0e-9]))) if np.any(x > 1.0e-9) else 0.0, max(0.0, -float(np.min(grad[x <= 1.0e-9]))) if np.any(x <= 1.0e-9) else 0.0)
    if kkt > 2.0e-7:
        raise RuntimeError(f"projected-ridge KKT residual {kkt:.3e}")
    return x


def constrained_quadratic_minimizer(
    V: np.ndarray,
    rhs: np.ndarray,
    btheta: float,
    fmax: float,
) -> np.ndarray:
    """Solve the fully constrained projected-ridge quadratic program.

    The feasible set is ``x >= 0``, ``||x||_2 <= btheta`` and
    ``||x||_1 <= fmax``.  A deterministic active-set solve first handles the
    nonnegative orthant.  When both norm constraints are inactive, that exact
    solution is returned unchanged.  Otherwise a small SLSQP problem is solved
    with analytic derivatives and then subjected to explicit feasibility and
    first-order checks.  Campaign dimensions are at most eight.
    """
    V = np.asarray(V, dtype=float)
    rhs = np.asarray(rhs, dtype=float)
    if V.ndim != 2 or V.shape[0] != V.shape[1] or rhs.shape != (V.shape[0],):
        raise ValueError("incompatible projected-ridge dimensions")
    if not np.all(np.isfinite(V)) or not np.all(np.isfinite(rhs)):
        raise ValueError("projected-ridge inputs must be finite")
    if btheta <= 0.0 or fmax <= 0.0:
        raise ValueError("btheta and fmax must be positive")
    if not np.allclose(V, V.T, atol=2.0e-12, rtol=2.0e-12):
        raise ValueError("projected-ridge matrix must be symmetric")
    try:
        np.linalg.cholesky(V)
    except np.linalg.LinAlgError as exc:
        raise ValueError("projected-ridge matrix must be positive definite") from exc

    x_nn = nonnegative_quadratic_minimizer(V, rhs)
    feasibility_tol = 2.0e-10
    if (
        float(np.linalg.norm(x_nn)) <= btheta + feasibility_tol
        and float(np.sum(x_nn)) <= fmax + feasibility_tol
    ):
        # Exact fast path when the nonnegative minimizer satisfies both norm bounds.
        return x_nn

    scale = 1.0
    norm_nn = float(np.linalg.norm(x_nn))
    sum_nn = float(np.sum(x_nn))
    if norm_nn > 0.0:
        scale = min(scale, btheta / norm_nn)
    if sum_nn > 0.0:
        scale = min(scale, fmax / sum_nn)
    x0 = np.maximum(0.0, x_nn * min(1.0, scale))

    def objective(x: np.ndarray) -> float:
        return 0.5 * float(x @ V @ x) - float(rhs @ x)

    def gradient(x: np.ndarray) -> np.ndarray:
        return V @ x - rhs

    constraints = (
        {
            "type": "ineq",
            "fun": lambda x: btheta * btheta - float(x @ x),
            "jac": lambda x: -2.0 * x,
        },
        {
            "type": "ineq",
            "fun": lambda x: fmax - float(np.sum(x)),
            "jac": lambda x: -np.ones_like(x),
        },
    )
    upper = min(float(btheta), float(fmax))
    result = minimize(
        objective,
        x0,
        jac=gradient,
        bounds=[(0.0, upper)] * len(rhs),
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1.0e-12, "maxiter": 1000, "disp": False},
    )
    x = np.maximum(np.asarray(result.x, dtype=float), 0.0)
    norm_x = float(np.linalg.norm(x))
    sum_x = float(np.sum(x))
    if (
        not result.success
        or not np.all(np.isfinite(x))
        or norm_x > btheta + 2.0e-8
        or sum_x > fmax + 2.0e-8
    ):
        raise RuntimeError(
            "full projected-ridge solve failed: "
            f"{result.message}; norm={norm_x:.12g}, sum={sum_x:.12g}"
        )
    if objective(x) > objective(x0) + 2.0e-9:
        raise RuntimeError("full projected-ridge solve did not improve its feasible start")

    # A deterministic variational check over the coordinate and pairwise
    # feasible tangent directions detects materially nonstationary solutions.
    grad = gradient(x)
    active_l2 = abs(norm_x - btheta) <= 2.0e-7
    active_l1 = abs(sum_x - fmax) <= 2.0e-7
    directional_violations: list[float] = []
    for j in range(len(x)):
        # Positive coordinate directions are feasible only away from both
        # active upper constraints; negative directions require x_j > 0.
        if not active_l1 and not active_l2:
            directional_violations.append(max(0.0, -float(grad[j])))
        if x[j] > 2.0e-7:
            directional_violations.append(max(0.0, float(grad[j])))
    for i in range(len(x)):
        if x[i] <= 2.0e-7:
            continue
        for j in range(len(x)):
            if i == j:
                continue
            dvec = np.zeros_like(x)
            dvec[i], dvec[j] = -1.0, 1.0
            # Pairwise transfers preserve the l1 constraint.  At an active l2
            # boundary they are feasible to first order only when x_j <= x_i.
            if (not active_l2) or x[j] <= x[i] + 2.0e-7:
                directional_violations.append(max(0.0, -float(grad @ dvec)))
    if directional_violations and max(directional_violations) > 2.0e-5:
        raise RuntimeError(
            f"full projected-ridge stationarity residual {max(directional_violations):.3e}"
        )
    return x


@dataclass
class GraphInfo:
    family: str
    edges: int
    diameter: int


def graph_info(n: int, family: str, radius: float, rng: np.random.Generator) -> GraphInfo:
    if family == "path":
        return GraphInfo(family, max(0, n - 1), max(0, n - 1))
    if family == "ring":
        return GraphInfo(family, n if n > 2 else max(0, n - 1), n // 2)
    if family == "complete":
        return GraphInfo(family, n * (n - 1) // 2, 1 if n > 1 else 0)
    if family != "connected_random_geometric":
        raise ValueError(f"unsupported graph {family}")
    # Reproduce a deterministic connected geometric graph; increase radius if needed.
    pts = rng.random((n, 2))
    r = radius
    while True:
        adj = np.linalg.norm(pts[:, None] - pts[None, :], axis=2) <= r
        np.fill_diagonal(adj, False)
        seen = {0}
        frontier = [0]
        while frontier:
            i = frontier.pop()
            for j in np.flatnonzero(adj[i]):
                if int(j) not in seen:
                    seen.add(int(j)); frontier.append(int(j))
        if len(seen) == n:
            break
        r += 0.03
        if r > 1.5:
            raise RuntimeError("could not construct connected graph")
    dist = np.full((n, n), n + 1, dtype=int)
    np.fill_diagonal(dist, 0)
    dist[adj] = 1
    for k in range(n):
        dist = np.minimum(dist, dist[:, k, None] + dist[None, k, :])
    return GraphInfo(family, int(np.triu(adj, 1).sum()), int(dist.max()))


@dataclass
class ElementBatch:
    owner: np.ndarray
    task: np.ndarray
    quota: np.ndarray
    element_id: np.ndarray
    p_h: np.ndarray
    p_1: np.ndarray
    robust_cost: np.ndarray
    actual_cost: np.ndarray
    demand: np.ndarray
    kinematic: np.ndarray
    quota_caps: np.ndarray

    @property
    def size(self) -> int:
        return len(self.owner)


@dataclass
class Assets:
    sim_root: Path
    campaign_path: Path
    cfg: dict[str, Any]
    tracking: TrackingBounds
    tracking_cfg: dict[str, Any]
    static: dict[str, np.ndarray]


def _as_float(rows: list[dict[str, str]], keys: Sequence[str]) -> np.ndarray:
    return np.asarray([[float(r[k]) for k in keys] for r in rows], dtype=float)


def load_assets(sim_root: Path, campaign_path: Path) -> Assets:
    cfg = load_json(campaign_path)
    tracking_cfg = load_json(sim_root / cfg["tracking_model_config"])
    tracking, _, _ = build_tracking_bounds(tracking_cfg)
    cid = cfg["campaign_id"]
    validate_assignment_template_map(cid, cfg, tuple(tracking.modes))
    ddir = sim_root / "datasets" / cid
    static: dict[str, np.ndarray] = {}
    if cid == "SAT-COV-V1":
        sats = load_csv(ddir / "satellites.csv")
        targets = load_csv(ddir / "targets.csv")
        quality = load_csv(ddir / "sensor_product_quality.csv")
        static["sat"] = _as_float(sats, ["orbital_phase_rad", "phase_rate_rad_per_epoch", "sensor_quality", "boresight_initial_rad"])
        rel_keys = [f"product_relevance_{j}" for j in range(1, 7)]
        static["target"] = _as_float(targets, ["longitude_deg", "latitude_deg", "spectral_channel", "priority", "cloud_baseline"] + rel_keys)
        q = np.zeros((len(sats), 6))
        for r in quality:
            q[int(r["satellite_id"]) - 1, int(r["product_id"]) - 1] = float(r["product_quality"])
        static["quality"] = q
    else:
        uavs = load_csv(ddir / "uavs.csv")
        cells = load_csv(ddir / "field_cells.csv")
        static["uav"] = _as_float(uavs, ["initial_x_m", "initial_y_m", "sensor_quality"])
        feat_keys = [f"agronomic_feature_{j}" for j in range(1, 9)]
        static["cell"] = _as_float(cells, ["x_m", "y_m", "zone_id", "priority"] + feat_keys)
    return Assets(sim_root, campaign_path, cfg, tracking, tracking_cfg, static)


class ContextGenerator:
    def __init__(self, assets: Assets, seed: int, n_agents: int, n_tasks: int, quota_active: bool = True):
        self.a = assets; self.rng = np.random.default_rng(seed); self.na=n_agents; self.nt=n_tasks; self.quota_active=quota_active
        if assets.cfg["campaign_id"] == "SAT-COV-V1":
            base = assets.static["target"][:n_tasks, 4]
            self.cloud = np.clip(base + self.rng.normal(0, .04, n_tasks), 0, .95)
            self.phase_shift = self.rng.uniform(-.12, .12, n_agents)
        else:
            self.wind = self.rng.normal(0, .6, 2)
            self.temp_shift = float(self.rng.normal(0, .6))
            self.patrol_phase = self.rng.uniform(0, 2*math.pi, n_agents)

    def next(self, epoch: int) -> ElementBatch:
        return self._sat(epoch) if self.a.cfg["campaign_id"] == "SAT-COV-V1" else self._uav(epoch)

    def _sat(self, epoch: int) -> ElementBatch:
        cfg=self.a.cfg; gen=cfg["generator"]; p=cfg["primary"]; sat=self.a.static["sat"][:self.na]; tar=self.a.static["target"][:self.nt]; qual=self.a.static["quality"][:self.na]
        ar=float(gen["cloud_ar_coefficient"]); innov=float(gen["cloud_innovation_std"])
        base=tar[:,4]
        self.cloud=np.clip(ar*self.cloud+(1-ar)*base+self.rng.normal(0,innov,self.nt),0,.98)
        solar=float(np.clip(.72+.22*math.sin(.037*epoch)+self.rng.normal(0,.025),.35,1.0))
        owner=np.repeat(np.arange(self.na),self.nt); task=np.tile(np.arange(self.nt),self.na)
        phase=(sat[:,0]+sat[:,1]*epoch+self.phase_shift)%(2*math.pi)
        lon=np.deg2rad(tar[:,0]); lat=np.deg2rad(tar[:,1])
        diff=np.abs((phase[:,None]-lon[None,:]+math.pi)%(2*math.pi)-math.pi)
        vis=np.maximum(0,np.cos(diff))*np.maximum(.25,np.cos(lat))[None,:]
        future=(phase[:,None]+sat[:,1,None]*self.a.tracking.horizon-lon[None,:]+math.pi)%(2*math.pi)-math.pi
        future_vis=np.maximum(0,np.cos(future))*np.maximum(.25,np.cos(lat))[None,:]
        slew=np.abs((sat[:,3,None]-lon[None,:]+math.pi)%(2*math.pi)-math.pi)/math.pi
        sensor=sat[:,2,None]
        priority=tar[:,3][None,:]
        clear=(1-self.cloud)[None,:]
        base_acq=sensor*priority*clear*solar*np.exp(-slew/float(gen["slew_decay_scale"]))
        current=np.clip(base_acq*(.22+.78*vis),0,1)
        horizon_gain=np.clip(.52*vis+.48*future_vis,0,1)
        h_acq=np.clip(base_acq*(.18+.82*horizon_gain),0,1)
        rel=tar[:,5:11]
        p1=np.clip(current[:,:,None]*rel[None,:,:]*qual[:,None,:],float(gen["minimum_coverage_probability"]),float(gen["maximum_coverage_probability"]))
        ph=np.clip(1-(1-np.clip(.16*h_acq[:,:,None]*rel[None,:,:]*qual[:,None,:],0,.95))**min(self.a.tracking.horizon,6),float(gen["minimum_coverage_probability"]),float(gen["maximum_coverage_probability"]))
        e_lo,e_hi=map(float,gen["energy_cost_range"]); d_lo,d_hi=map(float,gen["data_cost_range"])
        energy=e_lo+(e_hi-e_lo)*(.25+.75*slew)
        data=d_lo+(d_hi-d_lo)*(.25+.75*(1-clear))*np.ones((self.na,1))
        data=np.broadcast_to(data,(self.na,self.nt))
        actual=np.stack([energy*(.88+.06*self.rng.random((self.na,self.nt))),data*(.88+.06*self.rng.random((self.na,self.nt)))],axis=2)
        robust=np.stack([energy,data],axis=2)
        quota=tar[:,2].astype(int)-1
        caps=np.asarray(p["channel_capacities"] if self.quota_active else [self.na]*len(p["channel_capacities"]),int)
        kin=vis>float(gen["visibility_threshold"])
        return ElementBatch(owner,task,np.tile(quota,self.na),np.arange(self.na*self.nt),ph.reshape(-1,6),p1.reshape(-1,6),robust.reshape(-1,2),actual.reshape(-1,2),slew.reshape(-1),kin.reshape(-1),caps)

    def _uav(self, epoch: int) -> ElementBatch:
        cfg=self.a.cfg; gen=cfg["generator"]; p=cfg["primary"]; uav=self.a.static["uav"][:self.na]; cell=self.a.static["cell"][:self.nt]
        ar=float(gen["wind_ar_coefficient"]); innov=float(gen["wind_innovation_std_m_per_s"])
        self.wind=ar*self.wind+self.rng.normal(0,innov,2)
        Lx,Ly=map(float,gen["field_size_m"])
        ang=.025*epoch+self.patrol_phase
        pos=np.column_stack([(uav[:,0]+35*np.sin(ang))%Lx,(uav[:,1]+28*np.cos(.8*ang))%Ly])
        target=cell[:,:2]
        delta=target[None,:,:]-pos[:,None,:]
        wind_shift=self.wind[None,None,:]*self.a.tracking.horizon*self.a.tracking_cfg["sample_time"]
        d1=np.linalg.norm(delta,axis=2)
        dh=np.linalg.norm(delta-wind_shift,axis=2)
        scale=float(gen["transit_decay_scale_m"])
        sensor=uav[:,2,None]; priority=cell[:,3][None,:]
        current=sensor*priority*np.exp(-d1/(.55*scale))
        planned=sensor*priority*np.exp(-dh/scale)*(1+.08*np.cos(.03*epoch+self.temp_shift))
        feat=cell[:,4:12]
        p1=np.clip(current[:,:,None]*feat[None,:,:],float(gen["minimum_coverage_probability"]),float(gen["maximum_coverage_probability"]))
        ph=np.clip(1-(1-np.clip(.18*planned[:,:,None]*feat[None,:,:],0,.95))**min(self.a.tracking.horizon,6),float(gen["minimum_coverage_probability"]),float(gen["maximum_coverage_probability"]))
        demand=np.clip(dh/(math.hypot(Lx,Ly)),0,1)
        e_lo,e_hi=map(float,gen["energy_cost_range"]); y_lo,y_hi=map(float,gen["payload_cost_range"])
        energy=e_lo+(e_hi-e_lo)*(.2+.8*demand)
        payload=np.broadcast_to(y_lo+(y_hi-y_lo)*(.25+.75*cell[:,3][None,:]),(self.na,self.nt))
        actual=np.stack([energy*(.88+.06*self.rng.random((self.na,self.nt))),payload*(.88+.06*self.rng.random((self.na,self.nt)))],axis=2)
        robust=np.stack([energy,payload],axis=2)
        owner=np.repeat(np.arange(self.na),self.nt); task=np.tile(np.arange(self.nt),self.na)
        quota=cell[:,2].astype(int)-1
        caps=np.asarray(p["zone_capacities"] if self.quota_active else [self.na]*len(p["zone_capacities"]),int)
        kin=dh <= 1.35*scale
        return ElementBatch(owner,task,np.tile(quota,self.na),np.arange(self.na*self.nt),ph.reshape(-1,8),p1.reshape(-1,8),robust.reshape(-1,2),actual.reshape(-1,2),demand.reshape(-1),kin.reshape(-1),caps)


@dataclass
class Estimator:
    d: int; lam: float; sigma: float; delta: float; btheta: float; fmax: float
    V: np.ndarray=field(init=False); rhs: np.ndarray=field(init=False); theta: np.ndarray=field(init=False)
    def __post_init__(self):
        if self.d <= 0 or self.lam <= 0 or self.sigma < 0 or not (0 < self.delta < 1):
            raise ValueError("invalid estimator configuration")
        if self.btheta <= 0 or self.fmax <= 0:
            raise ValueError("estimator norm and value bounds must be positive")
        self.V=self.lam*np.eye(self.d); self.rhs=np.zeros(self.d); self.theta=np.zeros(self.d)
    def beta(self)->float: return beta_radius(self.V,self.lam,self.sigma,self.delta,self.btheta)
    def update(self,x:np.ndarray,y:float)->None:
        self.V += np.outer(x,x); self.rhs += x*y
        self.theta=constrained_quadratic_minimizer(self.V,self.rhs,self.btheta,self.fmax)


@dataclass
class PhysicalState:
    x: np.ndarray
    previous_modes: list[str]
    start_bounds: np.ndarray


@dataclass
class MethodState:
    resources: np.ndarray
    estimator: Estimator|None
    physical: PhysicalState
    cum_regret: float=0.0
    cum_gap: float=0.0


@dataclass
class Allocation:
    selected:list[int]; mismatch:int=0; width_sum:float=0.0; max_norm_error:float|None=None
    marginal_holds:bool|None=None; quant_eps:float=0.0; partial_violations:int=0


def feasible_mask(batch:ElementBatch,base:np.ndarray,selmask:np.ndarray,used_owner:np.ndarray,used_task:np.ndarray,counts:np.ndarray)->np.ndarray:
    m=base & ~selmask & ~used_owner[batch.owner] & ~used_task[batch.task]
    m &= counts[batch.quota] < batch.quota_caps[batch.quota]
    return m


def argmax_tie(scores:np.ndarray, ids:np.ndarray)->int:
    top=float(np.max(scores)); loc=np.flatnonzero(np.isclose(scores,top,rtol=0,atol=1e-12))
    return int(loc[np.argmin(ids[loc])])


def allocate(batch:ElementBatch,base:np.ndarray,method:str,theta_star:np.ndarray,est:Estimator|None,bits:int,graph:GraphInfo,rng:np.random.Generator,max_add:int)->Allocation:
    selected=[]; sm=np.zeros(batch.size,bool); no=int(batch.owner.max())+1; nt=int(batch.task.max())+1
    uo=np.zeros(no,bool); ut=np.zeros(nt,bool); qc=np.zeros(len(batch.quota_caps),int)
    fail_score=np.ones(batch.p_h.shape[1]); fail_true=np.ones(batch.p_h.shape[1]); mismatch=0;widthsum=0.;normerr=[];holds=True;partial=0;qeps=.5/(2**bits)
    pscore=batch.p_1 if method=="INSTANTANEOUS_MYOPIC" else batch.p_h
    while len(selected)<max_add:
        cand=base & ~sm if method=="UCB_WITHOUT_RESOURCE_FILTER" else feasible_mask(batch,base,sm,uo,ut,qc)
        idx=np.flatnonzero(cand)
        if not len(idx): break
        if method=="RANDOM_FEASIBLE":
            choice=int(idx[int(rng.integers(len(idx)))])
        else:
            psi=pscore[idx]*fail_score
            if method in {"CENTRAL_GREEDY_ORACLE","DISTRIBUTED_KNOWN_VALUE","INSTANTANEOUS_MYOPIC"}:
                scores=psi@theta_star; widths=np.zeros(len(idx))
            elif method in LEARNING_METHODS:
                if est is None: raise AssertionError("missing estimator")
                means=psi@est.theta
                if method in {"DISTRIBUTED_UCB","UCB_WITHOUT_RESOURCE_FILTER"}:
                    widths=np.sqrt(np.maximum(0,np.einsum('ij,ji->i',psi,np.linalg.solve(est.V,psi.T))))
                    scores=means+est.beta()*widths
                else:
                    widths=np.zeros(len(idx));scores=means
            else: raise ValueError(method)
            qs=np.rint(scores*(2**bits))/(2**bits)
            cpos=argmax_tie(qs,batch.element_id[idx]); central=int(idx[cpos])
            local=[]
            for o in range(no):
                po=np.flatnonzero(batch.owner[idx]==o)
                if len(po): local.append(int(idx[po[argmax_tie(qs[po],batch.element_id[idx[po]])]]))
            la=np.asarray(local,int); ls=np.asarray([qs[np.flatnonzero(idx==e)[0]] for e in la])
            distributed=int(la[argmax_tie(ls,batch.element_id[la])])
            mismatch += int(central!=distributed)
            choice=distributed if method in DISTRIBUTED_METHODS else central
            if method in {"DISTRIBUTED_UCB","UCB_WITHOUT_RESOURCE_FILTER"}:
                pos=int(np.flatnonzero(idx==choice)[0]); w=float(widths[pos]); widthsum+=w
                truepsi=batch.p_h[idx]*fail_true; tr=truepsi@theta_star
                selected_true=float((batch.p_h[choice]*fail_true)@theta_star)
                gap=max(0.,float(np.max(tr))-selected_true)
                denom=2*est.beta()*w+2*qeps
                normerr.append(gap/max(denom,1e-15))
                err=np.abs(truepsi@(est.theta-theta_star))
                bound=est.beta()*np.sqrt(np.maximum(0,np.einsum('ij,ji->i',truepsi,np.linalg.solve(est.V,truepsi.T))))
                holds=holds and bool(np.all(err<=bound+1e-8))
        selected.append(choice);sm[choice]=True
        if uo[batch.owner[choice]] or ut[batch.task[choice]] or qc[batch.quota[choice]]>=batch.quota_caps[batch.quota[choice]]: partial+=1
        uo[batch.owner[choice]]=True;ut[batch.task[choice]]=True;qc[batch.quota[choice]]+=1
        fail_score*=1-pscore[choice];fail_true*=1-batch.p_h[choice]
    return Allocation(selected,mismatch,widthsum,max(normerr) if normerr else None,holds if normerr else None,qeps,partial)


def assignment_violations(batch:ElementBatch, selected:Sequence[int])->int:
    if not selected:return 0
    o=batch.owner[list(selected)];t=batch.task[list(selected)];q=batch.quota[list(selected)]
    v=(len(o)-len(np.unique(o)))+(len(t)-len(np.unique(t)))
    for j,cap in enumerate(batch.quota_caps):v+=max(0,int(np.sum(q==j))-int(cap))
    return int(v)


def exact_optimum(batch:ElementBatch,base:np.ndarray,theta:np.ndarray,max_leaves:int)->tuple[list[int],float,int]:
    """Exact branch-and-bound over at most five owners, with a valid submodular bound."""
    nowners=int(batch.owner.max())+1
    options=[list(np.flatnonzero(base & (batch.owner==o))) for o in range(nowners)]
    # Owner order with fewest feasible options first improves pruning.
    order=sorted(range(nowners),key=lambda o:len(options[o]))
    best_sel: list[int]=[];best_val=0.;leaves=0
    def dfs(depth:int, sel:list[int], used_tasks:set[int], counts:np.ndarray, fail:np.ndarray, value:float)->None:
        nonlocal best_sel,best_val,leaves
        if leaves>max_leaves: raise RuntimeError(f"exact enumeration exceeded {max_leaves}")
        if depth==len(order):
            leaves+=1
            ids=tuple(int(batch.element_id[e]) for e in sel)
            bids=tuple(int(batch.element_id[e]) for e in best_sel)
            if value>best_val+1e-13 or (abs(value-best_val)<=1e-13 and ids<bids):best_val=value;best_sel=list(sel)
            return
        # Valid upper bound: sum best current marginal of each remaining owner, ignoring conflicts.
        ub=value
        for rr in order[depth:]:
            if options[rr]: ub+=max(float(theta@(batch.p_h[e]*fail)) for e in options[rr])
        if ub<best_val-1e-13:return
        owner=order[depth]
        ranked=[]
        for e in options[owner]:
            if int(batch.task[e]) in used_tasks or counts[batch.quota[e]]>=batch.quota_caps[batch.quota[e]]:continue
            ranked.append((float(theta@(batch.p_h[e]*fail)),int(batch.element_id[e]),int(e)))
        ranked.sort(key=lambda z:(-z[0],z[1]))
        for marg,_,e in ranked:
            counts[batch.quota[e]]+=1;used_tasks.add(int(batch.task[e]));sel.append(e)
            dfs(depth+1,sel,used_tasks,counts,fail*(1-batch.p_h[e]),value+marg)
            sel.pop();used_tasks.remove(int(batch.task[e]));counts[batch.quota[e]]-=1
        dfs(depth+1,sel,used_tasks,counts,fail,value)
    dfs(0,[],set(),np.zeros(len(batch.quota_caps),int),np.ones(batch.p_h.shape[1]),0.)
    return best_sel,best_val,leaves


def mode_for_demand(assets: Assets, demand: float, assigned: bool) -> str:
    return mode_from_assignment_map(
        assets.cfg["campaign_id"], assets.cfg, tuple(assets.tracking.modes), demand, assigned
    )


def physical_epoch(assets:Assets,state:PhysicalState,batch:ElementBatch,selected:Sequence[int],dist:np.ndarray,jumps:np.ndarray,uniforms:np.ndarray,dist_mult:float,jump_mult:float,burned:bool)->dict[str,Any]:
    cert=assets.tracking;n=len(state.x);assigned={int(batch.owner[e]):int(e) for e in selected}
    norms=[];mode_sl=math.inf;jump_sl=math.inf;env_sl=math.inf;ctrl=math.inf;valid=math.inf;fallback=False
    for a in range(n):
        old=cert.modes[state.previous_modes[a]]; e=assigned.get(a); target_name=mode_for_demand(assets,float(batch.demand[e]) if e is not None else 0.,e is not None);mode=cert.modes[target_name]
        if state.previous_modes[a]!=target_name or np.linalg.norm(jumps[a])>0:
            source_v=float(state.x[a]@old.p@state.x[a]); j=jump_mult*jumps[a]; nx=state.x[a]+j;target_v=float(nx@mode.p@nx)
            rhs=cert.mu*source_v+cert.c_jump*float(j@j);jump_sl=min(jump_sl,rhs-target_v);state.start_bounds[a]=cert.mu*state.start_bounds[a]+cert.c_jump*float(j@j);state.x[a]=nx
        state.previous_modes[a]=target_name
        bound=state.start_bounds[a]
        for tau in range(cert.horizon):
            x=state.x[a];v=float(x@mode.p@x);env_sl=min(env_sl,bound-v);norms.append(float(np.linalg.norm(x)))
            u=mode.k@x;ctrl=min(ctrl,cert.control_limit-float(np.linalg.norm(u)));valid=min(valid,cert.validity_radius-float(np.linalg.norm(x)))
            w=dist_mult*dist[tau,a];nx=mode.a_cl@x+mode.e_w@w;nv=float(nx@mode.p@nx);rhs=cert.lambda_c*v+cert.c_w*float(w@w);mode_sl=min(mode_sl,rhs-nv);bound=cert.lambda_c*bound+cert.c_w*float(w@w);state.x[a]=nx
        v=float(state.x[a]@mode.p@state.x[a]);env_sl=min(env_sl,bound-v);norms.append(float(np.linalg.norm(state.x[a])));state.start_bounds[a]=bound
    if math.isinf(jump_sl): jump_sl=float('nan')
    fallback = bool(ctrl < -1.0e-10 or valid < -1.0e-10)
    arr=np.asarray(norms)
    return {"fallback":fallback,"rms":float(np.sqrt(np.mean(arr**2))),"peak":float(arr.max()),"mode_slack":mode_sl,"jump_slack":jump_sl,"envelope_slack":env_sl,"uub_ratio":float(arr.max()/cert.all_step_radius) if burned else float('nan'),"control_margin":ctrl,"validity_margin":valid}


def make_states(methods:Sequence[str],assets:Assets,n_agents:int,initial:np.ndarray,sigma:float,resource_mult:float)->dict[str,MethodState]:
    p=assets.cfg["primary"];theta=np.asarray(p["theta_star"],float);lam=float(p["ridge_regularization"]);delta=float(p["confidence_delta"])
    res0=np.asarray(p["resource_initial"],float)*resource_mult
    idle=list(assets.tracking.modes)[0];P=assets.tracking.modes[idle].p
    bounds=np.asarray([float(x@P@x) for x in initial])
    out={}
    for m in methods:
        est=Estimator(len(theta),lam,sigma,delta,1.05*float(np.linalg.norm(theta)),float(theta.sum())) if m in LEARNING_METHODS else None
        out[m]=MethodState(np.tile(res0,(n_agents,1)),est,PhysicalState(initial.copy(),[idle]*n_agents,bounds.copy()))
    return out


def raw_name(cid:str,scale:str,partition:str,index:int,seed:int,method:str)->str:
    return f"baseline_{cid}_{scale}_{partition}_T{index:02d}_S{seed}_{method}.csv"


def run_trial(sim_root:Path,campaign_path:Path,seed_row:dict[str,Any],raw_dir:Path,engine_hash:str,*,scale:str="primary",methods:Sequence[str]|None=None,epoch_limit:int|None=None,feedback_sigma_override:float|None=None,resource_multiplier:float=1.0,graph_family_override:str|None=None,quant_bits_override:int|None=None,disturbance_multiplier:float=1.0,jump_multiplier:float=1.0,quota_layer_active:bool=True,write_raw:bool=True)->dict[str,Any]:
    assets=load_assets(sim_root,campaign_path);cfg=assets.cfg;p=cfg["primary"];spec=p if scale=="primary" else cfg["small_exact_oracle"]
    methods=tuple(methods or (PRIMARY_METHODS if scale=="primary" else SMALL_METHODS));na=int(spec["agents"]);nt=int(spec["tasks"]);epochs=min(int(spec["epochs"]),epoch_limit or int(spec["epochs"]));theta=np.asarray(p["theta_star"],float)
    sigma=float(p["feedback_noise_sigma"] if feedback_sigma_override is None else feedback_sigma_override);bits=int(p["score_quantization_bits"] if quant_bits_override is None else quant_bits_override);family=str(p["graph"] if graph_family_override is None else graph_family_override)
    grng=np.random.default_rng(int(seed_row["algorithm_seed"])^0xA13F09);graph=graph_info(na,family,float(p["graph_radius"]),grng);context=ContextGenerator(assets,int(seed_row["scenario_seed"]),na,nt,quota_layer_active)
    frng=np.random.default_rng(int(seed_row["feedback_seed"]));prng=np.random.default_rng(int(seed_row["physical_seed"]));arng=np.random.default_rng(int(seed_row["algorithm_seed"]));sdim=next(iter(assets.tracking.modes.values())).a_cl.shape[0];idim=next(iter(assets.tracking.modes.values())).e_w.shape[1]
    initial=random_ball(prng,(na,),sdim,assets.tracking.initial_error_bound);states=make_states(methods,assets,na,initial,sigma,resource_multiplier);rows={m:[] for m in methods};block={m:[] for m in methods};blockmin={m:None for m in methods};W=4*len(theta);burn=max(1,math.ceil(.2*epochs));start=time.perf_counter();chash=sha256_file(campaign_path)
    for k in range(1,epochs+1):
        batch=context.next(k);sign=1. if frng.random()<.5 else -1.;dist=random_ball(prng,(assets.tracking.horizon,na),idim,assets.tracking.w_bound);jumps=random_ball(prng,(na,),sdim,assets.tracking.jump_bound);uniforms=prng.random(na);randseed=int(arng.integers(0,2**32-1,dtype=np.uint32))
        allocs={}; oracle={}; exact_by={}; base_by={}; kappa={}
        for m in methods:
            st=states[m];aff=np.all(batch.robust_cost<=st.resources[batch.owner]+1e-12,axis=1);base=batch.kinematic&aff;base_by[m]=base;kappa[m]=curvature(theta,batch.p_h[base])
            if m=="CENTRAL_EXACT_OPTIMUM":
                sel,val,leaves=exact_optimum(batch,base,theta,int(spec["maximum_enumerated_feasible_sets"]));allocs[m]=Allocation(sel);exact_by[m]=(sel,val,leaves)
            else:
                allocs[m]=allocate(batch,base,m,theta,st.estimator,bits,graph,np.random.default_rng(randseed) if m=="RANDOM_FEASIBLE" else None,max_add=na)
            if m!="CENTRAL_GREEDY_ORACLE":oracle[m]=allocate(batch,base,"CENTRAL_GREEDY_ORACLE",theta,None,bits,graph,None,max_add=na)
        if "CENTRAL_GREEDY_ORACLE" in methods: oracle["CENTRAL_GREEDY_ORACLE"]=allocs["CENTRAL_GREEDY_ORACLE"]
        # On the small campaign, each method is compared with the exact
        # optimum on its own current residual-resource feasible family.
        if scale=="small":
            for m in methods:
                if m not in exact_by:
                    exact_by[m]=exact_optimum(batch,base_by[m],theta,int(spec["maximum_enumerated_feasible_sets"]))
        for m in methods:
            st=states[m];al=allocs[m];viol=assignment_violations(batch,al.selected);valid=viol==0
            if m=="UCB_WITHOUT_RESOURCE_FILTER":viol+=al.partial_violations;valid=viol==0
            selected=al.selected if valid else []
            true=coverage_value(theta,batch.p_h,selected);oval=coverage_value(theta,batch.p_h,oracle[m].selected);ratio=true/oval if oval>1e-15 else 1.; exact_record=exact_by.get(m); exactval=exact_record[1] if exact_record else None; exactratio=true/exactval if exactval and exactval>1e-15 else (1. if exactval==0 else None)
            feat=coverage_feature(batch.p_h,selected);noise=sign*min(sigma,.45*true) if true>0 else 0.;y=max(0.,true+noise)
            phys=physical_epoch(assets,st.physical,batch,selected,dist,jumps,uniforms,disturbance_multiplier,jump_multiplier,k>=burn)
            if phys["fallback"]: y=0.
            consume=np.zeros_like(st.resources)
            for e in selected:consume[batch.owner[e]]+=batch.actual_cost[e]
            repl=np.asarray(p["resource_replenishment_per_epoch"],float);initial_res=np.asarray(p["resource_initial"],float)*resource_multiplier
            st.resources=np.minimum(initial_res,st.resources-consume+repl);rv=int(np.sum(st.resources<-1e-10));minres=float(st.resources.min())
            est=st.estimator;beta=est.beta() if est else None;param=None
            if est: param=bool(float((est.theta-theta)@est.V@(est.theta-theta))<=beta*beta+1e-8);block[m].append(feat.copy());est.update(feat,y)
            if len(block[m])==W: blockmin[m]=float(eigvalsh(np.sum([np.outer(x,x) for x in block[m]],axis=0)).min());block[m]=[]
            alpha=1/(int(p["q"])+kappa[m]);q=int(p["q"]);errbudget=2*(beta or 0)*al.width_sum+2*len(al.selected)*al.quant_eps;lower=max(0.,(oval-q*errbudget)/(q+kappa[m]));slack=true-lower
            regret=(alpha*exactval-true) if exactval is not None else None;gap=max(0.,oval-true);st.cum_gap+=gap
            if regret is not None:st.cum_regret+=regret
            L=len(al.selected);rounds=(L+1)*graph.diameter;tx=2*graph.edges*rounds
            row={"schema_version":RAW_SCHEMA,"software_version":SOFTWARE_VERSION,"campaign_id":cfg["campaign_id"],"campaign_scale":scale,"partition":seed_row["partition"],"trial_index":int(seed_row["trial_index"]),"trial_seed":int(seed_row["trial_seed"]),"method_id":m,"epoch":k,"source_config_hash":chash,"engine_hash":engine_hash,"graph_family":family,"graph_edges":graph.edges,"graph_diameter":graph.diameter,"active_q":int(p["q"] if quota_layer_active else int(p["q"])-1),"selected_count":len(al.selected),"selected_elements":"|".join(map(str,[int(batch.element_id[e]) for e in al.selected])),"selection_hash":stable_hash("|".join(map(str,[int(batch.element_id[e]) for e in al.selected]))),"assignment_valid":valid,"true_value":true,"realized_return":y,"oracle_greedy_value":oval,"oracle_value_ratio":ratio,"exact_optimum_value":exactval,"exact_optimum_ratio":exactratio,"comparison_factor":alpha,"empirical_curvature":kappa[m],"scaled_regret_increment":regret,"cumulative_scaled_regret":st.cum_regret if regret is not None else None,"oracle_gap_upper_bound_increment":gap,"cumulative_oracle_gap_upper_bound":st.cum_gap,"allocation_lower_bound":lower,"allocation_bound_slack":slack,"beta":beta,"selected_width_sum":al.width_sum,"max_normalized_marginal_error":al.max_norm_error,"parameter_confidence_holds":param,"marginal_confidence_holds":al.marginal_holds,"block_gram_min_eigenvalue":blockmin[m],"central_distributed_mismatch":al.mismatch,"allocation_rounds":rounds,"round_law_expected":rounds,"directed_transmissions":tx,"resource_violations":rv,"complete_family_violations":viol,"minimum_residual_resource":minres,"resource_consumption_sum":float(consume.sum()),"fallback_episode":phys["fallback"],"tracking_rms":phys["rms"],"tracking_peak":phys["peak"],"minimum_mode_inequality_slack":phys["mode_slack"],"minimum_jump_inequality_slack":phys["jump_slack"],"finite_time_envelope_slack":phys["envelope_slack"],"uub_radius_ratio":phys["uub_ratio"],"control_limit_margin":phys["control_margin"],"model_validity_margin":phys["validity_margin"],"enumerated_feasible_sets":exact_record[2] if exact_record else None}
            rows[m].append(row)
    files=[]
    if write_raw:
        for m in methods:
            path = raw_dir / raw_name(
                cfg["campaign_id"], scale, str(seed_row["partition"]),
                int(seed_row["trial_index"]), int(seed_row["trial_seed"]), m
            )
            digest, n = write_csv_atomic(path, rows[m], RAW_FIELDS)
            try:
                recorded_path = path.relative_to(sim_root).as_posix()
            except ValueError:
                recorded_path = path.as_posix()
            files.append({
                "relative_path": recorded_path, "sha256": digest,
                "rows": n, "method": m
            })
    aggs={}
    for m,rr in rows.items():
        aggs[m]={"team_value":sum(float(r["realized_return"]) for r in rr),"true_value":sum(float(r["true_value"]) for r in rr),"mean_oracle_ratio":float(np.mean([r["oracle_value_ratio"] for r in rr])),"terminal_oracle_gap":rr[-1]["cumulative_oracle_gap_upper_bound"],"terminal_scaled_regret":rr[-1]["cumulative_scaled_regret"],"confidence_all":all(r["parameter_confidence_holds"] in (None,True) and r["marginal_confidence_holds"] in (None,True) for r in rr),"fallback_rate":float(np.mean([r["fallback_episode"] for r in rr])),"tracking_rms":float(np.mean([r["tracking_rms"] for r in rr])),"tracking_peak":max(float(r["tracking_peak"]) for r in rr),"min_envelope_slack":min(float(r["finite_time_envelope_slack"]) for r in rr),"max_uub_ratio":max([float(r["uub_radius_ratio"]) for r in rr if r["uub_radius_ratio"] is not None and math.isfinite(float(r["uub_radius_ratio"]))] or [float('nan')]),"resource_violations":sum(int(r["resource_violations"]) for r in rr),"family_violations":sum(int(r["complete_family_violations"]) for r in rr),"mismatches":sum(int(r["central_distributed_mismatch"]) for r in rr)}
    return {"campaign_id":cfg["campaign_id"],"scale":scale,"partition":seed_row["partition"],"trial_index":int(seed_row["trial_index"]),"trial_seed":int(seed_row["trial_seed"]),"epochs":epochs,"methods":list(methods),"files":files,"aggregates":aggs,"elapsed_seconds":time.perf_counter()-start,"graph":{"family":family,"edges":graph.edges,"diameter":graph.diameter}}
