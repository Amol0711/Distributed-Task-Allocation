"""Observable-certificate arithmetic for distributed task allocation.

This module accumulates raw and clipped episode charges and evaluates
fixed-trace transfers between declared return scales. The same primitives are
used by the simulation engine and the independent result validators.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

ABS_TOL = 2.0e-10
REL_TOL = 2.0e-10


class CertificateValidationError(ValueError):
    """Raised when a serialized certificate ledger fails deterministic replay."""


def _finite_nonnegative(name: str, value: float) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return result


def _positive_integer(name: str, value: int) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def certificate_cap(*, f_max: float, q: int) -> float:
    """Return the universal per-episode ceiling ``F_max/q``."""
    maximum = _finite_nonnegative("f_max", f_max)
    qq = _positive_integer("q", q)
    return maximum / qq


def fixed_comparator_factor(q: int) -> float:
    """Return the horizon-uniform approximation factor ``1/(q+1)``."""
    return 1.0 / (_positive_integer("q", q) + 1.0)


def contextual_comparator_factor(*, q: int, curvature: float) -> float:
    """Return ``1/(q+kappa)`` after validating ``kappa in [0,1]``."""
    qq = _positive_integer("q", q)
    kappa = float(curvature)
    if not math.isfinite(kappa) or kappa < -ABS_TOL or kappa > 1.0 + ABS_TOL:
        raise ValueError("curvature must be finite and lie in [0,1]")
    return 1.0 / (qq + min(1.0, max(0.0, kappa)))


@dataclass(frozen=True)
class EpisodeCertificate:
    """Raw and certified observable charges for one episode."""

    exploration: bool
    cap: float
    raw_exploitation_charge: float
    raw_charge: float
    certified_charge: float
    clipped: bool
    clip_excess: float


def certify_episode(
    *,
    exploration: bool,
    raw_exploitation_charge: float,
    f_max: float,
    q: int,
) -> EpisodeCertificate:
    """Certify one episode using exploration charging and exploitation clipping.

    For an exploration episode, the observable charge is exactly ``F_max/q``.
    For an exploitation episode with raw finite-channel charge ``z >= 0``, the
    certified charge is ``min(z, F_max/q)``.  The counterfactual raw
    exploitation charge is retained even on exploration episodes, but it is not
    accumulated by :class:`CertificateLedger`.
    """
    raw_exp = _finite_nonnegative(
        "raw_exploitation_charge", raw_exploitation_charge
    )
    cap = certificate_cap(f_max=f_max, q=q)
    if bool(exploration):
        raw_charge = cap
        certified = cap
        clipped = False
        excess = 0.0
    else:
        raw_charge = raw_exp
        certified = min(raw_exp, cap)
        excess = max(0.0, raw_exp - certified)
        clipped = excess > ABS_TOL
    return EpisodeCertificate(
        exploration=bool(exploration),
        cap=cap,
        raw_exploitation_charge=raw_exp,
        raw_charge=raw_charge,
        certified_charge=certified,
        clipped=clipped,
        clip_excess=excess,
    )


def clipped_transfer_increment(
    *, baseline_raw_charge: float, enlargement: float, cap: float
) -> float:
    """Return the exact episodewise increase under an enlarged fixed-trace radius.

    The identity

    ``min(b+d,c)-min(b,c) = min(d, max(c-b,0))``

    is used with ``b`` the baseline raw exploitation charge, ``d`` the
    nonnegative target-radius enlargement, and ``c=F_max/q``.  Retaining this
    episodewise residual capacity avoids replacing the exact transfer by the
    generally coarser global clipping of cumulative charges.
    """
    baseline = _finite_nonnegative("baseline_raw_charge", baseline_raw_charge)
    extra = _finite_nonnegative("enlargement", enlargement)
    ceiling = _finite_nonnegative("cap", cap)
    return min(extra, max(0.0, ceiling - baseline))


def universal_normalization(
    *, cumulative_charge: float, epoch: int, f_max: float, q: int
) -> float:
    """Return ``q B_k/(k F_max)`` for a clipped cumulative certificate."""
    charge = _finite_nonnegative("cumulative_charge", cumulative_charge)
    kk = _positive_integer("epoch", epoch)
    cap = certificate_cap(f_max=f_max, q=q)
    if cap <= 0.0:
        raise ValueError("f_max must be positive for certificate normalization")
    ratio = charge / (kk * cap)
    if ratio > 1.0 + 5.0e-9:
        raise ValueError(
            f"clipped certificate exceeds universal ceiling: ratio={ratio:.12g}"
        )
    if ratio < 0.0:
        raise ValueError("normalized certificate is negative")
    return min(1.0, max(0.0, ratio))


@dataclass
class CertificateLedger:
    """Cumulative direct or fixed-trace observable-certificate ledger."""

    episodes: int = 0
    exploration_episodes: int = 0
    exploitation_episodes: int = 0
    exploration: float = 0.0
    exploitation: float = 0.0
    raw_exploitation: float = 0.0
    total: float = 0.0
    clip_excess: float = 0.0
    clipped_episodes: int = 0

    def add(self, episode: EpisodeCertificate, *, exploration: bool) -> None:
        if bool(exploration) != episode.exploration:
            raise ValueError("episode branch disagrees with ledger branch")
        self.episodes += 1
        if episode.exploration:
            self.exploration_episodes += 1
            self.exploration += episode.certified_charge
        else:
            self.exploitation_episodes += 1
            self.raw_exploitation += episode.raw_exploitation_charge
            self.exploitation += episode.certified_charge
            self.clip_excess += episode.clip_excess
            self.clipped_episodes += int(episode.clipped)
        self.total += episode.certified_charge

    def check(self, *, f_max: float, q: int, atol: float = 5.0e-9) -> None:
        if self.episodes != self.exploration_episodes + self.exploitation_episodes:
            raise ValueError("episode counts do not close")
        if not 0 <= self.clipped_episodes <= self.exploitation_episodes:
            raise ValueError("clipped-episode count is inconsistent")
        cap = certificate_cap(f_max=f_max, q=q)
        if not math.isclose(
            self.exploration,
            self.exploration_episodes * cap,
            rel_tol=REL_TOL,
            abs_tol=atol,
        ):
            raise ValueError("exploration charge does not equal N_exp F_max/q")
        if self.exploitation > self.raw_exploitation + atol:
            raise ValueError("clipped exploitation total exceeds raw total")
        if not math.isclose(
            self.raw_exploitation - self.exploitation,
            self.clip_excess,
            rel_tol=REL_TOL,
            abs_tol=atol,
        ):
            raise ValueError("clip-excess identity failed")
        if not math.isclose(
            self.total,
            self.exploration + self.exploitation,
            rel_tol=REL_TOL,
            abs_tol=atol,
        ):
            raise ValueError("cumulative certificate decomposition failed")
        universal_normalization(
            cumulative_charge=self.total,
            epoch=self.episodes,
            f_max=f_max,
            q=q,
        )


def _float(row: Mapping[str, Any], field: str, row_number: int) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise CertificateValidationError(
            f"row {row_number}: invalid or missing numeric field {field!r}"
        ) from exc
    if not math.isfinite(value):
        raise CertificateValidationError(
            f"row {row_number}: nonfinite field {field!r}"
        )
    return value


def _int(row: Mapping[str, Any], field: str, row_number: int) -> int:
    value = _float(row, field, row_number)
    if value != int(value):
        raise CertificateValidationError(
            f"row {row_number}: noninteger field {field!r}"
        )
    return int(value)


def _assert_close(
    *, actual: float, expected: float, field: str, row_number: int, atol: float = 5.0e-9
) -> None:
    if not math.isclose(actual, expected, rel_tol=REL_TOL, abs_tol=atol):
        raise CertificateValidationError(
            f"row {row_number}: {field}={actual:.17g}, expected {expected:.17g}"
        )


def replay_certificate_rows(
    rows: Sequence[Mapping[str, Any]], *, prefix: str = ""
) -> dict[str, Any]:
    """Replay one serialized certificate route and reject any altered arithmetic.

    ``prefix`` is ``""`` for the direct route, ``"support_"`` for the
    support-derived fixed-trace route, and ``"prior_"`` for the prior-only
    fixed-trace route.  The common per-episode ceiling is stored once as
    ``certificate_cap``.
    """
    if prefix not in {"", "support_", "prior_"}:
        raise ValueError("unsupported certificate prefix")
    if not rows:
        raise CertificateValidationError("empty certificate trace")

    raw_exp_field = f"{prefix}raw_exploitation_charge"
    raw_inc_field = f"{prefix}raw_observable_bound_increment"
    inc_field = f"{prefix}observable_bound_increment"
    clip_flag_field = (
        "certificate_clip_indicator"
        if not prefix
        else f"{prefix}certificate_clip_indicator"
    )
    clip_excess_field = (
        "certificate_clip_excess"
        if not prefix
        else f"{prefix}certificate_clip_excess"
    )
    cum_exp_field = f"{prefix}cumulative_exploration_bound"
    cum_ucb_field = f"{prefix}cumulative_ucb_bound"
    cum_raw_ucb_field = f"{prefix}cumulative_raw_ucb_bound"
    cum_total_field = f"{prefix}cumulative_observable_bound"
    cum_clip_field = f"{prefix}cumulative_clip_excess"
    cum_count_field = f"{prefix}cumulative_clipped_episodes"
    normalized_field = f"{prefix}universal_normalized_observable_bound"

    ledger = CertificateLedger()
    first_q: int | None = None
    first_f_max: float | None = None
    for index, row in enumerate(rows, start=1):
        row_number = index + 1
        epoch = _int(row, "epoch", row_number)
        if epoch != index:
            raise CertificateValidationError(
                f"row {row_number}: epoch {epoch} is not contiguous"
            )
        q = _int(row, "active_q", row_number)
        f_max = _float(row, "f_max", row_number)
        if first_q is None:
            first_q = q
            first_f_max = f_max
        elif q != first_q or not math.isclose(
            f_max, float(first_f_max), rel_tol=0.0, abs_tol=ABS_TOL
        ):
            raise CertificateValidationError(
                f"row {row_number}: q or F_max changed within one trace"
            )
        explore = bool(_int(row, "exploration_indicator", row_number))
        raw_exp = _float(row, raw_exp_field, row_number)
        episode = certify_episode(
            exploration=explore,
            raw_exploitation_charge=raw_exp,
            f_max=f_max,
            q=q,
        )
        _assert_close(
            actual=_float(row, "certificate_cap", row_number),
            expected=episode.cap,
            field="certificate_cap",
            row_number=row_number,
        )
        _assert_close(
            actual=_float(row, raw_inc_field, row_number),
            expected=episode.raw_charge,
            field=raw_inc_field,
            row_number=row_number,
        )
        _assert_close(
            actual=_float(row, inc_field, row_number),
            expected=episode.certified_charge,
            field=inc_field,
            row_number=row_number,
        )
        if _int(row, clip_flag_field, row_number) != int(episode.clipped):
            raise CertificateValidationError(
                f"row {row_number}: {clip_flag_field} disagrees with replay"
            )
        _assert_close(
            actual=_float(row, clip_excess_field, row_number),
            expected=episode.clip_excess,
            field=clip_excess_field,
            row_number=row_number,
        )

        ledger.add(episode, exploration=explore)
        ledger.check(f_max=f_max, q=q)
        expected_fields = {
            cum_exp_field: ledger.exploration,
            cum_ucb_field: ledger.exploitation,
            cum_raw_ucb_field: ledger.raw_exploitation,
            cum_total_field: ledger.total,
            cum_clip_field: ledger.clip_excess,
            normalized_field: universal_normalization(
                cumulative_charge=ledger.total,
                epoch=epoch,
                f_max=f_max,
                q=q,
            ),
        }
        for field, expected in expected_fields.items():
            _assert_close(
                actual=_float(row, field, row_number),
                expected=expected,
                field=field,
                row_number=row_number,
            )
        if _int(row, cum_count_field, row_number) != ledger.clipped_episodes:
            raise CertificateValidationError(
                f"row {row_number}: {cum_count_field} disagrees with replay"
            )

    assert first_q is not None and first_f_max is not None
    return {
        "route": prefix[:-1] if prefix else "direct",
        "episodes": ledger.episodes,
        "q": first_q,
        "f_max": first_f_max,
        "exploration_episodes": ledger.exploration_episodes,
        "exploitation_episodes": ledger.exploitation_episodes,
        "clipped_episodes": ledger.clipped_episodes,
        "exploration_charge": ledger.exploration,
        "raw_exploitation_charge": ledger.raw_exploitation,
        "clipped_exploitation_charge": ledger.exploitation,
        "observable_certificate": ledger.total,
        "clip_excess": ledger.clip_excess,
        "universal_normalized_certificate": universal_normalization(
            cumulative_charge=ledger.total,
            epoch=ledger.episodes,
            f_max=first_f_max,
            q=first_q,
        ),
        "status": "PASS",
    }
