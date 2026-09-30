"""WIP aging-risk monitoring.

A lot is at aging risk when the time it has spent waiting at a route step
*after* excluding planned holds exceeds the step's versioned aging limit.

The limit bound to a lot is the revision effective when the lot entered the
step (the "当时版本" / grandfathering rule): a limit published later never
rewrites the clock a lot started under.

Events may arrive late or out of order; the timeline is rebuilt from
``occurred_at`` so reconstruction is order-independent.  Overlapping hold
windows are merged before subtraction so paused time is never double
counted.  Incomplete data (a release without a start, an event after the
evaluation time) fails closed instead of producing a quietly wrong number.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from fabweave.core.timebox import TimeBox, covered_duration
from fabweave.core.versioning import RevisionHistory

STEP_ENTERED = "step_entered"
STEP_EXITED = "step_exited"
HOLD_STARTED = "hold_started"
HOLD_RELEASED = "hold_released"
_KINDS = frozenset({STEP_ENTERED, STEP_EXITED, HOLD_STARTED, HOLD_RELEASED})


@dataclass(frozen=True, slots=True)
class AgingLimit:
    """A route step's aging limit under a specific revision."""

    route_step: str
    max_minutes: int
    revision: int

    def __post_init__(self) -> None:
        if not self.route_step:
            raise ValueError("route step must not be empty")
        if self.max_minutes <= 0 or self.revision <= 0:
            raise ValueError("aging limit and revision must be positive")


@dataclass(frozen=True, slots=True)
class WipAgingEvent:
    """An append-only WIP lifecycle event.

    ``hold_id`` pairs a hold start with its release; a release whose start
    has not arrived yet is a late event and is still reconstructed once the
    full stream is sorted by ``occurred_at``.
    """

    lot_id: str
    occurred_at: int
    kind: str
    route_step: str | None = None
    hold_id: str | None = None

    def __post_init__(self) -> None:
        if not self.lot_id:
            raise ValueError("lot id must not be empty")
        if self.occurred_at < 0:
            raise ValueError("event time must be non-negative")
        if self.kind not in _KINDS:
            raise ValueError(f"unknown event kind: {self.kind}")
        if self.kind == STEP_ENTERED and not self.route_step:
            raise ValueError("step_entered requires a route step")
        if self.kind in (HOLD_STARTED, HOLD_RELEASED) and not self.hold_id:
            raise ValueError(f"{self.kind} requires a hold id")


@dataclass(frozen=True, slots=True)
class AgingRecord:
    """Per-lot aging detail: elapsed, paused, effective time and hit rules."""

    lot_id: str
    route_step: str
    entered_at: int
    evaluated_at: int
    elapsed_minutes: int
    paused_minutes: int
    effective_minutes: int
    max_minutes: int
    limit_revision: int
    remaining_minutes: int
    overdue_minutes: int
    risk: float
    due_at: int | None
    remaining_to_due: int | None
    hit_rules: tuple[str, ...]


def _hold_boxes(events: list[WipAgingEvent], evaluated_at: int) -> list[TimeBox]:
    """Pair hold starts with releases; an open hold runs to evaluation time."""

    starts: dict[str, int] = {}
    ends: dict[str, int] = {}
    for event in events:
        if event.kind == HOLD_STARTED:
            # Duplicate start for the same hold is idempotent: keep the first.
            starts.setdefault(event.hold_id, event.occurred_at)
        elif event.kind == HOLD_RELEASED:
            if event.hold_id not in starts:
                raise ValueError(f"hold_released for {event.hold_id} arrived without a matching start")
            # Duplicate release for the same hold is idempotent: keep the first.
            ends.setdefault(event.hold_id, event.occurred_at)

    boxes: list[TimeBox] = []
    for hold_id, start in starts.items():
        end = ends.get(hold_id, evaluated_at)
        if end < start:
            raise ValueError(f"hold {hold_id} ends before it starts")
        if end > start:
            boxes.append(TimeBox(start, end))
    return boxes


def build_aging_record(
    lot_id: str,
    events: list[WipAgingEvent],
    limits: Mapping[str, RevisionHistory[AgingLimit]],
    evaluated_at: int,
    due_at: int | None = None,
) -> AgingRecord | None:
    """Rebuild one lot's aging clock from its (possibly late) event stream.

    Returns ``None`` when the lot has left the step (no open ``step_entered``).
    The limit used is the revision effective at ``entered_at``.
    """

    if evaluated_at < 0:
        raise ValueError("evaluation time must be non-negative")

    lot_events = sorted(
        (event for event in events if event.lot_id == lot_id),
        key=lambda event: event.occurred_at,
    )
    if not lot_events:
        return None
    if any(event.occurred_at > evaluated_at for event in lot_events):
        raise ValueError(f"event for {lot_id} occurs after evaluation time")

    entered: WipAgingEvent | None = None
    seen_entered = False
    for event in lot_events:
        if event.kind == STEP_ENTERED:
            entered = event
            seen_entered = True
        elif event.kind == STEP_EXITED:
            if entered is not None:
                entered = None
            elif not seen_entered:
                raise ValueError(f"step_exited for {lot_id} without a preceding step_entered")
            # Duplicate exit after the lot already left: idempotent no-op.

    if entered is None:
        return None

    route_step = entered.route_step
    entered_at = entered.occurred_at
    storage = TimeBox(entered_at, evaluated_at)
    paused = covered_duration(_hold_boxes(lot_events, evaluated_at), storage)
    elapsed = evaluated_at - entered_at
    effective = elapsed - paused

    if route_step not in limits:
        raise LookupError(f"no aging limits defined for step {route_step}")
    limit = limits[route_step].at(entered_at).value

    remaining = limit.max_minutes - effective
    overdue = max(0, effective - limit.max_minutes)
    risk = effective / limit.max_minutes
    hit = effective > limit.max_minutes
    rules = (f"aging_limit_exceeded:{route_step}",) if hit else ()

    return AgingRecord(
        lot_id=lot_id,
        route_step=route_step,
        entered_at=entered_at,
        evaluated_at=evaluated_at,
        elapsed_minutes=elapsed,
        paused_minutes=paused,
        effective_minutes=effective,
        max_minutes=limit.max_minutes,
        limit_revision=limit.revision,
        remaining_minutes=remaining,
        overdue_minutes=overdue,
        risk=risk,
        due_at=due_at,
        remaining_to_due=None if due_at is None else due_at - evaluated_at,
        hit_rules=rules,
    )


def rank_aging_risk(
    records: list[AgingRecord],
    *,
    only_breached: bool = True,
) -> list[AgingRecord]:
    """Stably rank breached lots: risk desc, remaining due asc, lot id asc.

    ``lot_id`` is an explicit tie-breaker so the order is deterministic even
    when two lots share the same risk and due date.  Lots without a due date
    sort after lots with one.
    """

    candidates = [record for record in records if record.hit_rules] if only_breached else list(records)
    return sorted(
        candidates,
        key=lambda record: (
            -record.risk,
            record.remaining_to_due is None,
            record.remaining_to_due if record.remaining_to_due is not None else 0,
            record.lot_id,
        ),
    )


def aging_risk_report(
    events: list[WipAgingEvent],
    limits: Mapping[str, RevisionHistory[AgingLimit]],
    evaluated_at: int,
    due_dates: Mapping[str, int] | None = None,
) -> list[AgingRecord]:
    """Group a WIP event stream by lot, rebuild each clock and rank the risks."""

    by_lot: dict[str, list[WipAgingEvent]] = {}
    for event in events:
        by_lot.setdefault(event.lot_id, []).append(event)

    records: list[AgingRecord] = []
    for lot_id in sorted(by_lot):
        record = build_aging_record(
            lot_id,
            by_lot[lot_id],
            limits,
            evaluated_at,
            (due_dates or {}).get(lot_id),
        )
        if record is not None:
            records.append(record)
    return rank_aging_risk(records)
