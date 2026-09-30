from __future__ import annotations

import pytest

from fabweave.aging.monitor import (
    HOLD_RELEASED,
    HOLD_STARTED,
    STEP_ENTERED,
    STEP_EXITED,
    AgingLimit,
    WipAgingEvent,
    aging_risk_report,
    build_aging_record,
    rank_aging_risk,
)
from fabweave.core.versioning import RevisionHistory


def _limits(*specs: tuple[str, int, int]) -> dict[str, RevisionHistory[AgingLimit]]:
    histories: dict[str, RevisionHistory[AgingLimit]] = {}
    for step, max_minutes, revision in specs:
        history = histories.setdefault(step, RevisionHistory())
        history.publish(revision, 0, AgingLimit(step, max_minutes, revision))
    return histories


def _enter(lot: str, step: str, at: int) -> WipAgingEvent:
    return WipAgingEvent(lot, at, STEP_ENTERED, route_step=step)


def _exit(lot: str, at: int) -> WipAgingEvent:
    return WipAgingEvent(lot, at, STEP_EXITED)


def _hold(lot: str, hold_id: str, start: int, end: int | None = None) -> list[WipAgingEvent]:
    events = [WipAgingEvent(lot, start, HOLD_STARTED, hold_id=hold_id)]
    if end is not None:
        events.append(WipAgingEvent(lot, end, HOLD_RELEASED, hold_id=hold_id))
    return events


def test_within_limit_is_not_breached():
    record = build_aging_record("LOT-A", [_enter("LOT-A", "ETCH", 0)], _limits(("ETCH", 100, 1)), 80)
    assert record is not None
    assert (record.elapsed_minutes, record.paused_minutes, record.effective_minutes) == (80, 0, 80)
    assert record.remaining_minutes == 20 and record.overdue_minutes == 0
    assert record.hit_rules == () and record.risk == pytest.approx(0.8)


def test_effective_time_over_limit_breaches_and_reports_rule():
    record = build_aging_record("LOT-A", [_enter("LOT-A", "ETCH", 0)], _limits(("ETCH", 100, 1)), 150)
    assert record is not None
    assert record.effective_minutes == 150 and record.overdue_minutes == 50
    assert record.hit_rules == ("aging_limit_exceeded:ETCH",)
    assert record.risk == pytest.approx(1.5)


def test_planned_hold_time_is_excluded_from_effective_clock():
    events = [_enter("LOT-A", "ETCH", 0), *_hold("LOT-A", "H1", 20, 50)]
    record = build_aging_record("LOT-A", events, _limits(("ETCH", 100, 1)), 120)
    assert record is not None
    assert record.elapsed_minutes == 120 and record.paused_minutes == 30
    assert record.effective_minutes == 90
    assert record.hit_rules == () and record.remaining_minutes == 10


def test_overlapping_holds_are_merged_not_double_counted():
    events = [
        _enter("LOT-A", "ETCH", 0),
        *_hold("LOT-A", "H1", 20, 60),
        *_hold("LOT-A", "H2", 40, 80),
    ]
    record = build_aging_record("LOT-A", events, _limits(("ETCH", 100, 1)), 120)
    assert record is not None
    # Union of [20,60) and [40,80) is [20,80): 60 paused minutes, not 80.
    assert record.paused_minutes == 60 and record.effective_minutes == 60


def test_late_out_of_order_events_reconstruct_the_same_timeline():
    ordered = [_enter("LOT-A", "ETCH", 0), *_hold("LOT-A", "H1", 20, 50)]
    shuffled = [*_hold("LOT-A", "H1", 20, 50), _enter("LOT-A", "ETCH", 0)]
    limits = _limits(("ETCH", 100, 1))
    first = build_aging_record("LOT-A", ordered, limits, 120)
    second = build_aging_record("LOT-A", shuffled, limits, 120)
    assert first == second


def test_open_hold_runs_to_evaluation_time():
    events = [_enter("LOT-A", "ETCH", 0), *_hold("LOT-A", "H1", 30)]
    record = build_aging_record("LOT-A", events, _limits(("ETCH", 100, 1)), 120)
    assert record is not None
    assert record.paused_minutes == 90 and record.effective_minutes == 30


def test_limit_revision_is_bound_at_entry_not_at_evaluation():
    histories: dict[str, RevisionHistory[AgingLimit]] = {"ETCH": RevisionHistory()}
    histories["ETCH"].publish(1, 0, AgingLimit("ETCH", 100, 1))
    histories["ETCH"].publish(2, 60, AgingLimit("ETCH", 40, 2))
    # Entered at t=0 under the 100-minute limit; the tighter t=60 limit must
    # not retroactively apply.
    record = build_aging_record("LOT-A", [_enter("LOT-A", "ETCH", 0)], histories, 120)
    assert record is not None
    assert record.max_minutes == 100 and record.limit_revision == 1
    assert record.effective_minutes == 120 and record.overdue_minutes == 20


def test_exited_lot_is_no_longer_in_wip():
    events = [_enter("LOT-A", "ETCH", 0), _exit("LOT-A", 50)]
    assert build_aging_record("LOT-A", events, _limits(("ETCH", 100, 1)), 120) is None


def test_reentered_lot_uses_the_latest_entry():
    events = [_enter("LOT-A", "ETCH", 0), _exit("LOT-A", 20), _enter("LOT-A", "ETCH", 30)]
    record = build_aging_record("LOT-A", events, _limits(("ETCH", 100, 1)), 120)
    assert record is not None and record.entered_at == 30 and record.effective_minutes == 90


def test_duplicate_hold_events_are_idempotent():
    events = [
        _enter("LOT-A", "ETCH", 0),
        WipAgingEvent("LOT-A", 20, HOLD_STARTED, hold_id="H1"),
        WipAgingEvent("LOT-A", 20, HOLD_STARTED, hold_id="H1"),
        WipAgingEvent("LOT-A", 50, HOLD_RELEASED, hold_id="H1"),
        WipAgingEvent("LOT-A", 50, HOLD_RELEASED, hold_id="H1"),
    ]
    record = build_aging_record("LOT-A", events, _limits(("ETCH", 100, 1)), 120)
    assert record is not None and record.paused_minutes == 30


def test_hold_entirely_before_entry_is_clipped_to_zero():
    events = [*_hold("LOT-A", "H1", 0, 10), _enter("LOT-A", "ETCH", 20)]
    record = build_aging_record("LOT-A", events, _limits(("ETCH", 100, 1)), 120)
    assert record is not None and record.paused_minutes == 0


def test_future_event_fails_closed():
    events = [_enter("LOT-A", "ETCH", 0), WipAgingEvent("LOT-A", 200, HOLD_STARTED, hold_id="H1")]
    with pytest.raises(ValueError):
        build_aging_record("LOT-A", events, _limits(("ETCH", 100, 1)), 120)


def test_hold_release_without_matching_start_fails_closed():
    events = [_enter("LOT-A", "ETCH", 0), WipAgingEvent("LOT-A", 50, HOLD_RELEASED, hold_id="H1")]
    with pytest.raises(ValueError):
        build_aging_record("LOT-A", events, _limits(("ETCH", 100, 1)), 120)


def test_exit_without_entry_fails_closed():
    events = [_exit("LOT-A", 50)]
    with pytest.raises(ValueError):
        build_aging_record("LOT-A", events, _limits(("ETCH", 100, 1)), 120)


def test_unknown_step_fails_closed():
    with pytest.raises(LookupError):
        build_aging_record("LOT-A", [_enter("LOT-A", "CMP", 0)], _limits(("ETCH", 100, 1)), 120)


def test_rank_orders_by_risk_then_due_then_lot_stably():
    records = [
        build_aging_record("LOT-C", [_enter("LOT-C", "ETCH", 0)], _limits(("ETCH", 100, 1)), 120, due_at=200),
        build_aging_record("LOT-A", [_enter("LOT-A", "ETCH", 0)], _limits(("ETCH", 100, 1)), 120, due_at=150),
        build_aging_record("LOT-B", [_enter("LOT-B", "ETCH", 0)], _limits(("ETCH", 100, 1)), 120, due_at=150),
        build_aging_record("LOT-D", [_enter("LOT-D", "ETCH", 0)], _limits(("ETCH", 100, 1)), 105, due_at=150),
    ]
    ranked = rank_aging_risk(records)
    # LOT-A/B/C all at 1.2x risk: earliest due first, then lot id.
    # LOT-D at 1.05x risk ranks last.
    assert [record.lot_id for record in ranked] == ["LOT-A", "LOT-B", "LOT-C", "LOT-D"]


def test_rank_without_due_date_sorts_after_dated_lots():
    records = [
        build_aging_record("LOT-Z", [_enter("LOT-Z", "ETCH", 0)], _limits(("ETCH", 100, 1)), 120),
        build_aging_record("LOT-A", [_enter("LOT-A", "ETCH", 0)], _limits(("ETCH", 100, 1)), 120, due_at=150),
    ]
    ranked = rank_aging_risk(records)
    assert [record.lot_id for record in ranked] == ["LOT-A", "LOT-Z"]


def test_rank_can_include_watch_list():
    records = [
        build_aging_record("LOT-HOT", [_enter("LOT-HOT", "ETCH", 0)], _limits(("ETCH", 100, 1)), 120),
        build_aging_record("LOT-COOL", [_enter("LOT-COOL", "ETCH", 0)], _limits(("ETCH", 100, 1)), 50),
    ]
    assert [record.lot_id for record in rank_aging_risk(records, only_breached=False)] == ["LOT-HOT", "LOT-COOL"]
    assert [record.lot_id for record in rank_aging_risk(records)] == ["LOT-HOT"]


def test_report_groups_lots_and_ranks_breached():
    events = [
        _enter("LOT-A", "ETCH", 0),
        *_hold("LOT-A", "H1", 20, 50),
        _enter("LOT-B", "ETCH", 0),
        _enter("LOT-C", "ETCH", 0),
        _exit("LOT-C", 30),
    ]
    due = {"LOT-A": 140, "LOT-B": 200}
    report = aging_risk_report(events, _limits(("ETCH", 100, 1)), 120, due)
    # LOT-A: 120 elapsed - 30 hold = 90 effective, not breached.
    # LOT-B: 120 effective, breached. LOT-C: exited, excluded.
    assert [record.lot_id for record in report] == ["LOT-B"]
    record = report[0]
    assert record.elapsed_minutes == 120 and record.paused_minutes == 0
    assert record.effective_minutes == 120 and record.overdue_minutes == 20
    assert record.remaining_to_due == 80 and record.hit_rules == ("aging_limit_exceeded:ETCH",)


def test_report_ignores_events_from_the_future_in_arrival_order():
    # Events arrive in a scrambled order with a late hold; reconstruction must
    # be identical to the ordered stream.
    events = [
        _enter("LOT-A", "ETCH", 0),
        WipAgingEvent("LOT-A", 50, HOLD_RELEASED, hold_id="H1"),
        WipAgingEvent("LOT-A", 20, HOLD_STARTED, hold_id="H1"),
    ]
    record = build_aging_record("LOT-A", events, _limits(("ETCH", 100, 1)), 120)
    assert record is not None
    assert record.paused_minutes == 30 and record.effective_minutes == 90
    assert record.hit_rules == ()
