#pysi/plugins/one_node/after_run/detect_trust_events.py

#簡易テスト（最低限）
#
#python -c "from pysi.plugins.one_node.after_run.detect_trust_events import detect_trust_events; print(detect_trust_events(r'PATH_TO_OUTPUT'))"
#
#実行後に trust_events.json が生成され、events が list、summary.count_total == len(events) になっていること。

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from pysi.core.hooks.core import HookBus


PLUGIN_META = {
    "name": "detect_trust_events",
    "version": "0.1.0",
    "category": "one_node.after_run",
    "inputs": ["one_node_result_timeseries.csv", "kpi_summary.json"],
    "outputs": ["trust_events.json"],
    "description": "Detect trust-related events from one-node output and export trust_events.json.",
}


REQUIRED_COLUMNS = [
    "month",
    "demand",
    "production",
    "sales",
    "inventory",
    "backlog",
    "waste",
    "over_i",
    "cap_P",
    "cap_S",
    "cap_I",
]


def register(bus: HookBus) -> None:
    bus.add_action("one_node.after_run", _detect_trust_events_action, priority=60)


def detect_trust_events(output_dir: str) -> Dict[str, Any]:
    out_dir = Path(output_dir)
    csv_path = out_dir / "one_node_result_timeseries.csv"
    kpi_path = out_dir / "kpi_summary.json"

    if not csv_path.exists():
        raise FileNotFoundError(f"Missing input file: {csv_path}")
    if not kpi_path.exists():
        raise FileNotFoundError(f"Missing input file: {kpi_path}")

    with kpi_path.open("r", encoding="utf-8") as f:
        json.load(f)

    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in reader.fieldnames]
        if missing_cols:
            raise ValueError(
                "Missing required columns in one_node_result_timeseries.csv: "
                + ", ".join(missing_cols)
            )
        rows = list(reader)

    events: List[Dict[str, Any]] = []

    for idx, row in enumerate(rows):
        month = str(row["month"])
        demand = _to_float(row.get("demand"))
        production = _to_float(row.get("production"))
        sales = _to_float(row.get("sales"))
        inventory = _to_float(row.get("inventory"))
        backlog = _to_float(row.get("backlog"))
        waste = _to_float(row.get("waste"))
        over_i = _to_float(row.get("over_i"))
        cap_p = _to_float(row.get("cap_P"))
        cap_i = _to_float(row.get("cap_I"))

        if backlog > 0:
            events.append(
                _make_event(
                    event_id="E_BACKLOG_POSITIVE",
                    severity="high",
                    month=month,
                    metric="backlog",
                    value=backlog,
                    threshold=0.0,
                    direction=">",
                    message="Backlog detected (stockout risk / promise break).",
                    row_index=idx,
                    fields={"backlog": backlog, "demand": demand, "sales": sales},
                )
            )

        if inventory > cap_i:
            events.append(
                _make_event(
                    event_id="E_INVENTORY_CAP_EXCEEDED",
                    severity="medium",
                    month=month,
                    metric="inventory",
                    value=inventory,
                    threshold=cap_i,
                    direction=">",
                    message="Inventory exceeds capacity (storage/quality risk).",
                    row_index=idx,
                    fields={"inventory": inventory, "cap_I": cap_i},
                )
            )

        if over_i > 0:
            events.append(
                _make_event(
                    event_id="E_OVER_I_HIGH",
                    severity="low",
                    month=month,
                    metric="over_i",
                    value=over_i,
                    threshold=0.0,
                    direction=">",
                    message="Over-inventory detected (obsolescence / holding risk).",
                    row_index=idx,
                    fields={"over_i": over_i, "inventory": inventory, "cap_I": cap_i},
                )
            )

        if waste > 0:
            events.append(
                _make_event(
                    event_id="E_WASTE_POSITIVE",
                    severity="high",
                    month=month,
                    metric="waste",
                    value=waste,
                    threshold=0.0,
                    direction=">",
                    message="Waste detected (resource loss / trust hit).",
                    row_index=idx,
                    fields={"waste": waste, "production": production, "inventory": inventory},
                )
            )

        if demand > 0 and (sales / demand) < 0.9:
            ratio = sales / demand
            events.append(
                _make_event(
                    event_id="E_SALES_BELOW_DEMAND",
                    severity="medium",
                    month=month,
                    metric="sales/demand",
                    value=ratio,
                    threshold=0.9,
                    direction="<",
                    message="Sales below demand (service level drop).",
                    row_index=idx,
                    fields={"demand": demand, "sales": sales, "ratio": ratio},
                )
            )

        if cap_p > 0 and (production / cap_p) >= 0.98:
            ratio = production / cap_p
            events.append(
                _make_event(
                    event_id="E_PRODUCTION_NEAR_CAP",
                    severity="low",
                    month=month,
                    metric="production/cap_P",
                    value=ratio,
                    threshold=0.98,
                    direction=">=",
                    message="Production near capacity (fragile supply).",
                    row_index=idx,
                    fields={"production": production, "cap_P": cap_p, "ratio": ratio},
                )
            )

    payload = {
        "schema_version": "0.1",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "inputs": {
            "one_node_result_timeseries": "one_node_result_timeseries.csv",
            "kpi_summary": "kpi_summary.json",
        },
        "events": events,
        "summary": {
            "count_total": len(events),
            "count_high": sum(1 for e in events if e["severity"] == "high"),
            "count_medium": sum(1 for e in events if e["severity"] == "medium"),
            "count_low": sum(1 for e in events if e["severity"] == "low"),
        },
    }

    out_path = out_dir / "trust_events.json"
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _detect_trust_events_action(**ctx: Any) -> None:
    run_dir_raw = str(ctx.get("run_dir", "")).strip()
    if not run_dir_raw:
        return
    run_dir = Path(run_dir_raw)
    detect_trust_events(str(run_dir / "output"))


def _to_float(v: Any) -> float:
    if v in (None, ""):
        return 0.0
    return float(v)


def _make_event(
    *,
    event_id: str,
    severity: str,
    month: str,
    metric: str,
    value: float,
    threshold: float,
    direction: str,
    message: str,
    row_index: int,
    fields: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "event_id": event_id,
        "severity": severity,
        "scope": "monthly",
        "month": month,
        "metric": metric,
        "value": float(value),
        "threshold": float(threshold),
        "direction": direction,
        "message": message,
        "evidence": {
            "row_index": row_index,
            "fields": {k: float(v) for k, v in fields.items()},
        },
    }
