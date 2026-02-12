#pysi/plugins/one_node/before_load/demand_adjust_csv.py

# pysi/plugins/one_node/before_load/demand_adjust_csv.py
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


def register(bus):
    bus.add_action("one_node.before_load", before_load, priority=10)


def _find_col(df: pd.DataFrame, candidates: Sequence[str]) -> Optional[str]:
    """
    Find a column by case-insensitive exact match first,
    then by "contains" match as a fallback.
    """
    cols = list(df.columns)
    lower_map = {c.lower(): c for c in cols}

    # exact match
    for key in candidates:
        if key.lower() in lower_map:
            return lower_map[key.lower()]

    # contains fallback
    for c in cols:
        cl = c.lower()
        for key in candidates:
            if key.lower() in cl:
                return c

    return None


def before_load(**ctx):
    """
    CSV-level demand adjustment before model load.
    Supports BOTH:
      A) wide format: month + demand column
      B) long format: month, item, value (item indicates demand)

    Also writes a per-run report:
      meta/demand_adjust_report.json
    """
    run_ctx = ctx.get("run_ctx", {}) or {}

    meta_dir = Path(run_ctx["meta_dir"])
    input_dir = Path(run_ctx["input_dir"])
    original_data_dir = Path(run_ctx.get("data_dir") or "")

    if not original_data_dir.exists():
        print(f"[before_load] original data_dir not found: {original_data_dir}, skip")
        return

    meta_path = meta_dir / "run_meta.json"
    if not meta_path.exists():
        print("[before_load] run_meta.json not found, skip")
        return

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    cfg = meta.get("config", {})
    da = cfg.get("demand_adjust", {}) or {}

    months = da.get("months", []) or []
    rate = da.get("rate", 0.0)
    try:
        rate = float(rate)
    except Exception:
        print(f"[before_load] invalid rate={rate}, skip")
        return

    if (not months) or (rate == 0.0):
        print("[before_load] no demand_adjust config (months empty or rate=0), skip")
        return

    input_dir.mkdir(parents=True, exist_ok=True)

    # copy all CSVs to input_dir
    copied = 0
    for p in original_data_dir.glob("*.csv"):
        shutil.copy(p, input_dir / p.name)
        copied += 1

    ts_name = da.get("timeseries_name", "virtual_node_timeseries.csv")
    ts_path = input_dir / ts_name
    if not ts_path.exists():
        print(f"[before_load] {ts_name} not found, skip")
        return

    df = pd.read_csv(ts_path)

    # Common columns
    month_col = _find_col(df, ("month", "ym", "year_month", "yyyymm"))
    if not month_col:
        print(f"[before_load] month column not found. cols={list(df.columns)}")
        return

    # Detect long format first: item/value
    item_col = _find_col(df, ("item", "metric", "kind", "type"))
    value_col = _find_col(df, ("value", "val", "amount", "qty"))
    node_col = _find_col(df, ("node_name", "node", "location", "site"))

    factor = 1.0 + rate
    touched = 0
    changes: List[Dict[str, Any]] = []

    if item_col and value_col:
        # long format: adjust rows where item indicates demand
        demand_items = da.get("items", ["demand", "demand_lots", "demand_qty"])
        demand_items_lc = {str(x).lower() for x in demand_items}

        # ensure numeric dtype (avoid dtype warnings when multiplying)
        df[value_col] = pd.to_numeric(df[value_col], errors="coerce").astype(float)

        for m in months:
            mask_m = df[month_col].astype(str) == str(m)
            mask_d = df[item_col].astype(str).str.lower().isin(demand_items_lc)
            mask = mask_m & mask_d
            if mask.any():
                # capture before
                before_vals = df.loc[mask, value_col].tolist()
                # apply
                df.loc[mask, value_col] = df.loc[mask, value_col] * factor
                after_vals = df.loc[mask, value_col].tolist()

                # build per-row changes (keep index order)
                idxs = list(df.index[mask])
                for i, ridx in enumerate(idxs):
                    rec: Dict[str, Any] = {
                        "month": str(df.at[ridx, month_col]),
                        "item": str(df.at[ridx, item_col]),
                        "before": float(before_vals[i]) if i < len(before_vals) else None,
                        "after": float(after_vals[i]) if i < len(after_vals) else None,
                    }
                    if node_col:
                        rec["node_name"] = str(df.at[ridx, node_col])
                    changes.append(rec)

                touched += int(mask.sum())

        df.to_csv(ts_path, index=False)

        print(
            f"[before_load.demand_adjust_csv] fmt=long copied_csv={copied}, file={ts_name}, "
            f"months={months}, rate={rate}, rows_touched={touched}, "
            f"item_col={item_col}, value_col={value_col}"
        )

        # write report to meta/
        try:
            report = {
                "plugin": "one_node.before_load.demand_adjust_csv",
                "format": "long",
                "timeseries_name": ts_name,
                "months": months,
                "rate": rate,
                "factor": factor,
                "matched_items": list(demand_items),
                "columns": {
                    "month_col": month_col,
                    "node_col": node_col,
                    "item_col": item_col,
                    "value_col": value_col,
                },
                "rows_touched": touched,
                "changes": changes,
            }
            (meta_dir / "demand_adjust_report.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            print(f"[before_load.demand_adjust_csv] report -> {meta_dir / 'demand_adjust_report.json'}")
        except Exception as e:
            print(f"[before_load.demand_adjust_csv] report write failed: {e}")

        run_ctx["data_dir_effective"] = str(input_dir)
        return

    # Otherwise: wide format
    demand_col = _find_col(df, ("demand", "demand_lots", "demand_qty", "sales", "sales_lots"))
    if not demand_col:
        print(f"[before_load] demand column not found (wide fmt). cols={list(df.columns)}")
        return

    for m in months:
        mask = df[month_col].astype(str) == str(m)
        if mask.any():
            # capture before
            before_vals = pd.to_numeric(df.loc[mask, demand_col], errors="coerce").astype(float).tolist()

            df.loc[mask, demand_col] = (
                pd.to_numeric(df.loc[mask, demand_col], errors="coerce").astype(float) * factor
            )

            after_vals = pd.to_numeric(df.loc[mask, demand_col], errors="coerce").astype(float).tolist()

            idxs = list(df.index[mask])
            for i, ridx in enumerate(idxs):
                rec2: Dict[str, Any] = {
                    "month": str(df.at[ridx, month_col]),
                    "before": float(before_vals[i]) if i < len(before_vals) else None,
                    "after": float(after_vals[i]) if i < len(after_vals) else None,
                }
                if node_col:
                    rec2["node_name"] = str(df.at[ridx, node_col])
                changes.append(rec2)

            touched += int(mask.sum())

    df.to_csv(ts_path, index=False)
    print(
        f"[before_load.demand_adjust_csv] fmt=wide copied_csv={copied}, file={ts_name}, "
        f"months={months}, rate={rate}, rows_touched={touched}, demand_col={demand_col}"
    )

    # write report to meta/
    try:
        report = {
            "plugin": "one_node.before_load.demand_adjust_csv",
            "format": "wide",
            "timeseries_name": ts_name,
            "months": months,
            "rate": rate,
            "factor": factor,
            "columns": {
                "month_col": month_col,
                "node_col": node_col,
                "demand_col": demand_col,
            },
            "rows_touched": touched,
            "changes": changes,
        }
        (meta_dir / "demand_adjust_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[before_load.demand_adjust_csv] report -> {meta_dir / 'demand_adjust_report.json'}")
    except Exception as e:
        print(f"[before_load.demand_adjust_csv] report write failed: {e}")

    run_ctx["data_dir_effective"] = str(input_dir)
