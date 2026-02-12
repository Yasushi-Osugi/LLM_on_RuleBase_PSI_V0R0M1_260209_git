# pysi/plugins/one_node/before_load/business_to_virtual_node_adapter.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


def register(bus) -> None:
    """
    Business Table (wide) -> virtual_node_timeseries.csv (long) adapter.

    Intended order in one_node.before_load:
      1) apply_operator_spec (priority=5)
      2) business-table operators (e.g., production_adjust_business_table..., priority~8)
      3) this adapter (priority=9)  ← generates virtual_node_timeseries.csv
    """
    bus.add_action("one_node.before_load", before_load, priority=9)


def _month_to_yyyymm(s: str) -> Optional[str]:
    s = (s or "").strip().replace("/", "-")
    if not s:
        return None
    # Accept YYYY-MM or YYYY-MM-DD
    if len(s) >= 7 and s[4] == "-":
        return s[:7]
    return None


def _to_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return default
        return float(x)
    except Exception:
        return default


def before_load(**ctx: Any) -> None:
    """
    Reads:
      run_ctx["input_dir"]/business_timeseries.csv (default)
    Writes:
      run_ctx["input_dir"]/virtual_node_timeseries.csv

    Mapping (minimal set for one-node PSI engine):
      demand_plan   -> item="demand"
      ship_plan     -> item="planning_S"
      purchase_plan -> item="production"
      sell_price_plan (if exists) -> item="sell_price"

    Notes:
      - This adapter only *creates/overwrites* virtual_node_timeseries.csv.
      - Business Table itself is NOT modified here (that's Apply/Operate responsibility).
    """
    run_ctx = ctx.get("run_ctx") or {}
    if not isinstance(run_ctx, dict):
        return

    input_dir = Path(str(run_ctx.get("input_dir") or ""))
    if not input_dir.exists():
        print(f"[BT_ADAPTER] input_dir missing: {input_dir}; skip")
        return

    cfg: Dict[str, Any] = run_ctx.get("config") or {}
    ad: Dict[str, Any] = cfg.get("business_to_virtual_node_adapter") or {}

    bt_file = str(ad.get("business_file") or "business_timeseries.csv").strip() or "business_timeseries.csv"
    out_file = str(ad.get("virtual_file") or "virtual_node_timeseries.csv").strip() or "virtual_node_timeseries.csv"

    bt_path = input_dir / bt_file
    out_path = input_dir / out_file

    if not bt_path.exists():
        print(f"[BT_ADAPTER] business table not found: {bt_path}; skip")
        return

    df = pd.read_csv(bt_path)
    if "month" not in df.columns:
        print("[BT_ADAPTER] missing required column: month; skip")
        return

    # Columns in Business Table we can map (sell_price_plan optional)
    required = ["demand_plan", "ship_plan", "purchase_plan"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[BT_ADAPTER] missing required columns: {missing}; skip")
        return

    has_price = "sell_price_plan" in df.columns

    rows: List[Dict[str, Any]] = []
    NODE = "GLOBAL"  # single-node assumption (temporary)

    for _, r in df.iterrows():
        m = _month_to_yyyymm(str(r.get("month", "")))
        if not m:
            continue

        demand = _to_float(r.get("demand_plan", 0.0))
        ship = _to_float(r.get("ship_plan", 0.0))
        prod = _to_float(r.get("purchase_plan", 0.0))

        rows.append({"month": m, "node_name": NODE, "item": "demand", "value": demand})
        rows.append({"month": m, "node_name": NODE, "item": "planning_S", "value": ship})
        rows.append({"month": m, "node_name": NODE, "item": "production", "value": prod})

        if has_price:
            price = _to_float(r.get("sell_price_plan", 0.0))
            rows.append({"month": m, "node_name": NODE, "item": "sell_price", "value": price})

    if not rows:
        print("[BT_ADAPTER] no rows generated; skip")
        return

    out_df = pd.DataFrame(rows, columns=["month", "node_name", "item", "value"])
    out_df = out_df.sort_values(["month", "node_name", "item"]).reset_index(drop=True)

    # Backup existing virtual_node_timeseries.csv once (safe-first)
    if out_path.exists():
        bk = out_path.with_name(out_path.stem + "_BK_from_business.csv")
        try:
            if not bk.exists():
                out_path.replace(bk)
                print(f"[BT_ADAPTER] backup existing virtual_node_timeseries -> {bk.name}")
            else:
                # If backup already exists, do not overwrite it; just overwrite out_path later
                pass
        except Exception as e:
            print(f"[BT_ADAPTER] backup failed ({e}); will overwrite {out_path.name} anyway")

    out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(
        f"[BT_ADAPTER] generated {out_path.name} from {bt_file} "
        f"rows={len(out_df)} months={out_df['month'].nunique()} items={sorted(out_df['item'].unique())}"
    )
