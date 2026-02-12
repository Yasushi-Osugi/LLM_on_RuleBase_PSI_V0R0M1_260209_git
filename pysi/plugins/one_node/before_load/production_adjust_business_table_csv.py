# pysi/plugins/one_node/before_load/production_adjust_business_table_csv.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def register(bus) -> None:
    # apply_operator_spec(priority=5) の後に走らせる想定
    bus.add_action("one_node.before_load", before_load, priority=8)


def _month_to_iso(s: str) -> str:
    s = (s or "").strip()
    if not s:
        return ""
    s = s.replace("/", "-")
    if len(s) == 7 and s[4] == "-":  # YYYY-MM
        return s
    if len(s) >= 10:  # YYYY-MM-xx → YYYY-MM
        return s[:7]
    return ""


def before_load(**ctx: Any) -> None:
    """
    Reads:
      run_ctx["config"]["production_adjust_business_table"] (dict)
    Writes:
      run_ctx["input_dir"]/<file> (default: business_timeseries.csv)
    Contract:
      - Only modifies INPUT column: purchase_plan
      - Filters by month in params.months
    """
    run_ctx = ctx.get("run_ctx") or {}
    if not isinstance(run_ctx, dict):
        return

    cfg: Dict[str, Any] = run_ctx.get("config") or {}
    pa: Dict[str, Any] = cfg.get("production_adjust_business_table") or {}
    if not isinstance(pa, dict):
        return

    months_raw = pa.get("months") or []
    rate = float(pa.get("rate") or 0.0)
    delta = float(pa.get("delta") or 0.0)

    if not isinstance(months_raw, list) or not months_raw:
        print("[PROD_ADJ_BT] months empty; skip")
        return
    if rate == 0.0 and delta == 0.0:
        print("[PROD_ADJ_BT] rate=0 and delta=0; skip")
        return

    file_name = str(pa.get("file") or "business_timeseries.csv").strip() or "business_timeseries.csv"
    input_dir = Path(str(run_ctx.get("input_dir") or ""))
    ts_path = input_dir / file_name
    if not input_dir.exists() or not ts_path.exists():
        print(f"[PROD_ADJ_BT] missing input: {ts_path}; skip")
        return

    target = {m for m in (_month_to_iso(str(x)) for x in months_raw) if m}
    if not target:
        print("[PROD_ADJ_BT] months parse failed; skip")
        return

    df = pd.read_csv(ts_path)
    need = {"month", "purchase_plan"}
    if not need.issubset(df.columns):
        print(f"[PROD_ADJ_BT] missing cols {sorted(need - set(df.columns))}; skip")
        return

    m = df["month"].astype(str).str.slice(0, 7).isin(target)
    if not m.any():
        print(f"[PROD_ADJ_BT] no target rows for months={sorted(target)}; skip")
        return

    before = df.loc[m, "purchase_plan"].astype(float).copy()
    if delta != 0.0:
        df.loc[m, "purchase_plan"] = before + delta
        mode = f"delta(+{delta})"
    else:
        df.loc[m, "purchase_plan"] = before * (1.0 + rate)
        mode = f"rate(*{1.0 + rate})"

    df.to_csv(ts_path, index=False)
    print(f"[PROD_ADJ_BT] updated purchase_plan {mode} months={sorted(target)} file={file_name}")

