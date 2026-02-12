#pysi/plugins/one_node/before_load/price_adjust_business_table_csv.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd


def register(bus) -> None:
    bus.add_action("one_node.before_load", before_load, priority=8)


def _norm_yyyymm(x: Any) -> str:
    s = ("" if x is None else str(x)).strip().replace("/", "-")
    if len(s) >= 7 and s[4] == "-":
        return s[:7]
    return ""


def before_load(**ctx: Any) -> None:
    run_ctx = ctx.get("run_ctx") or {}
    if not isinstance(run_ctx, dict):
        return

    cfg: Dict[str, Any] = run_ctx.get("config") or {}
    pa: Dict[str, Any] = cfg.get("price_adjust_business_table") or {}
    if not isinstance(pa, dict):
        return

    enabled = bool(pa.get("enabled", True))
    if not enabled:
        print("[PRICE_ADJ_BT] disabled; skip")
        return

    months = pa.get("months") or []
    rate = float(pa.get("rate", 0.0) or 0.0)
    delta = float(pa.get("delta", 0.0) or 0.0)

    if not isinstance(months, list) or not months:
        print("[PRICE_ADJ_BT] months empty; skip")
        return
    if rate == 0.0 and delta == 0.0:
        print("[PRICE_ADJ_BT] rate=0 and delta=0; skip")
        return

    file_name = str(pa.get("file") or "business_timeseries.csv").strip() or "business_timeseries.csv"
    input_dir = Path(str(run_ctx.get("input_dir") or ""))
    bt_path = input_dir / file_name
    if not bt_path.exists():
        print(f"[PRICE_ADJ_BT] not found: {bt_path.name}; skip")
        return

    df = pd.read_csv(bt_path)
    if "month" not in df.columns or "sell_price_plan" not in df.columns:
        print("[PRICE_ADJ_BT] missing required columns: month/sell_price_plan; skip")
        return

    target = {_norm_yyyymm(m) for m in months if _norm_yyyymm(m)}
    if not target:
        print("[PRICE_ADJ_BT] months parse failed; skip")
        return

    m = df["month"].astype(str).str.slice(0, 7).isin(target)
    if not m.any():
        print(f"[PRICE_ADJ_BT] no target rows for months={sorted(target)}; skip")
        return

    before = df.loc[m, "sell_price_plan"].astype(float).copy()
    if delta != 0.0:
        df.loc[m, "sell_price_plan"] = before + delta
        mode = f"delta(+{delta})"
    else:
        df.loc[m, "sell_price_plan"] = before * (1.0 + rate)
        mode = f"rate(*{1.0 + rate})"

    df.to_csv(bt_path, index=False, encoding="utf-8-sig")
    print(f"[PRICE_ADJ_BT] updated sell_price_plan {mode} months={sorted(target)} file={file_name}")
