# pysi/plugins/one_node/after_run/operate_writeback_business_table.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd


def register(bus) -> None:
    # export_money_timeseries の後に動かしたいので priority を大きめに
    bus.add_action("one_node.after_run", after_run, priority=90)


def _norm_yyyymm(s: Any) -> str:
    if s is None:
        return ""
    x = str(s).strip().replace("/", "-")
    if len(x) >= 7 and x[4] == "-":
        return x[:7]
    return ""


def _safe_read_csv(p: Path) -> pd.DataFrame:
    return pd.read_csv(p)


def _pick_first_existing(paths: List[Path]) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


def after_run(**ctx: Any) -> None:
    """
    Operate stage:
      - Read PSI result (one_node_result_timeseries.csv) from output_dir
      - Read money result (money_timeseries.csv) from output_dir (if exists)
      - Write back STATE/CALC columns into Business Table (business_timeseries.csv) in input_dir
      - Also write a copy into output_dir for audit: business_timeseries_operated.csv

    We DO NOT touch the original file under data_dir; only run/input and run/output.
    """
    run_ctx = ctx.get("run_ctx") or {}
    if not isinstance(run_ctx, dict):
        return

    input_dir = Path(str(run_ctx.get("input_dir") or ""))
    output_dir = Path(str(run_ctx.get("output_dir") or ""))

    if not input_dir.exists() or not output_dir.exists():
        print(f"[OPERATE_BT] missing dirs input_dir={input_dir} output_dir={output_dir}; skip")
        return

    cfg: Dict[str, Any] = run_ctx.get("config") or {}
    op_cfg: Dict[str, Any] = cfg.get("operate_writeback_business_table") or {}
    if not isinstance(op_cfg, dict):
        op_cfg = {}

    bt_file = str(op_cfg.get("business_file") or "business_timeseries.csv").strip() or "business_timeseries.csv"
    bt_path = input_dir / bt_file
    if not bt_path.exists():
        print(f"[OPERATE_BT] business table not found: {bt_path}; skip")
        return

    # Required result files
    psi_path = output_dir / "one_node_result_timeseries.csv"
    if not psi_path.exists():
        print(f"[OPERATE_BT] missing PSI result: {psi_path}; skip")
        return

    money_path = output_dir / "money_timeseries.csv"  # optional

    bt = _safe_read_csv(bt_path)
    if "month" not in bt.columns:
        print("[OPERATE_BT] business table missing 'month'; skip")
        return

    psi = _safe_read_csv(psi_path)
    need_psi = {"month", "demand", "production", "sales", "inventory", "backlog", "over_i", "cap_P", "cap_S", "cap_I"}
    if not need_psi.issubset(set(psi.columns)):
        print(f"[OPERATE_BT] PSI result missing cols: {sorted(need_psi - set(psi.columns))}; skip")
        return

    money = None
    if money_path.exists():
        money = _safe_read_csv(money_path)
        # money columns are typically: month, Revenue, Process_Cost, Purchase_Cost, Profit, ProfitRate
        if "month" not in money.columns:
            money = None

    # Normalize month keys
    bt["_yyyymm"] = bt["month"].map(_norm_yyyymm)
    psi["_yyyymm"] = psi["month"].map(_norm_yyyymm)
    if money is not None:
        money["_yyyymm"] = money["month"].map(_norm_yyyymm)

    # Build lookup tables
    psi_keyed = psi.set_index("_yyyymm")
    money_keyed = money.set_index("_yyyymm") if money is not None else None

    # Writeback targets (STATE/CALC)
    # - Keep INPUT columns untouched: demand_plan/ship_plan/purchase_plan/sell_price_plan/promotion_spend...
    # - Update computed/actual/state columns when they exist in BT.
    updates: List[Tuple[str, str]] = [
        ("ship_capped", "sales"),
        ("purchase_capped", "production"),
        ("inv_ctrl", "inventory"),
        ("bo_ctrl", "backlog"),
        ("inv_overflow", "over_i"),
        ("cap_P", "cap_P"),
        ("cap_S", "cap_S"),
        ("cap_I", "cap_I"),
    ]

    # Apply PSI writeback
    hit = 0
    for i in range(len(bt)):
        k = bt.at[i, "_yyyymm"]
        if not k or k not in psi_keyed.index:
            continue
        row = psi_keyed.loc[k]

        for bt_col, psi_col in updates:
            if bt_col in bt.columns and psi_col in psi.columns:
                bt.at[i, bt_col] = float(row[psi_col])
        hit += 1

    # Apply money writeback (if BT has these columns)
    # BT columns are lowercase in your sheet: revenue/process_cost/purchase_cost/profit/profit_rate
    if money_keyed is not None:
        money_map = [
            ("revenue", "Revenue"),
            ("process_cost", "Process_Cost"),
            ("purchase_cost", "Purchase_Cost"),
            ("profit", "Profit"),
            ("profit_rate", "ProfitRate"),
        ]
        for i in range(len(bt)):
            k = bt.at[i, "_yyyymm"]
            if not k or k not in money_keyed.index:
                continue
            row = money_keyed.loc[k]
            for bt_col, m_col in money_map:
                if bt_col in bt.columns and m_col in money.columns:
                    bt.at[i, bt_col] = float(row[m_col])

    bt = bt.drop(columns=["_yyyymm"])

    # Persist (overwrite run/input copy) + export audit copy
    bt.to_csv(bt_path, index=False, encoding="utf-8-sig")
    out_copy = output_dir / "business_timeseries_operated.csv"
    bt.to_csv(out_copy, index=False, encoding="utf-8-sig")

    print(
        f"[OPERATE_BT] writeback done rows_hit={hit} "
        f"input_updated={bt_path.name} output_copy={out_copy.name}"
    )
