#pysi/plugins/one_node/after_run/export_money_timeseries_from_business_table.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


def register(bus) -> None:
    # operate_writeback_business_table の後に走らせる（大きめpriority）
    # 既存 export_money_timeseries.py(priority=60) の後にもなる
    bus.add_action("one_node.after_run", export_money_timeseries_from_business_table, priority=90)


def _norm_yyyymm_series(s: pd.Series) -> pd.Series:
    # "YYYY-MM" or "YYYY-MM-DD" を "YYYY-MM" へ
    ss = s.astype(str).str.strip()
    ss = ss.str.replace("/", "-", regex=False)
    ss = ss.str.slice(0, 7)
    ss = ss.where(ss.str.match(r"^\d{4}-\d{2}$", na=False), np.nan)
    return ss


def _pick_col(df: pd.DataFrame, candidates: list[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def export_money_timeseries_from_business_table(**ctx: Any) -> Dict[str, Any]:
    """
    Business Table を正として金額計算し、
      1) business_timeseries.csv の金額列を更新（run/input 側）
      2) output/money_timeseries.csv を出力（月次集計）

    前提（ユーザー合意済み）:
      - Shipは cap_S 反映後の値（ship_capped 等）を使う
      - Purchaseは cap_P 反映後の値（purchase_capped 等）を使う
      - promotion_cost は簡易に Revenue * 0.03（または promo_ratio があればそれ）
    """
    run_ctx = ctx.get("run_ctx") or {}
    if not isinstance(run_ctx, dict):
        return ctx

    input_dir = Path(str(run_ctx.get("input_dir") or ""))
    output_dir = Path(str(run_ctx.get("output_dir") or ""))
    if not input_dir.exists() or not output_dir.exists():
        return ctx

    bt_path = input_dir / "business_timeseries.csv"
    if not bt_path.exists():
        print("[BT_MONEY] business_timeseries.csv not found; skip")
        return ctx

    df = pd.read_csv(bt_path, encoding="utf-8-sig")

    if "month" not in df.columns:
        print("[BT_MONEY] missing column: month; skip")
        return ctx

    # drop blank tail rows
    df = df.copy()
    df["month"] = _norm_yyyymm_series(df["month"])
    df = df[df["month"].notna()].copy()

    # --- pick columns (safety first) ---
    ship_col = _pick_col(df, ["ship_actual", "ship_capped", "ship_ctrl", "ship_plan"])
    pur_col = _pick_col(df, ["purchase_actual", "purchase_capped", "purchase_ctrl", "purchase_plan"])
    price_col = _pick_col(df, ["sell_price_eff", "sell_price_plan"])
    upc = _pick_col(df, ["unit_purchase_cost"])
    prc = _pick_col(df, ["unit_process_cost"])
    uhc = _pick_col(df, ["unit_holding_cost"])

    if ship_col is None or pur_col is None or price_col is None:
        print(f"[BT_MONEY] missing qty/price cols ship={ship_col} purchase={pur_col} price={price_col}; skip")
        return ctx
    if upc is None or prc is None:
        print(f"[BT_MONEY] missing unit cost cols unit_purchase_cost={upc} unit_process_cost={prc}; skip")
        return ctx

    # numeric normalize
    def fnum(x: pd.Series) -> pd.Series:
        return pd.to_numeric(x, errors="coerce").fillna(0.0).astype(float)

    ship_qty = fnum(df[ship_col])
    purchase_qty = fnum(df[pur_col])
    sell_price = fnum(df[price_col])
    unit_process_cost = fnum(df[prc])
    unit_purchase_cost = fnum(df[upc])

    # holding_cost: prefer existing holding_cost column, else compute inv_open  unit_holding_cost
    if "holding_cost" in df.columns:
        holding_cost = fnum(df["holding_cost"])
    else:
        inv_open = fnum(df["inv_open"]) if "inv_open" in df.columns else 0.0
        unit_holding_cost = fnum(df[uhc]) if uhc is not None else 0.0
        holding_cost = inv_open * unit_holding_cost

    revenue = ship_qty * sell_price
    process_cost = ship_qty * unit_process_cost
    purchase_cost = purchase_qty * unit_purchase_cost

    # promotion_cost: promo_ratioがあればそれ（0〜1想定）、なければ 3%
    if "promo_ratio" in df.columns:
        promo_ratio = fnum(df["promo_ratio"])
        # 0〜1に軽くクリップ（暴走防止）
        promo_ratio = promo_ratio.clip(lower=0.0, upper=1.0)
        promotion_cost = revenue * promo_ratio
    else:
        promotion_cost = revenue * 0.03

    profit = revenue - process_cost - purchase_cost - holding_cost - promotion_cost
    profit_rate = np.where(revenue > 0, profit / revenue, 0.0)

    # --- write back to Business Table (run/input) ---
    df["revenue"] = revenue
    df["process_cost"] = process_cost
    df["purchase_cost"] = purchase_cost
    df["holding_cost"] = holding_cost
    df["promotion_cost"] = promotion_cost
    df["profit"] = profit
    df["profit_rate"] = profit_rate

    df.to_csv(bt_path, index=False, encoding="utf-8-sig")

    # also keep a copy under output
    out_bt = output_dir / "business_timeseries_money_updated.csv"
    df.to_csv(out_bt, index=False, encoding="utf-8-sig")

    # --- monthly summary -> money_timeseries.csv ---
    
    #@STOP
    #agg_cols = ["revenue", "process_cost", "purchase_cost", "holding_cost", "promotion_cost", "profit"]

    agg_cols = [
        "revenue",
        "process_cost",
        "purchase_cost",
        "holding_cost",
        "promotion_cost",
        "profit",
    ]

    g = df.groupby("month", as_index=False)[agg_cols].sum()
    g = g.rename(
        columns={
            "revenue": "Revenue",
            "process_cost": "Process_Cost",
            "purchase_cost": "Purchase_Cost",
            "holding_cost": "Holding_Cost",
            "promotion_cost": "Promotion_Cost",
            "profit": "Profit",
        }
    )
    # ProfitRate is recomputed from aggregated values (safe)
    g["ProfitRate"] = np.where(g["Revenue"] > 0, g["Profit"] / g["Revenue"], 0.0)

    out_money = output_dir / "money_timeseries.csv"
    g.to_csv(out_money, index=False, encoding="utf-8-sig")

    print(
        f"[BT_MONEY] updated BT money cols and wrote money_timeseries.csv "
        f"(ship={ship_col}, purchase={pur_col}, price={price_col})"
    )
    return ctx


