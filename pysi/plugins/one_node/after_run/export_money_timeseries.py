# pysi/plugins/one_node/after_run/export_money_timeseries.py
# pysi/plugins/one_node/after_run/export_money_timeseries.py
from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import json

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _norm_month_series(s: pd.Series) -> pd.Series:
    """
    Normalize month strings into YYYY-MM.
    Accepts: '2025-1', '2025-01', 'Jan-25'
    """
    x = s.astype(str).str.strip()

    # 1) YYYY-M -> YYYY-0M
    x = x.str.replace(r"^(\d{4})-(\d{1})$", r"\1-0\2", regex=True)

    # 2) Try parse formats
    m1 = pd.to_datetime(x, format="%Y-%m", errors="coerce")
    m2 = pd.to_datetime(x, format="%b-%y", errors="coerce")
    
    dt = m1.fillna(m2)
    
    if dt.isna().any():
        bad = x[dt.isna()].unique()[:5]
        # ログ出力に留めるか、必要なら raise してください
        print(f"[export_money_timeseries] month parse failed. examples={bad}")
        return x

    return dt.dt.strftime("%Y-%m")

def _mark_artifact(run_dir: Path, meta_dir: Path, output_dir: Path, key: str, artifact_path: Path) -> None:
    """
    Update meta/run_meta.json to include artifacts.
    """
    run_meta_path = meta_dir / "run_meta.json"
    if not run_meta_path.exists():
        return

    try:
        meta = json.loads(run_meta_path.read_text(encoding="utf-8"))
    except Exception:
        return

    artifacts = meta.get("artifacts", {}) or {}
    artifacts[key] = str(artifact_path)
    meta["artifacts"] = artifacts

    payload = json.dumps(meta, ensure_ascii=False, indent=2)
    run_meta_path.write_text(payload, encoding="utf-8")

    out_run_meta = output_dir / "run_meta.json"
    if out_run_meta.exists():
        out_run_meta.write_text(payload, encoding="utf-8")

# -----------------------------------------------------------------------------
# Data Loading & Processing
# -----------------------------------------------------------------------------
def _load_unit_table(unit_path: Path) -> pd.DataFrame:
    """
    Read unit price/cost table and normalize item names.
    """
    df = pd.read_csv(unit_path)
    required = {"month", "item", "value"}
    if not required.issubset(set(df.columns)):
        raise ValueError(f"{unit_path} missing columns {required}")

    df["month"] = _norm_month_series(df["month"])
    df["item"] = df["item"].astype(str).str.strip()

    # BK版に合わせたエイリアス設定
    alias = {
        "price": "unit_price",
        "sell_price": "unit_price",
        "unit_sell_price": "unit_price",
        "process_cost": "unit_process_cost",
        "unit_pro_cost": "unit_process_cost",
        "pro_cost": "unit_process_cost",
        "purchase_cost": "unit_purchase_cost",
        "unit_pur_cost": "unit_purchase_cost",
        "pur_cost": "unit_purchase_cost",
    }
    df["item"] = df["item"].replace(alias)

    # Pivot to wide format
    wide = df.pivot_table(index="month", columns="item", values="value", aggfunc="first").reset_index()
    
    # 欠損カラムの補完
    for c in ["unit_price", "unit_process_cost", "unit_purchase_cost"]:
        if c not in wide.columns:
            wide[c] = 0.0
            
    return wide

def _compute_money(ts: pd.DataFrame, unit: pd.DataFrame) -> pd.DataFrame:
    """
    Core logic from BK version.
    """
    # 月で結合
    df = ts.merge(unit, on="month", how="left")

    # データ型の変換と欠損埋め
    sales = df["sales"].fillna(0).astype(float)
    production = df["production"].fillna(0).astype(float)
    unit_price = df["unit_price"].fillna(0).astype(float)
    unit_proc = df["unit_process_cost"].fillna(0).astype(float)
    unit_pur = df["unit_purchase_cost"].fillna(0).astype(float)

    # --- 計算ロジック ---
    # Revenue = Sales * unit_price
    revenue = sales * unit_price

    # Process_Cost = Sales * unit_proc (Ship_Qty基準)
    process_cost = sales * unit_proc

    # Purchase_Cost = Production * unit_pur
    purchase_cost = production * unit_pur

    profit = revenue - process_cost - purchase_cost

    # ProfitRate (分母0回避)
    profit_rate = np.where(revenue != 0, profit / revenue, 0.0)

    return pd.DataFrame({
        "month": df["month"],
        "Revenue": revenue,
        "Process_Cost": process_cost,
        "Purchase_Cost": purchase_cost,
        "Profit": profit,
        "ProfitRate": profit_rate,
    })


def _load_unit_from_timeseries(ts: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    If timeseries already has price/cost columns, build a unit table from it.
    Expected columns (either name is acceptable):
      - unit_price or sell_price
      - unit_process_cost (optional)
      - unit_purchase_cost (optional)
    """
    cols = set(ts.columns)
    if "unit_price" in cols:
        up = "unit_price"
    elif "sell_price" in cols:
        up = "sell_price"
    else:
        return None

    unit = pd.DataFrame({
        "month": ts["month"],
        "unit_price": ts[up].fillna(0).astype(float),
    })
    # optional unit costs
    if "unit_process_cost" in cols:
        unit["unit_process_cost"] = ts["unit_process_cost"].fillna(0).astype(float)
    else:
        unit["unit_process_cost"] = 0.0
    if "unit_purchase_cost" in cols:
        unit["unit_purchase_cost"] = ts["unit_purchase_cost"].fillna(0).astype(float)
    else:
        unit["unit_purchase_cost"] = 0.0

    # keep first (month-level)
    unit = unit.groupby("month", as_index=False).first()
    return unit


# -----------------------------------------------------------------------------
# Hook entry
# -----------------------------------------------------------------------------
def export_money_timeseries(**ctx: Any) -> Dict[str, Any]:
    run_dir = Path(str(ctx.get("run_dir", "")))
    if not run_dir:
        return ctx

    meta_dir = Path(str(ctx.get("meta_dir", run_dir / "meta")))
    out_dir = run_dir / "output"
    in_dir = run_dir / "input"
    data_dir = Path(str(ctx.get("data_dir_effective", ctx.get("data_dir", "")) or ""))

    # 1) パスの特定 (BK版の探索ロジックを統合)
    ts_path = out_dir / "one_node_result_timeseries.csv"
    
    # 候補から単価ファイルを探す
    unit_candidates = [
        in_dir / "virtual_node_unit_price_cost.csv",
        data_dir / "virtual_node_unit_price_cost.csv",
        in_dir / "virtual_node_timeseries_unit_Price_Cost.csv",
        data_dir / "virtual_node_timeseries_unit_Price_Cost.csv",
    ]
    unit_path = next((p for p in unit_candidates if p.exists()), None)

    #@STOP
    #if not ts_path.exists() or unit_path is None:
    #    print(f"[export_money_timeseries] missing input files. skip.")
    #    return ctx
    
    if not ts_path.exists():
        print(f"[export_money_timeseries] missing timeseries. skip.")
        return ctx

    # 2) 読み込みと計算
    ts = pd.read_csv(ts_path)
    if "month" not in ts.columns or "sales" not in ts.columns:
        print(f"[export_money_timeseries] missing columns in timeseries. skip.")
        return ctx
    
    ts["month"] = _norm_month_series(ts["month"])

    #@STOP
    #unit = _load_unit_table(unit_path)

    # Prefer price embedded in timeseries (e.g., from Business Table adapter)
    unit = _load_unit_from_timeseries(ts)
    if unit is None:
        if unit_path is None:
            print(f"[export_money_timeseries] unit file not found and no price in timeseries. skip.")
            return ctx
        unit = _load_unit_table(unit_path)


    df_out = _compute_money(ts, unit)

    # 3) 書き出し
    money_path = out_dir / "money_timeseries.csv"
    out_dir.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(money_path, index=False, encoding="utf-8-sig")
    print(f"[export_money_timeseries] wrote: {money_path}")

    # 4) run_meta.json への登録
    try:
        rel = money_path.relative_to(run_dir)
    except Exception:
        rel = Path("output") / money_path.name
    _mark_artifact(run_dir, meta_dir, out_dir, "money_timeseries", rel)

    return ctx

def register(bus) -> None:
    # export_bundle (priority 50) の後に実行
    bus.add_action("one_node.after_run", export_money_timeseries, priority=60)