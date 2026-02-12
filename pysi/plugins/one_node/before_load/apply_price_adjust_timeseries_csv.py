# pysi/plugins/one_node/before_load/apply_price_adjust_timeseries_csv.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


def _pick_cfg(run_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    config の取り方がズレても拾えるようにする（最低限）
    優先順位:
      1) run_ctx["config"]["apply_price_adjust_timeseries"]
      2) run_ctx["config"]["apply_price_adjust_timeseries_csv"]  # 旧名があるなら
      3) run_ctx["apply_price_adjust_timeseries"]                # 直下に来る実装なら
    """
    cfg_root = run_ctx.get("config", {}) or {}

    cfg = cfg_root.get("apply_price_adjust_timeseries")
    if isinstance(cfg, dict):
        return cfg

    cfg = cfg_root.get("apply_price_adjust_timeseries_csv")
    if isinstance(cfg, dict):
        return cfg

    cfg = run_ctx.get("apply_price_adjust_timeseries")
    if isinstance(cfg, dict):
        return cfg

    return {}


def _norm_yyyymm(s: Any) -> str:
    """
    '2025-11' / '2025-1' / 'Jan-25' 等を、可能なら 'YYYY-MM' に寄せる。
    失敗したら元文字列を返す（＝比較で落ちるので安全側）。
    """
    if s is None:
        return ""
    ss = str(s).strip()
    if not ss:
        return ""

    # まず YYYY-M(M) 系を優先的に処理
    if "-" in ss:
        parts = ss.split("-")
        if len(parts) >= 2 and parts[0].isdigit() and parts[1].isdigit():
            y = int(parts[0])
            m = int(parts[1])
            if 1 <= m <= 12:
                return f"{y:04d}-{m:02d}"

    # 次に datetime に任せる（Jan-25 など）
    dt = pd.to_datetime(ss, errors="coerce")
    if pd.isna(dt):
        return ss
    return f"{dt.year:04d}-{dt.month:02d}"


def _resolve_price_csv_path(run_ctx: Dict[str, Any]) -> Path | None:
    """
    編集対象のCSVをどこで持つか：
      - run_dir/input にあればそれを優先（runごとの成果物が閉じる）
      - なければ data_dir を見に行く
    """
    run_dir = Path(run_ctx.get("run_dir", "") or "")
    data_dir = Path(run_ctx.get("data_dir", "") or "")

    # run_dir/input/virtual_node_unit_price_cost.csv
    if str(run_dir):
        p = run_dir / "input" / "virtual_node_unit_price_cost.csv"
        if p.exists():
            return p

    # data_dir/virtual_node_unit_price_cost.csv
    if str(data_dir):
        p = data_dir / "virtual_node_unit_price_cost.csv"
        if p.exists():
            return p

    return None


def _apply_price_adjust_long(df: pd.DataFrame, months: List[str], rate: float, price_items: List[str]) -> tuple[pd.DataFrame, int]:
    """
    long形式: columns include [month, item, value] を想定
    """
    need_cols = {"month", "item", "value"}
    if not need_cols.issubset(set(df.columns)):
        return df, 0

    months_norm = set(_norm_yyyymm(m) for m in months)
    if "" in months_norm:
        months_norm.discard("")

    # month正規化列
    mcol = df["month"].map(_norm_yyyymm)

    # item候補（sell_price を優先、unit_price も受ける）
    items_norm = set(str(x).strip() for x in price_items if str(x).strip())

    mask = mcol.isin(months_norm) & df["item"].astype(str).str.strip().isin(items_norm)
    n = int(mask.sum())
    if n <= 0:
        return df, 0

    df.loc[mask, "value"] = pd.to_numeric(df.loc[mask, "value"], errors="coerce").fillna(0.0) * (1.0 + rate)

    return df, n


def _apply_price_adjust_wide(df: pd.DataFrame, months: List[str], rate: float) -> tuple[pd.DataFrame, int]:
    """
    wide形式: 'unit_price' 列がある場合（念のため残す）
    期待列: month, unit_price
    """
    if ("month" not in df.columns) or ("unit_price" not in df.columns):
        return df, 0

    months_norm = set(_norm_yyyymm(m) for m in months)
    if "" in months_norm:
        months_norm.discard("")

    mcol = df["month"].map(_norm_yyyymm)
    mask = mcol.isin(months_norm)
    n = int(mask.sum())
    if n <= 0:
        return df, 0

    df.loc[mask, "unit_price"] = pd.to_numeric(df.loc[mask, "unit_price"], errors="coerce").fillna(0.0) * (1.0 + rate)
    return df, n


def _on_before_load(run_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """
    Before-load hook:
      - virtual_node_unit_price_cost.csv の価格を、指定月だけ (1+rate) 倍する
      - long形式（month,item,value）なら item='sell_price'（+ unit_price）を更新
      - wide形式（unit_price列）でも一応対応
    """
    cfg = _pick_cfg(run_ctx)

    enabled = bool(cfg.get("enabled", False))
    months = cfg.get("months", []) or []
    try:
        rate = float(cfg.get("rate", 0.0) or 0.0)
    except Exception:
        rate = 0.0

    if (not enabled) or (not months) or (rate == 0.0):
        print("[before_load.apply_price_adjust_timeseries_csv] disabled or months empty or rate=0; skip")
        return run_ctx

    csv_path = _resolve_price_csv_path(run_ctx)
    if csv_path is None:
        print("[before_load.apply_price_adjust_timeseries_csv] not found: virtual_node_unit_price_cost.csv; skip")
        return run_ctx

    df = pd.read_csv(csv_path)

    # long形式の item 名は、現状データからすると sell_price が本命
    # ただし将来 unit_price に寄せても動くように両方受ける
    price_items = cfg.get("items") or cfg.get("price_items") or ["sell_price", "unit_price"]

    if {"month", "item", "value"}.issubset(set(df.columns)):
        df2, n = _apply_price_adjust_long(df, months, rate, list(price_items))
        if n <= 0:
            print(f"[before_load.apply_price_adjust_timeseries_csv] no matching rows (items={list(price_items)}, months={months}); skip")
            return run_ctx
        df2.to_csv(csv_path, index=False)
        print(f"[STEP:PRICE_ADJ_TS] updated {n} row(s) in {csv_path} months={months} rate={rate} items={list(price_items)}")
        return run_ctx

    # wide形式 fallback
    if "unit_price" in df.columns:
        df2, n = _apply_price_adjust_wide(df, months, rate)
        if n <= 0:
            print(f"[before_load.apply_price_adjust_timeseries_csv] no matching rows (unit_price, months={months}); skip")
            return run_ctx
        df2.to_csv(csv_path, index=False)
        print(f"[STEP:PRICE_ADJ_TS] updated {n} row(s) in {csv_path} months={months} rate={rate} (unit_price)")
        return run_ctx

    print(f"[before_load.apply_price_adjust_timeseries_csv] unsupported schema in {csv_path}; columns={list(df.columns)}; skip")
    return run_ctx


def apply_price_adjust_timeseries_csv(**ctx: Any) -> Dict[str, Any]:
    run_ctx = dict(ctx)
    _on_before_load(run_ctx)
    return ctx


def register(bus) -> None:
    """
    Register plugin to hook bus.
    """
    bus.add_action(
        "one_node.before_load",
        apply_price_adjust_timeseries_csv,
        priority=40,
    )
