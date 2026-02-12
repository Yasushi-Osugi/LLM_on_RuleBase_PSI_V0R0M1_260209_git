# pysi/plugins/one_node/before_load/production_shift_business_table_csv.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd


def register(bus) -> None:
    # apply_operator_spec(優先度=5) の後に動かしたい
    bus.add_action("one_node.before_load", before_load, priority=35)


def _as_list(x) -> List[Any]:
    if x is None:
        return []
    if isinstance(x, list):
        return x
    return [x]


def _pick_col(df: pd.DataFrame, col: str) -> str:
    if col in df.columns:
        return col
    # ありがちな別名を保険で当てる
    for cand in ["production", "production_plan", "planning_P", "P_plan"]:
        if cand in df.columns:
            return cand
    raise KeyError(f"production column not found. requested={col}, columns={list(df.columns)}")


def _normalize_months(df: pd.DataFrame) -> None:
    if "month" not in df.columns:
        raise KeyError("business_timeseries.csv must have 'month' column")
    df["month"] = df["month"].astype(str)


def _apply_one_shift(df: pd.DataFrame, cfg: Dict[str, Any]) -> Tuple[bool, str]:
    file = str(cfg.get("file") or "business_timeseries.csv")
    col_req = str(cfg.get("col") or "production")

    from_months = [str(m) for m in _as_list(cfg.get("from_months"))]
    to_months = [str(m) for m in _as_list(cfg.get("to_months"))]

    from_rate = cfg.get("from_rate", None)
    to_rate = cfg.get("to_rate", None)
    preserve_total = bool(cfg.get("preserve_total", True))

    if not from_months or not to_months:
        return False, "from_months/to_months empty; skip"
    if from_rate is None and cfg.get("from_delta") is None:
        return False, "missing from_rate (or from_delta); skip"

    # rate優先（deltaは必要なら後で拡張）
    from_rate = float(from_rate)

    # preserve_total の場合、to_rateが未指定なら自動計算
    if preserve_total and to_rate is None:
        # ざっくり：1ヶ月ずつ同じrateで増減する前提（あなたの例と整合）
        # 例: from -0.20 x1, to_monthsが2なら +0.10
        to_rate = (-from_rate) * (len(from_months) / max(len(to_months), 1))
    if to_rate is None:
        return False, "missing to_rate (and preserve_total=false); skip"
    to_rate = float(to_rate)

    _normalize_months(df)
    col = _pick_col(df, col_req)

    # 対象行
    m_from = df["month"].isin(from_months)
    m_to = df["month"].isin(to_months)

    if not m_from.any() or not m_to.any():
        return False, f"months not found in data: from={from_months}, to={to_months}"

    # 現在値
    base_from = pd.to_numeric(df.loc[m_from, col], errors="coerce").fillna(0.0).astype(float)
    base_to = pd.to_numeric(df.loc[m_to, col], errors="coerce").fillna(0.0).astype(float)

    # 変更後
    df.loc[m_from, col] = (base_from * (1.0 + from_rate)).values
    df.loc[m_to, col] = (base_to * (1.0 + to_rate)).values

    # 総量保存チェック（完全一致でなく誤差許容）
    if preserve_total:
        before = float(base_from.sum() + base_to.sum())
        after = float(df.loc[m_from, col].sum() + df.loc[m_to, col].sum())
        if abs(after - before) > max(1e-6, abs(before) * 1e-6):
            return False, f"preserve_total violated: before={before}, after={after}"

    return True, f"shifted {col}: from {from_months}({from_rate:+.3f}) to {to_months}({to_rate:+.3f})"


def before_load(**ctx: Any) -> Dict[str, Any]:
    run_ctx = ctx.get("run_ctx") or {}
    if not isinstance(run_ctx, dict):
        return ctx

    data_dir = Path(run_ctx.get("data_dir") or "")
    cfg = (run_ctx.get("config") or {}).get("production_shift_business_table") or {}
    shifts = _as_list(cfg.get("shifts") if isinstance(cfg, dict) else cfg)

    if not data_dir or not data_dir.exists():
        print("[PROD_SHIFT_BT] data_dir missing; skip")
        return ctx
    if not shifts:
        print("[PROD_SHIFT_BT] no shifts; skip")
        return ctx

    # 入力
    file = None
    for it in shifts:
        if isinstance(it, dict) and it.get("file"):
            file = it.get("file")
            break
    file = file or "business_timeseries.csv"

    path = data_dir / file
    if not path.exists():
        print(f"[PROD_SHIFT_BT] missing file: {path}; skip")
        return ctx

    df = pd.read_csv(path)
    ok_any = False
    for it in shifts:
        if not isinstance(it, dict):
            continue
        ok, msg = _apply_one_shift(df, it)
        print(f"[PROD_SHIFT_BT] {msg}")
        ok_any = ok_any or ok

    if ok_any:
        df.to_csv(path, index=False)
        print(f"[PROD_SHIFT_BT] wrote: {path}")

    return ctx
