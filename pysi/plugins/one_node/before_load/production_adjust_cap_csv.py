#pysi/plugins/one_node/before_load/production_adjust_cap_csv.py

# production_adjust_cap_csv.py（生産P調整オペレータ）
# これは capacity_P.csv を months 指定で rate（倍率）/delta（加算） します。
# 米モデルなら「収穫期（cap_P>0の月）」に対して増産する想定です。
# pysi/plugins/one_node/before_load/production_adjust_cap_csv.py
# pysi/plugins/one_node/before_load/production_adjust_cap_csv.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def register(bus) -> None:
    """
    Hook registration.
    - apply_operator_spec(priority=5) の後に走らせたいので priority=20 にする。
    - bus API は環境差があるので add_action/register の両対応にする。
    """
    # Preferred in your environment (you asked this form)
    if hasattr(bus, "add_action"):
        # one_node.before_load という “フェーズ名” を使う設計に合わせる
        bus.add_action("one_node.before_load", production_adjust_cap_csv, priority=20)
        return

    # Fallback: some buses use register("before_load", ...)
    if hasattr(bus, "register"):
        try:
            bus.register("before_load", production_adjust_cap_csv, name="production_adjust_cap_csv", priority=20)
        except Exception:
            # 最後の保険：何もしない（ロードはされるが実行されない可能性あり）
            pass


def _get_paths(ctx: Dict[str, Any]) -> Tuple[Optional[Path], Optional[Path]]:
    """
    Return (run_dir, input_dir).
    - Preferred: ctx["run_ctx"] dict providing {"run_dir","input_dir"}
    - Fallback: ctx["run_dir"] and ctx["data_dir"] or ctx["input_dir"]
    - Last resort: run_dir/"data"
    """
    run_ctx = ctx.get("run_ctx") or {}
    run_dir = run_ctx.get("run_dir") or ctx.get("run_dir")
    input_dir = run_ctx.get("input_dir") or ctx.get("input_dir") or ctx.get("data_dir")

    run_dir_p = Path(str(run_dir)) if run_dir else None
    input_dir_p = Path(str(input_dir)) if input_dir else None

    if input_dir_p is None and run_dir_p is not None:
        cand = run_dir_p / "data"
        input_dir_p = cand

    return run_dir_p, input_dir_p


def _get_production_adjust(meta: Dict[str, Any]) -> Dict[str, Any]:
    """
    Accept both:
      meta["production_adjust"] = {...}
    and (optional)
      meta["config"]["production_adjust"] = {...}
    """
    cfg = meta.get("production_adjust")
    if isinstance(cfg, dict):
        return cfg
    cfg2 = (meta.get("config") or {}).get("production_adjust")
    if isinstance(cfg2, dict):
        return cfg2
    return {}


def _scale_capacity_p_long(df: pd.DataFrame, months: List[str], rate: float) -> Tuple[pd.DataFrame, int]:
    """
    Long format examples:
      month,node_name,value
      month,value
      ym,value
    We try to detect month/value columns.
    """
    cols = {c.lower(): c for c in df.columns}
    month_col = cols.get("month") or cols.get("ym") or cols.get("period")
    if month_col is None:
        month_col = df.columns[0]

    value_col = cols.get("value") or cols.get("cap") or cols.get("capacity")
    if value_col is None:
        if len(df.columns) >= 2:
            value_col = df.columns[1]
        else:
            raise ValueError("capacity_P.csv: cannot detect value column")

    mask = df[month_col].astype(str).isin(months)
    touched = int(mask.sum())
    if touched > 0:
        df.loc[mask, value_col] = pd.to_numeric(df.loc[mask, value_col], errors="coerce") * (1.0 + rate)
    return df, touched


def _scale_capacity_p_wide(df: pd.DataFrame, months: List[str], rate: float) -> Tuple[pd.DataFrame, int]:
    """
    Wide format example:
      item, 2025-01, 2025-02, ...
    or:
      2025-01, 2025-02, ...
    We scale matching month columns.
    """
    touched = 0
    for m in months:
        if m in df.columns:
            df[m] = pd.to_numeric(df[m], errors="coerce") * (1.0 + rate)
            touched += len(df)
    return df, touched


def production_adjust_cap_csv(**ctx: Any) -> None:
    """
    capacity_P.csv を months 指定で rate (倍率: 1+rate) する。
    """
    meta = ctx.get("meta") or {}
    cfg = _get_production_adjust(meta)

    #@ADD
    print("[STEP:PROD_ADJ_CAP] purpose=adjust capacity_P.csv by months/rate from run_meta.production_adjust")
    print(f"[STEP:PROD_ADJ_CAP] production_adjust cfg={cfg}")


    months = cfg.get("months") or []
    rate = float(cfg.get("rate", 0.0) or 0.0)

    if not months or rate == 0.0:
        print("[before_load.production_adjust_cap_csv] no production_adjust config; skip")
        print("[STEP:PROD_ADJ_CAP] no production_adjust config; skip")
        return

    run_dir, input_dir = _get_paths(ctx)
    # ★ログ強化：パス解決を確認
    print(f"[before_load.production_adjust_cap_csv] resolved run_dir={run_dir} input_dir={input_dir}")

    if run_dir is None or input_dir is None or not input_dir.exists():
        print("[before_load.production_adjust_cap_csv] run_dir/input_dir missing; skip")
        return

    cap_path = input_dir / "capacity_P.csv"
    if not cap_path.exists():
        print(f"[before_load.production_adjust_cap_csv] missing: {cap_path}; skip")
        return

    df = pd.read_csv(cap_path)

    touched = 0
    try:
        lowered = [c.lower() for c in df.columns]
        if ("month" in lowered) or ("ym" in lowered) or ("period" in lowered):
            df, touched = _scale_capacity_p_long(df, list(months), rate)
        else:
            df, touched = _scale_capacity_p_wide(df, list(months), rate)

    except Exception as e:
        print(f"[before_load.production_adjust_cap_csv] failed to apply: {e}; skip")
        #@ADD
        print(f"[STEP:PROD_ADJ_CAP] failed to apply: {e}; skip")
        return

    if touched == 0:
        print(f"[before_load.production_adjust_cap_csv] months not found; months={months}; skip")
        #@ADD
        print(f"[STEP:PROD_ADJ_CAP] months not found; months={months}; skip")
        return

    df.to_csv(cap_path, index=False)
    print(
        f"[before_load.production_adjust_cap_csv] applied: months={months}, rate={rate}, touched={touched}, file={cap_path}"
    )
    #@ADD
    print(f"[STEP:PROD_ADJ_CAP] rows_touched={touched} file={cap_path.name}")
    print(f"[STEP:PROD_ADJ_CAP] applied months={months} rate={rate}")