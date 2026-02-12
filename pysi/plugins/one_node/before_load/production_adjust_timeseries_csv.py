# pysi/plugins/one_node/before_load/production_adjust_timeseries_csv.py
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


def register(bus) -> None:
    # apply_operator_spec(priority=5) の後、demand_adjust_csv(priority=10) より前に走らせる想定
    bus.add_action("one_node.before_load", before_load, priority=8)


def _month_to_iso(s: str) -> Optional[str]:
    """
    '2026-09' -> '2026-09'
    'Sep-26'  -> '2026-09'
    """
    s = (s or "").strip()
    if not s:
        return None
    # ISO: YYYY-MM
    if len(s) == 7 and s[4] == "-":
        yyyy, mm = s.split("-", 1)
        if yyyy.isdigit() and mm.isdigit():
            return f"{int(yyyy):04d}-{int(mm):02d}"
    # 'Sep-26' 形式
    try:
        dt = datetime.strptime(s, "%b-%y")
        return f"{dt.year:04d}-{dt.month:02d}"
    except Exception:
        return None


def _read_json(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def before_load(**ctx: Any) -> None:
    """
    Reads:
      run_ctx["config"]["production_adjust_timeseries"] (dict)
    Writes:
      run_ctx["input_dir"]/<file> (default: virtual_node_timeseries.csv)
    """
    run_ctx = ctx.get("run_ctx") or {}
    if not isinstance(run_ctx, dict):
        return

    cfg = run_ctx.get("config") or {}
    if not isinstance(cfg, dict):
        return

    pa = cfg.get("production_adjust_timeseries") or {}
    if not isinstance(pa, dict):
        return

    # DEBUG: confirm config source
    print("[STEP:PROD_ADJ_TS] cfg source keys:", list((run_ctx.get("config") or {}).keys()))
    print("[STEP:PROD_ADJ_TS] cfg raw:", (run_ctx.get("config") or {}).get("production_adjust_timeseries"))



    enabled = bool(pa.get("enabled", True))
    if not enabled:
        print("[before_load.production_adjust_timeseries_csv] disabled; skip")
        return

    months_raw = pa.get("months") or []
    rate = float(pa.get("rate", 0.0) or 0.0)
    delta = float(pa.get("delta", 0.0) or 0.0)


    # DEBUG: confirm what config this plugin is actually using
    print(f"[STEP:PROD_ADJ_TS] rate={rate} (1+rate={1.0+float(rate)}) delta={delta} cfg={pa}")



    if not isinstance(months_raw, list) or not months_raw:
        print("[before_load.production_adjust_timeseries_csv] months empty; skip")
        return
    if rate == 0.0 and delta == 0.0:
        print("[before_load.production_adjust_timeseries_csv] rate=0 and delta=0; skip")
        return

    file_name = str(pa.get("file") or "virtual_node_timeseries.csv").strip() or "virtual_node_timeseries.csv"
    input_dir = Path(str(run_ctx.get("input_dir") or ""))
    if not input_dir.exists():
        print("[before_load.production_adjust_timeseries_csv] input_dir missing; skip")
        return
    ts_path = input_dir / file_name
    if not ts_path.exists():
        print(f"[before_load.production_adjust_timeseries_csv] missing {ts_path}; skip")
        return

    # 変換: 指定 months を ISO に正規化
    target_iso = {x for x in (_month_to_iso(str(m)) for m in months_raw) if x}
    if not target_iso:
        print("[before_load.production_adjust_timeseries_csv] months parse failed; skip")
        return

    # CSVの month 列も ISO に変換して照合する
    df = pd.read_csv(ts_path)
    if not {"month", "item", "value"}.issubset(set(df.columns)):
        print(f"[before_load.production_adjust_timeseries_csv] unexpected cols={list(df.columns)}; skip")
        return

    month_iso = df["month"].astype(str).map(lambda x: _month_to_iso(x) or "")
    mask = (df["item"].astype(str) == "production") & (month_iso.isin(list(target_iso)))

    hit = int(mask.sum())
    if hit <= 0:
        print(
            "[before_load.production_adjust_timeseries_csv] no matching rows "
            f"(item=production, months={sorted(target_iso)}); skip"
        )
        return

    # 更新ロジック:
    # - delta が指定されていれば加算
    # - それ以外は rate で倍率
    if delta != 0.0:
        df.loc[mask, "value"] = df.loc[mask, "value"].astype(float) + float(delta)
        mode = f"delta(+{delta})"
    else:
        df.loc[mask, "value"] = df.loc[mask, "value"].astype(float) * (1.0 + float(rate))
        mode = f"rate(*{1.0+rate})"

    df.to_csv(ts_path, index=False)
    print(
        "[STEP:PROD_ADJ_TS] purpose=adjust virtual_node_timeseries.csv production rows "
        f"by months/{mode}"
    )
    print(f"[STEP:PROD_ADJ_TS] file={ts_path}")
    print(f"[STEP:PROD_ADJ_TS] months_iso={sorted(target_iso)} matched_rows={hit}")
