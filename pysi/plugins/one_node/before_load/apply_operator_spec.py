# pysi/plugins/one_node/before_load/apply_operator_spec.py
# pysi/plugins/one_node/before_load/apply_operator_spec.py

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def register(bus) -> None:
    # demand_adjust_csv(priority=10) より前に走らせて、run_meta.json を更新する
    bus.add_action("one_node.before_load", before_load, priority=5)


@dataclass
class ApplyResult:
    applied: int = 0
    skipped: int = 0
    errors: int = 0


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _normalize_operator_specs(meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    operator_spec の受け口を広めにする：
      - meta["operator_spec"] が dict なら 1件
      - meta["operator_spec"] が list なら 複数
      - meta["plan"]["operator_spec"] / ["operator_specs"] も許容
      - meta["operator_specs"] も許容
    """
    spec = meta.get("operator_spec")
    if isinstance(spec, dict):
        return [spec]
    if isinstance(spec, list):
        return [x for x in spec if isinstance(x, dict)]

    plan = meta.get("plan")
    if isinstance(plan, dict):
        spec2 = plan.get("operator_spec")
        if isinstance(spec2, dict):
            return [spec2]
        if isinstance(spec2, list):
            return [x for x in spec2 if isinstance(x, dict)]
        spec3 = plan.get("operator_specs")
        if isinstance(spec3, list):
            return [x for x in spec3 if isinstance(x, dict)]

    spec4 = meta.get("operator_specs")
    if isinstance(spec4, list):
        return [x for x in spec4 if isinstance(x, dict)]

    return []


def _flatten_params(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    decide_operator_spec が出す形式:
      {"operator":"xxx","params":{...}, ...}
    を、
      {"operator":"xxx", ...params..., ...}
    に平坦化して、既存apply関数が months/rate を取り出せるようにする。
    """
    op = spec.get("operator") or spec.get("name")
    params = spec.get("params")
    if isinstance(params, dict) and op:
        flat = {"operator": op}
        flat.update(params)
        # reason などは残す（任意）
        if "reason" in spec:
            flat["reason"] = spec["reason"]
        if "enabled" in spec:
            flat["enabled"] = spec["enabled"]
        return flat
    return spec


def _find_previous_run_with_operator_spec(run_dir: Path) -> Optional[Path]:
    """
    runs/one_node/<run_id>/meta/operator_spec.json を「直近」から探して返す。
    """
    runs_root = run_dir.parent if run_dir else None
    if runs_root is None or not runs_root.exists():
        return None

    run_dirs = [d for d in runs_root.iterdir() if d.is_dir()]
    # 新しい順（YYYYMMDD_HHMMSS の文字列順でOK）
    run_dirs.sort(reverse=True)

    current_name = run_dir.name
    for d in run_dirs:
        if d.name == current_name:
            continue
        cand = d / "meta" / "operator_spec.json"
        if cand.exists():
            return cand
    return None




def _find_latest_operator_spec(run_dir: Path) -> Optional[Path]:
    """
    runs/one_node/_latest/meta/operator_spec.json を優先参照する（debug向け）。
    """
    runs_root = run_dir.parent if run_dir else None
    if runs_root is None or not runs_root.exists():
        return None
    cand = runs_root / "_latest" / "meta" / "operator_spec.json"
    return cand if cand.exists() else None



def _apply_demand_adjust_csv(meta: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[bool, str]:
    """
    operator_spec(demand_adjust_csv) -> meta["config"]["demand_adjust"] へ展開
    期待 spec 例：
      {"operator":"demand_adjust_csv","months":["2025-07"],"rate":0.2, "items":[...], "file":"virtual_node_timeseries.csv"}
    """
    months = spec.get("months")
    rate = spec.get("rate")
    if not isinstance(months, list) or rate is None:
        return False, "missing months(list) or rate"

    da: Dict[str, Any] = {}
    da["months"] = [str(x) for x in months]
    da["rate"] = float(rate)

    if "items" in spec and isinstance(spec["items"], list):
        da["items"] = [str(x) for x in spec["items"]]
    if "file" in spec and isinstance(spec["file"], str) and spec["file"].strip():
        da["file"] = spec["file"].strip()
    if "format" in spec and isinstance(spec["format"], str) and spec["format"].strip():
        da["format"] = spec["format"].strip()

    da["enabled"] = bool(spec.get("enabled", True))

    meta.setdefault("config", {})
    if isinstance(meta["config"], dict):
        meta["config"]["demand_adjust"] = da
    else:
        meta["config"] = {"demand_adjust": da}

    return True, "expanded to meta.config.demand_adjust"


def _apply_production_adjust_cap_csv(meta: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[bool, str]:
    """
    operator_spec(production_adjust_cap_csv / production_adjust_csv) -> meta["config"]["production_adjust"] へ展開
    Expected spec example:
      {"operator":"production_adjust_cap_csv","months":["2025-09","2025-10"],"rate":0.10}
    """
    months = spec.get("months") or []
    rate = float(spec.get("rate", 0.0) or 0.0)

    if not isinstance(months, list) or not months or rate == 0.0:
        return False, "months empty or rate=0"

    pa: Dict[str, Any] = {
        "months": [str(x) for x in months],
        "rate": float(rate),
        "enabled": bool(spec.get("enabled", True)),
    }

    if "file" in spec and isinstance(spec["file"], str) and spec["file"].strip():
        pa["file"] = spec["file"].strip()
    if "format" in spec and isinstance(spec["format"], str) and spec["format"].strip():
        pa["format"] = spec["format"].strip()

    meta.setdefault("config", {})
    if isinstance(meta["config"], dict):
        meta["config"]["production_adjust"] = pa
    else:
        meta["config"] = {"production_adjust": pa}

    return True, f"production_adjust set: months={pa['months']}, rate={pa['rate']}"


def _apply_production_adjust_timeseries_csv(meta: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[bool, str]:
    """
    operator_spec(production_adjust_timeseries_csv) -> meta["config"]["production_adjust_timeseries"] へ展開

    Expected spec example:
      {"operator":"production_adjust_timeseries_csv","months":["2026-09","2026-10"],"rate":0.10,"file":"virtual_node_timeseries.csv"}
      {"operator":"production_adjust_timeseries_csv","months":["2026-09"],"delta":50}
    """
    months = spec.get("months") or []
    rate = float(spec.get("rate", 0.0) or 0.0)
    delta = float(spec.get("delta", 0.0) or 0.0)

    if not isinstance(months, list) or not months:
        return False, "months empty"
    if rate == 0.0 and delta == 0.0:
        return False, "rate=0 and delta=0"

    pa: Dict[str, Any] = {
        "months": [str(x) for x in months],
        "rate": float(rate),
        "delta": float(delta),
        "enabled": bool(spec.get("enabled", True)),
    }

    if "file" in spec and isinstance(spec["file"], str) and spec["file"].strip():
        pa["file"] = spec["file"].strip()
    if "format" in spec and isinstance(spec["format"], str) and spec["format"].strip():
        pa["format"] = spec["format"].strip()

    meta.setdefault("config", {})
    if isinstance(meta["config"], dict):
        meta["config"]["production_adjust_timeseries"] = pa
    else:
        meta["config"] = {"production_adjust_timeseries": pa}

    return True, "expanded to meta.config.production_adjust_timeseries"


def _apply_price_adjust_business_table_csv(meta: Dict[str, Any], spec: Dict[str, Any]) -> Tuple[bool, str]:
    """
    operator_spec(price_adjust_business_table_csv) -> meta["config"]["price_adjst_business_table"] へ展開

    Expected spec example:
      {"operator":"price_adjust_business_table_csv","months":["2026-04","2026-0"],"rate":-0.10,"file":"business_timeseries.csv"}
      {"operator":"price_adjust_business_table_csv","months":["2026-04"],"delta:-5000}
    """
    months = spec.get("months") or []
    rate = float(spec.get("rate", 0.0) or 0.0)
    delta = float(spec.get("delta", 0.0) or 0.0)

    if not isinstance(months, list) or not months:
        return False, "months empty"
    if rate == 0.0 and delta == 0.0:
        return False, "rate=0 and delta=0"

    pa: Dict[str, Any] = {
        "months": [str(x) for x in months],
        "rate": rate,
        "delta": delta,
        "enabled": bool(spec.get("enabled", True)),
    }

    if "file" in spec and isinstance(spec["file"], str) and spec["file"].strip():
        pa["file"] = spec["file"].strip()

    meta.setdefault("config", {})
    if isinstance(meta["config"], dict):
        meta["config"]["price_adjust_business_table"] = pa
    else:
        meta["config"] = {"price_adjust_business_table": pa}

    return True, "expanded to meta.config.price_adjust_business_table"





def _apply_production_adjust_business_table_csv(meta: Dict[str, Any], spec: Dic[str, Any]) -> Tuple[bool, str]:
    """
    operator_spec(production_adjust_business_table_csv)
      -> meta["config"]["production_adjust_business_table"] へ展開

    Expected spec example:
      {
        "operator":"production_adjust_business_table_csv",
        "months":["2026-09","2026-10"],
        "rate":0.10,
        "file":"business_timeseries.csv"
      }
    """
    months = spec.get("months") or []
    rate = float(spec.get("rate", 0.0) or 0.0)
    delta = float(spec.get("delta", 0.0) or 0.0)

    if not isinstance(months, list) or not months:
        return False, "months empty"
    if rate == 0.0 and delta == 0.0:
        return False, "rate=0 and delta=0"

    pa: Dict[str, Any] = {
        "months": [str(x) for x in months],
        "rate": rate,
        "delta": delta,
        "enabled": bool(spec.get("enabled", True)),
    }

    if "file" in spec and isinstance(spec["file"], str) and spec["file"].strip():
        pa["file"] = spec["file"].strip()

    meta.setdefault("config", {})
    if isinstance(meta["config"], dict):
        meta["config"]["production_adjust_business_table"] = pa
    else:
        meta["config"] = {"production_adjust_business_table": pa}

    return True, "expanded to meta.config.production_adjust_business_table"




def _dispatch_apply(meta: Dict[str, Any], spec_in: Dict[str, Any]) -> Tuple[bool, str]:
    spec = _flatten_params(spec_in)

    op = str(spec.get("operator") or spec.get("name") or "").strip()
    if not op:
        return False, "operator name missing"

    # alias（入力の揺れを吸収）
    op_alias = {
        # demand
        "demand_adjust": "demand_adjust_csv",
        "demand_adjust_csv": "demand_adjust_csv",
        # production(capacity)
        "production_adjust": "production_adjust_cap_csv",
        "production_adjust_csv": "production_adjust_cap_csv",
        "production_adjust_cap_csv": "production_adjust_cap_csv",
        # production(timeseries) ←今回追加
        "production_adjust_timeseries": "production_adjust_timeseries_csv",
        "production_adjust_timeseries_csv": "production_adjust_timeseries_csv",
        # production(business table) ←今回追加
        "production_adjust_business_table": "production_adjust_business_table_csv",
        "production_adjust_business_table_csv": "production_adjust_business_table_csv",
        # price(business table)
        "price_adjust_business_table": "price_adjust_business_table_csv",
        "price_adjust_business_table_csv": "price_adjust_business_table_csv",
    }

    op2 = op_alias.get(op, op)

    if op2 == "demand_adjust_csv":
        return _apply_demand_adjust_csv(meta, spec)
    if op2 == "production_adjust_cap_csv":
        return _apply_production_adjust_cap_csv(meta, spec)
    if op2 == "production_adjust_timeseries_csv":
        return _apply_production_adjust_timeseries_csv(meta, spec)

    if op2 == "production_adjust_business_table_csv":
        return _apply_production_adjust_business_table_csv(meta, spec)

    if op2 == "price_adjust_business_table_csv":
        return _apply_price_adjust_business_table_csv(meta, spec)

    return False, f"unknown operator: {op2}"


def before_load(**ctx: Any) -> None:
    """
    Purpose:
      meta/operator_spec.json（または前回runのそれ）を読み取り、run_meta.json に展開して後続before_loadが読める形にする。
    """

    # meta_dir が先に取れているなら（run_ctx 経由で取れてる）
    # run_dir が空でも meta_dir.parent で復元できる

    run_ctx = ctx.get("run_ctx") or {}
    meta_dir = Path(run_ctx.get("meta_dir") or "")

    run_dir = Path(str(ctx.get("run_dir") or ""))

    # --- FIX: run_dir が '.' になるケースを “不正” 扱いして meta_dir.parent へ寄せる ---
    try:
        run_dir_s = str(run_dir).strip()
    except Exception:
        run_dir_s = ""

    if (not run_dir_s) or (run_dir_s in (".", ".\\", "./")) or (run_dir.name == "."):
        # meta_dir は runs/.../<run_id>/meta のはずなので、その親が run_dir
        if str(meta_dir):
            cand = meta_dir.parent
            if cand.exists():
                run_dir = cand

    # --- robust run_dir detection (RunCtx first, then meta_dir parent) ---
    if (not str(run_dir)) or (not run_dir.exists()):
        if str(meta_dir):
            cand = meta_dir.parent
            if cand.exists():
                run_dir = cand

    if not str(meta_dir):
        if run_dir:
            meta_dir = run_dir / "meta"
        else:
            print("[before_load.apply_operator_spec] meta_dir missing; skip")
            return

    meta_dir.mkdir(parents=True, exist_ok=True)

    print("[STEP:APPLY] purpose=expand operator_spec into run_meta for before_load")
    print(f"[STEP:APPLY] meta_dir={meta_dir} run_dir={run_dir}")

    meta_path = meta_dir / "run_meta.json"

    print("[before_load.apply_operator_spec] meta_path=", meta_path)
    print("[before_load.apply_operator_spec] meta_dir exists=", meta_dir.exists())

    
    meta = _read_json(meta_path)
    if not isinstance(meta, dict):
        print("[before_load.apply_operator_spec] meta/run_meta.json missing; skip")
        return

    # 1) まず「今回runの meta/operator_spec.json」を優先

    spec_path = meta_dir / "operator_spec.json"
    spec_obj: Optional[Dict[str, Any]] = None

    source_current = None
    source_prev = None

    if spec_path.exists():
        spec_obj = _read_json(spec_path)
        source_current = str(spec_path)


    # 2) 無ければ「固定 latest」を探す（debug向け）
    if spec_obj is None and run_dir and run_dir.exists():
        latest = _find_latest_operator_spec(run_dir)
        if latest:
            spec_obj = _read_json(latest)
            source_prev = str(latest)

    # 3) さらに無ければ「前回runの meta/operator_spec.json」を探す
    if spec_obj is None and run_dir and run_dir.exists():
        prev = _find_previous_run_with_operator_spec(run_dir)
        if prev:
            spec_obj = _read_json(prev)
            source_prev = str(prev)

    #@STOP
    ## 2) 無ければ「前回runの meta/operator_spec.json」を探す
    #if spec_obj is None and run_dir and run_dir.exists():
    #    prev = _find_previous_run_with_operator_spec(run_dir)
    #    if prev:
    #        spec_obj = _read_json(prev)
    #        source_prev = str(prev)
    #
    #specs: List[Dict[str, Any]] = []
    #if isinstance(spec_obj, dict):
    #    specs = [spec_obj]
    #else:
    #    # 3) さらに無ければ run_meta.json 内の operator_spec を探す（後方互換）
    #    specs = _normalize_operator_specs(meta)

    #
    #if not specs:
    #    print("[before_load.apply_operator_spec] operator_spec not found (current & previous); skip")
    #    return

    # build specs list
    specs: List[Dict[str, Any]] = []
    if isinstance(spec_obj, dict):
        specs = [spec_obj]
    else:
        # fallback: operator_spec inside run_meta.json
        specs = _normalize_operator_specs(meta)

    if not specs:
        print("[before_load.apply_operator_spec] operator_spec not found (current & latest & previous); skip")
        return


    print(f"[STEP:APPLY] source current={source_current} prev={source_prev}")

    result = ApplyResult()
    details: List[Dict[str, Any]] = []

    for s in specs:
        ok, msg = _dispatch_apply(meta, s)
        if ok:
            result.applied += 1
        else:
            result.skipped += 1
        details.append({"spec": s, "ok": ok, "message": msg})

    try:
        _write_json(meta_path, meta)
    except Exception as e:
        result.errors += 1
        print(f"[before_load.apply_operator_spec] FAILED to write run_meta.json: {e}")
        return

    # --- FIX: also update in-memory run_ctx so subsequent before_load plugins can see changes ---
    try:
        run_ctx2 = ctx.get("run_ctx")
        if isinstance(run_ctx2, dict):
            run_ctx2["meta"] = meta
            if isinstance(meta.get("config"), dict):
                run_ctx2["config"] = meta.get("config")
    except Exception:
        pass

    report = {
        "plugin": "one_node.before_load.apply_operator_spec",
        "meta_path": str(meta_path),
        "operator_specs_count": len(specs),
        "result": {"applied": result.applied, "skipped": result.skipped, "errors": result.errors},
        "details": details,
        "source": {
            "current_operator_spec": str(spec_path) if spec_path.exists() else None,
        },
    }
    try:
        _write_json(meta_dir / "apply_operator_spec_report.json", report)
    except Exception:
        pass

    print(
        f"[STEP:APPLY] result applied={result.applied} skipped={result.skipped} "
        f"errors={result.errors} meta_updated={meta_path.name}"
    )
