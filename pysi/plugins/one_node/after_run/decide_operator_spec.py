# pysi/plugins/one_node/after_run/decide_operator_spec.py

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pysi.core.hooks.core import HookBus


def _bus_register(bus: HookBus, event_name: str, fn) -> None:
    """
    HookBus の実装差（on / add_action など）を吸収して登録する。
    V0R2_template4single_node では bus.on が無いので add_action を使う。
    """
    if hasattr(bus, "on") and callable(getattr(bus, "on")):
        bus.on(event_name, fn)
        return
    if hasattr(bus, "add_action") and callable(getattr(bus, "add_action")):
        bus.add_action(event_name, fn)
        return
    raise AttributeError("HookBus has neither .on nor .add_action")


def register(bus: HookBus) -> None:
    _bus_register(bus, "after_run", _on_after_run)


def _get_issue_severity(issue: Optional[str]) -> Optional[str]:
    if not issue:
        return None
    if "SHORTAGE" in issue or "STOCKOUT" in issue:
        return "HIGH"
    if "INV_OVER_CAP" in issue:
        return "MID"
    return "MID"


def _pick_operator_spec(
    issues: List[str],
    config: Dict[str, Any],
    output_dir: Path,
) -> Dict[str, Any]:
    """
    Decide operator_spec.json from diagnosis issues and config.

    Priority:
      1) output/diagnosis.json issues[*].recommend (if available)
      2) config.decide.recommend (fallback)
    """

    # Issue priority (smaller is higher priority)
    ISSUE_PRIORITY = {
        # supply shortage first
        "TOTAL_SHORTAGE": 10,
        "SHORTAGE": 11,
        "STOCKOUT": 12,
        # then inventory / demand mismatch
        "DEMAND_MISMATCH": 20,
        "SALES_SLUMP": 21,
        "OVER_STOCK_RISK": 22,
        "INV_OVER_CAP": 23,
        # others
        "SEASONAL_PREBUILD": 50,
    }

    def _issue_pri(code: Optional[str]) -> int:
        if not code:
            return 999
        return int(ISSUE_PRIORITY.get(str(code), 999))

    def _load_recommend_from_diagnosis_json(out_dir: Path) -> List[Dict[str, Any]]:
        """
        Prefer output/diagnosis.json (produced by diagnose_one_node).
        Expected shape:
          {"issues":[{"code":"SHORTAGE", "recommend":[{"operator":..., "hint":...}, ...]}, ...]}
        """
        try:
            p = out_dir / "diagnosis.json"
            if not p.exists():
                return []
            obj = json.loads(p.read_text(encoding="utf-8"))
            iss = obj.get("issues") or []
            if not isinstance(iss, list):
                return []

            # collect recs with issue metadata, then sort by issue priority then rec priority
            collected: List[Dict[str, Any]] = []
            for it in iss:
                if not isinstance(it, dict):
                    continue
                code = it.get("code") or it.get("id") or it.get("issue")  # tolerate variants
                sev = it.get("severity") if isinstance(it.get("severity"), (str, int, float)) else None

                recs = it.get("recommend") or []
                if not isinstance(recs, list):
                    continue
                for r in recs:
                    if not (isinstance(r, dict) and r.get("operator")):
                        continue
                    rr = dict(r)
                    rr["_issue_code"] = code
                    rr["_issue_severity"] = sev
                    rr["_issue_priority"] = _issue_pri(code)
                    # optional per-recommend priority
                    try:
                        rr["_rec_priority"] = int(rr.get("priority", 999))
                    except Exception:
                        rr["_rec_priority"] = 999
                    collected.append(rr)

            collected.sort(key=lambda x: (x.get("_issue_priority", 999), x.get("_rec_priority", 999)))
            return collected
        except Exception:
            return []

    # 1) recommendations: prefer output/diagnosis.json
    recommend = _load_recommend_from_diagnosis_json(output_dir)

    # 2) fallback: config.decide.recommend
    if not recommend:
        decide_cfg = (config.get("decide", {}) or {})
        recommend = decide_cfg.get("recommend", []) or []
        if not isinstance(recommend, list):
            recommend = []

    if not recommend:
        return {}

    # pick first recommendation (already sorted by issue priority)
    top = recommend[0]
    operator = top.get("operator")
    hint = top.get("hint", {}) or {}
    months = hint.get("months") or []
    rate = hint.get("rate", None)
    delta = hint.get("delta", None)
    picked_issue_code = top.get("_issue_code")
    picked_issue_sev = top.get("_issue_severity")

    # special handler: rice seasonal prebuild smoothing (equal distribute shortages across window)
    if operator == "production_adjust_timeseries_csv":
        # if we have total shortage and a production window config, distribute delta equally
        apply_cfg = (config.get("apply", {}) or {})
        prod_window = apply_cfg.get("prod_window") or []
        if not prod_window:
            # fallback to hint.months
            prod_window = months or []

        gap_total = hint.get("gap_total", None)
        if gap_total is None:
            # fallback compute from diagnosis_report? (not mandatory)
            gap_total = hint.get("gap", None)

        if isinstance(prod_window, list) and prod_window and gap_total is not None:
            try:
                n = len(prod_window)
                delta_each = float(gap_total) / float(n) if n > 0 else float(gap_total)
            except Exception:
                delta_each = float(gap_total)

            return {
                "operator": operator,
                "params": {
                    "months": prod_window,
                    "delta": delta_each,
                    "enabled": True,
                },
                "reason": {
                    "picked_from_issue": picked_issue_code or (issues[0] if issues else None),
                    "severity": picked_issue_sev or (_get_issue_severity(issues[0]) if issues else None),
                    "summary_status": "WARN" if issues else "OK",
                    "strategy": "rice_gap_smoothing_equal",
                    "gap_total": gap_total,
                    "prod_window": prod_window,
                },
            }

    # default operator_spec
    return {
        "operator": operator,
        "params": {
            "months": months,
            **({"rate": float(rate)} if rate is not None else {}),
            **({"delta": float(delta)} if delta is not None else {}),
            "enabled": True,
        },
        "reason": {
            "picked_from_issue": picked_issue_code or (issues[0] if issues else None),
            "severity": picked_issue_sev or (_get_issue_severity(issues[0]) if issues else None),
            "summary_status": "WARN" if issues else "OK",
        },
    }


def _on_after_run(run_ctx: dict) -> dict:
    """
    Decide operator_spec.json from diagnosis issues and config.
    Output: runs/.../meta/operator_spec.json
    """
    cfg = (run_ctx.get("config") or {})
    decide_cfg = cfg.get("decide") or {}
    if isinstance(decide_cfg, dict) and decide_cfg.get("enabled") is False:
        print("[one_node.decide_spec] disabled by config.decide.enabled=false; skip")
        return run_ctx

    meta_dir = Path(run_ctx["meta_dir"])
    output_dir = Path(run_ctx.get("output_dir", meta_dir.parent / "output"))
    output_dir.mkdir(parents=True, exist_ok=True)

    config = run_ctx.get("config", {}) or {}

    # load diagnosis_report.json (meta) for issues list (backward compatibility)
    issues: List[str] = []
    try:
        diag_report = meta_dir / "diagnosis_report.json"
        if diag_report.exists():
            obj = json.loads(diag_report.read_text(encoding="utf-8"))
            issues = obj.get("issues", []) or []
            if not isinstance(issues, list):
                issues = []
    except Exception:
        issues = []

    operator_spec = _pick_operator_spec(issues=issues, config=config, output_dir=output_dir)
    if not operator_spec or not operator_spec.get("operator"):
        print("[one_node.decide_spec] no valid operator picked; skip writing operator_spec.json")
        return run_ctx

    # override rate by config.diagnose.seasonal_prebuild_reco_rate if present (legacy behavior)
    rate = operator_spec.get("params", {}).get("rate", None)
    delta = operator_spec.get("params", {}).get("delta", None)

    try:
        diagnose_cfg = (config.get("diagnose", {}) or {})
        if diagnose_cfg.get("seasonal_prebuild_reco_rate", None) is not None:
            rate = float(diagnose_cfg.get("seasonal_prebuild_reco_rate"))
            try:
                print(
                    "[STEP:DECIDE] override rate by "
                    f"config.diagnose.seasonal_prebuild_reco_rate={rate}"
                )
            except Exception:
                pass
    except Exception:
        pass

    # write meta/operator_spec.json
    out_path = meta_dir / "operator_spec.json"
    out_path.write_text(json.dumps(operator_spec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[one_node.decide_spec] wrote: {out_path}")

    # also update _latest pointer (by copy) if your pipeline expects it elsewhere (optional)
    # NOTE: apply_operator_spec reads runs/_latest/meta/operator_spec.json, so it is up to the runner to copy/update it.
    # In this repo, runner / pipeline already maintains _latest. Keep this plugin side-effect free.

    if rate is not None:
        operator_spec["params"]["rate"] = rate
    if delta is not None:
        operator_spec["params"]["delta"] = delta

    return operator_spec

