#/pysi/plugins/one_node/after_run/diagnose_promotion_effect.py

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd


def register(bus) -> None:
    #@STOP
    ## 既存 diagnose_one_node の後に評価したい
    #bus.add_action("one_node.after_run", diagnose_promotion_effect, priority=70)

    # BT->money の後に評価したい（Promotion_Cost 等が揃った money_timeseries を見る）
    bus.add_action("one_node.after_run", diagnose_promotion_effect, priority=95)

def _load_money_ts(run_ctx: Dict[str, Any]) -> pd.DataFrame:
    out_dir = Path(str(run_ctx.get("output_dir") or ""))
    path = out_dir / "money_timeseries.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df


def diagnose_promotion_effect(**ctx: Any) -> Dict[str, Any]:
    run_ctx = ctx.get("run_ctx") or {}
    if not isinstance(run_ctx, dict):
        return ctx

    df = _load_money_ts(run_ctx)
    if df.empty:
        print("[DIAG_PROMO] money_timeseries.csv not found; skip")
        return ctx

    required = {
        "month",
        "Revenue",
        "Process_Cost",
        "Purchase_Cost",
        "Promotion_Cost",
        "Profit",
    }
    if not required.issubset(set(df.columns)):
        print(f"[DIAG_PROMO] missing columns: {required - set(df.columns)}; skip")
        return ctx

    # numeric normalize
    for c in ["Revenue", "Process_Cost", "Purchase_Cost", "Promotion_Cost", "Profit"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0).astype(float)

    # Contribution = 変動利益（販促判断用）
    df["Contribution"] = (
        df["Revenue"]
        - df["Process_Cost"]
        - df["Purchase_Cost"]
        - df["Promotion_Cost"]
    )

    issues: List[Dict[str, Any]] = []

    # A) 売れば売るほど損 → 販促停止/縮小
    m_lossmaking = (df["Revenue"] > 0) & (df["Contribution"] < 0)
    if m_lossmaking.any():
        months = df.loc[m_lossmaking, "month"].astype(str).tolist()
        issues.append(
            {
                "code": "PROMO_STOP_LOSSMAKING",
                "severity": "high",
                "months": months,
                "evidence": {
                    "rule": "Contribution < 0 with Revenue > 0",
                    "count": len(months),
                },
                "recommend": [
                    {
                        "operator": "promotion_adjust_business_table",
                        "hint": {
                            "months": months,
                            "promo_ratio": 0.0,
                        },
                        "note": "Selling more increases loss; stop or reduce promotion.",
                    }
                ],
            }
        )

    # B) 売れば改善 → 売り切り強化（在庫圧縮）
    m_push = (df["Revenue"] > 0) & (df["Profit"] < 0) & (df["Contribution"] >= 0)
    if m_push.any():
        months = df.loc[m_push, "month"].astype(str).tolist()
        issues.append(
            {
                "code": "PROMO_PUSH_TO_CLEAR",
                "severity": "medium",
                "months": months,
                "evidence": {
                    "rule": "Profit < 0 but Contribution >= 0",
                    "count": len(months),
                },
                "recommend": [
                    {
                        "operator": "promotion_adjust_business_table",
                        "hint": {
                            "months": months,
                            "promo_ratio": 0.08,
                        },
                        "note": "Selling improves result; strengthen promotion to clear inventory.",
                    }
                ],
            }
        )

    if not issues:
        print("[DIAG_PROMO] no promotion-related issues detected")
        return ctx

    # append to existing diagnosis.json if present
    out_dir = Path(str(run_ctx.get("output_dir") or ""))
    diag_path = out_dir / "diagnosis.json"
    if diag_path.exists():
        try:
            base = pd.read_json(diag_path)
            base_issues = base.get("issues", [])
        except Exception:
            base_issues = []
    else:
        base_issues = []

    merged = base_issues + issues
    pd.Series({"issues": merged}).to_json(diag_path, force_ascii=False, indent=2)

    print(f"[DIAG_PROMO] detected issues={len(issues)}")
    return ctx
