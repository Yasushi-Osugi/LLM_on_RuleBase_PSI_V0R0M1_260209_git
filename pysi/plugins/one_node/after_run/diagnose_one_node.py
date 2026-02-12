#pysi.plugins.one_node.after_run.diagnose_one_node.py

# ********
# SEASONAL_PREBUILD追加
# ********
#**端境期（cap_P==0 が続く期間）**に
#backlog>0 または inventory が需要に対して薄い（=在庫が底を打つ）を検知
#その直前の **収穫期（cap_P>0 の期間）**で
#production/cap_P が低く 作り溜め余地（slack）があったを検知
#すると「前年のPを前倒しで積むべき」＝ SEASONAL_PREBUILD を出す

#pysi.plugins.one_node.after_run.diagnose_one_node.py

# ********
# SEASONAL_PREBUILD追加
# ********
#**端境期（cap_P==0 が続く期間）**に
#backlog>0 または inventory が需要に対して薄い（=在庫が底を打つ）を検知
#その直前の **収穫期（cap_P>0 の期間）**で
#production/cap_P が低く 作り溜め余地（slack）があったを検知
#すると「前年のPを前倒しで積むべき」＝ SEASONAL_PREBUILD を出す
#pysi.plugins.one_node.after_run.diagnose_one_node.py

# **** このpluginの意図 ****
#  - one_node_result_timeseries.csv を読み
#  - 経営的に「何が起きているか」を issue として列挙し
#  - Decide が拾える形（issues[*].recommend）で operator 推奨を渡す
#
# NOTE:
#  Decide側は diagnosis.json の `issues[*].recommend` を読む実装のため
#  本pluginは recommend を issue 内に格納する（A案）

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pysi.core.hooks.core import HookBus


def _bus_register(bus: HookBus, event_name: str, fn) -> None:
    """
    HookBus の実装差（on / add_action など）を吸収して登録する。
    """
    if hasattr(bus, "on"):
        bus.on(event_name, fn)
    elif hasattr(bus, "add_action"):
        bus.add_action(event_name, fn)
    else:
        raise RuntimeError("Unsupported HookBus API (missing on/add_action)")

#@STOP
#def register(bus) -> None:
#    # export_bundle(priority=5) の後に実行される想定
#    _bus_register(bus, "after_run", diagnose_after_run)

def register(bus) -> None:
    # export_bundle(priority=5) の後、decide(priority=70) の前に動かしたいなら 60 あたり
    if hasattr(bus, "add_action"):
        bus.add_action("one_node.after_run", diagnose_after_run, priority=60)
    else:
        # 互換用（もし bus.on がある実装なら）
        bus.on("one_node.after_run", diagnose_after_run)


def _safe_mean(x: np.ndarray) -> Optional[float]:
    try:
        v = x[~np.isnan(x)]
        if len(v) == 0:
            return None
        return float(np.mean(v))
    except Exception:
        return None


def _to_float_array(df: pd.DataFrame, col: str, default: float = 0.0) -> np.ndarray:
    if col not in df.columns:
        return np.array([default] * len(df), dtype=float)
    return df[col].astype(float).to_numpy()


def _streak_months(months: List[str], mask: List[bool]) -> Tuple[int, List[str]]:
    """
    mask=True が連続する最大 streak と、その該当 months を返す。
    """
    best_n = 0
    best_months: List[str] = []
    cur_n = 0
    cur_months: List[str] = []
    for m, ok in zip(months, mask):
        if ok:
            cur_n += 1
            cur_months.append(m)
            if cur_n > best_n:
                best_n = cur_n
                best_months = cur_months.copy()
        else:
            cur_n = 0
            cur_months = []
    return best_n, best_months


def _ym(s: str) -> str:
    return str(s)[:7]


def _infer_harvest_month_numbers(months: List[str], cap_p: np.ndarray, config: Dict[str, Any]) -> List[int]:
    """
    収穫期（cap_p>0 が立つ月）の月番号(1-12)を推定。
    config 側に明示があればそれを優先。
    """
    try:
        sc = (config or {}).get("scenario") or {}
        harvest = sc.get("harvest_month_numbers")
        if isinstance(harvest, list) and harvest:
            return [int(x) for x in harvest]
    except Exception:
        pass

    mnums: List[int] = []
    for m, cp in zip(months, cap_p.tolist()):
        if cp > 0:
            try:
                mnums.append(int(m.split("-")[1]))
            except Exception:
                continue
    return sorted(list(set(mnums)))


def _crop_year_label(month: str) -> str:
    # crop-year: 便宜的に年単位ラベル
    return str(month)[:4]


def _crop_year_of_month(month: str, harvest_month_numbers: List[int]) -> str:
    """
    収穫が年跨ぎするケースを考慮し crop-year を割当て。
    超簡易：収穫月より前は前年 crop-year 扱い等。
    """
    try:
        y = int(month[:4])
        m = int(month[5:7])
        if not harvest_month_numbers:
            return str(y)
        harvest_min = min(harvest_month_numbers)
        if m < harvest_min:
            return str(y - 1)
        return str(y)
    except Exception:
        return _crop_year_label(month)


def _recommend_production_by_crop_year(
    months: List[str],
    demand: np.ndarray,
    sales: np.ndarray,
    production: np.ndarray,
    cap_p: np.ndarray,
    config: Dict[str, Any],
    th: Any,
) -> Dict[str, Any]:
    """
    backlog が出る crop-year の端境期に対し、収穫期（cap_p>0）へ前倒し生産を推奨。
    既存の RICE 推奨ロジック（簡略）を温存。
    """
    harvest_month_numbers = _infer_harvest_month_numbers(months, cap_p, config)
    # backlogが立つ月を拾う
    backlog_mask = (demand - sales) > 0
    backlog_months = [m for m, ok in zip(months, backlog_mask.tolist()) if ok]
    if not backlog_months:
        return {"operator": "production_adjust_timeseries_csv", "hint": {"months": months[-2:], "rate": 0.10}}

    # backlog月の crop-year を特定
    cy = _crop_year_of_month(backlog_months[-1], harvest_month_numbers)

    # 同一 crop-year の収穫期候補 month を抽出
    cand: List[str] = []
    for m, cp in zip(months, cap_p.tolist()):
        if cp <= 0:
            continue
        if _crop_year_of_month(m, harvest_month_numbers) == cy:
            cand.append(m)

    if not cand:
        cand = months[-2:]

    return {
        "operator": "production_adjust_timeseries_csv",
        "hint": {
            "months": cand[:2] if len(cand) >= 2 else cand,
            "rate": float(getattr(th, "seasonal_prebuild_reco_rate", 0.10)),
        },
    }


@dataclass
class Thresholds:
    service_level_target: float = 0.98
    inv_over_cap_streak_n: int = 2
    demand_cv_threshold: float = 0.30

    seasonal_prebuild_lookback_months: int = 6
    seasonal_prebuild_inv_months: float = 1.0
    seasonal_prebuild_reco_rate: float = 0.10
    seasonal_prebuild_backlog_threshold: float = 0.0


def _read_json(p: Path) -> Dict[str, Any]:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _repo_root_from_run_dir(run_dir: Path) -> Path:
    # runs/one_node/YYYYMMDD_xxxxxx から repo root へ
    # run_dir = .../runs/one_node/20260130_...
    # -> repo_root = .../
    try:
        return run_dir.parent.parent.parent
    except Exception:
        return run_dir


def _resolve_path(run_dir: Path, rel: str) -> Path:
    root = _repo_root_from_run_dir(run_dir)
    return (root / rel).resolve()


def _pick_run_meta(run_dir: Path) -> Dict[str, Any]:
    meta = run_dir / "meta" / "run_meta.json"
    if meta.exists():
        return _read_json(meta)
    out = run_dir / "output" / "run_meta.json"
    if out.exists():
        return _read_json(out)
    return {}


def _load_scenario_config(run_dir: Path) -> Dict[str, Any]:
    """
    run_meta.json 内の scenario を手掛かりに configs を読む（無ければ空）。
    """
    rm = _pick_run_meta(run_dir)
    scenario = rm.get("scenario")
    if not scenario:
        # run_meta の構造が違う場合にも耐える
        scenario = (rm.get("meta") or {}).get("scenario")

    # configs/<scenario>.json を想定
    if scenario:
        p = _resolve_path(run_dir, f"configs/{scenario}.json")
        if p.exists():
            obj = _read_json(p)
            # run_meta.json と同じ形（topに scenario/config）が多いのでそのまま返す
            cfg = obj.get("config")
            if isinstance(cfg, dict):
                return cfg
            # すでに config 部分ならそれを返す
            if isinstance(obj, dict):
                return obj
    return {}


def _load_thresholds(run_dir: Path) -> Thresholds:
    # run_meta/config から必要な値だけ拾って Thresholds に反映
    cfg = _load_scenario_config(run_dir)
    d = (cfg.get("diagnose") or {}) if isinstance(cfg, dict) else {}
    th = Thresholds()

    # RICE系 threshold（既存）
    if "seasonal_prebuild_lookback_months" in d:
        th.seasonal_prebuild_lookback_months = int(d["seasonal_prebuild_lookback_months"])
    if "seasonal_prebuild_inv_months" in d:
        th.seasonal_prebuild_inv_months = float(d["seasonal_prebuild_inv_months"])
    if "seasonal_prebuild_reco_rate" in d:
        th.seasonal_prebuild_reco_rate = float(d["seasonal_prebuild_reco_rate"])
    if "seasonal_prebuild_backlog_threshold" in d:
        th.seasonal_prebuild_backlog_threshold = float(d["seasonal_prebuild_backlog_threshold"])

    # 共通
    if "service_level_target" in d:
        th.service_level_target = float(d["service_level_target"])
    if "inv_over_cap_streak_n" in d:
        th.inv_over_cap_streak_n = int(d["inv_over_cap_streak_n"])
    if "demand_cv_threshold" in d:
        th.demand_cv_threshold = float(d["demand_cv_threshold"])

    return th


def diagnose_after_run(**ctx: Any) -> None:
    """
    Diagnosis plugin for one-node PSI.

    Reads:
      - output/one_node_result_timeseries.csv   (PSI result)
    Writes:
      - output/diagnosis.json                  (for Decide)
      - meta/diagnosis_report.json             (traceability)

    IMPORTANT:
      - Decide側は diagnosis.json の `issues[*].recommend` を読む実装。
        したがって本pluginは、SmartPhone/Riceどちらのケースでも
        recommend は各 issue の中に格納する（A案）。
    """

    run_dir = Path(str(ctx.get("run_dir", "")))
    if not run_dir.exists():
        print("[one_node.diagnose] run_dir missing; skip")
        return

    out_dir = run_dir / "output"
    meta_dir = run_dir / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    # thresholds (shared)
    th = _load_thresholds(run_dir)

    # scenario config (to pick case & optional domain params)
    config = ctx.get("config")
    if not isinstance(config, dict) or not config:
        config = _load_scenario_config(run_dir)

    print("[STEP:DIAGNOSE] purpose=make issues from PSI result")

    diagnose_cfg = ((config or {}).get("diagnose") or {}) if isinstance(config, dict) else {}
    case = str(diagnose_cfg.get("case", "") or "").strip().lower()

    # ----------------------------
    # Load one-node result timeseries
    # ----------------------------
    ts_path = out_dir / "one_node_result_timeseries.csv"
    if not ts_path.exists():
        # some runners may put it in meta; try fallback
        alt = meta_dir / "one_node_result_timeseries.csv"
        if alt.exists():
            ts_path = alt

    if not ts_path.exists():
        print(f"[one_node.diagnose] missing timeseries: {ts_path}; skip")
        return

    df = pd.read_csv(ts_path)

    # Normalize month
    months = df.get("month")
    if months is None:
        months = df.iloc[:, 0]
    months = [_ym(m) for m in months.tolist()]

    # series
    demand = _to_float_array(df, "demand", 0.0)
    sales = _to_float_array(df, "sales", 0.0)
    production = _to_float_array(df, "production", 0.0)
    inv = _to_float_array(df, "inventory", 0.0)
    backlog = _to_float_array(df, "backlog", 0.0)
    waste = _to_float_array(df, "waste", 0.0)
    over_i = _to_float_array(df, "over_i", 0.0)

    cap_s = _to_float_array(df, "cap_s", 0.0)
    cap_p = _to_float_array(df, "cap_p", 0.0)
    cap_i = _to_float_array(df, "cap_i", 0.0)

    #@STOP
    #ship_util = np.where(cap_s > 0, sales / cap_s, np.nan)
    #prod_util = np.where(cap_p > 0, production / cap_p, np.nan)
    #ship_util_avg = _safe_mean(ship_util)
    #prod_util_avg = _safe_mean(prod_util)
    # NOTE: utilization is computed under np.errstate below (to avoid RuntimeWarning)

    backlog_mask = [b > 0 for b in backlog.tolist()]
    bl_streak, bl_months = _streak_months(months, backlog_mask)

    #@STOP
    #sl = np.where(demand > 0, sales / demand, np.nan)

    # safety: avoid noisy RuntimeWarning when cap_* or demand contains 0
    with np.errstate(divide="ignore", invalid="ignore"):
        ship_util = np.where(cap_s > 0, sales / cap_s, np.nan)
        prod_util = np.where(cap_p > 0, production / cap_p, np.nan)
        sl = np.where(demand > 0, sales / demand, np.nan)

    ship_util_avg = _safe_mean(ship_util)
    prod_util_avg = _safe_mean(prod_util)

    service_level_avg = _safe_mean(sl)

    # ----------------------------
    # PHONE / SMART_PHONE branch
    # ----------------------------
    if case in {"phone", "smart_phone", "smartphone"}:
        target_inv_m = float(diagnose_cfg.get("target_inventory_months", 2.0))

        def _trend_last_n(arr: np.ndarray, n: int = 3) -> Optional[float]:
            x = np.array(arr, dtype=float)
            x = x[~np.isnan(x)]
            if len(x) < n:
                return None
            y = x[-n:]
            return float((y[-1] - y[0]) / max(1, (n - 1)))

        inv_months = None
        avg_sales = None
        s = sales.astype(float)
        s = s[~np.isnan(s)]
        if len(s) >= 3:
            avg_sales = float(np.mean(s[-3:]))
        elif len(s) >= 1:
            avg_sales = float(s[-1])

        inv_last = float(inv[~np.isnan(inv)][-1]) if np.any(~np.isnan(inv)) else 0.0
        if avg_sales is not None and avg_sales > 1e-9:
            inv_months = inv_last / avg_sales

        #@STOP
        #sales_trend = _trend_last_n(sales, 3)
        #inv_trend = _trend_last_n(inv, 3)

        #@STOP
        #inv_months = inv_last / avg_sales

        # ---- phone sell-out / slump thresholds (overrideable by config.diagnose.*) ----
        target_inv_m = float(diagnose_cfg.get("target_inventory_months", 2.0))
        lookback_m = int(diagnose_cfg.get("sales_trend_lookback_months", 3))
        overstock_ratio = float(diagnose_cfg.get("overstock_ratio", 1.10))  # targetの何倍で過多扱い
        # 傾き（units/month）がこれ以下なら「販売不振」候補（※欠品が無い時だけ発火）
        sales_slump_slope = float(diagnose_cfg.get("sales_slump_slope", -1.0))

        sales_trend = _trend_last_n(sales, lookback_m)
        inv_trend = _trend_last_n(inv, lookback_m)

        # 値引き余地（将来拡張用：現時点は config に入っている場合だけ診断根拠に載せる）
        discount_room = None
        try:
            if "discount_room" in diagnose_cfg:
                discount_room = float(diagnose_cfg.get("discount_room"))
            elif "margin_buffer" in diagnose_cfg:
                discount_room = float(diagnose_cfg.get("margin_buffer"))
        except Exception:
            discount_room = None



        lifecycle_phase = str(diagnose_cfg.get("lifecycle_phase", "MID")).upper()

        issues: List[Dict[str, Any]] = []

        # TOTAL_SHORTAGE
        backlog_end = float(backlog[~np.isnan(backlog)][-1]) if np.any(~np.isnan(backlog)) else 0.0
        if backlog_end > 0 or bl_streak >= 2:
            focus_months = bl_months[-2:] if len(bl_months) >= 2 else bl_months
            issues.append(
                {
                    "code": "TOTAL_SHORTAGE",
                    "severity": "HIGH",
                    "evidence": {
                        "backlog_end": backlog_end,
                        "backlog_streak": bl_streak,
                        "months": bl_months,
                        "service_level_avg": service_level_avg,
                        "service_level_target": float(diagnose_cfg.get("service_level_target", 0.98)),
                        "ship_util_avg": ship_util_avg,
                        "prod_util_avg": prod_util_avg,
                    },
                    "message": "供給不足（売り逃し/欠品）が顕在化。供給増・優先供給・価格の見直しが必要。",
                    "recommend": [
                        {"operator": "production_adjust_timeseries_csv", "hint": {"months": focus_months, "rate": 0.20}},
                        {
                            "operator": "apply_price_adjust_timeseries_csv",
                            "hint": {"months": focus_months, "price_rate": +0.05, "reason": "scarcity pricing"},
                        },
                    ],
                }
            )

        # ------------------------------------------------------------
        # PHONE: 「販売不振（在庫過多）」分岐
        # 重要：欠品（backlog>0 / service level悪化）があるときは販売不振に誤判定しない
        # ------------------------------------------------------------
        # service level（需要>0の期）

        #@STOP
        #sl = np.where(demand > 0, sales / demand, np.nan)

        with np.errstate(divide="ignore", invalid="ignore"):
            sl = np.where(demand > 0, sales / demand, np.nan)

        service_level_avg = float(np.nanmean(sl)) if np.any(~np.isnan(sl)) else None
        sl_target = float(diagnose_cfg.get("service_level_target", 0.98))

        no_shortage = (backlog_end <= 0.0) and (service_level_avg is None or service_level_avg >= sl_target)

        if no_shortage:
            # 1) OVER_STOCK_RISK: 在庫月数が目標を超えている
            if inv_months is not None and inv_months > target_inv_m * overstock_ratio:
                issues.append({
                    "code": "OVER_STOCK_RISK",
                    "severity": "HIGH" if inv_months > target_inv_m * 1.30 else "MED",
                    "evidence": {
                        "inventory_months": inv_months,
                        "target_inventory_months": target_inv_m,
                        "discount_room": discount_room,
                        "lookback_months": lookback_m,
                    },
                    "message": "欠品ではなく在庫過多。売り切り（価格/販促/チャネル）か減産の判断が必要。",
                    "recommend": [
                        {
                            "operator": "apply_price_adjust_timeseries_csv",
                            "hint": {
                                "months": months[-2:] if len(months) >= 2 else months,
                                "price_rate": -0.10,
                                "reason": "sell-out discount",
                            },
                        }
                    ],
                })

            # 2) SALES_SLUMP: 売上トレンド悪化（欠品でない）
            if sales_trend is not None and sales_trend <= sales_slump_slope:
                issues.append({
                    "code": "SALES_SLUMP",
                    "severity": "HIGH",
                    "evidence": {
                        "sales_trend": float(sales_trend),
                        "sales_slump_slope": float(sales_slump_slope),
                        "lookback_months": lookback_m,
                        "discount_room": discount_room,
                    },
                    "message": "欠品ではなく販売が鈍化。値付け/販促/チャネルの見直し余地。",
                    "recommend": [
                        {
                            "operator": "apply_price_adjust_timeseries_csv",
                            "hint": {
                                "months": months[-2:] if len(months) >= 2 else months,
                                "price_rate": -0.05,
                                "reason": "stimulate demand",
                            },
                        }
                    ],
                })

            # 3) DEMAND_MISMATCH: 売上↓×在庫↑（欠品でない）
            if (sales_trend is not None and inv_trend is not None) and (sales_trend < 0.0 and inv_trend > 0.0):
                issues.append({
                    "code": "DEMAND_MISMATCH",
                    "severity": "HIGH",
                    "evidence": {
                        "sales_trend": float(sales_trend),
                        "inventory_trend": float(inv_trend),
                        "lookback_months": lookback_m,
                    },
                    "message": "需要読みと供給（作り込み）がズレている可能性。チャネル配分/作り込み量の再設計。",
                    "recommend": [
                        {
                            "operator": "apply_channel_realloc_timeseries_csv",
                            "hint": {
                                "months": months[-2:] if len(months) >= 2 else months,
                                "from_channel": "RETAIL",
                                "to_channel": "ONLINE",
                                "rate": 0.20,
                                "reason": "shift to faster channel",
                            },
                        }
                    ],
                })

        # INV_OVER_CAP（phoneでも同名を維持）
        over_i_sum = float(np.nansum(over_i))
        if over_i_sum > 0:
            over_mask = [oi > 0 for oi in over_i.tolist()]
            over_streak, over_months = _streak_months(months, over_mask)
            issues.append(
                {
                    "code": "INV_OVER_CAP",
                    "severity": "HIGH" if over_streak >= 2 else "MED",
                    "evidence": {
                        "over_i_sum": over_i_sum,
                        "over_i_streak": over_streak,
                        "months": over_months,
                        "inventory_last": inv_last,
                        "cap_i_last": float(cap_i[~np.isnan(cap_i)][-1]) if np.any(~np.isnan(cap_i)) else None,
                        "inventory_months": inv_months,
                        "target_inventory_months": target_inv_m,
                    },
                    "message": "在庫が保管容量（soft含む）を超過。値付け/チャネル/減産の意思決定が必要。",
                    "recommend": [
                        {"operator": "apply_price_adjust_timeseries_csv", "hint": {"months": over_months[-2:] if len(over_months) >= 2 else over_months, "price_rate": -0.10}},
                        {"operator": "apply_channel_realloc_timeseries_csv", "hint": {"months": over_months[-2:] if len(over_months) >= 2 else over_months, "from_channel": "RETAIL", "to_channel": "ONLINE"}},
                    ],
                }
            )

        # OVER_STOCK_RISK
        if inv_months is not None and inv_months > target_inv_m:
            sev = "HIGH" if inv_months > target_inv_m * float(diagnose_cfg.get("over_stock_severity_ratio_high", 1.3)) else "MED"
            focus_months = months[-2:] if len(months) >= 2 else months
            issues.append(
                {
                    "code": "OVER_STOCK_RISK",
                    "severity": sev,
                    "evidence": {
                        "inventory_months": float(inv_months),
                        "target_inventory_months": float(target_inv_m),
                        "inventory_last": inv_last,
                        "avg_sales_3m": avg_sales,
                        "sales_trend_3m": sales_trend,
                        "inventory_trend_3m": inv_trend,
                    },
                    "message": "在庫月数が目標を超過。売り切り（値付け・チャネル・減産）判断点。",
                    "recommend": [
                        {"operator": "apply_price_adjust_timeseries_csv", "hint": {"months": focus_months, "price_rate": -0.05}},
                        {"operator": "production_adjust_timeseries_csv", "hint": {"months": focus_months, "rate": -0.20}},
                    ],
                }
            )

        # DEMAND_MISMATCH
        if (sales_trend is not None and inv_trend is not None) and (sales_trend < 0 and inv_trend > 0):
            focus_months = months[-2:] if len(months) >= 2 else months
            issues.append(
                {
                    "code": "DEMAND_MISMATCH",
                    "severity": "HIGH",
                    "evidence": {"sales_trend_3m": sales_trend, "inventory_trend_3m": inv_trend},
                    "message": "販売が減速し在庫が増加。需要ズレが顕在化。価格/チャネル施策の検討が必要。",
                    "recommend": [
                        {"operator": "apply_price_adjust_timeseries_csv", "hint": {"months": focus_months, "price_rate": -0.10}},
                        {"operator": "apply_channel_realloc_timeseries_csv", "hint": {"months": focus_months, "from_channel": "RETAIL", "to_channel": "ONLINE"}},
                    ],
                }
            )



        # SALES_SLUMP (在庫過多＝売り逃し ではなく 販売不振 の分岐)
        # 条件例:
        #  - lifecycle_phase が LATE/END ではない（＝売り切り局面ではない）
        #  - sales_trend がマイナス
        #  - 在庫月数が目標超 or over_i が出ている
        if (
            lifecycle_phase not in {"LATE", "END"}
            and (sales_trend is not None and sales_trend < 0)
            and (
                (inv_months is not None and inv_months > target_inv_m)
                or float(np.nansum(over_i)) > 0
            )
        ):
            focus_months = months[-2:] if len(months) >= 2 else months
            issues.append(
                {
                    "code": "SALES_SLUMP",
                    "severity": "HIGH",
                    "evidence": {
                        "lifecycle_phase": lifecycle_phase,
                        "sales_trend_3m": sales_trend,
                        "inventory_months": inv_months,
                        "target_inventory_months": target_inv_m,
                        "over_i_sum": float(np.nansum(over_i)),
                    },
                    "message": "販売不振が主因で在庫が積み上がっている可能性。売り切り局面ではなく『需要刺激＋減産』の両面で手当て。",
                    "recommend": [
                        # 需要刺激（キャンペーン等のWhat-if）…既存の demand_adjust_csv を使う
                        {"operator": "demand_adjust_csv", "hint": {"months": focus_months, "rate": +0.10}},
                        # 値付けの打ち手（過度な売り切りではなく調整）
                        {"operator": "apply_price_adjust_timeseries_csv", "hint": {"months": focus_months, "price_rate": -0.05, "mode": "stimulate"}},
                        # 減産（積み上がり防止）
                        {"operator": "production_adjust_timeseries_csv", "hint": {"months": focus_months, "rate": -0.10}},
                        # チャネル再配分（詰まり解消）
                        {"operator": "apply_channel_realloc_timeseries_csv", "hint": {"months": focus_months, "from_channel": "RETAIL", "to_channel": "ONLINE"}},
                    ],
                }
            )




        # SELL_OUT_DECISION_POINT
        if lifecycle_phase in {"LATE", "END"} and inv_months is not None and inv_months > max(target_inv_m, 2.0):
            focus_months = months[-2:] if len(months) >= 2 else months
            issues.append(
                {
                    "code": "SELL_OUT_DECISION_POINT",
                    "severity": "HIGH",
                    "evidence": {"lifecycle_phase": lifecycle_phase, "inventory_months": float(inv_months)},
                    "message": "ライフサイクル終盤で在庫滞留。売り切り（値付け・チャネル・処分）判断点。",
                    "recommend": [
                        {"operator": "apply_price_adjust_timeseries_csv", "hint": {"months": focus_months, "price_rate": -0.15, "mode": "sell_out"}},
                    ],
                }
            )

        if not issues:
            summary = "重大アラームなし。週次で継続監視し、売り切り/補給の判断点を早期検知。"
        else:
            summary = f"主要論点: {issues[0]['code']}。在庫は結果ではなく意思決定のアラーム。"

        diagnosis = {
            "summary": summary,
            "case": "smart_phone",
            "months": months,
            "metrics": {
                "service_level_avg": service_level_avg,
                "ship_util_avg": ship_util_avg,
                "prod_util_avg": prod_util_avg,
                "inventory_months": inv_months,
                "target_inventory_months": target_inv_m,
                "sales_trend_3m": sales_trend,
                "inventory_trend_3m": inv_trend,
                "lifecycle_phase": lifecycle_phase,
            },
            "issues": issues,
        }

        out_path = out_dir / "diagnosis.json"
        out_path.write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")

        meta_path = meta_dir / "diagnosis_report.json"
        meta_path.write_text(
            json.dumps(
                {
                    "run_dir": str(run_dir),
                    "out_path": str(out_path),
                    "case": "smart_phone",
                    "issue_codes": [i.get("code") for i in issues],
                    "thresholds": {
                        "service_level_target": float(diagnose_cfg.get("service_level_target", 0.98)),
                        "target_inventory_months": float(target_inv_m),
                    },
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        print(f"[one_node.diagnose] wrote: {out_path}")
        print(f"[one_node.diagnose] wrote: {meta_path}")
        return

    # ----------------------------
    # RICE / default branch (existing behavior)
    # ----------------------------
    print(
        f"[STEP:DIAGNOSE] thresholds seasonal_prebuild_lookback_months={th.seasonal_prebuild_lookback_months} "
        f"inv_months={th.seasonal_prebuild_inv_months} backlog_th={th.seasonal_prebuild_backlog_threshold} "
        f"reco_rate={th.seasonal_prebuild_reco_rate}"
    )

    issues: List[Dict[str, Any]] = []

    # SHORTAGE
    shortage = (bl_streak >= 2) or (service_level_avg is not None and service_level_avg < th.service_level_target)
    if shortage:
        severity = "HIGH" if bl_streak >= 3 else "MED"
        prod_reco = {"operator": "production_adjust_timeseries_csv", "hint": {"months": bl_months[:2], "rate": 0.10}}
        try:
            prod_reco = _recommend_production_by_crop_year(months, demand, sales, production, cap_p, config, th)
        except Exception as e:
            print(f"[one_node.diagnose] production recommend fallback due to error: {e}")

        issues.append(
            {
                "code": "SHORTAGE",
                "severity": severity,
                "evidence": {
                    "backlog_streak": bl_streak,
                    "months": bl_months,
                    "service_level_avg": service_level_avg,
                    "service_level_target": th.service_level_target,
                },
                "recommend": [
                    {"operator": "demand_adjust_csv", "hint": {"months": bl_months[:2], "rate": -0.10}},
                    prod_reco,
                ],
            }
        )

    # INV_OVER_CAP
    inv_over_mask: List[bool] = []
    for i in range(len(months)):
        inv_over_mask.append((cap_i[i] > 0 and inv[i] > cap_i[i]) or (over_i[i] > 0))
    over_streak, over_months = _streak_months(months, inv_over_mask)
    over_i_sum = float(np.nansum(over_i))
    if (over_streak >= th.inv_over_cap_streak_n) or (over_i_sum > 0):
        issues.append(
            {
                "code": "INV_OVER_CAP",
                "severity": "HIGH" if over_streak >= 3 else "MED",
                "evidence": {
                    "over_streak": over_streak,
                    "months": over_months,
                    "over_i_sum": over_i_sum,
                },
                "recommend": [
                    {"operator": "demand_adjust_csv", "hint": {"months": over_months[:2], "rate": +0.10}},
                    {"operator": "production_adjust_timeseries_csv", "hint": {"months": over_months[:2], "rate": -0.10}},
                ],
            }
        )

    # DEMAND_VOLATILITY
    demand_cv = float(np.nanstd(demand) / np.nanmean(demand)) if np.nanmean(demand) > 0 else 0.0
    if demand_cv > th.demand_cv_threshold:
        issues.append(
            {
                "code": "DEMAND_VOLATILITY",
                "severity": "MED",
                "evidence": {"demand_cv": demand_cv, "threshold": th.demand_cv_threshold},
                "recommend": [{"operator": "demand_adjust_csv", "hint": {"months": months[-2:], "rate": -0.05}}],
            }
        )

    # SEASONAL_PREBUILD（既存）
    try:
        prebuild_month_numbers = _infer_harvest_month_numbers(months, cap_p, config)
        if prebuild_month_numbers:
            inv_lean_threshold = float(th.seasonal_prebuild_inv_months)
            stress_months: List[str] = []
            for i in range(len(months)):
                inv_m = 0.0
                if sales[i] > 1e-9:
                    inv_m = inv[i] / sales[i]
                if (demand[i] > sales[i]) and (inv_m < inv_lean_threshold):
                    stress_months.append(months[i])

            if stress_months:
                issues.append(
                    {
                        "code": "SEASONAL_PREBUILD",
                        "severity": "MED",
                        "evidence": {
                            "stress_months": stress_months,
                            "inv_lean_threshold": inv_lean_threshold,
                            "harvest_month_numbers": prebuild_month_numbers,
                        },
                        "recommend": [
                            {
                                "operator": "production_adjust_timeseries_csv",
                                "hint": {
                                    "months": [m for m in months if int(m.split("-")[1]) in prebuild_month_numbers],
                                    "rate": float(th.seasonal_prebuild_reco_rate),
                                },
                            }
                        ],
                    }
                )
    except Exception as e:
        print(f"[one_node.diagnose] seasonal_prebuild skipped due to error: {e}")

    if not issues:
        summary = "重大アラームなし。継続監視で十分。"
    else:
        summary = f"issues={[i.get('code') for i in issues]}"

    diagnosis = {
        "summary": summary,
        "case": "rice",
        "months": months,
        "metrics": {
            "service_level_avg": service_level_avg,
            "ship_util_avg": ship_util_avg,
            "prod_util_avg": prod_util_avg,
            "demand_cv": demand_cv,
        },
        "issues": issues,
    }

    out_path = out_dir / "diagnosis.json"
    out_path.write_text(json.dumps(diagnosis, ensure_ascii=False, indent=2), encoding="utf-8")

    meta_path = meta_dir / "diagnosis_report.json"
    meta_path.write_text(
        json.dumps(
            {
                "run_dir": str(run_dir),
                "out_path": str(out_path),
                "issue_codes": [i.get("code") for i in issues],
                "seasonal_prebuild_lookback_months": th.seasonal_prebuild_lookback_months,
                "seasonal_prebuild_inv_months": th.seasonal_prebuild_inv_months,
                "seasonal_prebuild_reco_rate": th.seasonal_prebuild_reco_rate,
                "seasonal_prebuild_backlog_threshold": th.seasonal_prebuild_backlog_threshold,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[one_node.diagnose] wrote: {out_path}")
    print(f"[one_node.diagnose] wrote: {meta_path}")
