# pysi/plugins/one_node/after_run/export_bundle.py
# pysi/plugins/one_node/after_run/export_bundle.py
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, List

import numpy as np
import pandas as pd

from pysi.core.hooks.core import HookBus
from pysi.tutorial.plot_virtual_node_v0 import plot_phone_v0


PLUGIN_META = {
    "name": "export_bundle",
    "version": "0.2.1",
    "category": "one_node.after_run",
    "inputs": ["model", "run_dir"],
    "outputs": ["one_node_result_timeseries.csv", "kpi_summary.json", "psi_overview.png", "run_meta.json"],
    "description": "Export one_node run bundle (csv/json/png/meta). Save png without blocking GUI (no plt.show()).",
}


def register(bus: HookBus) -> None:
    # one_node.after_run で呼ばれる
    bus.add_action("one_node.after_run", export_bundle, priority=50)


def _as_list(x: Any, months: List[str]) -> List[float]:
    """Duck-typing: pandas Series/DataFrame列/リストなどを months 長の float list にする。"""
    if x is None:
        return [0.0] * len(months)

    # pandas Series: index が month のケース
    try:
        return [float(x.loc[m]) for m in months]  # type: ignore[attr-defined]
    except Exception:
        pass

    # list-like
    try:
        xs = list(x)
        if len(xs) == len(months):
            return [float(v) for v in xs]
        # 長さ不一致なら、可能なら切り詰め/パディング
        out = [float(v) for v in xs[: len(months)]]
        if len(out) < len(months):
            out.extend([0.0] * (len(months) - len(out)))
        return out
    except Exception:
        return [0.0] * len(months)


def export_bundle(**ctx: Any) -> None:
    run_dir = Path(str(ctx.get("run_dir", "")))
    if not run_dir:
        return

    out_dir = run_dir / "output"
    meta_dir = run_dir / "meta"
    out_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)

    model = ctx.get("model")
    if model is None:
        return

    months: List[str] = list(getattr(model, "months", []))
    if not months:
        return

    # --- collect series (duck-typing) ---
    demand = getattr(model, "demand", None)
    inv = getattr(model, "inv", getattr(model, "inventory", None))
    backlog = getattr(model, "backlog", None)
    waste = getattr(model, "waste", None)
    over_i = getattr(model, "over_i", None)

    production = getattr(
        model,
        "production",
        getattr(model, "production_plan", getattr(model, "prod_plan", None)),
    )
    sales = getattr(model, "sales", getattr(model, "ship", getattr(model, "S", None)))

    # caps (optional)
    pcap = getattr(model, "p_cap_series", getattr(model, "p_cap", getattr(model, "P_cap", None)))
    scap = getattr(model, "s_cap_series", getattr(model, "s_cap", getattr(model, "S_cap", None)))
    icap = getattr(model, "i_cap_series", getattr(model, "i_cap", getattr(model, "I_cap", None)))

    df = pd.DataFrame(
        {
            "month": months,
            "demand": _as_list(demand, months),
            "production": _as_list(production, months),
            "sales": _as_list(sales, months),
            "inventory": _as_list(inv, months),
            "backlog": _as_list(backlog, months),
            "waste": _as_list(waste, months),
            "over_i": _as_list(over_i, months),
            "cap_P": _as_list(pcap, months) if pcap is not None else [np.nan] * len(months),
            "cap_S": _as_list(scap, months) if scap is not None else [np.nan] * len(months),
            "cap_I": _as_list(icap, months) if icap is not None else [np.nan] * len(months),
        }
    )
    df.to_csv(out_dir / "one_node_result_timeseries.csv", index=False)

    kpi = {
        "total_demand": float(df["demand"].sum()),
        "total_production": float(df["production"].sum()),
        "ending_inventory": float(df["inventory"].iloc[-1]) if len(df) else 0.0,
        "backlog_end": float(df["backlog"].iloc[-1]) if len(df) else 0.0,
        "total_waste": float(df["waste"].sum()),
        "total_over_i": float(df["over_i"].sum()),
    }
    (out_dir / "kpi_summary.json").write_text(json.dumps(kpi, indent=2, ensure_ascii=False), encoding="utf-8")

    meta = {
        "data_dir": ctx.get("data_dir"),
        "args": ctx.get("args", {}),
        "run_dir": str(run_dir),
        "plugin": PLUGIN_META,
    }
    (meta_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    # --- copy meta/*.json to output/ (for easy artifact sharing) ---
    try:
        copied_meta = 0
        for p in meta_dir.glob("*.json"):
            shutil.copy(p, out_dir / p.name)
            copied_meta += 1
        if copied_meta:
            print(f"[one_node.export_bundle] copied meta json -> output/: {copied_meta} file(s)")
    except Exception as e:
        # meta copy failure should not break run bundle
        print(f"[one_node.export_bundle] WARN: meta json copy failed: {e}")

    # --- IMPORTANT PART ---
    # plot_phone_v0() が plt.show() する前提を利用しつつ、
    # ここでは「保存だけして show しない（=ブロックしない）」に切り替える。
    try:
        import matplotlib.pyplot as plt  # import はここ（backendのブレを最小化）

        # ★変更点：run_id 付きファイル名で保存
        run_id = Path(str(run_dir)).name if run_dir else ""
        suffix = f"__{run_id}" if run_id else ""
        png_path = out_dir / f"psi_overview{suffix}.png"

        orig_show = plt.show

        def _save_only_show(*args: Any, **kwargs: Any) -> None:
            """Save the current figure but DO NOT open GUI window (non-blocking)."""
            fig = None
            try:
                fig = plt.gcf()
                fig.savefig(png_path, dpi=160)
                print(f"[one_node.export_bundle] wrote: {png_path}")
            except Exception as e:
                print(f"[one_node.export_bundle] WARN: failed to save {png_path}: {e}")
            finally:
                # Do not call orig_show() -> avoid blocking GUI.
                try:
                    if fig is not None:
                        plt.close(fig)
                except Exception:
                    pass
            return None

        plt.show = _save_only_show  # type: ignore[assignment]
        try:
            plot_phone_v0(model, title="One-node PSI overview (run bundle)")
        finally:
            plt.show = orig_show  # type: ignore[assignment]

    except Exception as e:
        # ここで落ちても run bundle の csv/json/meta は残る
        print(f"[one_node.export_bundle] WARN: plotting failed: {e}")
