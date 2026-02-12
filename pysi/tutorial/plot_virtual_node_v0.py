
# pysi/tutorial/plot_virtual_node_v0.py

# pysi/tutorial/plot_virtual_node_v0.py
from __future__ import annotations

from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 既存互換：model から直接描画（そのまま残す）
# ============================================================
def plot_phone_v0(model, title: str = "PSI Overview (Demand / Supply / Inventory)") -> None:
    months: List[str] = list(model.months)
    x = np.arange(len(months))

    demand = model.demand.reindex(months).astype(float).values
    sales = model.sales.reindex(months).astype(float).values
    backlog = model.backlog.reindex(months).astype(float).values
    prod = model.production_plan.reindex(months).astype(float).values
    inv_total = model.inv.reindex(months).astype(float).values
    waste = model.waste.reindex(months).astype(float).values

    p_cap = model.p_cap_series.reindex(months).astype(float).values
    s_cap = model.s_cap_series.reindex(months).astype(float).values
    i_cap = model.i_cap_series.reindex(months).astype(float).values

    cap_mode = getattr(model, "cap_mode", "soft")
    shelf_life = getattr(model, "shelf_life", None)

    # ---- Inventory split ----
    if cap_mode == "soft":
        i_over = np.maximum(inv_total - i_cap, 0.0)
        i_norm = np.minimum(inv_total, i_cap)
    else:
        i_norm = inv_total
        i_over = None

    if shelf_life:
        title = f"{title} (FIFO+Expiration: {int(shelf_life)}m)"

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Lots")

    ax.fill_between(x, 0, i_cap, alpha=0.12, label="I_cap", zorder=0)
    ax.plot(x, s_cap, linestyle="--", linewidth=3, label="S_cap", zorder=2)

    ax.plot(x, demand, marker="o", label="Demand", zorder=3)
    ax.plot(x, sales, marker="o", label="Sales", zorder=3)
    ax.plot(x, backlog, marker="o", label="Backlog", zorder=3)

    bar_w = 0.22
    ax.bar(x - bar_w, prod, width=bar_w, label="Production", zorder=1)
    ax.bar(x, i_norm, width=bar_w, label="Inventory", zorder=1)
    ax.bar(x + bar_w, waste, width=bar_w, label="Waste", zorder=1)

    if i_over is not None:
        ax.bar(x, i_over, bottom=i_cap, alpha=0.25, label="over_i", zorder=2)

    ax.scatter(x, p_cap, marker="^", s=90, label="P_cap", zorder=6)

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()


# ============================================================
# CSV utilities
# ============================================================
def _read_result_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "month" not in df.columns:
        raise ValueError(f"{path} must contain 'month' column")
    df["month"] = df["month"].astype(str)
    return df


def _align_series(df: pd.DataFrame, months: List[str], col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(months))
    m2v = dict(zip(df["month"], df[col].astype(float)))
    return np.array([m2v.get(m, 0.0) for m in months])


# ============================================================
# CSV BASED : BEFORE / AFTER (上下2段)
# ============================================================
def plot_before_after_from_csv(
    before_csv: str | Path,
    after_csv: str | Path,
    title: str = "PSI Overview (Before / After)",
) -> None:
    before = _read_result_csv(Path(before_csv))
    after = _read_result_csv(Path(after_csv))

    months = list(dict.fromkeys(before["month"].tolist() + after["month"].tolist()))
    x = np.arange(len(months))

    def _plot(ax, df: pd.DataFrame, subtitle: str):
        demand = _align_series(df, months, "demand")
        sales = _align_series(df, months, "sales")
        backlog = _align_series(df, months, "backlog")
        prod = _align_series(df, months, "production")
        inv = _align_series(df, months, "inventory")
        waste = _align_series(df, months, "waste")
        cap_p = _align_series(df, months, "cap_P")
        cap_s = _align_series(df, months, "cap_S")
        cap_i = _align_series(df, months, "cap_I")
        over_i = _align_series(df, months, "over_i")

        ax.set_title(subtitle)
        ax.set_ylabel("Lots")

        if np.nanmax(cap_i) > 0:
            ax.fill_between(x, 0, cap_i, alpha=0.12, label="I_cap", zorder=0)
        if np.nanmax(cap_s) > 0:
            ax.plot(x, cap_s, linestyle="--", linewidth=3, label="S_cap", zorder=2)

        ax.plot(x, demand, marker="o", label="Demand", zorder=3)
        ax.plot(x, sales, marker="o", label="Sales", zorder=3)
        ax.plot(x, backlog, marker="o", label="Backlog", zorder=3)

        bar_w = 0.22
        ax.bar(x - bar_w, prod, width=bar_w, label="Production", zorder=1)
        ax.bar(x, inv, width=bar_w, label="Inventory", zorder=1)
        ax.bar(x + bar_w, waste, width=bar_w, label="Waste", zorder=1)

        if np.nanmax(over_i) > 0:
            ax.bar(x, over_i, bottom=cap_i, alpha=0.25, label="over_i", zorder=2)

        if np.nanmax(cap_p) > 0:
            ax.scatter(x, cap_p, marker="^", s=90, label="P_cap", zorder=6)

        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=45, ha="right")
        ax.legend(loc="upper right")

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(title)

    _plot(axes[0], before, "BEFORE (input_snapshot)")
    _plot(axes[1], after, "AFTER  (output)")

    fig.tight_layout()
    plt.show()


# ============================================================
# run_dir BASED helper
# ============================================================
def plot_run_dir_before_after(run_dir: str | Path) -> None:
    run_dir = Path(run_dir)

    before = run_dir / "input_snapshot" / "one_node_result_timeseries.csv"
    after = run_dir / "output" / "one_node_result_timeseries.csv"

    if not before.exists():
        raise FileNotFoundError(f"input_snapshot not found: {before}")
    if not after.exists():
        raise FileNotFoundError(f"output not found: {after}")

    plot_before_after_from_csv(
        before,
        after,
        title=f"PSI Overview (Before / After) : {run_dir.name}",
    )
