# pysi/tutorial/plot_virtual_node_v0.py
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

import matplotlib
# Headless(Codex等)でも確実に描画できるようにする（pyplot importより前）
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CSV reader (legacy helper)
# ============================================================
def _read_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "month" not in df.columns:
        raise ValueError(f"missing 'month' column: {csv_path}")
    return df


# ============================================================
# Model plot (run bundle)
# ============================================================
def plot_phone_v0(
    model,
    title: str = "PSI Overview (Demand / Supply / Inventory)",
    save_path: str | Path | None = None,
    show: bool | None = None,
) -> None:
    """
    PhoneV0Model (attribute access) を前提にした PSI overview。
    - save_path を渡すと PNG 保存
    - show は None の場合、save_path が無ければ True / あれば False に自動判定
    """
    # PhoneV0Model は dict ではない想定（.get() は使わない）
    months: List[str] = list(getattr(model, "months", []))
    if not months:
        months = []

    demand = np.array(getattr(model, "demand", []), dtype=float)
    sales = np.array(getattr(model, "sales", []), dtype=float)
    backlog = np.array(getattr(model, "backlog", []), dtype=float)

    production = np.array(getattr(model, "production", []), dtype=float)
    inventory = np.array(getattr(model, "inventory", []), dtype=float)
    waste = np.array(getattr(model, "waste", []), dtype=float)

    s_cap = np.array(getattr(model, "S_cap", []), dtype=float)
    p_cap = np.array(getattr(model, "P_cap", []), dtype=float)
    i_cap = np.array(getattr(model, "I_cap", []), dtype=float)
    over_i = np.array(getattr(model, "over_i", []), dtype=float)

    x = np.arange(len(months)) if months else np.arange(max(len(demand), len(sales), len(production), 0))

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Lots")

    # inventory cap (area)
    if len(i_cap) == len(x):
        ax.fill_between(x, 0, i_cap, alpha=0.15, label="I_cap")

    # S cap (line)
    if len(s_cap) == len(x):
        ax.plot(x, s_cap, linestyle="--", linewidth=3, label="S_cap")

    # demand/sales/backlog
    if len(demand) == len(x):
        ax.plot(x, demand, marker="o", label="Demand")
    if len(sales) == len(x):
        ax.plot(x, sales, marker="o", label="Sales")
    if len(backlog) == len(x):
        ax.plot(x, backlog, marker="o", label="Backlog")

    # P cap (marker)
    if len(p_cap) == len(x):
        ax.plot(x, p_cap, marker="^", linestyle="None", markersize=10, label="P_cap")

    # production bars
    if len(production) == len(x):
        ax.bar(x - 0.15, production, width=0.3, label="Production")

    # inventory bars
    if len(inventory) == len(x):
        ax.bar(x + 0.15, inventory, width=0.3, label="Inventory")

    # waste bars
    if len(waste) == len(x):
        ax.bar(x, waste, width=0.25, label="Waste")

    # over_i (optional)
    if len(over_i) == len(x) and np.nanmax(over_i) > 0:
        ax.bar(x, over_i, width=0.25, alpha=0.25, label="over_i")

    if months:
        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=45, ha="right")

    ax.legend(loc="upper right")
    fig.tight_layout()

    if show is None:
        show = save_path is None

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_psi] saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# CSV utilities (current)
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
    # col が無い場合はゼロ系列（落ちないのが正義）
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
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    before = _read_result_csv(Path(before_csv))
    after = _read_result_csv(Path(after_csv))

    # ------------------------------------------------------------
    # Column normalization
    #   - accept both legacy TitleCase and current lowercase
    #   - unify to "current" columns used in this function:
    #       month,demand,production,sales,inventory,backlog,waste,over_i,cap_P,cap_S,cap_I
    # ------------------------------------------------------------
    def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
        # case-insensitive mapping
        mapping = {
            "month": "month",
            "demand": "demand",
            "production": "production",
            "sales": "sales",
            "inventory": "inventory",
            "backlog": "backlog",
            "waste": "waste",
            "over_i": "over_i",
            "cap_p": "cap_P",
            "cap_s": "cap_S",
            "cap_i": "cap_I",

            # legacy TitleCase variants
            "demand ": "demand",  # just in case of trailing spaces
        }

        # additional legacy names (exact but handled case-insensitively)
        legacy = {
            "Demand": "demand",
            "Production": "production",
            "Sales": "sales",
            "Inventory": "inventory",
            "Backlog": "backlog",
            "Waste": "waste",
            "Over_I": "over_i",
            "overI": "over_i",
            "P_cap": "cap_P",
            "S_cap": "cap_S",
            "I_cap": "cap_I",
        }

        rename: dict[str, str] = {}
        for c in df.columns:
            raw = str(c).strip()
            low = raw.lower()
            if raw in legacy:
                rename[c] = legacy[raw]
                continue
            if low in mapping:
                rename[c] = mapping[low]

        if rename:
            df = df.rename(columns=rename)

        # month は常に str に統一
        if "month" in df.columns:
            df["month"] = df["month"].astype(str)

        return df

    before = _normalize_cols(before)
    after = _normalize_cols(after)

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

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_psi] saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


#
# --- compatibility alias (過去コードが plot_before_after を呼んでも死なないように) ---
#
plot_before_after = plot_before_after_from_csv


# ============================================================
# run_dir BASED helper
# ============================================================
def plot_run_dir_before_after(run_dir: str | Path) -> None:
    """
    Backward/CLI compatibility for tools.run_operator_queue.py
    Expected signature: plot_run_dir_before_after(run_dir)
    """
    run_dir = Path(run_dir)

    before = run_dir / "input_snapshot" / "one_node_result_timeseries.csv"
    after = run_dir / "output" / "one_node_result_timeseries.csv"

    if not before.exists():
        raise FileNotFoundError(f"input_snapshot not found: {before}")
    if not after.exists():
        raise FileNotFoundError(f"output not found: {after}")

    out_png = run_dir / "output" / f"psi_before_after__{run_dir.name}.png"

    # plot_before_after は alias でも残すが、明示的に本体を呼ぶ
    plot_before_after_from_csv(
        before,
        after,
        title=f"PSI overview (Before / After) : {run_dir.name}",
        save_path=out_png,
        show=False,
    )
