# pysi/tutorial/plot_virtual_node_v0_money.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

import matplotlib
# Headless(Codex等)でも確実に描画できるようにする（pyplot importより前）
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ============================================================
# CSV reader
# ============================================================
def _read_money_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "month" not in df.columns:
        raise ValueError(f"missing 'month' column: {csv_path}")
    return df


def _align_series(df: pd.DataFrame, months: list[str], col: str) -> np.ndarray:
    m = dict(zip(df["month"].astype(str), df[col].astype(float)))
    return np.array([m.get(mm, 0.0) for mm in months], dtype=float)


# ============================================================
# CSV BASED : run bundle money (single)
# ============================================================
def plot_money_timeseries(
    csv_path: str | Path,
    title: str = "Money overview (run bundle)",
    save_path: str | Path | None = None,
    show: bool | None = None,
):
    """
    run_one_node4plugin などから呼ばれる想定の「単体」金額グラフ。
    旧挙動互換:
      - save_path が None なら show=True 相当（GUI環境向け）
      - save_path を渡すと保存優先（Codex/Headless向け）
    """
    df = _read_money_csv(Path(csv_path))

    months = df["month"].astype(str).tolist()
    x = np.arange(len(months))

    revenue = df["Revenue"].astype(float).to_numpy() if "Revenue" in df.columns else np.zeros_like(x, dtype=float)
    proc = df["Process_Cost"].astype(float).to_numpy() if "Process_Cost" in df.columns else np.zeros_like(x, dtype=float)
    purch = df["Purchase_Cost"].astype(float).to_numpy() if "Purchase_Cost" in df.columns else np.zeros_like(x, dtype=float)
    profit = df["Profit"].astype(float).to_numpy() if "Profit" in df.columns else np.zeros_like(x, dtype=float)
    pr = df["ProfitRate"].astype(float).to_numpy() if "ProfitRate" in df.columns else np.zeros_like(x, dtype=float)

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_title(title)
    ax.set_ylabel("Amount")

    # cost stack
    ax.bar(x, proc, label="Process_Cost")
    ax.bar(x, purch, bottom=proc, label="Purchase_Cost")

    # profit
    ax.bar(x, profit, label="Profit")

    # revenue
    ax.plot(x, revenue, marker="o", label="Revenue")

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")

    # profit rate
    ax2 = ax.twinx()
    ax2.plot(x, pr * 100.0, marker="o", label="ProfitRate(%)")
    ax2.set_ylabel("ProfitRate (%)")

    # merge legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")

    plt.tight_layout()

    if show is None:
        show = save_path is None

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_money] saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# CSV BASED : BEFORE / AFTER（上下2段）
# ============================================================
def plot_money_before_after(
    before_csv: str | Path,
    after_csv: str | Path,
    title: str = "Money overview (Before / After)",
    save_path: str | Path | None = None,
    show: bool | None = None,
):
    before = _read_money_csv(Path(before_csv))
    after = _read_money_csv(Path(after_csv))

    # 時間軸を統合（順序保持）
    months = list(dict.fromkeys(before["month"].tolist() + after["month"].tolist()))
    x = np.arange(len(months))

    def _plot(ax, df: pd.DataFrame, subtitle: str):
        revenue = _align_series(df, months, "Revenue") if "Revenue" in df.columns else np.zeros_like(x, dtype=float)
        proc = _align_series(df, months, "Process_Cost") if "Process_Cost" in df.columns else np.zeros_like(x, dtype=float)
        purch = _align_series(df, months, "Purchase_Cost") if "Purchase_Cost" in df.columns else np.zeros_like(x, dtype=float)
        profit = _align_series(df, months, "Profit") if "Profit" in df.columns else np.zeros_like(x, dtype=float)
        pr = _align_series(df, months, "ProfitRate") if "ProfitRate" in df.columns else np.zeros_like(x, dtype=float)

        ax.set_title(subtitle)
        ax.set_ylabel("Amount")

        # cost stack
        ax.bar(x, proc, label="Process_Cost")
        ax.bar(x, purch, bottom=proc, label="Purchase_Cost")

        # profit (can be negative)
        ax.bar(x, profit, label="Profit")

        # revenue
        ax.plot(x, revenue, marker="o", label="Revenue")

        ax.set_xticks(x)
        ax.set_xticklabels(months, rotation=45, ha="right")

        # profit rate (secondary axis)
        ax2 = ax.twinx()
        ax2.plot(x, pr * 100.0, marker="o", label="ProfitRate(%)")
        ax2.set_ylabel("ProfitRate (%)")

        # merge legends
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right")

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.suptitle(title)

    _plot(axes[0], before, "BEFORE (input_snapshot)")
    _plot(axes[1], after, "AFTER  (output)")

    fig.tight_layout()

    if show is None:
        show = save_path is None

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"[plot_money] saved: {save_path}")

    if show:
        plt.show()

    plt.close(fig)


# ============================================================
# run_dir BASED helper
# ============================================================
def plot_money_run_dir_before_after(run_dir: str | Path):
    run_dir = Path(run_dir)

    before = run_dir / "input_snapshot" / "money_timeseries.csv"
    after = run_dir / "output" / "money_timeseries.csv"

    if not before.exists():
        raise FileNotFoundError(f"input_snapshot not found: {before}")
    if not after.exists():
        raise FileNotFoundError(f"output not found: {after}")

    out_png = run_dir / "output" / f"money_before_after__{run_dir.name}.png"

    plot_money_before_after(
        before,
        after,
        title=f"Money overview (Before / After) : {run_dir.name}",
        save_path=out_png,
        show=False,
    )


# --- compatibility alias (run_operator_queue expects this name) ---
plot_run_dir_before_after = plot_money_run_dir_before_after
