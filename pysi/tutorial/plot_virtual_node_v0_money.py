# pysi/tutorial/plot_virtual_node_v0_money.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 既存互換：単一CSVを描画（そのまま残す）
# ============================================================
def plot_money_timeseries(csv_path: str | Path, title: str = "Money overview (run bundle)"):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # expected columns:
    # month, Revenue, Process_Cost, Purchase_Cost, Profit, ProfitRate
    months = df["month"].astype(str).tolist()
    x = np.arange(len(months))

    revenue = df["Revenue"].to_numpy()
    proc = df["Process_Cost"].to_numpy()
    purch = df["Purchase_Cost"].to_numpy()
    profit = df["Profit"].to_numpy()
    pr = df["ProfitRate"].to_numpy()  # ratio

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title(title)

    ax.bar(x, proc, label="Process_Cost")
    ax.bar(x, purch, bottom=proc, label="Purchase_Cost")
    ax.bar(x, profit, label="Profit")

    ax.plot(x, revenue, marker="o", label="Revenue")

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount")

    ax2 = ax.twinx()
    ax2.plot(x, pr * 100.0, marker="o", label="ProfitRate(%)")
    ax2.set_ylabel("ProfitRate (%)")

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")

    plt.tight_layout()
    plt.show()


# ============================================================
# CSV utilities
# ============================================================
def _read_money_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "month" not in df.columns:
        raise ValueError(f"{path} must contain 'month' column")
    df["month"] = df["month"].astype(str)
    return df


def _align_series(df: pd.DataFrame, months: list[str], col: str) -> np.ndarray:
    if col not in df.columns:
        return np.zeros(len(months))
    m2v = dict(zip(df["month"], df[col].astype(float)))
    return np.array([m2v.get(m, 0.0) for m in months])


# ============================================================
# CSV BASED : BEFORE / AFTER（上下2段）
# ============================================================
def plot_money_before_after(
    before_csv: str | Path,
    after_csv: str | Path,
    title: str = "Money overview (Before / After)",
):
    before = _read_money_csv(Path(before_csv))
    after = _read_money_csv(Path(after_csv))

    # 時間軸を統合（順序保持）
    months = list(dict.fromkeys(before["month"].tolist() + after["month"].tolist()))
    x = np.arange(len(months))

    def _plot(ax, df: pd.DataFrame, subtitle: str):
        revenue = _align_series(df, months, "Revenue")
        proc = _align_series(df, months, "Process_Cost")
        purch = _align_series(df, months, "Purchase_Cost")
        profit = _align_series(df, months, "Profit")
        pr = _align_series(df, months, "ProfitRate")

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
    plt.show()


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

    plot_money_before_after(
        before,
        after,
        title=f"Money overview (Before / After) : {run_dir.name}",
    )


# ============================================================
# CLI entry (従来どおり latest を描く)
# ============================================================
def main():
    latest = Path("runs/one_node/_latest/output/money_timeseries.csv")
    if latest.exists():
        plot_money_timeseries(latest, title="Money overview (latest run)")
        return

    raise SystemExit("money_timeseries.csv not found. Pass a path or run from repo root.")


if __name__ == "__main__":
    main()
