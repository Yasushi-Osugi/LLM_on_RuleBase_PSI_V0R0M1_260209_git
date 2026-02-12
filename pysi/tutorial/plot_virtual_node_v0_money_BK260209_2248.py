# pysi/tutorial/plot_virtual_node_v0_money.py
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_money_timeseries(csv_path: str | Path, title: str = "Money overview (run bundle)"):
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)

    # expected columns: month, Revenue, Process_Cost, Purchase_Cost, Profit, ProfitRate
    months = df["month"].astype(str).tolist()
    x = np.arange(len(months))

    revenue = df["Revenue"].to_numpy()
    proc = df["Process_Cost"].to_numpy()
    purch = df["Purchase_Cost"].to_numpy()
    profit = df["Profit"].to_numpy()
    pr = df["ProfitRate"].to_numpy()  # ratio

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_title(title)

    # bars (costs & profit)
    ax.bar(x, proc, label="Process_Cost")
    ax.bar(x, purch, bottom=proc, label="Purchase_Cost")
    # Profit is shown as separate bar (can be negative)
    ax.bar(x, profit, label="Profit")

    # revenue as line (primary axis)
    ax.plot(x, revenue, marker="o", label="Revenue")

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")
    ax.set_xlabel("Month")
    ax.set_ylabel("Amount")

    # profit rate on secondary axis (%)
    ax2 = ax.twinx()
    ax2.plot(x, pr * 100.0, marker="o", label="ProfitRate(%)")
    ax2.set_ylabel("ProfitRate (%)")

    # merge legends
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right")

    plt.tight_layout()
    plt.show()


def main():
    # default: latest run
    latest = Path("runs/one_node/_latest/output/money_timeseries.csv")
    if latest.exists():
        plot_money_timeseries(latest, title="Money overview (latest run)")
        return

    raise SystemExit("money_timeseries.csv not found. Pass a path or run from repo root.")


if __name__ == "__main__":
    main()
