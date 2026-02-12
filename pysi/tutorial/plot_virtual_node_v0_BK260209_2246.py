
# pysi/tutorial/plot_virtual_node_v0.py


from __future__ import annotations

from typing import List
import numpy as np
import matplotlib.pyplot as plt


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

    # ---- Inventory split (<= I_cap) and over_i (> I_cap) ----
    i_over = None
    if cap_mode == "soft":
        if hasattr(model, "over_i") and model.over_i is not None:
            i_over = model.over_i.reindex(months).astype(float).values
            i_over = np.maximum(i_over, 0.0)
        else:
            i_over = np.maximum(inv_total - i_cap, 0.0)

        i_norm = np.minimum(inv_total, i_cap)
    else:
        i_norm = inv_total
        i_over = None

    # タイトル調整（Pharmaのときだけ）
    if shelf_life:
        title = f"{title}  (FIFO+Expiration: {int(shelf_life)}m)"

    fig, ax = plt.subplots(figsize=(16, 6))
    ax.set_title(title)
    ax.set_xlabel("Month")
    ax.set_ylabel("Lots")

    # --- I_cap (BOX) ---
    ax.fill_between(x, 0, i_cap, alpha=0.12, label="I_cap: Inventory capacity (BOX)", zorder=0)

    # --- S_cap (dashed line) ---
    ax.plot(x, s_cap, linestyle="--", linewidth=3, label="S_cap: Service capacity (ceiling)", zorder=2)

    # --- demand / sales / backlog ---
    ax.plot(x, demand, marker="o", label="Demand (MKT)", zorder=3)
    ax.plot(x, sales, marker="o", label="Sales (S, fulfilled)", zorder=3)
    ax.plot(x, backlog, marker="o", label="Backlog (CO)", zorder=3)

    # --- bars: P / I / waste ---
    bar_w = 0.22
    ax.bar(x - bar_w, prod, width=bar_w, label="Production (P)", zorder=1)

    ax.bar(x, i_norm, width=bar_w, label="Inventory (≤ I_cap)", zorder=1)

    waste_label = "Waste"
    if shelf_life:
        waste_label = "Waste (expired + overflow)"
    ax.bar(x + bar_w, waste, width=bar_w, label=waste_label, zorder=1)

    if cap_mode == "soft" and i_over is not None:
        ax.bar(
            x,
            i_over,
            width=bar_w,
            bottom=i_cap,
            alpha=0.25,
            label="over_i: Excess over I_cap (soft)",
            zorder=2,
        )

    # --- P_cap (▲) ---
    ax.scatter(x, p_cap, marker="^", s=90, label="P_cap (▲)", zorder=6)

    # --- P_cap binding highlight (▲) ---
    bind = set(getattr(model, "p_cap_binding_months", []) or [])
    if bind:
        idx = [i for i, m in enumerate(months) if m in bind]
        if idx:
            ax.scatter(
                np.array(idx),
                p_cap[np.array(idx)],
                marker="^",
                s=140,
                label="P_cap binding (▲)",
                zorder=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha="right")

    ax2 = ax.twinx()
    ax2.set_ylabel("Inventory")
    ymax = max(np.nanmax(i_cap), np.nanmax(inv_total))
    if cap_mode == "soft" and i_over is not None:
        ymax = max(ymax, np.nanmax(i_cap + i_over))
    ax2.set_ylim(0, ymax * 1.2 if len(months) else 1)

    ax.legend(loc="upper right")
    fig.tight_layout()
    plt.show()
