# pysi/tutorial/phone_v0_adapter.py
# pysi/tutorial/virtual_node_v0_adapter.py

from __future__ import annotations

from pathlib import Path
import pandas as pd


def business_to_virtual_node_adapter(
    data_dir: str,
    business_fname: str = "business_timeseries.csv",
    out_fname: str = "virtual_node_timeseries.csv",
    node_name: str = "GLOBAL",
):
    """
    Convert business_timeseries.csv into long-format virtual_node_timeseries.csv

    Output columns:
      - month
      - node_name
      - item        (demand, planning_S, production, sell_price)
      - value
    """

    data_dir = Path(data_dir)
    src = data_dir / business_fname
    dst = data_dir / out_fname

    if not src.exists():
        raise FileNotFoundError(src)

    df = pd.read_csv(src)

    required_cols = ["month", "demand", "planning_S", "production"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"{business_fname} must have column '{c}'")

    has_price = "sell_price" in df.columns

    rows = []
    NODE = node_name  # ★ 追加：node_name を明示

    for _, r in df.iterrows():
        m = r["month"]

        demand = float(r["demand"])
        ship = float(r["planning_S"])
        prod = float(r["production"])

        rows.append({"month": m, "node_name": NODE, "item": "demand", "value": demand})
        rows.append({"month": m, "node_name": NODE, "item": "planning_S", "value": ship})
        rows.append({"month": m, "node_name": NODE, "item": "production", "value": prod})

        if has_price:
            price = float(r["sell_price"])
            rows.append({"month": m, "node_name": NODE, "item": "sell_price", "value": price})

    # ★ columns を明示（順序保証）
    out_df = pd.DataFrame(rows, columns=["month", "node_name", "item", "value"])

    dst.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(dst, index=False)

    print(f"[BT_ADAPTER] wrote {dst} rows={len(out_df)} node={NODE}")
