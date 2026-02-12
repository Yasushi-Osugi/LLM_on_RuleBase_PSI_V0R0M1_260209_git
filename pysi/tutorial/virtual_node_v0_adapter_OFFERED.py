# pysi/tutorial/phone_v0_adapter.py
# pysi/tutorial/virtual_node_v0_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def _read_timeseries_long(data_dir: str, timeseries_name: str) -> pd.DataFrame:
    path = Path(data_dir) / timeseries_name
    if not path.exists():
        raise FileNotFoundError(f"timeseries file not found: {path}")

    df = _read_csv(str(path))
    required = ["month", "node_name", "item", "value"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"{path} must have columns {required}, got {list(df.columns)}"
        )
    return df


def _read_capacity(data_dir: str, name: str) -> pd.DataFrame:
    path = Path(data_dir) / name
    if not path.exists():
        raise FileNotFoundError(f"capacity file not found: {path}")

    df = _read_csv(str(path))
    # expected columns: month, cap
    # tolerate variations (Month/capacity etc.) by simple aliasing if needed
    cols = [c.strip() for c in df.columns]
    df.columns = cols
    if "month" not in df.columns:
        # try common variants
        for alt in ["Month", "MONTH"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "month"})
                break
    if "cap" not in df.columns:
        for alt in ["capacity", "CAP", "Cap"]:
            if alt in df.columns:
                df = df.rename(columns={alt: "cap"})
                break
    if "month" not in df.columns or "cap" not in df.columns:
        raise ValueError(
            f"{path} must have columns ['month','cap'], got {list(df.columns)}"
        )
    return df[["month", "cap"]].copy()


@dataclass
class PhoneV0Model:
    months: List[str]
    node_name: str

    demand: np.ndarray
    production: np.ndarray
    sales: np.ndarray
    inv: np.ndarray
    backlog: np.ndarray
    waste: np.ndarray

    cap_p: np.ndarray
    cap_s: np.ndarray
    cap_i: np.ndarray
    over_i: np.ndarray


def load_phone_v0(
    data_dir: str,
    cap_mode: str = "soft",
    timeseries_name: Optional[str] = None,
    shelf_life: Optional[int] = None,
) -> PhoneV0Model:
    """
    Virtual single-node monthly PSI.
    - demand & production_plan come from virtual_node_timeseries_PSI.csv (preferred, long format)
    - fallback: virtual_node_timeseries.csv
    - capacity_* come from capacity_P/S/I.csv

    If shelf_life is provided (e.g. 3), enable FIFO inventory buckets + expiration each month:
      - inventory is held in age buckets (0..shelf_life-1)
      - each month, after sales and capacity handling, the oldest bucket expires (added to waste)
    """
    if timeseries_name is None:
        # Prefer split PSI input if present
        cand = "virtual_node_timeseries_PSI.csv"
        timeseries_name = cand if (Path(data_dir) / cand).exists() else "virtual_node_timeseries.csv"

    df = _read_timeseries_long(data_dir, timeseries_name).copy()

    # normalize month as string
    df["month"] = df["month"].astype(str).str.strip()
    df["node_name"] = df["node_name"].astype(str).str.strip()
    df["item"] = df["item"].astype(str).str.strip()

    # single node assumption: take the first node_name in file
    node_name = df["node_name"].iloc[0]
    # if multiple node_name exist, keep only the first (for now)
    df = df[df["node_name"] == node_name].copy()

    # Build month index (sorted unique)
    months = sorted(df["month"].unique().tolist())

    def _series(item: str) -> np.ndarray:
        sub = df[df["item"] == item][["month", "value"]].copy()
        if sub.empty:
            return np.zeros(len(months), dtype=float)
        m2v = dict(zip(sub["month"], sub["value"]))
        return np.array([float(m2v.get(m, 0.0)) for m in months], dtype=float)

    demand = _series("demand")
    production_plan = _series("production")

    # capacities
    cap_p_df = _read_capacity(data_dir, "capacity_P.csv")
    cap_s_df = _read_capacity(data_dir, "capacity_S.csv")
    cap_i_df = _read_capacity(data_dir, "capacity_I.csv")

    def _cap_array(cap_df: pd.DataFrame) -> np.ndarray:
        cap_df = cap_df.copy()
        cap_df["month"] = cap_df["month"].astype(str).str.strip()
        m2c = dict(zip(cap_df["month"], cap_df["cap"]))
        # if month not present => treat as 0 (explicitly binding) or large?
        # keep current behavior: 0 means "no capacity"
        return np.array([float(m2c.get(m, 0.0)) for m in months], dtype=float)

    cap_p = _cap_array(cap_p_df)
    cap_s = _cap_array(cap_s_df)
    cap_i = _cap_array(cap_i_df)

    # simulation
    n = len(months)

    sales = np.zeros(n, dtype=float)
    inv = np.zeros(n, dtype=float)
    backlog = np.zeros(n, dtype=float)
    waste = np.zeros(n, dtype=float)
    over_i = np.zeros(n, dtype=float)

    if shelf_life is not None and shelf_life > 0:
        inv_buckets = np.zeros((shelf_life, ), dtype=float)  # age 0..shelf_life-1
    else:
        inv_buckets = None

    # initial inventory can be provided as item "inventory_init" (optional)
    inv_init = _series("inventory_init")
    if inv_init.sum() > 0:
        inv0 = float(inv_init[0])
        if inv_buckets is not None:
            inv_buckets[0] += inv0
        else:
            inv[0] = inv0

    for t in range(n):
        # carry over inventory and backlog
        if t > 0:
            backlog[t] = backlog[t - 1]
            if inv_buckets is not None:
                # buckets already carry over
                pass
            else:
                inv[t] = inv[t - 1]

        # add new demand to backlog
        backlog[t] += demand[t]

        # production limited by cap_p
        p = production_plan[t]
        if cap_p[t] > 0:
            p = min(p, cap_p[t])
        else:
            # cap_p=0 => binding; produce 0
            p = 0.0

        # put produced into inventory
        if inv_buckets is not None:
            inv_buckets = np.roll(inv_buckets, 1)  # age all inventory by 1 month
            # the oldest bucket expires
            expired = inv_buckets[-1]
            if expired > 0:
                waste[t] += expired
            inv_buckets[-1] = 0.0
            inv_buckets[0] += p
            inv_total = float(inv_buckets.sum())
        else:
            inv_total = float(inv[t] + p)

        # ship/sell limited by cap_s and available (inv + backlog)
        ship_cap = cap_s[t] if cap_s[t] > 0 else 0.0
        ship_demand = backlog[t]  # try to fulfill backlog
        ship_avail = inv_total
        ship = min(ship_demand, ship_avail)
        if ship_cap > 0:
            ship = min(ship, ship_cap)

        sales[t] = ship
        backlog[t] -= ship
        inv_total -= ship

        # inventory capacity handling
        if cap_i[t] > 0:
            if inv_total > cap_i[t]:
                if cap_mode == "hard":
                    # throw away excess
                    excess = inv_total - cap_i[t]
                    waste[t] += excess
                    inv_total = cap_i[t]
                else:
                    # soft: keep but record overage
                    over_i[t] = inv_total - cap_i[t]
        # write back inventory
        if inv_buckets is not None:
            # put inv_total back into buckets without changing ages:
            # keep distribution as-is, just reconcile to inv_total by adjusting age0
            current = float(inv_buckets.sum())
            if abs(current - inv_total) > 1e-9:
                inv_buckets[0] += (inv_total - current)
            inv[t] = float(inv_buckets.sum())
        else:
            inv[t] = inv_total

    return PhoneV0Model(
        months=months,
        node_name=node_name,
        demand=demand,
        production=production_plan,
        sales=sales,
        inv=inv,
        backlog=backlog,
        waste=waste,
        cap_p=cap_p,
        cap_s=cap_s,
        cap_i=cap_i,
        over_i=over_i,
    )
