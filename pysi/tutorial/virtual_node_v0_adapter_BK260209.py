# pysi/tutorial/phone_v0_adapter.py
# pysi/tutorial/virtual_node_v0_adapter.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class PhoneV0Model:
    months: List[str]

    demand: pd.Series          # index=months
    production_plan: pd.Series # index=months

    # capacities (may contain NaN -> means "no cap that month")
    p_cap_series: pd.Series    # index=months
    s_cap_series: pd.Series    # index=months
    i_cap_series: pd.Series    # index=months

    # results
    sales: pd.Series           # index=months
    inv: pd.Series             # index=months
    backlog: pd.Series         # index=months
    waste: pd.Series           # index=months

    # diagnostics
    over_i: pd.Series          # index=months
    cap_mode: str              # "soft" | "hard"

    # NEW: if set (months), enable FIFO + expiration (pharma-style)
    shelf_life: Optional[int] = None

    # binding months (for teaching)
    p_cap_binding_months: List[str] = None
    s_cap_binding_months: List[str] = None
    i_cap_binding_months: List[str] = None

    @property
    def inventory(self):
        return self.inv


def _read_timeseries_long(data_dir: str, fname: str) -> pd.DataFrame:
    path = f"{data_dir}/{fname}"
    df = pd.read_csv(path)

    # --- 修正: 全てが空、または month が空の行を削除 ---
    df = df.dropna(subset=["month", "value"], how="any")
    # ---------------------------------------------

    # expected cols: month,node_name,item,value
    for c in ["month", "node_name", "item", "value"]:
        if c not in df.columns:
            raise ValueError(f"{path} must have columns {['month','node_name','item','value']}, got {list(df.columns)}")
    return df


# ********
# 日付のSORT
# ********
import re

def _month_to_ym(s: str) -> str:
    """
    Normalize month string into 'YYYY-MM'.
    Supports:
      - '2025-1' / '2025-01'
      - 'Jan-25' (MMM-YY)
      - '25-Jan' (YY-MMM)  <-- Excelで出がち
    """
    s = str(s).strip()
    if not s or s.lower() == "nan":
        return s

    # already YYYY-MM
    if re.match(r"^\d{4}-\d{2}$", s):
        return s

    # YYYY-M -> YYYY-0M
    m = re.match(r"^(\d{4})-(\d{1})$", s)
    if m:
        return f"{m.group(1)}-0{m.group(2)}"

    month_map = {
        "Jan": "01", "Feb": "02", "Mar": "03", "Apr": "04", "May": "05", "Jun": "06",
        "Jul": "07", "Aug": "08", "Sep": "09", "Oct": "10", "Nov": "11", "Dec": "12",
    }

    # MMM-YY  (Jan-25)
    m = re.match(r"^([A-Za-z]{3})-(\d{2})$", s)
    if m:
        mon, yy = m.group(1).capitalize(), m.group(2)
        if mon in month_map:
            return f"20{yy}-{month_map[mon]}"

    # YY-MMM  (25-Apr)  <-- 追加
    m = re.match(r"^(\d{2})-([A-Za-z]{3})$", s)
    if m:
        yy, mon = m.group(1), m.group(2).capitalize()
        if mon in month_map:
            return f"20{yy}-{month_map[mon]}"

    # last resort: pandas
    try:
        dt = pd.to_datetime(s, errors="raise")
        return dt.strftime("%Y-%m")
    except Exception:
        return s



def _pivot_item(df: pd.DataFrame, months: List[str], node: str, item: str, default: float = 0.0) -> pd.Series:
    sub = df[(df["node_name"] == node) & (df["item"] == item)].copy()
    if sub.empty:
        return pd.Series([default for _ in months], index=months, dtype=float)

    sub["month"] = sub["month"].map(_month_to_ym)
    # last one wins (so you can intentionally override by appending)
    sub = sub.groupby("month", as_index=True)["value"].last().astype(float)
    out = pd.Series([default for _ in months], index=months, dtype=float)
    for m, v in sub.items():
        if m in out.index:
            out.loc[m] = float(v)
    return out


def _read_capacity_series(data_dir: str, fname: str, months: List[str]) -> pd.Series:
    path = f"{data_dir}/{fname}"
    df = pd.read_csv(path)
    # expected cols: month,value  (month can be '2025-01' or 'Jan-25')
    if "month" not in df.columns or "value" not in df.columns:
        raise ValueError(f"{path} must have columns ['month','value'], got {list(df.columns)}")
    df = df.copy()
    df["month"] = df["month"].map(_month_to_ym)
    s = df.groupby("month", as_index=True)["value"].last().astype(float)
    out = pd.Series([np.nan for _ in months], index=months, dtype=float)
    for m, v in s.items():
        if m in out.index:
            out.loc[m] = float(v)
    return out


def load_phone_v0(
    data_dir: str,
    timeseries_name: Optional[str] = None,
    cap_mode: str = "soft",
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
    df["month"] = df["month"].map(_month_to_ym)

    node = "VIRTUAL_NODE"
    if "node_name" in df.columns and df["node_name"].nunique() == 1:
        node = str(df["node_name"].iloc[0])

    # months domain
    months = sorted(df["month"].unique().tolist())

    demand = _pivot_item(df, months, node, "demand", default=0.0)
    production_plan = _pivot_item(df, months, node, "production", default=0.0)

    # initial inventory (take last if multiple exist)
    initial_inventory_series = _pivot_item(df, months, node, "initial_inventory", default=0.0)
    initial_inventory = float(initial_inventory_series.loc[months[0]])

    # capacities
    p_cap_series = _read_capacity_series(data_dir, "capacity_P.csv", months)
    s_cap_series = _read_capacity_series(data_dir, "capacity_S.csv", months)
    i_cap_series = _read_capacity_series(data_dir, "capacity_I.csv", months)

    # normalize shelf_life
    if shelf_life is not None:
        try:
            shelf_life = int(shelf_life)
        except Exception:
            shelf_life = None
    if shelf_life is not None and shelf_life <= 0:
        shelf_life = None

    prev_inv = initial_inventory
    prev_backlog = 0.0

    # FIFO buckets when enabled: buckets[0] newest, buckets[-1] oldest
    buckets: Optional[List[float]] = None
    if shelf_life:
        buckets = [0.0 for _ in range(shelf_life)]
        buckets[0] = prev_inv

    sales: List[float] = []
    inv: List[float] = []
    backlog: List[float] = []
    waste: List[float] = []
    over_i: List[float] = []

    p_bind: List[str] = []
    s_bind: List[str] = []
    i_bind: List[str] = []

    for m in months:
        d = float(demand.loc[m])
        p_plan = float(production_plan.loc[m])

        # ---- P_cap ----
        pcap = float(p_cap_series.loc[m]) if np.isfinite(p_cap_series.loc[m]) else p_plan
        p_act = min(p_plan, pcap)
        if p_plan > p_act + 1e-9 and np.isfinite(p_cap_series.loc[m]):
            p_bind.append(m)

        if shelf_life:
            assert buckets is not None

            # 1) arrival into newest bucket
            buckets[0] += p_act

            # 2) need includes backlog
            need = prev_backlog + d

            # 3) available inventory
            available = float(sum(buckets))

            # ---- S_cap ----
            scap = float(s_cap_series.loc[m]) if np.isfinite(s_cap_series.loc[m]) else need
            s_act = min(need, scap, available)
            if min(need, available) > s_act + 1e-9 and np.isfinite(s_cap_series.loc[m]):
                s_bind.append(m)

            # 4) consume FIFO (oldest first)
            to_ship = s_act
            for i in reversed(range(shelf_life)):
                take = min(buckets[i], to_ship)
                buckets[i] -= take
                to_ship -= take
                if to_ship <= 1e-12:
                    break

            # 5) backlog update
            new_backlog = need - s_act

            # 6) I_cap handling (after sales)
            inv_raw = float(sum(buckets))
            icap = float(i_cap_series.loc[m]) if np.isfinite(i_cap_series.loc[m]) else inv_raw
            oi = max(0.0, inv_raw - icap)
            if oi > 1e-9 and np.isfinite(i_cap_series.loc[m]):
                i_bind.append(m)

            waste_month = 0.0
            if cap_mode == "hard":
                # overflow is wasted (cut from newest)
                overflow = oi
                if overflow > 0:
                    buckets[0] -= overflow
                waste_month += overflow
                # after cut, recompute raw (for consistency)
                inv_raw = float(sum(buckets))
            else:
                # soft: overflow is NOT wasted; record as over_i only
                pass

            # 7) expiration at month end (oldest bucket)
            expired = buckets[-1]
            if expired > 0:
                waste_month += expired
            # shift age: new month -> everything gets 1 month older
            buckets = [0.0] + buckets[:-1]

            inv_act = float(sum(buckets))
            w = float(waste_month)

        else:
            # Simple inventory (no FIFO/expiration)
            available = prev_inv + p_act
            need = prev_backlog + d

            # ---- S_cap ----
            scap = float(s_cap_series.loc[m]) if np.isfinite(s_cap_series.loc[m]) else need
            s_act = min(need, scap, available)
            if min(need, available) > s_act + 1e-9 and np.isfinite(s_cap_series.loc[m]):
                s_bind.append(m)

            new_backlog = need - s_act

            inv_raw = available - s_act
            icap = float(i_cap_series.loc[m]) if np.isfinite(i_cap_series.loc[m]) else inv_raw
            oi = max(0.0, inv_raw - icap)
            if oi > 1e-9 and np.isfinite(i_cap_series.loc[m]):
                i_bind.append(m)

            if cap_mode == "hard":
                w = oi
                inv_act = min(inv_raw, icap)
            else:
                w = 0.0
                inv_act = inv_raw

        # store
        sales.append(float(s_act))
        inv.append(float(inv_act))
        backlog.append(float(new_backlog))
        waste.append(float(w))
        over_i.append(float(oi))

        prev_inv = float(inv_act)
        prev_backlog = float(new_backlog)

        # keep updated buckets reference
        if shelf_life:
            assert buckets is not None

    model = PhoneV0Model(
        months=months,
        demand=demand,
        production_plan=production_plan,
        p_cap_series=p_cap_series,
        s_cap_series=s_cap_series,
        i_cap_series=i_cap_series,
        sales=pd.Series(sales, index=months),
        inv=pd.Series(inv, index=months),
        backlog=pd.Series(backlog, index=months),
        waste=pd.Series(waste, index=months),
        over_i=pd.Series(over_i, index=months),
        cap_mode=cap_mode,
        shelf_life=shelf_life,
        p_cap_binding_months=p_bind,
        s_cap_binding_months=s_bind,
        i_cap_binding_months=i_bind,
    )
    return model
