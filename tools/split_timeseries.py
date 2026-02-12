# tools/split_timeseries.py

# STARTER
#python tools/split_timeseries.py --input data/phone_v0/virtual_node_timeseries.csv
#出力：
#data/phone_v0/virtual_node_timeseries_PSI.csv
#data/phone_v0/virtual_node_unit_price_cost.csv


#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Split a mixed timeseries file into:
  1) virtual_node_timeseries_PSI.csv        (long: month,item,value)
  2) virtual_node_unit_price_cost.csv       (long: month,item,value)

Supported input formats:
  - CSV (long):   columns include (month,item,value)  OR similar
  - CSV (wide):   first col month, other cols are items
  - XLSX:         a sheet that contains either long or wide table
                 (e.g., PSI_Image-like layout with row labels and month columns)

Usage examples:
  python tools/split_timeseries.py --input data/phone_v0/virtual_node_timeseries.csv
  python tools/split_timeseries.py --input data/phone_v0/virtual_node_timeseries_money_ready.csv
  python tools/split_timeseries.py --input data/phone_v0/virtual_node_timeseries_money_IMAGE.xlsx --sheet PSI_Image
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


DEFAULT_PSI_ITEMS = [
    # core PSI
    "demand", "S", "s", "sales", "ship", "shipment", "sold", "S_actual",
    "production", "P", "p", "purchase", "procure",
    "inventory", "I", "i",
    "backlog", "CO", "co",
    # optional helpful signals
    "S_accume", "P_accume", "Supply_Accume", "I_accume",
]

DEFAULT_UNIT_ITEMS = [
    # unit economics
    "unit_price", "price", "Price",
    "unit_process_cost", "process_cost", "Process_Cost",
    "unit_purchase_cost", "purchase_cost", "Purchase_Cost",
]


def _norm_month(x) -> str:
    s = str(x).strip()
    return s[:7]


def _looks_like_month(s: str) -> bool:
    return bool(re.match(r"^\d{4}-\d{2}$", s.strip()[:7]))


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0).astype(float)


def _detect_long_columns(df: pd.DataFrame) -> Optional[Tuple[str, str, str]]:
    """Try to find month/item/value columns in a CSV-like df."""
    cols = {c.lower(): c for c in df.columns}
    m = cols.get("month") or cols.get("yyyymm") or cols.get("ym") or cols.get("period") or None
    it = cols.get("item") or cols.get("kpi") or cols.get("name") or cols.get("variable") or None
    val = cols.get("value") or cols.get("val") or cols.get("qty") or cols.get("amount") or None

    if m and it and val:
        return m, it, val

    # heuristic: first 3 columns
    if len(df.columns) >= 3:
        c0, c1, c2 = df.columns[:3]
        # check if c0 looks like month
        sample = df[c0].astype(str).head(10).tolist()
        if sum(_looks_like_month(x) for x in sample) >= max(1, len(sample) // 2):
            return c0, c1, c2

    return None


def _wide_to_long(df: pd.DataFrame) -> pd.DataFrame:
    """Convert wide format: month + item columns -> long month,item,value."""
    if df.shape[1] < 2:
        raise ValueError("Wide table must have at least 2 columns (month + item columns).")
    month_col = df.columns[0]
    out = df.melt(id_vars=[month_col], var_name="item", value_name="value")
    out.rename(columns={month_col: "month"}, inplace=True)
    out["month"] = out["month"].apply(_norm_month)
    out["item"] = out["item"].astype(str)
    out["value"] = _coerce_numeric(out["value"])
    return out


def _load_input(path: Path, sheet: Optional[str]) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xlsm", ".xls"]:
        # Read sheet; if none given, read first sheet.
        df = pd.read_excel(path, sheet_name=sheet or 0, header=None)
        return df
    else:
        return pd.read_csv(path)


def _xlsx_grid_to_table(df_grid: pd.DataFrame) -> pd.DataFrame:
    """
    Parse an Excel grid like PSI_Image:
      - somewhere there is a row with "month"
      - the next row is month labels across columns
      - below, row labels like S, S_actual, CO, I, P ... with values across month columns
    Return a wide table: month + items as columns.
    """
    # Find a cell that equals "month" (case-insensitive)
    month_positions = []
    for r in range(df_grid.shape[0]):
        for c in range(df_grid.shape[1]):
            v = df_grid.iat[r, c]
            if isinstance(v, str) and v.strip().lower() == "month":
                month_positions.append((r, c))
    if not month_positions:
        raise ValueError("Could not find a 'month' header cell in the sheet.")

    # Use the first occurrence
    r0, c0 = month_positions[0]

    # Month labels are expected on row r0+1 (same column start c0+1..)
    months = []
    month_cols = []
    for c in range(c0 + 1, df_grid.shape[1]):
        v = df_grid.iat[r0 + 1, c]
        if pd.isna(v):
            continue
        ms = str(v).strip()
        ms7 = ms[:7]
        if _looks_like_month(ms7):
            months.append(ms7)
            month_cols.append(c)

    if not months:
        raise ValueError("Could not detect month labels on the row below 'month'.")

    # Row labels for items are expected in column c0 (rows r0+2..)
    items = []
    values = {}
    for r in range(r0 + 2, df_grid.shape[0]):
        label = df_grid.iat[r, c0]
        if pd.isna(label):
            continue
        item = str(label).strip()
        if item == "":
            continue

        row_vals = []
        has_any = False
        for c in month_cols:
            v = df_grid.iat[r, c]
            if pd.isna(v):
                row_vals.append(0.0)
            else:
                try:
                    row_vals.append(float(v))
                    has_any = True
                except Exception:
                    # non-numeric; treat as 0
                    row_vals.append(0.0)
        if has_any:
            items.append(item)
            values[item] = row_vals

    if not items:
        raise ValueError("Could not detect item rows under the month block.")

    wide = pd.DataFrame({"month": months})
    for item in items:
        wide[item] = values[item]
    return wide


def _classify_item(item: str, psi_set: set, unit_set: set) -> str:
    # exact match first (case-insensitive)
    it = item.strip()
    it_l = it.lower()

    if it in psi_set or it_l in psi_set:
        return "psi"
    if it in unit_set or it_l in unit_set:
        return "unit"

    # heuristic patterns
    if re.search(r"(price|cost)", it_l):
        return "unit"
    if re.fullmatch(r"(s|p|i|co)", it_l):
        return "psi"
    if re.search(r"(demand|sales|ship|prod|invent|backlog)", it_l):
        return "psi"

    # default: psi (safer for planning)
    return "psi"


def split_timeseries(
    df_long: pd.DataFrame,
    psi_items: List[str],
    unit_items: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    psi_set = {x for x in psi_items} | {x.lower() for x in psi_items}
    unit_set = {x for x in unit_items} | {x.lower() for x in unit_items}

    df = df_long.copy()
    df["month"] = df["month"].apply(_norm_month)
    df["item"] = df["item"].astype(str).str.strip()
    df["value"] = _coerce_numeric(df["value"])

    cls = df["item"].apply(lambda x: _classify_item(x, psi_set, unit_set))
    df_psi = df.loc[cls == "psi", ["month", "item", "value"]].copy()
    df_unit = df.loc[cls == "unit", ["month", "item", "value"]].copy()

    # normalize unit item names to the canonical ones if possible
    rename_map = {
        "price": "unit_price",
        "process_cost": "unit_process_cost",
        "purchase_cost": "unit_purchase_cost",
        "process_cost ": "unit_process_cost",
        "purchase_cost ": "unit_purchase_cost",
        "Process_Cost".lower(): "unit_process_cost",
        "Purchase_Cost".lower(): "unit_purchase_cost",
    }
    df_unit["item_norm"] = df_unit["item"].str.lower().map(rename_map).fillna(df_unit["item"])
    df_unit["item"] = df_unit["item_norm"]
    df_unit.drop(columns=["item_norm"], inplace=True)

    # Keep last value if duplicates exist
    df_psi = df_psi.groupby(["month", "item"], as_index=False)["value"].last()
    df_unit = df_unit.groupby(["month", "item"], as_index=False)["value"].last()

    return df_psi, df_unit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to mixed input file (csv/xlsx).")
    ap.add_argument("--sheet", default=None, help="Excel sheet name (if input is xlsx).")
    ap.add_argument("--out_dir", default=None, help="Output directory (default: same as input).")
    ap.add_argument("--psi_out", default="virtual_node_timeseries_PSI.csv")
    ap.add_argument("--unit_out", default="virtual_node_unit_price_cost.csv")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_dir = Path(args.out_dir) if args.out_dir else in_path.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = _load_input(in_path, args.sheet)

    # If xlsx grid (header=None), parse to wide then to long
    if in_path.suffix.lower() in [".xlsx", ".xlsm", ".xls"]:
        wide = _xlsx_grid_to_table(raw)
        df_long = _wide_to_long(wide)
    else:
        # CSV
        cols = _detect_long_columns(raw)
        if cols:
            m, it, val = cols
            df_long = raw[[m, it, val]].copy()
            df_long.columns = ["month", "item", "value"]
            df_long["month"] = df_long["month"].apply(_norm_month)
            df_long["item"] = df_long["item"].astype(str)
            df_long["value"] = _coerce_numeric(df_long["value"])
        else:
            # treat as wide
            df_long = _wide_to_long(raw)

    df_psi, df_unit = split_timeseries(df_long, DEFAULT_PSI_ITEMS, DEFAULT_UNIT_ITEMS)

    psi_path = out_dir / args.psi_out
    unit_path = out_dir / args.unit_out
    df_psi.to_csv(psi_path, index=False, encoding="utf-8-sig")
    df_unit.to_csv(unit_path, index=False, encoding="utf-8-sig")

    print(f"[OK] PSI  -> {psi_path}  rows={len(df_psi)}")
    print(f"[OK] UNIT -> {unit_path} rows={len(df_unit)}")
    print("Tip: ensure UNIT contains items: unit_price, unit_process_cost, unit_purchase_cost (per month).")


if __name__ == "__main__":
    main()
