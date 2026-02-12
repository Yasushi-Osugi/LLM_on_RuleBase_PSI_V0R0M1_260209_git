
# tools/csv_YYYY-MM_update.py

import pandas as pd

for fn in [
  "data/phone_v0/virtual_node_timeseries_PSI.csv",
  "data/phone_v0/virtual_node_unit_price_cost.csv",
]:
  df = pd.read_csv(fn)
  # YYYY-M or YYYY-MM を YYYY-MM に正規化
  s = df["month"].astype(str).str.strip()
  # "2026-1" -> "2026-01"
  df["month"] = s.str.replace(r"^(\d{4})-(\d{1})$", r"\1-0\2", regex=True)
  df.to_csv(fn, index=False, encoding="utf-8-sig")
  print("[fixed month]", fn, "unique months:", sorted(df["month"].unique())[:5], "...")
