# tools/normalize_months.py
import re
from pathlib import Path
import pandas as pd

MONTH_MAP = {
    "jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06",
    "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12",
}

def month_to_ym(x: str) -> str:
    s = str(x).strip()
    if not s or s.lower() == "nan":
        return s

    # YYYY-MM
    if re.match(r"^\d{4}-\d{2}$", s):
        return s

    # YYYY-M -> YYYY-0M
    m = re.match(r"^(\d{4})-(\d{1})$", s)
    if m:
        return f"{m.group(1)}-0{m.group(2)}"

    # MMM-YY (Jan-25)
    m = re.match(r"^([A-Za-z]{3})-(\d{2})$", s)
    if m:
        mon = m.group(1).lower()
        yy = m.group(2)
        if mon in MONTH_MAP:
            return f"20{yy}-{MONTH_MAP[mon]}"

    # YY-MMM (25-Jan)  ← ここが今回の肝
    m = re.match(r"^(\d{2})-([A-Za-z]{3})$", s)
    if m:
        yy = m.group(1)
        mon = m.group(2).lower()
        if mon in MONTH_MAP:
            return f"20{yy}-{MONTH_MAP[mon]}"

    # pandas fallback
    try:
        dt = pd.to_datetime(s, errors="raise")
        return dt.strftime("%Y-%m")
    except Exception:
        return s

def normalize_file(path: Path) -> None:
    df = pd.read_csv(path)
    if "month" not in df.columns:
        raise ValueError(f"{path} missing 'month' column")
    before = df["month"].astype(str).unique()[:5]
    df["month"] = df["month"].astype(str).map(month_to_ym)
    after = df["month"].astype(str).unique()[:5]
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[OK] {path}")
    print("  before:", before)
    print("  after :", after)
    print("  min/max:", df["month"].min(), df["month"].max(), "n_months:", df["month"].nunique())

if __name__ == "__main__":
    targets = [
        Path("data/phone_v0/virtual_node_timeseries_PSI.csv"),
        Path("data/phone_v0/virtual_node_unit_price_cost.csv"),
    ]
    for p in targets:
        normalize_file(p)
