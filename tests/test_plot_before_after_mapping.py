import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from pysi.tutorial.plot_virtual_node_v0 import (
    _prepare_before_after_plot_series,
    plot_before_after_from_csv,
)


class TestBeforeAfterPlotMapping(unittest.TestCase):
    def test_series_mapping_matches_dataframe_columns_for_selected_month(self):
        df = pd.DataFrame(
            {
                "month": ["2026-01", "2026-02"],
                "demand": [100, 110],
                "production": [90, 95],
                "sales": [80, 88],
                "inventory": [40, 35],
                "backlog": [20, 22],
                "waste": [2, 1],
                "over_i": [0, 0],
                "cap_P": [120, 120],
                "cap_S": [130, 130],
                "cap_I": [50, 50],
            }
        )
        months = df["month"].tolist()
        series = _prepare_before_after_plot_series(df, months)

        i = months.index("2026-02")
        self.assertEqual(series["Production"][i], df.loc[i, "production"])
        self.assertEqual(series["Sales"][i], df.loc[i, "sales"])
        self.assertEqual(series["Backlog"][i], df.loc[i, "backlog"])
        self.assertEqual(series["Inventory"][i], df.loc[i, "inventory"])

    def test_inventory_component_and_legend_labels_are_consistent(self):
        df = pd.DataFrame(
            {
                "month": ["2026-01", "2026-02"],
                "demand": [100, 110],
                "production": [90, 95],
                "sales": [80, 88],
                "inventory": [40, 70],
                "backlog": [20, 22],
                "waste": [2, 1],
                "over_i": [0, 15],
                "cap_P": [120, 120],
                "cap_S": [130, 130],
                "cap_I": [50, 50],
            }
        )
        months = df["month"].tolist()
        series = _prepare_before_after_plot_series(df, months)

        # over-cap month should split inventory and over_i components (not overwrite Inventory)
        i = months.index("2026-02")
        self.assertEqual(series["Inventory"][i], 50)
        self.assertEqual(series["over_i"][i], 15)

        with tempfile.TemporaryDirectory() as td:
            tdp = Path(td)
            before_csv = tdp / "before.csv"
            after_csv = tdp / "after.csv"
            out_png = tdp / "out.png"
            df.to_csv(before_csv, index=False)
            df.to_csv(after_csv, index=False)

            original_close = plt.close
            plt.close = lambda *args, **kwargs: None
            try:
                plot_before_after_from_csv(before_csv, after_csv, save_path=out_png, show=False)
                fig = plt.gcf()
                ax = fig.axes[0]
                labels = ax.get_legend_handles_labels()[1]
            finally:
                plt.close = original_close
                plt.close("all")

        expected_labels = {
            "I_cap",
            "S_cap",
            "Demand",
            "Sales",
            "Backlog",
            "Production",
            "Inventory",
            "Waste",
            "over_i",
            "P_cap",
        }
        self.assertTrue(expected_labels.issubset(set(labels)))


if __name__ == "__main__":
    unittest.main()
