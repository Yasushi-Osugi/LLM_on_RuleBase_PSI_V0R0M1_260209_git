# tools/run_cycle.py

# tools/run_cycle.py

from pathlib import Path
import json
import shutil
import datetime
from textwrap import dedent


def find_latest_completed_run(base_dir: Path) -> Path:
    """
    Find the most recent run directory that has input_snapshot.
    """
    candidates = []

    for p in base_dir.iterdir():
        if not p.is_dir():
            continue
        if (p / "input_snapshot").exists():
            candidates.append(p)

    if not candidates:
        raise FileNotFoundError("No completed runs (with input_snapshot) found.")

    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def export_reports(run_dir: Path) -> Path:
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

    reports_root = Path("reports")
    reports_root.mkdir(exist_ok=True)

    report_dir = reports_root / timestamp
    report_dir.mkdir()

    # --- CSV ---
    before_csv = run_dir / "input_snapshot" / "one_node_result_timeseries.csv"
    after_csv = run_dir / "output" / "one_node_result_timeseries.csv"

    shutil.copy(before_csv, report_dir / "before_timeseries.csv")
    shutil.copy(after_csv, report_dir / "after_timeseries.csv")

    # --- PNG ---
    psi_png = list((run_dir / "output").glob("psi_before_after__*.png"))
    money_png = list((run_dir / "output").glob("money_before_after__*.png"))

    if psi_png:
        shutil.copy(psi_png[0], report_dir / "psi_before_after.png")

    if money_png:
        shutil.copy(money_png[0], report_dir / "money_before_after.png")

    # --- diagnosis ---
    diagnosis_src = run_dir / "output" / "diagnosis.json"
    if diagnosis_src.exists():
        shutil.copy(diagnosis_src, report_dir / "diagnosis.json")

    # --- operator template ---
    template = [
        {
            "operator": "sales_adjust_business_table_csv",
            "params": {
                "months": ["YYYY-MM"],
                "rate": 0.00,
                "file": "business_timeseries.csv",
            },
            "reason": "describe_reason",
        }
    ]

    with open(report_dir / "operator_spec.template.json", "w", encoding="utf-8") as f:
        json.dump(template, f, indent=2)

    # --- Prompt ---
    prompt = dedent(
        f"""\
        You are a supply chain planning AI for a monthly PSI (Production / Sales / Inventory) model.

        You will be given these files from a single run:
        - psi_before_after.png          (PSI before/after comparison)
        - money_before_after.png        (Revenue/Profit before/after comparison, if exists)
        - diagnosis.json                (rule-based diagnosis output, if exists)
        - operator_spec.template.json   (JSON format template)

        Task:
        1) Read the graphs and diagnosis and identify key issues (excess inventory, backlog, capacity overage, profit deterioration, etc.).
        2) Propose a *sequence* of planning operators (operator_spec.json) to improve the plan.
        3) Output ONLY a valid JSON array (no prose, no markdown fences).

        Constraints:
        - Use only operators that exist in this repository.
        - If you shift production, keep total production volume constant across the horizon (pair minus/plus months).
        - Respect capacity (P_cap/S_cap/I_cap) unless your operator explicitly changes capacity.
        - Prefer small, explainable adjustments (rate within +/-20% unless absolutely necessary).

        Output format:
        - JSON array of objects: {{ "operator": "...", "params": {{...}}, "reason": "..." }}
        """
    ).strip()

    # NOTE: Use UTF-8 with BOM to avoid mojibake in Windows `type`
    with open(report_dir / "PROMPT_TO_LLM.md", "w", encoding="utf-8-sig", newline="\n") as f:
        f.write(prompt + "\n")

    return report_dir


def main():
    runs_root = Path("runs/one_node")

    latest_run = find_latest_completed_run(runs_root)

    print("Using run:", latest_run)

    report_dir = export_reports(latest_run)

    print("\n================================")
    print("Diagnostic Package Created")
    print("Location:", report_dir)
    print("Next:")
    print("1. Open PROMPT_TO_LLM.md")
    print("2. Upload PNG to ChatGPT/Gemini")
    print("3. Save JSON as candidate_operator_spec.json")
    print("================================")


if __name__ == "__main__":
    main()
