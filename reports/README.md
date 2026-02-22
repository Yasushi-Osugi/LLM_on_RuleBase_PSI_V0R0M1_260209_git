# LLM on Rule Base PSI – Diagnostic Loop

This directory contains diagnostic input packages for LLM planning cycles.

Each timestamp folder includes:

- psi_before_after.png
- before_timeseries.csv
- after_timeseries.csv
- diagnosis.json
- operator_spec.template.json
- PROMPT_TO_LLM.md

---

## Human-in-the-Loop Procedure

### Step 1: Run simulation
```bash
python -m tools.run_operator_queue ...
Step 2: Export diagnostic package
python tools/run_cycle.py
Step 3: LLM Diagnosis
Open PROMPT_TO_LLM.md

Upload psi_before_after.png to ChatGPT/Gemini

Paste prompt

Copy JSON-only output

Step 4: Save as candidate
Save output as:

runs/one_node/_latest/meta/candidate_operator_spec.json
Step 5: Approve
After reviewing:

Rename to:

operator_spec.json
Step 6: Re-run queue
python -m tools.run_operator_queue ...
This creates a reproducible, auditable planning loop.


---

# ③ operator_spec.template.json  
（LLMに貼りやすい雛形）

```json
[
  {
    "operator": "sales_adjust_business_table_csv",
    "params": {
      "months": ["YYYY-MM"],
      "rate": 0.00,
      "file": "business_timeseries.csv"
    },
    "reason": "inventory_adjustment"
  },
  {
    "operator": "price_adjust_business_table_csv",
    "params": {
      "months": ["YYYY-MM"],
      "rate": 0.00,
      "file": "business_timeseries.csv"
    },
    "reason": "profit_optimization"
  },
  {
    "operator": "production_adjust_business_table_csv",
    "params": {
      "months": ["YYYY-MM"],
      "rate": 0.00,
      "file": "business_timeseries.csv"
    },
    "reason": "production_shift"
  }
]
