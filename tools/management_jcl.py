# tools/management_jcl.py

#2) 使い方（あなたの現行コマンド体系に合わせる）
#2.1 One-node を承認→実行→index追記
#
#例（スマホ）：
#
#python -m tools.management_jcl one_node ^
#  --candidate runs\one_node\_latest\meta\operator_spec.json ^
#  --data_dir data\phone_v0 ^
#  --runs_dir runs\one_node ^
#  --run_meta configs\phone_sellout_demo.json ^
#  --index_path runs\one_node\_index.jsonl

#
#Approve and dispatch? [y/N]: で y を押すと実行します
#
#実行完了後に
#
#runs/one_node/_index.jsonl へ追記
#
#runs/one_node/<RUN_ID>/meta/status_report.json を保存
#
#2.2 Queue を承認→実行→index追記（最小版）
#python -m tools.management_jcl queue ^
#  --candidate reports\20260223_074956\candidate_operator_spec.json ^
#  --queue configs\operator_queue_phone_demo.json ^
#  --stop_on_error ^
#  --plot_each_step ^
#  --index_path runs\one_node\_index.jsonl
#
#※この最小版は step_run_id の列挙まではしません（run_operator_queue 側が「作成
#したstep一覧」を機械可読で吐いていないため）。
#次のステップで、run_operator_queue.py に「生成したstep_run_id一覧を meta に出す#」追記をすると、一気に堅くできます。
#
#

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# =========================
# Constants / Defaults
# =========================

DEFAULT_INDEX_PATH = Path("runs/one_node/_index.jsonl")
DEFAULT_RUNS_ROOT = Path("runs/one_node")
DEFAULT_REPORTS_ROOT = Path("reports")

# Operator allow-list (minimum guardrail)
# ここは「まず止めない」ために緩めに。
# 厳密化は次のステップで（projectポリシーが固まってから）。
ALLOWED_OPERATOR_PREFIXES = [
    "production_",
    "demand_",
    "price_",
    "apply_",
    "operate_",
    "business_",
]

# Hard constraints (guardrail)
MAX_ABS_RATIO = 10.0  # 例えば「価格を10倍」みたいなのは一旦止める等の荒いガード


# =========================
# Utilities
# =========================

def utc_now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def dump_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def append_jsonl(path: Path, record: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False)
    with path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def is_timestamp_id(s: str) -> bool:
    return bool(re.fullmatch(r"\d{8}_\d{6}", s))


def safe_relpath(p: Path) -> str:
    # Windowsでも / を使うと比較が楽。ここではPOSIX表記に寄せる。
    return p.as_posix()


@dataclass
class ValidationResult:
    ok: bool
    errors: List[str]
    warnings: List[str]


# =========================
# Candidate Spec Validation
# =========================

def extract_operator_names(spec: Dict[str, Any]) -> List[str]:
    """
    あなたの operator_spec の形式が変遷中でも壊れにくいように、
    ありがちなキーを複数探索する。
    """
    ops: List[str] = []

    # Pattern 1: spec["operators"] = [{"name": "..."}]
    if isinstance(spec.get("operators"), list):
        for x in spec["operators"]:
            if isinstance(x, dict) and isinstance(x.get("name"), str):
                ops.append(x["name"])

    # Pattern 2: spec["steps"] = [{"operator": "..."}]
    if isinstance(spec.get("steps"), list):
        for x in spec["steps"]:
            if isinstance(x, dict) and isinstance(x.get("operator"), str):
                ops.append(x["operator"])

    # Pattern 3: spec["operator"] single
    if isinstance(spec.get("operator"), str):
        ops.append(spec["operator"])

    # Pattern 4: spec["issue"]["recommend"]["operator"]
    issue = spec.get("issue")
    if isinstance(issue, dict):
        rec = issue.get("recommend")
        if isinstance(rec, dict) and isinstance(rec.get("operator"), str):
            ops.append(rec["operator"])

    # unique
    uniq = []
    for o in ops:
        if o not in uniq:
            uniq.append(o)
    return uniq


def validate_operator_allowlist(operator_names: List[str]) -> ValidationResult:
    errors, warnings = [], []

    if not operator_names:
        warnings.append("No operator names detected in candidate spec (allowed but suspicious).")
        return ValidationResult(ok=True, errors=errors, warnings=warnings)

    for op in operator_names:
        if not any(op.startswith(prefix) for prefix in ALLOWED_OPERATOR_PREFIXES):
            errors.append(f"Operator '{op}' is not allowed by prefix allow-list: {ALLOWED_OPERATOR_PREFIXES}")

    return ValidationResult(ok=(len(errors) == 0), errors=errors, warnings=warnings)


def validate_numeric_sanity(spec: Dict[str, Any]) -> ValidationResult:
    """
    “危険そうな値”だけ雑に止める。
    例: ratioやdeltaが異常に大きい、負の価格、など。
    形式が揺れていても拾えるように “それっぽいキー” を軽く見る。
    """
    errors, warnings = [], []

    def walk(obj: Any, path: str = ""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, f"{path}.{k}" if path else k)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                walk(v, f"{path}[{i}]")
        else:
            # scalar
            return

    # Quick heuristics: check common fields
    # ratio / rate / factor / multiplier
    def check_ratio_like(val: Any, p: str):
        if isinstance(val, (int, float)):
            if abs(val) > MAX_ABS_RATIO:
                warnings.append(f"Suspicious large ratio-like value at '{p}': {val} (>|{MAX_ABS_RATIO}|)")

    def check_non_negative_price(val: Any, p: str):
        if isinstance(val, (int, float)):
            if val < 0:
                errors.append(f"Negative price-like value at '{p}': {val}")

    def scan(obj: Any, path: str = ""):
        if isinstance(obj, dict):
            for k, v in obj.items():
                p = f"{path}.{k}" if path else k
                lk = k.lower()
                if any(x in lk for x in ["ratio", "rate", "factor", "multiplier"]):
                    check_ratio_like(v, p)
                if any(x in lk for x in ["price", "unit_price", "selling_price"]):
                    check_non_negative_price(v, p)
                scan(v, p)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                scan(v, f"{path}[{i}]")

    scan(spec)
    return ValidationResult(ok=(len(errors) == 0), errors=errors, warnings=warnings)


def validate_candidate_spec(path: Path) -> Tuple[Dict[str, Any], ValidationResult]:
    errors, warnings = [], []
    try:
        spec = load_json(path)
    except Exception as e:
        return {}, ValidationResult(ok=False, errors=[f"JSON parse error: {e}"], warnings=[])

    operator_names = extract_operator_names(spec)

    r1 = validate_operator_allowlist(operator_names)
    errors += r1.errors
    warnings += r1.warnings

    r2 = validate_numeric_sanity(spec)
    errors += r2.errors
    warnings += r2.warnings

    ok = (len(errors) == 0)
    return spec, ValidationResult(ok=ok, errors=errors, warnings=warnings)


# =========================
# Approval (CLI)
# =========================

def ask_approval(candidate_path: Path, operator_names: List[str], warnings: List[str]) -> bool:
    print("\n==============================")
    print(" MANAGEMENT JCL: APPROVAL GATE")
    print("==============================")
    print(f"Candidate: {candidate_path}")
    if operator_names:
        print("Operators:")
        for op in operator_names:
            print(f"  - {op}")
    else:
        print("Operators: (not detected)")

    if warnings:
        print("\nWarnings:")
        for w in warnings:
            print(f"  * {w}")

    while True:
        ans = input("\nApprove and dispatch? [y/N]: ").strip().lower()
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no", ""):
            return False
        print("Please type 'y' or 'n'.")


# =========================
# Dispatch (subprocess)
# =========================

def dispatch_one_node(data_dir: Path, runs_dir: Path, run_meta: Path) -> int:
    """
    tools/run_one_node4plugin.py を呼ぶ。
    NOTE: REPOの実行方法に合わせて `python -m tools.run_one_node4plugin` を使う。
    """
    cmd = [
        sys.executable, "-m", "tools.run_one_node4plugin",
        "--data_dir", str(data_dir),
        "--runs_dir", str(runs_dir),
        "--run_meta", str(run_meta),
    ]
    print("\n[dispatch] " + " ".join(cmd))
    return subprocess.call(cmd)


def dispatch_queue(queue_path: Path, stop_on_error: bool = True, plot_each_step: bool = False) -> int:
    cmd = [
        sys.executable, "-m", "tools.run_operator_queue",
        "--queue", str(queue_path),
    ]
    if stop_on_error:
        cmd.append("--stop_on_error")
    if plot_each_step:
        cmd.append("--plot_each_step")

    print("\n[dispatch] " + " ".join(cmd))
    return subprocess.call(cmd)


# =========================
# Run detection + indexing
# =========================

def find_latest_run_id(runs_root: Path) -> Optional[str]:
    """
    runs/one_node 配下の YYYYMMDD_HHMMSS のうち、最新を探す。
    """
    if not runs_root.exists():
        return None
    run_ids = []
    for p in runs_root.iterdir():
        if p.is_dir() and is_timestamp_id(p.name):
            run_ids.append(p.name)
    if not run_ids:
        return None
    run_ids.sort()
    return run_ids[-1]


def build_index_record_for_one_node(run_id: str, data_dir: Path, run_root: Path) -> Dict[str, Any]:
    out_dir = run_root / "output"
    in_dir = run_root / "input"

    plots = []
    if out_dir.exists():
        for p in out_dir.glob("*.png"):
            plots.append(safe_relpath(p))

    record = {
        "schema_version": "1.0",
        "kind": "one_node",
        "run_id": run_id,
        "created_at": utc_now_iso(),
        "data_dir": safe_relpath(data_dir),
        "run_root": safe_relpath(run_root),
        "inputs": {
            "business_timeseries": safe_relpath(in_dir / "business_timeseries.csv") if (in_dir / "business_timeseries.csv").exists() else None,
            "capacity_P": safe_relpath(in_dir / "capacity_P.csv") if (in_dir / "capacity_P.csv").exists() else None,
            "capacity_S": safe_relpath(in_dir / "capacity_S.csv") if (in_dir / "capacity_S.csv").exists() else None,
            "capacity_I": safe_relpath(in_dir / "capacity_I.csv") if (in_dir / "capacity_I.csv").exists() else None,
            "virtual_node_timeseries": safe_relpath(in_dir / "virtual_node_timeseries.csv") if (in_dir / "virtual_node_timeseries.csv").exists() else None,
            "unit_price_cost": safe_relpath(in_dir / "virtual_node_unit_price_cost.csv") if (in_dir / "virtual_node_unit_price_cost.csv").exists() else None,
        },
        "outputs": {
            "psi_result": safe_relpath(out_dir / "one_node_result_timeseries.csv") if (out_dir / "one_node_result_timeseries.csv").exists() else None,
            "money_result": safe_relpath(out_dir / "money_timeseries.csv") if (out_dir / "money_timeseries.csv").exists() else None,
            "kpi_summary": safe_relpath(out_dir / "kpi_summary.json") if (out_dir / "kpi_summary.json").exists() else None,
            "diagnosis": safe_relpath(out_dir / "diagnosis.json") if (out_dir / "diagnosis.json").exists() else None,
            "plots": plots
        },
        "status": {
            "kernel_status": None
        }
    }
    return record


def build_status_report(
    jcl_status: str,
    run_kind: str,
    artifacts: Dict[str, Any],
    candidate_path: Optional[Path],
    approved: Optional[bool],
    kernel_status: Optional[str],
    warnings: List[str],
    errors: List[str],
) -> Dict[str, Any]:
    return {
        "schema_version": "1.0",
        "time_utc": utc_now_iso(),
        "jcl_status": jcl_status,
        "run_kind": run_kind,
        "candidate": None if candidate_path is None else {
            "source": "external_llm",
            "path": safe_relpath(candidate_path),
            "hash": sha256_file(candidate_path) if candidate_path.exists() else None
        },
        "approval": None if approved is None else {
            "required": True,
            "approved_by": "user" if approved else None,
            "approved_at": utc_now_iso() if approved else None,
            "comment": ""
        },
        "kernel_status": None if kernel_status is None else {
            "status": kernel_status,
            "message": "",
            "warnings": warnings
        },
        "validation": {
            "errors": errors,
            "warnings": warnings
        },
        "artifacts": artifacts
    }


# =========================
# Main
# =========================

def main() -> int:
    parser = argparse.ArgumentParser(description="Management JCL prototype (approval -> dispatch -> index).")

    sub = parser.add_subparsers(dest="mode", required=True)

    # ---- one_node ----
    p1 = sub.add_parser("one_node", help="Dispatch one_node run after approval.")
    p1.add_argument("--candidate", type=str, required=True, help="Path to candidate operator spec json.")
    p1.add_argument("--data_dir", type=str, required=True, help="data directory (e.g. data/phone_v0).")
    p1.add_argument("--runs_dir", type=str, default=str(DEFAULT_RUNS_ROOT), help="runs directory (default runs/one_node).")
    p1.add_argument("--run_meta", type=str, required=True, help="run meta json (e.g. configs/phone_sellout_demo.json).")
    p1.add_argument("--index_path", type=str, default=str(DEFAULT_INDEX_PATH), help="runs index jsonl path.")

    # ---- queue ----
    p2 = sub.add_parser("queue", help="Dispatch operator queue after approval.")
    p2.add_argument("--candidate", type=str, required=True, help="Path to candidate operator spec json.")
    p2.add_argument("--queue", type=str, required=True, help="queue config json (e.g. configs/operator_queue_phone_demo.json).")
    p2.add_argument("--stop_on_error", action="store_true", help="stop on error")
    p2.add_argument("--plot_each_step", action="store_true", help="plot each step")
    p2.add_argument("--index_path", type=str, default=str(DEFAULT_INDEX_PATH), help="runs index jsonl path.")

    args = parser.parse_args()

    candidate_path = Path(args.candidate)
    if not candidate_path.exists():
        print(f"[error] candidate not found: {candidate_path}")
        return 2

    spec, vr = validate_candidate_spec(candidate_path)
    operator_names = extract_operator_names(spec)

    if not vr.ok:
        print("\n[validation] FAILED")
        for e in vr.errors:
            print(f"  - {e}")
        # status_report (where to store? candidate is outside run yet => store under reports/)
        status = build_status_report(
            jcl_status="JCL_POLICY_VIOLATION",
            run_kind=args.mode,
            artifacts={"run_root": None},
            candidate_path=candidate_path,
            approved=False,
            kernel_status=None,
            warnings=vr.warnings,
            errors=vr.errors,
        )
        # write into reports/<timestamp>/status_report.json next to candidate
        out = DEFAULT_REPORTS_ROOT / dt.datetime.now().strftime("%Y%m%d_%H%M%S") / "status_report.json"
        dump_json(out, status)
        print(f"[status] written: {out}")
        return 3

    approved = ask_approval(candidate_path, operator_names, vr.warnings)
    if not approved:
        status = build_status_report(
            jcl_status="JCL_REJECTED",
            run_kind=args.mode,
            artifacts={"run_root": None},
            candidate_path=candidate_path,
            approved=False,
            kernel_status=None,
            warnings=vr.warnings,
            errors=[],
        )
        out = DEFAULT_REPORTS_ROOT / dt.datetime.now().strftime("%Y%m%d_%H%M%S") / "status_report.json"
        dump_json(out, status)
        print(f"\n[jcl] rejected. status written: {out}")
        return 0

    # APPROVED -> DISPATCH
    if args.mode == "one_node":
        data_dir = Path(args.data_dir)
        runs_dir = Path(args.runs_dir)
        run_meta = Path(args.run_meta)
        index_path = Path(args.index_path)

        status_pre = build_status_report(
            jcl_status="JCL_DISPATCHED",
            run_kind="one_node",
            artifacts={
                "run_root": None,
                "runs_dir": safe_relpath(runs_dir),
                "data_dir": safe_relpath(data_dir),
                "run_meta": safe_relpath(run_meta),
            },
            candidate_path=candidate_path,
            approved=True,
            kernel_status=None,
            warnings=vr.warnings,
            errors=[],
        )
        out_pre = DEFAULT_REPORTS_ROOT / dt.datetime.now().strftime("%Y%m%d_%H%M%S") / "status_report.json"
        dump_json(out_pre, status_pre)
        print(f"\n[jcl] dispatched. pre-status written: {out_pre}")

        rc = dispatch_one_node(data_dir=data_dir, runs_dir=runs_dir, run_meta=run_meta)
        if rc != 0:
            print(f"[dispatch] failed rc={rc}")
            return rc

        # Detect latest run_id
        latest_run_id = find_latest_run_id(runs_dir)
        if not latest_run_id:
            print("[error] cannot find latest run_id after dispatch.")
            return 4

        run_root = runs_dir / latest_run_id
        record = build_index_record_for_one_node(latest_run_id, data_dir, run_root)
        record["status"]["kernel_status"] = "RUN_OK"
        append_jsonl(index_path, record)

        # Write status_report into run/meta
        status_post = build_status_report(
            jcl_status="JCL_INDEX_UPDATED",
            run_kind="one_node",
            artifacts={
                "run_root": safe_relpath(run_root),
                "input_dir": safe_relpath(run_root / "input"),
                "output_dir": safe_relpath(run_root / "output"),
                "meta_dir": safe_relpath(run_root / "meta"),
                "index_path": safe_relpath(index_path),
            },
            candidate_path=candidate_path,
            approved=True,
            kernel_status="RUN_OK",
            warnings=vr.warnings,
            errors=[],
        )
        dump_json(run_root / "meta" / "status_report.json", status_post)

        print("\n[jcl] DONE")
        print(f"  run_id: {latest_run_id}")
        print(f"  index : {index_path}")
        print(f"  status: {run_root / 'meta' / 'status_report.json'}")
        return 0

    elif args.mode == "queue":
        queue_path = Path(args.queue)
        index_path = Path(args.index_path)
        rc = dispatch_queue(queue_path=queue_path, stop_on_error=args.stop_on_error, plot_each_step=args.plot_each_step)
        if rc != 0:
            print(f"[dispatch] failed rc={rc}")
            return rc

        # Queue run indexing:
        # この段階では「どの step_run_id が作られたか」を厳密に追うには
        # run_operator_queue 側で “生成したstep一覧” を出力するのが理想。
        # 最小版では index には queue dispatched の事実だけ記録。
        record = {
            "schema_version": "1.0",
            "kind": "queue",
            "queue_config": safe_relpath(queue_path),
            "created_at": utc_now_iso(),
            "status": {"kernel_status": "RUN_OK"},
            "note": "Minimal JCL prototype does not enumerate step_run_ids yet."
        }
        append_jsonl(index_path, record)

        out = DEFAULT_REPORTS_ROOT / dt.datetime.now().strftime("%Y%m%d_%H%M%S") / "status_report.json"
        status = build_status_report(
            jcl_status="JCL_INDEX_UPDATED",
            run_kind="queue",
            artifacts={
                "run_root": None,
                "queue_config": safe_relpath(queue_path),
                "index_path": safe_relpath(index_path),
            },
            candidate_path=candidate_path,
            approved=True,
            kernel_status="RUN_OK",
            warnings=vr.warnings,
            errors=[],
        )
        dump_json(out, status)

        print("\n[jcl] DONE (queue)")
        print(f"  index : {index_path}")
        print(f"  status: {out}")
        return 0

    else:
        print(f"[error] unknown mode: {args.mode}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())