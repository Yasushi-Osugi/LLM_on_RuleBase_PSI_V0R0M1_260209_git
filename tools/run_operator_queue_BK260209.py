# tools/run_operator_queue.py

# STARTER

# “計画だけ表示”（安全）
#python -m tools.run_operator_queue --queue configs/operator_queue_phone_demo.json --dry_run

# 計画状態が変化するので、REPO/. directoryをbackupしておかないと初期状態に戻れない???
# 実行（1手ずつRUN、棋譜保存）
#python -m tools.run_operator_queue --queue configs/operator_queue_phone_demo.json --stop_on_error


from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class StepSpec:
    step_id: str
    operator: str
    params: Dict[str, Any]
    reason: Any  # str or dict


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _collect_run_dirs(runs_dir: Path) -> Dict[str, float]:
    """
    Return {dir_name: mtime} for run dirs like YYYYMMDD_HHMMSS
    """
    d: Dict[str, float] = {}
    if not runs_dir.exists():
        return d
    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue
        name = p.name
        if len(name) == 15 and name[8] == "_":
            try:
                d[name] = p.stat().st_mtime
            except Exception:
                pass
    return d


def _detect_new_run_dir(before: Dict[str, float], after: Dict[str, float]) -> Optional[str]:
    new_names = [k for k in after.keys() if k not in before]
    if new_names:
        # if multiple, pick most recent mtime
        new_names.sort(key=lambda n: after.get(n, 0.0), reverse=True)
        return new_names[0]
    # fallback: pick most recently modified (sometimes run dir already existed in edge cases)
    if after:
        names = list(after.keys())
        names.sort(key=lambda n: after.get(n, 0.0), reverse=True)
        return names[0]
    return None


def _normalize_steps(queue_obj: Dict[str, Any]) -> Tuple[List[StepSpec], str, Optional[str]]:
    """
    queue_obj format:
    {
      "queue_name": "...",
      "run_meta": "configs/xxx__queue.json",   # optional
      "operators": [ { "id": "...", "operator": "...", "params": {...}, "reason": ... }, ... ]
    }
    """
    queue_name = str(queue_obj.get("queue_name") or "queue")
    run_meta = queue_obj.get("run_meta")  # may be None
    ops = queue_obj.get("operators") or []
    if not isinstance(ops, list) or not ops:
        raise ValueError("operator_queue.json must have non-empty list: operators[]")

    steps: List[StepSpec] = []
    for i, it in enumerate(ops, start=1):
        if not isinstance(it, dict):
            raise ValueError(f"operators[{i}] must be an object")
        step_id = str(it.get("id") or f"step_{i:02d}")
        operator = str(it.get("operator") or "").strip()
        if not operator:
            raise ValueError(f"operators[{i}] missing operator")
        params = it.get("params") or {}
        if not isinstance(params, dict):
            raise ValueError(f"operators[{i}].params must be an object")
        reason = it.get("reason", "")
        steps.append(StepSpec(step_id=step_id, operator=operator, params=params, reason=reason))

    return steps, queue_name, (str(run_meta) if run_meta else None)


def _write_latest_operator_spec(
    runs_dir: Path,
    step: StepSpec,
    queue_name: str,
) -> Path:
    """
    Write ONE operator spec to: runs/one_node/_latest/meta/operator_spec.json
    """
    latest_meta = runs_dir / "_latest" / "meta"
    latest_meta.mkdir(parents=True, exist_ok=True)
    out = latest_meta / "operator_spec.json"

    # Keep shape compatible with apply_operator_spec: {"operator":..., "params":..., "reason":...}
    spec = {
        "operator": step.operator,
        "params": step.params,
        "reason": {
            "queue_name": queue_name,
            "step_id": step.step_id,
            "detail": step.reason,
        },
    }
    _write_json(out, spec)
    return out


def _snapshot_run(
    runs_dir: Path,
    run_id: str,
    queue_name: str,
    step: StepSpec,
) -> Path:
    """
    Copy meta/ and output/ into:
      runs/one_node/steps/<queue_name>/<step_id>__<run_id>/
    """
    src = runs_dir / run_id
    if not src.exists():
        raise FileNotFoundError(f"run dir not found: {src}")

    dst = runs_dir / "steps" / queue_name / f"{step.step_id}__{run_id}"
    dst.mkdir(parents=True, exist_ok=True)

    # Copy meta and output (merge if exists)
    for sub in ["meta", "output"]:
        s = src / sub
        if s.exists():
            shutil.copytree(s, dst / sub, dirs_exist_ok=True)

    # Also store the operator_spec used for this step
    used_spec = runs_dir / "_latest" / "meta" / "operator_spec.json"
    if used_spec.exists():
        shutil.copy2(used_spec, dst / "operator_spec_used.json")

    return dst


def _call_run_one_node(
    data_dir: str,
    runs_dir: Path,
    run_meta: str,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "tools.run_one_node4plugin",
        "--data_dir",
        data_dir,
        "--runs_dir",
        str(runs_dir),
        "--run_meta",
        run_meta,
    ]
    subprocess.check_call(cmd)


def main() -> int:
    ap = argparse.ArgumentParser(description="Run operator queue step-by-step (one operator per run).")
    ap.add_argument("--queue", required=True, help="Path to operator_queue.json (e.g. configs/operator_queue_phone_demo.json)")
    ap.add_argument("--data_dir", default="data/phone_v0", help="Data directory passed to run_one_node4plugin")
    ap.add_argument("--runs_dir", default="runs/one_node", help="Runs directory passed to run_one_node4plugin")
    ap.add_argument("--run_meta", default=None, help="Run meta JSON path. If omitted, use queue.run_meta")
    ap.add_argument("--start", type=int, default=1, help="Start step index (1-based)")
    ap.add_argument("--end", type=int, default=None, help="End step index (1-based, inclusive)")
    ap.add_argument("--dry_run", action="store_true", help="Do not execute; just show planned steps")
    ap.add_argument("--stop_on_error", action="store_true", help="Stop queue on first failure (recommended)")
    args = ap.parse_args()

    queue_path = Path(args.queue)
    if not queue_path.exists():
        print(f"[queue] ERROR: queue file not found: {queue_path}")
        return 2

    queue_obj = _read_json(queue_path)
    steps, queue_name, queue_run_meta = _normalize_steps(queue_obj)

    run_meta = args.run_meta or queue_run_meta
    if not run_meta:
        print("[queue] ERROR: run_meta not provided. Set --run_meta or queue.run_meta")
        return 2

    runs_dir = Path(args.runs_dir)
    start = max(1, int(args.start))
    end = int(args.end) if args.end is not None else len(steps)
    end = min(end, len(steps))

    plan = steps[start - 1 : end]
    print(f"[queue] queue_name={queue_name}")
    print(f"[queue] run_meta={run_meta}")
    print(f"[queue] steps={len(steps)}  run_range={start}..{end}  planned={len(plan)}")

    for idx, step in enumerate(plan, start=start):
        print(f"\n[queue] --- STEP {idx:02d}/{len(steps):02d} : {step.step_id} ---")
        print(f"[queue] operator={step.operator}")
        print(f"[queue] params={step.params}")

        spec_path = _write_latest_operator_spec(runs_dir, step, queue_name)
        print(f"[queue] wrote _latest operator_spec: {spec_path}")

        if args.dry_run:
            continue

        before = _collect_run_dirs(runs_dir)
        try:
            _call_run_one_node(args.data_dir, runs_dir, run_meta)
        except subprocess.CalledProcessError as e:
            print(f"[queue] ERROR: run_one_node4plugin failed at step {step.step_id} (exit={e.returncode})")
            if args.stop_on_error:
                return e.returncode or 1
            else:
                continue

        after = _collect_run_dirs(runs_dir)
        run_id = _detect_new_run_dir(before, after)
        if not run_id:
            print("[queue] ERROR: could not detect new run_id")
            if args.stop_on_error:
                return 1
            else:
                continue

        snap = _snapshot_run(runs_dir, run_id, queue_name, step)
        print(f"[queue] detected run_id={run_id}")
        print(f"[queue] saved snapshot: {snap}")

    print("\n[queue] DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
