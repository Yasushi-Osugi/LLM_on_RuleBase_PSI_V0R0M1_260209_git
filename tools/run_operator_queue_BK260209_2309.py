# tools/run_operator_queue.py

# STARTER

# “計画だけ表示”（安全）
#python -m tools.run_operator_queue --queue configs/operator_queue_phone_demo.json --dry_run

# 計画状態が変化するので、REPO/. directoryをbackupしておかないと初期状態に戻れない???
# 実行（1手ずつRUN、棋譜保存）
#python -m tools.run_operator_queue --queue configs/operator_queue_phone_demo.json --stop_on_error


# NEW STARTER
#python -m tools.run_operator_queue --queue configs/operator_queue_phone_demo.json --stop_on_error

# tools/run_operator_queue.py

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
class QueueStep:
    step_id: str
    operator: str
    params: Dict[str, Any]

# --- ヘルパー関数 -----------------------------------------------------------

def _safe_copy(src: Path, dst: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        return True
    except Exception as e:
        print(f"[queue] Warning: copy failed {src.name} -> {dst.name}: {e}")
        return False

def _copy_input_snapshot_from_prev(prev_run_dir: Path, cur_run_dir: Path) -> None:
    """
    直前の run の output から主要な CSV を現在の run の input_snapshot/ にコピーする。
    これにより、グラフ描画等で Before/After の比較が可能になる。
    """
    snapshot_dir = cur_run_dir / "input_snapshot"
    prev_out = prev_run_dir / "output"
    
    targets = [
        "one_node_result_timeseries.csv",
        "money_timeseries.csv",
        "business_timeseries.csv"
    ]
    
    for t in targets:
        src = prev_out / t
        if src.exists():
            _safe_copy(src, snapshot_dir / t)

def _collect_run_dirs(runs_dir: Path) -> List[Path]:
    """run_id (timestamp) 形式のディレクトリ一覧を取得"""
    return sorted([d for d in runs_dir.iterdir() if d.is_dir() and d.name.isdigit()])

def _detect_new_run_dir(before: List[Path], after: List[Path]) -> Optional[Path]:
    """実行後に増えた最新のディレクトリを特定する"""
    new_dirs = [d for d in after if d not in before]
    return new_dirs[-1] if new_dirs else None

def _write_latest_operator_spec(runs_dir: Path, step: QueueStep, queue_name: str) -> Path:
    """
    次の run が読み込むための runs/one_node/_latest/meta/operator_spec.json を作成する。
    """
    latest_meta_dir = runs_dir / "_latest" / "meta"
    latest_meta_dir.mkdir(parents=True, exist_ok=True)
    
    spec = {
        "operator": step.operator,
        "params": step.params,
        "reason": f"queue_work: {queue_name} / step: {step.step_id}"
    }
    
    out_path = latest_meta_dir / "operator_spec.json"
    out_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path

def _call_run_one_node(data_dir: str, runs_dir: Path, run_meta: str) -> None:
    """外部プロセスとして run_one_node4plugin.py を実行"""
    cmd = [
        sys.executable, "-m", "tools.run_one_node4plugin",
        "--data_dir", data_dir,
        "--runs_dir", str(runs_dir),
        "--run_meta", run_meta
    ]
    # 実行。エラー時は CalledProcessError を投げる
    subprocess.check_call(cmd)

# --- メインロジック ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run a sequence of operators as a queue work.")
    parser.add_argument("--queue", type=str, required=True, help="Path to operator_queue_xxx.json")
    parser.add_argument("--data_dir", type=str, default="data/phone_v0", help="Data directory")
    parser.add_argument("--runs_dir", type=str, default="runs/one_node", help="Runs root directory")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop queue if a step fails")
    parser.add_argument("--dry_run", action="store_true", help="Show plan only")
    args = parser.parse_args()

    queue_path = Path(args.queue)
    if not queue_path.exists():
        print(f"Error: queue file not found: {args.queue}")
        sys.exit(1)

    with open(queue_path, "r", encoding="utf-8") as f:
        queue_data = json.load(f)

    queue_name = queue_data.get("queue_name", "unnamed_queue")
    run_meta = queue_data.get("run_meta", "configs/phone_sellout_demo.json")
    operators_raw = queue_data.get("operators", [])

    steps = [
        QueueStep(
            step_id=op.get("name", f"step_{i:02d}"),
            operator=op.get("operator", ""),
            params=op.get("params", {})
        )
        for i, op in enumerate(operators_raw)
    ]

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[queue] Starting queue: {queue_name}")
    print(f"[queue] steps={len(steps)}  run_meta={run_meta}")

    last_run_dir: Optional[Path] = None

    for idx, step in enumerate(steps, start=1):
        print(f"\n[queue] --- STEP {idx:02d}/{len(steps):02d} : {step.step_id} ---")
        print(f"[queue] operator={step.operator}")
        
        if args.dry_run:
            print(f"[queue] (dry-run) would apply: {step.params}")
            continue

        # 1. 実行前に「次に適用すべき指示書」を _latest に配置
        _write_latest_operator_spec(runs_dir, step, queue_name)

        # 2. 実行前のディレクトリ状態を記録
        before_runs = _collect_run_dirs(runs_dir)

        # 3. シミュレーション実行
        try:
            _call_run_one_node(args.data_dir, runs_dir, run_meta)
        except subprocess.CalledProcessError as e:
            print(f"[queue] ERROR: Step {step.step_id} failed with exit code {e.returncode}")
            if args.stop_on_error:
                sys.exit(e.returncode)
            continue

        # 4. 生成された run ディレクトリを特定
        after_runs = _collect_run_dirs(runs_dir)
        current_run_dir = _detect_new_run_dir(before_runs, after_runs)

        if current_run_dir:
            print(f"[queue] Step completed. Run directory: {current_run_dir}")
            
            # 5. 前回の結果がある場合、input_snapshot を作成して Before/After 比較を可能にする
            if last_run_dir:
                _copy_input_snapshot_from_prev(last_run_dir, current_run_dir)
            
            # 6. 次のステップのために、今回の output を data_dir に書き戻す必要がある場合は、
            #    各プラグイン（operate_writeback_business_table等）が既に行っている想定。
            #    ここでは、今回の実行ディレクトリを「次回の直前ディレクトリ」として保持。
            last_run_dir = current_run_dir
        else:
            print("[queue] Warning: Could not detect new run directory.")

    print(f"\n[queue] All steps finished for queue: {queue_name}")

if __name__ == "__main__":
    main()
