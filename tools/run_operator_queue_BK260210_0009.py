# tools/run_operator_queue.py

# STARTER

#python -m tools.run_operator_queue --queue configs/operator_queue_phone_demo.json --stop_on_error --plot_each_step


#1) いつもの実行（プロット無し）
#python -m tools.run_operator_queue --queue configs/operator_queue_phone_demo.json --stop_on_error

#2) 各手番ごとに「上下2段（数量＋金額）」を自動表示
#python -m tools.run_operator_queue --queue configs/operator_queue_phone_demo.json --stop_on_error --plot_each_step

#3) 量だけ（PSIだけ）表示したい
#python -m tools.run_operator_queue --queue configs/operator_queue_phone_demo.json --stop_on_error --plot_each_step --plot_psi

#4) 金額だけ表示したい
#python -m tools.run_operator_queue --queue configs/operator_queue_phone_demo.json --stop_on_error --plot_each_step --plot_money

#注意点（1行だけ）
#STEP1 は “直前手” が無いので input_snapshot が作れず、上下比較はスキップされ得ます（STEP2以降が本番）。
#スキップ時は [queue] plot(...) skipped: input_snapshot not found と出ます。

# tools/run_operator_queue.py

# STARTER
#1) いつもの実行（プロット無し）
#python -m tools.run_operator_queue --queue configs/operator_queue_phone_demo.json --stop_on_error
#
#2) 各手番ごとに「上下2段（数量＋金額）」を自動表示
#python -m tools.run_operator_queue --queue configs/operator_queue_phone_demo.json --stop_on_error --plot_each_step
#
#3) 量だけ（PSIだけ）表示したい
#python -m tools.run_operator_queue --queue configs/operator_queue_phone_demo.json --stop_on_error --plot_each_step --plot_psi
#
#4) 金額だけ表示したい
#python -m tools.run_operator_queue --queue configs/operator_queue_phone_demo.json --stop_on_error --plot_each_step --plot_money
#
#注意点（1行だけ）
#STEP1 は “直前手” が無いので input_snapshot が作れず、上下比較はスキップされ得ます（STEP2以降が本番）。
#スキップ時は [queue] plot(...) skipped: input_snapshot not found と出ます。


from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


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
        print(f"[queue] Warning: copy failed {src} -> {dst}: {e}")
        return False


def _copy_input_snapshot_from_prev(prev_run_dir: Path, cur_run_dir: Path) -> None:
    """
    直前の run の output から主要な CSV を現在の run の input_snapshot/ にコピーする。
    これにより、グラフ描画等で Before/After の比較が可能になる。

    重要:
    - plot側は input_snapshot/one_node_result_timeseries.csv と input_snapshot/money_timeseries.csv を探す
    - business_timeseries は実装・運用でファイル名が揺れがちなので候補を複数見る
    """
    snapshot_dir = cur_run_dir / "input_snapshot"
    snapshot_dir.mkdir(parents=True, exist_ok=True)  # ★必ず作る

    prev_out = prev_run_dir / "output"

    # ★実際の output に合わせた候補（存在するものだけコピー）
    candidates = [
        "one_node_result_timeseries.csv",
        "money_timeseries.csv",
        # business系（運用に合わせて候補を広げる）
        "business_timeseries_operated.csv",
        "business_timeseries_money_updated.csv",
        "business_timeseries.csv",
    ]

    copied: List[str] = []
    missing: List[str] = []

    for name in candidates:
        src = prev_out / name
        if src.exists():
            ok = _safe_copy(src, snapshot_dir / name)
            if ok:
                copied.append(name)
        else:
            missing.append(name)

    # ★デバッグログ（これで「コピーが動いた/動かない」が即わかる）
    print(f"[queue] input_snapshot prepared: cur={cur_run_dir.name} from prev={prev_run_dir.name}")
    if copied:
        print(f"[queue] input_snapshot copied: {copied}")
    else:
        print(f"[queue] Warning: input_snapshot copied nothing (prev_out={prev_out})")
    # missing は多くなりがちなので、必要ならコメントアウト可
    # print(f"[queue] input_snapshot missing candidates: {missing}")


def _collect_run_dirs(runs_dir: Path) -> List[Path]:
    """
    run_id (timestamp) 形式のディレクトリ一覧を取得
    例: 20260209_225028
    """
    pat = re.compile(r"^\d{8}_\d{6}$")  # YYYYMMDD_HHMMSS
    return sorted([d for d in runs_dir.iterdir() if d.is_dir() and pat.match(d.name)])


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
        "reason": f"queue_work: {queue_name} / step: {step.step_id}",
    }

    out_path = latest_meta_dir / "operator_spec.json"
    out_path.write_text(json.dumps(spec, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def _call_run_one_node(data_dir: str, runs_dir: Path, run_meta: str) -> None:
    """外部プロセスとして run_one_node4plugin.py を実行"""
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


def _maybe_plot_before_after(run_dir: Path, plot_money: bool, plot_psi: bool) -> None:
    """
    各手番の run_dir に対して Before/After（上下2段）を描画する。
    - input_snapshot が無いと比較できないので、無い場合はスキップ（警告だけ出す）
    """
    if plot_psi:
        try:
            from pysi.tutorial.plot_virtual_node_v0 import plot_run_dir_before_after  # lazy import
            plot_run_dir_before_after(run_dir)
        except FileNotFoundError as e:
            print(f"[queue] plot(psi) skipped: {e}")
        except Exception as e:
            print(f"[queue] plot(psi) failed: {e}")

    if plot_money:
        try:
            from pysi.tutorial.plot_virtual_node_v0_money import plot_money_run_dir_before_after  # lazy import
            plot_money_run_dir_before_after(run_dir)
        except FileNotFoundError as e:
            print(f"[queue] plot(money) skipped: {e}")
        except Exception as e:
            print(f"[queue] plot(money) failed: {e}")


# --- メインロジック ---------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Run a sequence of operators as a queue work.")
    parser.add_argument("--queue", type=str, required=True, help="Path to operator_queue_xxx.json")
    parser.add_argument("--data_dir", type=str, default="data/phone_v0", help="Data directory")
    parser.add_argument("--runs_dir", type=str, default="runs/one_node", help="Runs root directory")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop queue if a step fails")
    parser.add_argument("--dry_run", action="store_true", help="Show plan only")

    # ★追加：各手番ごとに上下2段プロット
    parser.add_argument(
        "--plot_each_step",
        action="store_true",
        help="Plot Before/After (2 rows) at the end of each step (requires input_snapshot).",
    )
    parser.add_argument(
        "--plot_money",
        action="store_true",
        help="(with --plot_each_step) Also plot money_before/after.",
    )
    parser.add_argument(
        "--plot_psi",
        action="store_true",
        help="(with --plot_each_step) Also plot psi_before/after.",
    )
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
            params=op.get("params", {}),
        )
        for i, op in enumerate(operators_raw)
    ]

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    print(f"[queue] Starting queue: {queue_name}")
    print(f"[queue] steps={len(steps)}  run_meta={run_meta}")

    # plot flags:
    # 何も指定されないときは、--plot_each_step でも「両方」を描くのが便利なのでデフォルトを両方に寄せる
    # ただし、明示指定があればそれに従う
    plot_psi = True
    plot_money = True
    if args.plot_each_step:
        if args.plot_psi or args.plot_money:
            plot_psi = bool(args.plot_psi)
            plot_money = bool(args.plot_money)

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

        if not current_run_dir:
            print("[queue] Warning: Could not detect new run directory.")
            continue

        print(f"[queue] Step completed. Run directory: {current_run_dir}")

        # 5. 前回の結果がある場合、input_snapshot を作成して Before/After 比較を可能にする
        if last_run_dir:
            _copy_input_snapshot_from_prev(last_run_dir, current_run_dir)
        else:
            print("[queue] input_snapshot: STEP1 has no previous run; skip snapshot creation (expected).")

        # 6. plot（任意）
        if args.plot_each_step:
            _maybe_plot_before_after(current_run_dir, plot_money=plot_money, plot_psi=plot_psi)

        # 7. 次のステップのために保持
        last_run_dir = current_run_dir

    print(f"\n[queue] All steps finished for queue: {queue_name}")


if __name__ == "__main__":
    main()
