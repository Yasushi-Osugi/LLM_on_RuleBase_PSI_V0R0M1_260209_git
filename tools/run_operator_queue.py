# tools/run_operator_queue.py

# STARTER

# 実行
#python -m tools.run_operator_queue --queue configs/operator_queue_phone_demo.json --stop_on_err

# 確認
#type runs\one_node\queue_work\phone_demo_4steps\step_run_ids.json
#type runs\one_node\queue_work\phone_demo_4steps\step_run_ids.jsonl

#まずは起動オプション側のおすすめ
#A. “手ごとの効き” を見る（現状の設計）
#python -m tools.run_operator_queue --queue configs/operator_queue_phone_demo.json --stop_on_error --plot_each_step
#
#
#step1 は prev が無いので BEFORE が作れず skip（これは正常）
#
#step2 以降は「直前runの出力 vs 今回runの出力」

#B. “初期からの累積差” を毎回見る（おすすめ）
#
#後述のコードの --baseline_first を使う（新設）：
#
#python -m tools.run_operator_queue --queue configs/operator_queue_phone_demo.json --stop_on_error --plot_each_step --baseline_first


#C. 目で追うのが辛いので、PNG保存も併用（おすすめ）
#
#後述のコードの --save_plots を使う（新設）：
#
#python -m tools.run_operator_queue --queue configs/operator_queue_phone_demo.json --stop_on_error --plot_each_step --save_plots


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
from __future__ import annotations

import argparse
import json
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


# --- helpers ---------------------------------------------------------------

def _safe_copy(src: Path, dst: Path) -> bool:
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(src), str(dst))
        return True
    except Exception as e:
        print(f"[queue] Warning: copy failed {src} -> {dst}: {e}")
        return False


def _write_step_run_ids(work_dir: Path, step_run_ids: List[Dict[str, Any]]) -> None:
    """
    queue_work/<queue_name>/ 配下に stepごとのrun_id一覧を出力する。
      - step_run_ids.json  : まとめ（人間が読む用）
      - step_run_ids.jsonl : 1行1レコード（grep/集計用）
    """
    work_dir.mkdir(parents=True, exist_ok=True)

    out_json = work_dir / "step_run_ids.json"
    out_jsonl = work_dir / "step_run_ids.jsonl"

    out_json.write_text(
        json.dumps(step_run_ids, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    with out_jsonl.open("w", encoding="utf-8") as f:
        for row in step_run_ids:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"[queue] wrote step_run_ids: {out_json}")
    print(f"[queue] wrote step_run_ids: {out_jsonl}")


def _copy_input_snapshot_from_run(run_dir_src: Path, cur_run_dir: Path) -> List[str]:
    """
    指定 run_dir_src の output から主要CSVを、cur_run_dir/input_snapshot にコピーする。
    戻り値: コピーできたファイル名一覧
    """
    snapshot_dir = cur_run_dir / "input_snapshot"
    prev_out = run_dir_src / "output"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    # ここは「比較で使うもの」を増やしてOK（ログに合わせて拡張）
    targets = [
        "one_node_result_timeseries.csv",
        "money_timeseries.csv",
        "business_timeseries_operated.csv",
        "business_timeseries_money_updated.csv",
        # 互換用（古いrunに存在する場合がある）
        "business_timeseries.csv",
    ]

    copied: List[str] = []
    for t in targets:
        src = prev_out / t
        if src.exists():
            if _safe_copy(src, snapshot_dir / t):
                copied.append(t)

    return copied


def _collect_run_dirs(runs_dir: Path) -> List[Path]:
    """
    run_id (timestamp) 形式のディレクトリ一覧を取得
    例: 20260209_225028
    """
    import re

    pat = re.compile(r"^\d{8}_\d{6}$")  # YYYYMMDD_HHMMSS
    return sorted([d for d in runs_dir.iterdir() if d.is_dir() and pat.match(d.name)])


def _detect_new_run_dir(before: List[Path], after: List[Path]) -> Optional[Path]:
    new_dirs = [d for d in after if d not in before]
    return new_dirs[-1] if new_dirs else None


def _write_latest_operator_spec(runs_dir: Path, step: QueueStep, queue_name: str) -> Path:
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
    index_path = runs_dir / "_index.jsonl"
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
        "--index_path",
        str(index_path),
    ]
    subprocess.check_call(cmd)


def _plot_before_after(cur_run_dir: Path, save_plots: bool = False) -> None:
    """
    PSI と Money の before/after plot を呼ぶ。
    input_snapshot が無ければスキップ。
    """
    # PSI
    try:
        from pysi.tutorial.plot_virtual_node_v0 import plot_run_dir_before_after as plot_psi
    except Exception as e:
        print(f"[queue] plot(psi) skipped: cannot import plot module: {e}")
        plot_psi = None

    # Money
    try:
        from pysi.tutorial.plot_virtual_node_v0_money import plot_run_dir_before_after as plot_money
    except Exception as e:
        print(f"[queue] plot(money) skipped: cannot import plot module: {e}")
        plot_money = None

    # 期待ファイル（plot側がこれを参照している前提）
    psi_before = cur_run_dir / "input_snapshot" / "one_node_result_timeseries.csv"
    psi_after = cur_run_dir / "output" / "one_node_result_timeseries.csv"
    money_before = cur_run_dir / "input_snapshot" / "money_timeseries.csv"
    money_after = cur_run_dir / "output" / "money_timeseries.csv"

    if plot_psi is not None:
        if not psi_before.exists():
            print(f"[queue] plot(psi) skipped: input_snapshot not found: {psi_before}")
        elif not psi_after.exists():
            print(f"[queue] plot(psi) skipped: output not found: {psi_after}")
        else:
            plot_psi(str(cur_run_dir))
            if save_plots:
                # plot側が save しない場合の保険として、画像は output 側に残す想定。
                # ここで強制saveしたい場合は plot側に savefig hook を足すのが安全。
                print("[queue] plot(psi) shown (save_plots requested: plot module responsibility)")

    if plot_money is not None:
        if not money_before.exists():
            print(f"[queue] plot(money) skipped: input_snapshot not found: {money_before}")
        elif not money_after.exists():
            print(f"[queue] plot(money) skipped: output not found: {money_after}")
        else:
            plot_money(str(cur_run_dir))
            if save_plots:
                print("[queue] plot(money) shown (save_plots requested: plot module responsibility)")


# --- main ------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Run a sequence of operators as a queue work.")
    parser.add_argument("--queue", type=str, required=True, help="Path to operator_queue_xxx.json")
    parser.add_argument("--data_dir", type=str, default="data/phone_v0", help="Data directory")
    parser.add_argument("--runs_dir", type=str, default="runs/one_node", help="Runs root directory")
    parser.add_argument("--stop_on_error", action="store_true", help="Stop queue if a step fails")
    parser.add_argument("--dry_run", action="store_true", help="Show plan only")
    parser.add_argument("--plot_each_step", action="store_true", help="Plot before/after at each step")
    parser.add_argument(
        "--baseline_first",
        action="store_true",
        help="Use STEP1 run output as baseline for all later input_snapshot (cumulative compare)",
    )
    parser.add_argument(
        "--save_plots",
        action="store_true",
        help="Request saving plots (actual save handled in plot modules; kept for CLI symmetry)",
    )

    args = parser.parse_args()

    #@ADD
    print("[queue] args=", args)


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
        for i, op in enumerate(operators_raw, start=1)
    ]

    runs_dir = Path(args.runs_dir)
    runs_dir.mkdir(parents=True, exist_ok=True)

    # queue work dir（run_id一覧など、queue単位の成果物を置く）
    queue_work_dir = runs_dir / "queue_work" / queue_name
    step_run_ids: List[Dict[str, Any]] = []

    print(f"[queue] Starting queue: {queue_name}")
    print(f"[queue] steps={len(steps)}  run_meta={run_meta}")

    last_run_dir: Optional[Path] = None
    baseline_run_dir: Optional[Path] = None

    for idx, step in enumerate(steps, start=1):
        print(f"\n[queue] --- STEP {idx:02d}/{len(steps):02d} : {step.step_id} ---")
        print(f"[queue] operator={step.operator}")

        if args.dry_run:
            print(f"[queue] (dry-run) would apply params: {step.params}")
            continue

        # 1) 次に適用する operator_spec を _latest に配置
        _write_latest_operator_spec(runs_dir, step, queue_name)

        # 2) 実行前の runs_dir 状態
        before_runs = _collect_run_dirs(runs_dir)

        # 3) run_one_node 実行
        try:
            _call_run_one_node(args.data_dir, runs_dir, run_meta)
        except subprocess.CalledProcessError as e:
            print(f"[queue] ERROR: Step {step.step_id} failed with exit code {e.returncode}")
            if args.stop_on_error:
                sys.exit(e.returncode)
            continue

        # 4) 新規 run ディレクトリ検出
        after_runs = _collect_run_dirs(runs_dir)
        current_run_dir = _detect_new_run_dir(before_runs, after_runs)

        if not current_run_dir:
            print("[queue] Warning: Could not detect new run directory.")
            continue

        # run_id は “run dir名” として確定（YYYYMMDD_HHMMSS）
        run_id = current_run_dir.name
        print(f"[queue] Step completed. Run directory: {current_run_dir}")

        # --- stepごとのrun_idを収集（あとで一覧エクスポート）
        step_run_ids.append(
            {
                "step_index": idx,
                "step_id": step.step_id,
                "operator": step.operator,
                "run_id": run_id,
                "run_root": str(current_run_dir).replace("\\", "/"),
                "reason": f"queue_work: {queue_name} / step: {step.step_id}",
            }
        )

        # baseline を保持（STEP1）
        if idx == 1 and args.baseline_first:
            baseline_run_dir = current_run_dir

        # 5) input_snapshot 作成
        if idx == 1:
            print("[queue] input_snapshot: STEP1 has no previous run; skip snapshot creation (expected).")
        else:
            src_for_snapshot = baseline_run_dir if (args.baseline_first and baseline_run_dir) else last_run_dir
            if src_for_snapshot is None:
                print("[queue] input_snapshot skipped: no source run_dir (unexpected).")
            else:
                print(f"[queue] input_snapshot prepared: cur={current_run_dir.name} from prev={src_for_snapshot.name}")
                copied = _copy_input_snapshot_from_run(src_for_snapshot, current_run_dir)
                print(f"[queue] input_snapshot copied: {copied}")

        # 6) plot
        if args.plot_each_step:
            _plot_before_after(current_run_dir, save_plots=args.save_plots)

        # 7) 次の step 用に保持
        last_run_dir = current_run_dir

    if not args.dry_run:
        _write_step_run_ids(queue_work_dir, step_run_ids)

    print(f"\n[queue] All steps finished for queue: {queue_name}")


if __name__ == "__main__":
    main()