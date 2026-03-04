#tools/run_detect_trust_events.py

# tools/run_detect_trust_events.py
# Usage examples:
#   python -m tools.run_detect_trust_events --runs_dir runs\one_node --latest --write
#   python -m tools.run_detect_trust_events --output_dir runs/one_node/20260304_052713/output
#   python -m tools.run_detect_trust_events --run_dir    runs/one_node/20260304_052713


import argparse
import json
from pathlib import Path
from typing import Any, Dict

from pysi.plugins.one_node.after_run.detect_trust_events import detect_trust_events


def _resolve_latest_run_output_dir(runs_dir: Path, index_path: Path | None) -> Path:
    runs_dir = runs_dir.resolve()
    if index_path is None:
        index_path = runs_dir / "_index.jsonl"
    index_path = index_path.resolve()

    if not index_path.exists():
        raise FileNotFoundError(f"index jsonl not found: {index_path}")

    last_line = None
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                last_line = line
    if not last_line:
        raise RuntimeError(f"index jsonl is empty: {index_path}")

    obj = json.loads(last_line)
    # run_one_node4plugin の index 仕様が多少揺れても耐えるように候補を複数見る
    run_dir_str = (
        obj.get("run_dir")
        or obj.get("run_path")
        or obj.get("dir")
        or obj.get("path")
    )
    if not run_dir_str:
        raise KeyError(f"Cannot find run dir key in last index entry: keys={list(obj.keys())}")

    run_dir = Path(run_dir_str)
    # indexに相対パスが入るケースを想定
    if not run_dir.is_absolute():
        run_dir = (runs_dir / run_dir).resolve()

    out = run_dir / "output"
    return out

def _resolve_output_dir(run_dir: Path | None, output_dir: Path | None) -> Path:
    if output_dir is not None:
        out = output_dir
    elif run_dir is not None:
        # allow passing ".../output" itself
        out = run_dir if run_dir.name == "output" else (run_dir / "output")
    else:
        raise ValueError("Either --output_dir or --run_dir must be provided.")

    out = out.resolve()
    if not out.exists():
        raise FileNotFoundError(f"output_dir not found: {out}")

    # minimal sanity check
    required = ["one_node_result_timeseries.csv", "kpi_summary.json"]
    missing = [name for name in required if not (out / name).exists()]
    if missing:
        raise FileNotFoundError(f"Missing required file(s) under output_dir: {out} missing={missing}")

    return out


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--runs_dir", default=None, help="e.g., runs/one_node (contains _index.jsonl and run folders)")
    ap.add_argument("--latest", action="store_true", help="resolve latest run from runs_dir/_index.jsonl")
    ap.add_argument("--index_path", default=None, help="optional explicit path to _index.jsonl")

    ap.add_argument("--output_dir", default=None, help="path to run bundle output dir")
    ap.add_argument("--run_dir", default=None, help="path to run bundle dir (contains output/)")
    ap.add_argument(
        "--write",
        action="store_true",
        help="write trust_events.json into output_dir (in addition to printing)",
    )
    args = ap.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else None
    output_dir = Path(args.output_dir) if args.output_dir else None

    #@STOP
    #out = _resolve_output_dir(run_dir=run_dir, output_dir=output_dir)
    if args.latest:
        if not args.runs_dir:
            raise ValueError("--latest requires --runs_dir (e.g., runs/one_node)")
        out = _resolve_latest_run_output_dir(
            runs_dir=Path(args.runs_dir),
            index_path=Path(args.index_path) if args.index_path else None,
        )
    else:
        out = _resolve_output_dir(run_dir=run_dir, output_dir=output_dir)



    result: Dict[str, Any] = detect_trust_events(str(out))

    # print pretty
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.write:
        out_path = out / "trust_events.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[run_detect_trust_events] wrote: {out_path}")


if __name__ == "__main__":
    main()