# tools/run_detect_trust_events.py
#
# Usage examples:
#   # latest（indexから自動解決）
#   python -m tools.run_detect_trust_events --runs_dir runs\one_node --latest --write
#
#   # output_dirをoutputにして起動（今まで通り）
#   python -m tools.run_detect_trust_events --output_dir runs/one_node/20260304_052713/output
#
#   # output_dir に run_dir を渡してもOK（今回の落ち方を潰す）
#   python -m tools.run_detect_trust_events --output_dir runs/one_node/20260304_052713 --write

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from pysi.plugins.one_node.after_run.detect_trust_events import detect_trust_events

REQUIRED_FILES = ("one_node_result_timeseries.csv", "kpi_summary.json")


# ----------------------------
# Path helpers
# ----------------------------
def _path_from_cli(raw: str) -> Path:
    # Accept Windows-style backslash paths even on POSIX shells.
    return Path(raw.replace("\\", "/"))


def _has_required_files(p: Path) -> bool:
    return p.exists() and p.is_dir() and all((p / name).exists() for name in REQUIRED_FILES)


def _coerce_to_output_dir(p: Path) -> Path:
    """
    Accept either:
      - output_dir itself (contains required files)
      - run_dir (has output/ under it)
      - meta dir (p.name == 'meta' and sibling output/ exists)
    """
    # 1) already output_dir
    if _has_required_files(p):
        return p

    # 2) run_dir -> output/
    out = p / "output"
    if _has_required_files(out):
        return out

    # 3) mistakenly passed meta/
    if p.name == "meta":
        out2 = p.parent / "output"
        if _has_required_files(out2):
            return out2

    # Return original so caller can produce a good error.
    return p


# ----------------------------
# Index helpers
# ----------------------------
def _iter_last_json_line(index_path: Path) -> Dict[str, Any]:
    if not index_path.exists():
        raise FileNotFoundError(f"index file not found: {index_path}")

    last: Optional[str] = None
    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                last = s

    if not last:
        raise ValueError(f"index file is empty: {index_path}")

    try:
        return json.loads(last)
    except Exception as e:
        raise ValueError(f"failed to parse last json line in {index_path}: {e}") from e


def _candidate_dirs_from_index(obj: Dict[str, Any], runs_dir: Path) -> Iterable[Path]:
    """
    Your current index entry keys (observed):
      keys=['schema_version','kind','run_id','created_at','data_dir','run_root','inputs','outputs','status']
    We try multiple ways, in safe order.
    """

    # 1) If outputs include concrete artifact paths, use their parent dirs (most reliable)
    outputs = obj.get("outputs")
    if isinstance(outputs, dict):
        for k in ("psi_result", "kpi_summary", "money_result", "diagnosis"):
            v = outputs.get(k)
            if isinstance(v, str) and v.strip():
                p = _path_from_cli(v)
                if not p.is_absolute():
                    p = runs_dir / p
                # if it's a file -> parent should be output dir
                yield p.parent

        # (compat) if someone wrote output_dir/output/out_dir into outputs
        for k in ("output_dir", "output", "out_dir"):
            v = outputs.get(k)
            if isinstance(v, str) and v.strip():
                p = _path_from_cli(v)
                if not p.is_absolute():
                    p = runs_dir / p
                yield p

    run_id = obj.get("run_id")
    run_root = obj.get("run_root")

    # 2) run_root + run_id
    if isinstance(run_root, str) and run_root.strip() and isinstance(run_id, str) and run_id.strip():
        root = _path_from_cli(run_root)
        if not root.is_absolute():
            root = runs_dir / root
        yield root / run_id / "output"
        yield root / run_id  # run_dir

    # 3) runs_dir + run_id
    if isinstance(run_id, str) and run_id.strip():
        yield runs_dir / run_id / "output"
        yield runs_dir / run_id

    # 4) legacy compatibility
    for legacy_key in ("run_dir", "run_path", "dir", "path"):
        v = obj.get(legacy_key)
        if isinstance(v, str) and v.strip():
            p = _path_from_cli(v)
            if not p.is_absolute():
                p = runs_dir / p
            yield p / "output"
            yield p


def _resolve_latest_run_output_dir(runs_dir: str, index_name: str = "_index.jsonl", index_path: Optional[str] = None) -> Path:
    runs_dir_p = _path_from_cli(runs_dir).resolve()

    idx = _path_from_cli(index_path).resolve() if index_path else (runs_dir_p / index_name)
    obj = _iter_last_json_line(idx)

    tried: list[str] = []
    for cand in _candidate_dirs_from_index(obj, runs_dir_p):
        coerced = _coerce_to_output_dir(cand)
        tried.append(str(coerced))
        if _has_required_files(coerced):
            return coerced

    raise FileNotFoundError(
        "Cannot resolve latest output_dir with required files.\n"
        f"index={idx}\n"
        f"index keys={list(obj.keys())}\n"
        f"tried={tried}\n"
        f"required={list(REQUIRED_FILES)}"
    )


def _resolve_output_dir(run_dir: Optional[str], output_dir: Optional[str]) -> Path:
    if output_dir:
        out = _coerce_to_output_dir(_path_from_cli(output_dir))
        if _has_required_files(out):
            return out
        missing = [f for f in REQUIRED_FILES if not (out / f).exists()]
        raise FileNotFoundError(f"Missing required file(s) under output_dir: {out} missing={missing}")

    if run_dir:
        out = _coerce_to_output_dir(_path_from_cli(run_dir))
        if _has_required_files(out):
            return out
        missing = [f for f in REQUIRED_FILES if not (out / f).exists()]
        raise FileNotFoundError(f"Missing required file(s) under run_dir: {out} missing={missing}")

    raise ValueError("Either --output_dir or --run_dir must be provided.")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--runs_dir", default=None, help="e.g., runs/one_node (contains _index.jsonl and run folders)")
    ap.add_argument("--latest", action="store_true", help="resolve latest run from runs_dir/_index.jsonl")
    ap.add_argument("--index_path", default=None, help="optional explicit path to _index.jsonl")

    ap.add_argument("--output_dir", default=None, help="path to run bundle output dir OR run dir (we coerce)")
    ap.add_argument("--run_dir", default=None, help="path to run bundle dir (contains output/)")

    ap.add_argument("--write", action="store_true", help="write trust_events.json into output_dir (in addition to printing)")
    args = ap.parse_args()

    if args.latest:
        if not args.runs_dir:
            raise ValueError("--latest requires --runs_dir (e.g., runs/one_node)")
        out = _resolve_latest_run_output_dir(
            runs_dir=args.runs_dir,
            index_path=args.index_path,
        )
    else:
        out = _resolve_output_dir(run_dir=args.run_dir, output_dir=args.output_dir)

    result: Dict[str, Any] = detect_trust_events(str(out))

    # print pretty
    print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.write:
        out_path = out / "trust_events.json"
        out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[run_detect_trust_events] wrote: {out_path}")


if __name__ == "__main__":
    main()