# tools/show_index.py
# tools/show_index.py

# STARTER
#python -m tools.show_index --index_path runs\one_node\_index.jsonl --n 5
#python -m tools.show_index --index_path runs\one_node\_index.jsonl --n 20 --json

#python -m tools.show_index --index_path runs\one_node\_index.jsonl --n 5 --verify
#python -m tools.show_index --index_path runs\one_node\_index.jsonl --n 20 --json --verify


from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"index not found: {path}")
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for ln, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                # 壊れた行があっても運用を止めない（台帳は現場で育つ）
                rows.append(
                    {
                        "kind": "parse_error",
                        "line_no": ln,
                        "raw": line[:2000],
                        "error": str(e),
                    }
                )
    return rows


def _tail(items: List[Dict[str, Any]], n: int) -> List[Dict[str, Any]]:
    if n <= 0:
        return []
    return items[-n:]


def _get(d: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def _repo_root_from_index_path(index_path: Path) -> Path:
    """
    runs/one_node/_index.jsonl を想定し、
    indexの2階層上（repo root相当）を基準に解決する。
    例:
      index_path = <repo>/runs/one_node/_index.jsonl
      repo_root  = <repo>
    """
    p = index_path.resolve()
    # .../runs/one_node/_index.jsonl -> parents[2] == .../
    if len(p.parents) >= 3:
        return p.parents[2]
    return p.parent


def _resolve_path(p: Optional[str], base_dir: Path) -> Optional[Path]:
    if not p:
        return None
    pp = Path(p)
    if pp.is_absolute():
        return pp
    # JSONL内は "runs/one_node/..." のような repo相対を想定
    return (base_dir / pp).resolve()


def _exists(p: Optional[str], base_dir: Path) -> bool:
    rp = _resolve_path(p, base_dir)
    return bool(rp and rp.exists())


def _summarize_one(row: Dict[str, Any], *, verify: bool, base_dir: Path) -> Dict[str, Any]:
    run_id = row.get("run_id", "")
    created_at = row.get("created_at", "")
    kind = row.get("kind", "")
    data_dir = row.get("data_dir", "")
    run_root = row.get("run_root", "")

    kernel_status = _get(row, "status.kernel_status", "")
    plots = _get(row, "outputs.plots", []) or []
    plots_n = len(plots) if isinstance(plots, list) else 0

    psi_result = _get(row, "outputs.psi_result", "")
    money_result = _get(row, "outputs.money_result", "")
    diagnosis = _get(row, "outputs.diagnosis", "")
    kpi_summary = _get(row, "outputs.kpi_summary", "")

    return {
        "run_id": run_id,
        "created_at": created_at,
        "kind": kind,
        "data_dir": data_dir,
        "run_root": run_root,
        "kernel_status": kernel_status,
        "plots_n": plots_n,
        "has_psi": bool(psi_result),
        "has_money": bool(money_result),
        "has_diag": bool(diagnosis),
        "has_kpi": bool(kpi_summary),
        # optional existence checks (can be slow on network fs)
        "psi_exists": _exists(psi_result, base_dir) if verify else None,
        "money_exists": _exists(money_result, base_dir) if verify else None,
        "diag_exists": _exists(diagnosis, base_dir) if verify else None,
        "kpi_exists": _exists(kpi_summary, base_dir) if verify else None,
    }


def _is_ok(kernel_status: str) -> bool:
    return str(kernel_status).upper() in ("RUN_OK", "OK", "SUCCESS")


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarize recent runs from _index.jsonl")
    ap.add_argument("--index_path", default="runs/one_node/_index.jsonl")
    ap.add_argument("--n", type=int, default=10, help="show last N entries")
    ap.add_argument("--kind", default=None, help="filter by kind (e.g. one_node)")
    ap.add_argument("--json", action="store_true", help="output summary as json")
    ap.add_argument(
        "--verify",
        action="store_true",
        help="check output file existence (slower). default: off",
    )
    args = ap.parse_args()

    path = Path(args.index_path)
    rows = _read_jsonl(path)
    base_dir = _repo_root_from_index_path(path)

    # filter out parse_error rows unless user asks kind=parse_error
    if args.kind:
        rows = [r for r in rows if r.get("kind") == args.kind]
    else:
        rows = [r for r in rows if r.get("kind") != "parse_error"]

    recent = _tail(rows, args.n)
    items = [_summarize_one(r, verify=args.verify, base_dir=base_dir) for r in recent]

    ok = sum(1 for x in items if _is_ok(x.get("kernel_status", "")))
    ng = len(items) - ok

    result = {
        "index_path": str(path),
        "total_entries": len(rows),
        "shown": len(items),
        "ok": ok,
        "ng": ng,
        "latest": items[-1] if items else None,
        "items": items,
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # human-friendly output
    print(f"Index: {path}")
    print(f"Base : {base_dir}")
    print(f"Total entries (after filter): {len(rows)}   Shown: {len(items)}   OK: {ok}   NG: {ng}")
    if items:
        latest = items[-1]
        print(f"Latest: {latest['run_id']}  {latest['created_at']}  status={latest['kernel_status']}  plots={latest['plots_n']}")
    print("-" * 80)

    for x in items:
        flags = []
        flags.append("PSI" if x["has_psi"] else "psi-")
        flags.append("MNY" if x["has_money"] else "mny-")
        flags.append("DIA" if x["has_diag"] else "dia-")
        flags.append("KPI" if x["has_kpi"] else "kpi-")
        if args.verify:
            vflags = []
            vflags.append("psiOK" if x["psi_exists"] else "psi??")
            vflags.append("mnyOK" if x["money_exists"] else "mny??")
            vflags.append("diaOK" if x["diag_exists"] else "dia??")
            vflags.append("kpiOK" if x["kpi_exists"] else "kpi??")
            vtxt = "  " + "/".join(vflags)
        else:
            vtxt = ""
        print(
            f"{x['run_id']}  {x['created_at']}  {x['kernel_status']:<8}  plots={x['plots_n']:<2}  "
            f"{'/'.join(flags)}  {x.get('data_dir','')}{vtxt}"
        )


if __name__ == "__main__":
    main()