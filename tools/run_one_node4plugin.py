# tools/run_one_node4plugin.py

#STARTER
#python -m tools.run_one_node4plugin ^
#  --data_dir data/rice_v0_on_CAP ^
#  --runs_dir runs/one_node ^
#  --run_meta configs/rice_demand_adjust_demo.json

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from pysi.tutorial.virtual_node_v0_adapter import load_phone_v0
from pysi.tutorial.plot_virtual_node_v0 import plot_phone_v0

# HookBus (same boot sequence spirit as main.py)
from pysi.core.hooks.core import HookBus, autoload_plugins, set_global, call_register_if_present

from pysi.tutorial.plot_virtual_node_v0_money import plot_money_timeseries

def _read_json_OLD(path: Path) -> Dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _load_run_meta_OLD(repo_root_meta: Path, meta_dir_meta: Path, run_meta_path: Path | None) -> Dict[str, Any]:
    """
    設定 run_meta.json の読み込み優先順位:
      1) --run_meta で指定された JSON
      2) meta/run_meta.json（run bundle 側に置かれている場合）
      3) リポジトリ直下 run_meta.json
      4) 空
    """
    if run_meta_path is not None and run_meta_path.exists():
        return _read_json(run_meta_path)

    if meta_dir_meta.exists():
        return _read_json(meta_dir_meta)

    if repo_root_meta.exists():
        return _read_json(repo_root_meta)

    return {}




def _read_json(p: Path) -> Dict[str, Any]:
    # 1) read text (utf-8 -> cp932 fallback)
    try:
        txt = p.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        txt = p.read_text(encoding="cp932")

    # 2) parse json
    try:
        obj = json.loads(txt)
    except Exception as e:
        raise ValueError(f"JSON decode failed: {p} ({e})")

    if not isinstance(obj, dict):
        raise ValueError(f"JSON root is not an object(dict): {p} type={type(obj)}")

    return obj

def _load_run_meta(repo_root_meta: Path, meta_dir_meta: Path, run_meta_path: Path | None) -> Dict[str, Any]:
    """
    設定 run_meta.json の読み込み優先順位:
      1) --run_meta で指定された JSON
      2) meta/run_meta.json（run bundle 側に置かれている場合）
      3) リポジトリ直下 run_meta.json
      4) 空
    """
    candidates: list[tuple[str, Path]] = []
    if run_meta_path is not None:
        candidates.append(("--run_meta", run_meta_path))
    candidates.append(("run_bundle_meta", meta_dir_meta))
    candidates.append(("repo_root", repo_root_meta))

    for label, path in candidates:
        if path is None or not path.exists():
            continue
        try:
            meta = _read_json(path)
            # ここが超重要：空なら「成功扱い」にしない
            if not meta:
                print(f"[run_one_node4plugin] run_meta loaded but EMPTY: source={label} path={path}")
                continue
            print(f"[run_one_node4plugin] run_meta source={label} path={path} keys={list(meta.keys())}")
            return meta
        except Exception as e:
            # 握りつぶさず理由を出す（debug中はこれが命綱）
            print(f"[run_one_node4plugin] run_meta load FAILED: source={label} path={path} err={e}")
            continue

    print("[run_one_node4plugin] run_meta not found; use empty {}")
    return {}




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data_dir",
        required=True,
        help="data directory (contains virtual_node_timeseries.csv and capacity_*.csv)",
    )
    ap.add_argument(
        "--cap_mode",
        default="soft",
        choices=["soft", "hard"],
        help="I_cap mode: soft=record over_i, hard=clip to I_cap",
    )
    ap.add_argument(
        "--timeseries_name",
        default=None,
        help="timeseries csv filename (default: virtual_node_timeseries.csv)",
    )
    ap.add_argument(
        "--shelf_life",
        type=int,
        default=None,
        help="months; enable FIFO+expiration if set (e.g. 3)",
    )
    ap.add_argument(
        "--runs_dir",
        default="runs/one_node",
        help="run bundle base dir (default: runs/one_node)",
    )
    ap.add_argument(
        "--run_meta",
        default=None,
        help="path to scenario meta json (e.g. configs/rice_demand_adjust_demo.json)",
    )
    args = ap.parse_args()

    # run bundle dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.runs_dir) / ts
    input_dir = run_dir / "input"
    output_dir = run_dir / "output"
    meta_dir = run_dir / "meta"
    input_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)


    # ----------------------------
    # copy input dataset into run bundle (input/)
    # so before_load operators can safely rewrite CSVs
    # ----------------------------
    src_dir = Path(args.data_dir)
    if src_dir.exists():
        for p in src_dir.glob("*.csv"):
            try:
                shutil.copy2(p, input_dir / p.name)
            except Exception:
                pass


    # load run_meta.json (config)
    repo_root_meta = Path("run_meta.json")
    meta_dir_meta = meta_dir / "run_meta.json"
    run_meta_path = Path(args.run_meta) if args.run_meta else None
    meta = _load_run_meta(repo_root_meta, meta_dir_meta, run_meta_path)

    print("[run_one_node4plugin] run_meta_path=", run_meta_path)
    print("[run_one_node4plugin] loaded meta keys=", list(meta.keys()) if isinstance(meta, dict) else type(meta))


    # keep a copy in run bundle for traceability
    # IMPORTANT: plugin(before_load) expects run_meta.json to have {"config": {...}}
    normalized: Dict[str, Any] = {}
    if isinstance(meta, dict) and meta:
        if "config" in meta and isinstance(meta.get("config"), dict):
            normalized = meta
        else:
            normalized = {"config": meta}

    if normalized:
        try:
            meta_dir_meta.write_text(
                json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

        #@STOP
        #try:
        #    (meta_dir_meta / "run_meta.json").write_text(
        #        json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8"
        #    )
        #except Exception:
        #    pass


    # ----------------------------
    # copy input data -> run bundle input/
    # ----------------------------
    src_dir = Path(args.data_dir)
    if src_dir.exists():
        for p in src_dir.glob("*.csv"):
            try:
                shutil.copy(p, input_dir / p.name)
            except Exception:
                pass

    # ----------------------------
    # run_ctx (dict) for plugins
    # ----------------------------
    config = normalized.get("config", {}) if isinstance(normalized, dict) else {}
    run_ctx: Dict[str, Any] = {
        "run_dir": str(run_dir),
        "data_dir": str(args.data_dir),
        "input_dir": str(input_dir),
        "data_dir_effective": str(input_dir),  # use copied inputs
        "output_dir": str(output_dir),
        "meta_dir": str(meta_dir),
        "config": config,
        "meta": normalized,
    }

    # HookBus boot
    bus = HookBus()
    set_global(bus)
    autoload_plugins("pysi.plugins.one_node")
    call_register_if_present("pysi.plugins.one_node", bus)

    # ---- hook point: before_load (CSV rewrite etc.)
    bus.do_action(
        "one_node.before_load",
        run_ctx=run_ctx,
        meta=run_ctx.get("meta"),
        config=run_ctx.get("config"),
    )

    # load + run (use effective data_dir)
    m = load_phone_v0(
        run_ctx.get("data_dir_effective", args.data_dir),
        cap_mode=args.cap_mode,
        timeseries_name=args.timeseries_name,
        shelf_life=args.shelf_life,
    )

    # after_load hook
    bus.do_action("one_node.after_load", model=m, run_ctx=run_ctx)

    months = m.months
    print("months:", months[:3], ".", months[-3:])
    print("total demand:", float(m.demand.sum()))

    if hasattr(m, "production"):
        total_prod = float(m.production.sum())
    elif hasattr(m, "production_plan"):
        total_prod = float(m.production_plan.sum())
    elif hasattr(m, "prod_plan"):
        total_prod = float(m.prod_plan.sum())
    else:
        total_prod = 0.0
    print("total production:", total_prod)

    print("ending inventory:", float(m.inv.iloc[-1]) if len(months) else 0.0)
    print("total backlog(end):", float(m.backlog.iloc[-1]) if len(months) else 0.0)
    print("total waste:", float(m.waste.sum()))
    print("total over_i (soft cap overage):", float(m.over_i.sum()))

    # after_run hook (export run-bundle etc.)
    bus.do_action(
        "one_node.after_run",
        run_dir=str(run_dir),
        data_dir=str(args.data_dir),
        data_dir_effective=str(run_ctx.get("data_dir_effective", args.data_dir)),
        args=vars(args),
        model=m,
        run_ctx=run_ctx,
    )

    plot_phone_v0(m, title="One-node PSI overview (run bundle)")


    #@ADD
    #from pysi.tutorial.plot_virtual_node_v0_money import plot_money_timeseries

    money_csv = Path(output_dir) / "money_timeseries.csv"
    if money_csv.exists():
        plot_money_timeseries(money_csv, title="Money overview (run bundle)")



if __name__ == "__main__":
    main()
