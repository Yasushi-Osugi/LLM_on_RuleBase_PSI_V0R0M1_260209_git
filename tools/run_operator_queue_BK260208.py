#tools/run_operator_queue.py

# Starter
# python -m tools.run_operator_queue
# ※ run_meta の中に config.decide.enabled=false を入れた「queue専用run_meta」を別に用意するのが安全
# 例：configs/phone_sellout_demo__queue.json を作って、通常版と分ける

import json
import shutil
import subprocess
from pathlib import Path
from datetime import datetime

def write_latest_operator_spec(runs_dir: Path, spec: dict) -> Path:
    latest = runs_dir / "_latest" / "meta"
    latest.mkdir(parents=True, exist_ok=True)
    p = latest / "operator_spec.json"
    p.write_text(json.dumps(spec, ensure_ascii=False, indent=2), encoding="utf-8")
    return p

def find_latest_run_id(runs_dir: Path) -> str:
    # YYYYMMDD_HHMMSS のディレクトリを最新順で拾う（あなたの運用に合わせた簡易）
    cand = [d.name for d in runs_dir.iterdir() if d.is_dir() and len(d.name) == 15 and d.name[8] == "_"]
    cand.sort(reverse=True)
    return cand[0] if cand else ""

def snapshot_run(runs_dir: Path, run_id: str, step_id: str) -> Path:
    src = runs_dir / run_id
    dst = runs_dir / "steps" / f"{step_id}__{run_id}"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copytree(src / "meta", dst / "meta", dirs_exist_ok=True)
    shutil.copytree(src / "output", dst / "output", dirs_exist_ok=True)
    return dst

def main():
    # ---- adjust here ----
    queue_path = Path("configs/operator_queue_phone_demo.json")
    data_dir = "data/phone_v0"
    runs_dir = Path("runs/one_node")
    run_meta = "configs/phone_sellout_demo.json"

    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    ops = queue.get("operators", [])
    if not ops:
        raise SystemExit("queue has no operators")

    for op in ops:
        step_id = op.get("id") or "step"
        # operator_spec は “1件dict” にする（apply_operator_spec が扱いやすい）
        spec = {
            "operator": op["operator"],
            "params": op.get("params", {}),
            "reason": op.get("reason", ""),
        }
        write_latest_operator_spec(runs_dir, spec)

        # Decide無効で回す：run_meta 側に config.decide.enabled=false を入れる
        # すでに run_meta を編集したくないなら、「queue用run_meta」を別jsonで作るのが安全
        subprocess.check_call([
            "python", "-m", "tools.run_one_node4plugin",
            "--data_dir", data_dir,
            "--runs_dir", str(runs_dir),
            "--run_meta", run_meta,
        ])

        run_id = find_latest_run_id(runs_dir)
        if not run_id:
            raise RuntimeError("could not detect latest run_id")
        dst = snapshot_run(runs_dir, run_id, step_id)
        print(f"[queue] saved snapshot: {dst}")

if __name__ == "__main__":
    main()

