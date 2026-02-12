#tools/dump_contract.py
import json
from pathlib import Path

def _load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def _first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None

def summarize_operator_spec(op: dict) -> dict:
    s = {"top_keys": sorted(list(op.keys()))}
    s["operator"] = op.get("operator")
    s["has_params"] = "params" in op
    # 代表サンプル
    if isinstance(op.get("params"), dict):
        s["params_keys"] = sorted(list(op["params"].keys()))
    return s

def main():
    runs_dir = Path("runs/one_node")

    # 1) 最優先: _latest/meta
    p1 = runs_dir / "_latest" / "meta" / "operator_spec.json"

    # 2) meta配下を探索
    cand_meta = sorted(runs_dir.glob("**/meta/operator_spec.json"),
                       key=lambda p: p.stat().st_mtime, reverse=True)

    # 3) output配下も探索（将来）
    cand_out = sorted(runs_dir.glob("**/output/operator_spec.json"),
                      key=lambda p: p.stat().st_mtime, reverse=True)

    op_path = _first_existing([p1] + cand_meta + cand_out)

    if not op_path:
        print("[dump_contract] operator_spec.json not found under", runs_dir)
        print("  tried:", p1)
        print("  meta matches:", len(cand_meta), " output matches:", len(cand_out))
        return

    print("[dump_contract] operator_spec:", op_path)

    op = _load_json(op_path)
    print(json.dumps(summarize_operator_spec(op), ensure_ascii=False, indent=2))

    # diagnosis も同じ階層にあるとは限らないので、同runのoutputを推測
    # _latest/meta → _latest は meta しかないのでスキップしてOK
    if op_path.parent.name == "meta" and op_path.parent.parent.name != "_latest":
        out_dir = op_path.parent.parent / "output"
        diag_path = out_dir / "diagnosis.json"
        if diag_path.exists():
            diag = _load_json(diag_path)
            print("[dump_contract] diagnosis keys:", sorted(list(diag.keys()))[:50])
            issues = diag.get("issues") or []
            print("[dump_contract] issues_len:", len(issues))
            print("[dump_contract] issue_sample:", json.dumps(issues[:1], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
