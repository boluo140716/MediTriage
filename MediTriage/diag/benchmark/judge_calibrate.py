"""judge 校准：用已知正确答案的冻结 MCQ 集，量化两 judge 的可信度。

对每条校准题：让 DeepSeek、Gemini 各自"从 options 选正确字母"，与已知 answer 比对
→ 各自准确率（+ Wilson 置信区间）；再算两 judge 选择的 Cohen's κ 一致性。
准确率明显 > 随机（5 选 1 ≈ 0.2）且 > 0.7，方能支撑"用它们当裁判"。

用法:
    docker exec -e DEEPSEEK_API_KEY=... -e GEMINI_API_KEY=... \
        medix-fix python3 /workspace/MediTriage/diag/benchmark/judge_calibrate.py [--n N]
输出:
    /workspace/log/benchmark/judge_calibration.json + stdout 摘要
"""
import argparse
import json
import os
import sys

# --- 路径引导：向上定位 MediTriage 根（含 agent/meditriage/paths.py），从任意目录可运行 ---
import sys as _sys
from pathlib import Path as _Path
_ASK = next(
    p for p in _Path(__file__).resolve().parents
    if (p / 'agent' / 'meditriage' / 'paths.py').is_file()
)
for _p in (str(_ASK / 'agent'), str(_ASK / 'diag')):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import meditriage.paths as _paths
# --- end 引导 ---
from judge_lib import (  # noqa: E402
    cohen_kappa,
    judge_choose_letter,
    judge_deepseek,
    judge_gemini,
    wilson_ci,
)

CALIB_PATH = str(_paths.DATA_DIR / "benchmark/medical_eval_v1/calib_mcq.jsonl")
OUT_PATH = str(_paths.LOG_DIR / "benchmark/judge_calibration.json")


def load_calib(path, n=None):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if n is not None and len(rows) >= n:
                break
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None,
                    help="限量（默认全量）；小样可省 API")
    args = ap.parse_args()

    rows = load_calib(CALIB_PATH, args.n)
    n = len(rows)
    print(f"=== judge 校准：{n} 条 MCQ（{CALIB_PATH}）===", flush=True)

    ds_correct = gm_correct = 0
    ds_labels, gm_labels = [], []

    for i, r in enumerate(rows, 1):
        q, opts, gold = (
            r["question"], r["options"], str(r["answer"]).strip().upper())
        ds = judge_choose_letter(judge_deepseek, q, opts)
        gm = judge_choose_letter(judge_gemini, q, opts)
        ds_labels.append(ds)
        gm_labels.append(gm)
        ds_ok = ds == gold
        gm_ok = gm == gold
        ds_correct += ds_ok
        gm_correct += gm_ok
        print(f"[{i}/{n}] {r['id']} gold={gold} | DS={ds}({'✓' if ds_ok else '✗'}) "
              f"GM={gm}({'✓' if gm_ok else '✗'})", flush=True)

    ds_acc = ds_correct / n if n else 0.0
    gm_acc = gm_correct / n if n else 0.0
    ds_ci = wilson_ci(ds_correct, n)
    gm_ci = wilson_ci(gm_correct, n)
    # κ 仅在两边都给出有效字母的样本上算（剔除空选）
    paired = [(a, b) for a, b in zip(ds_labels, gm_labels) if a and b]
    kappa = cohen_kappa([a for a, _ in paired], [b for _, b in paired])

    result = {
        "deepseek": {"acc": round(ds_acc, 4),
                     "ci": [round(ds_ci[0], 4), round(ds_ci[1], 4)], "n": n},
        "gemini": {"acc": round(gm_acc, 4),
                   "ci": [round(gm_ci[0], 4), round(gm_ci[1], 4)], "n": n},
        "cohen_kappa": round(kappa, 4),
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print("\n=== 摘要 ===")
    print(f"DeepSeek 准确率: {ds_acc:.3f}  Wilson95%CI=[{ds_ci[0]:.3f},{ds_ci[1]:.3f}]  (n={n})")
    print(f"Gemini   准确率: {gm_acc:.3f}  Wilson95%CI=[{gm_ci[0]:.3f},{gm_ci[1]:.3f}]  (n={n})")
    print(f"两 judge Cohen's κ: {kappa:.3f}  (有效配对 {len(paired)}/{n})")
    print(f"已写出 {OUT_PATH}")


if __name__ == "__main__":
    main()
