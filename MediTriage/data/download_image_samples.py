"""抽取医学图像 VQA 样本到 data/med_image_samples/。

数据集：flaviagiammarino/vqa-rad（放射影像 VQA）
输出：图片 PNG + sample_manifest.jsonl
"""
import json
from pathlib import Path

from datasets import load_dataset

# --- 路径引导：向上定位 MediTriage 根（含 agent/meditriage/paths.py），从任意目录可运行 ---
import sys as _sys
from pathlib import Path as _Path
_ASK = next(
    p for p in _Path(__file__).resolve().parents
    if (p / 'agent' / 'meditriage' / 'paths.py').is_file()
)
for _p in (str(_ASK / 'agent'), str(_Path(__file__).resolve().parent)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import meditriage.paths as _paths
# --- end 引导 ---

OUT_DIR = (_paths.DATA_DIR / "med_image_samples")
VQA_RAD_DIR = OUT_DIR / "vqa_rad"
VQA_RAD_DIR.mkdir(parents=True, exist_ok=True)

N = 50  # 抽样数量

print("Loading flaviagiammarino/vqa-rad ...")
ds = load_dataset("flaviagiammarino/vqa-rad", split="test")
print(f"Dataset loaded: {len(ds)} examples, taking first {N}")

manifest = []
for i in range(min(N, len(ds))):
    ex = ds[i]
    img = ex["image"]  # PIL Image
    img_path = VQA_RAD_DIR / f"vqa_rad_{i:03d}.png"
    img.save(img_path)
    manifest.append({
        "image_path": str(img_path),
        "question": ex["question"],
        "answer": ex["answer"],
        "dataset": "vqa-rad",
        "license": "CC0-1.0 (flaviagiammarino/vqa-rad)",
    })

manifest_path = OUT_DIR / "sample_manifest.jsonl"
with open(manifest_path, "w", encoding="utf-8") as f:
    for m in manifest:
        f.write(json.dumps(m, ensure_ascii=False) + "\n")

print(f"DONE: {len(manifest)} samples saved to {VQA_RAD_DIR}")
print(f"Manifest: {manifest_path}")
print("Example:", json.dumps(manifest[0], ensure_ascii=False)[:200])
