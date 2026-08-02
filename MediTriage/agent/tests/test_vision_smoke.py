"""图像 VQA 链路 smoke test：vision_handler -> vLLM Vision -> MediX-R1。"""
import asyncio
import json
import sys
from pathlib import Path

# 路径引导：向上定位 MediTriage 根（含 agent/meditriage/paths.py），从任意目录可运行
import sys as _sys
from pathlib import Path as _Path
_ASK = next(
    p for p in _Path(__file__).resolve().parents
    if (p / "agent" / "meditriage" / "paths.py").is_file()
)
for _p in (str(_ASK / "agent"), str(_Path(__file__).resolve().parent)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import meditriage.paths as _paths

from meditriage.core.vision_handler import process_image_query

MANIFEST = str(_paths.DATA_DIR / "med_image_samples/sample_manifest.jsonl")


async def main():
    samples = [json.loads(l) for l in open(MANIFEST)][:3]
    correct = 0
    for s in samples:
        print("\n" + "=" * 70)
        print("IMG:", s["image_path"].split("/")[-1])
        print("Q:", s["question"])
        print("GROUND TRUTH:", s["answer"])
        r = await process_image_query(s["image_path"], s["question"])
        ans = r["answer"]
        print("MODEL:", ans[:300])
        # 粗略判断：ground truth 关键词是否在模型回答里
        gt = s["answer"].lower().strip()
        if gt in ans.lower() or (
            gt in ("yes", "no") and gt in ans.lower()[:80]
        ):
            correct += 1
            print("  [match]")
    print("\n" + "=" * 70)
    print("rough match: %d/%d" % (correct, len(samples)))


if __name__ == "__main__":
    asyncio.run(main())


def test_vision_smoke():
    """pytest smoke：整条链路跑通不抛即通过（需本地 vLLM/Milvus 已起）。"""
    asyncio.run(main())
