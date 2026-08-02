"""上下文边界 + 长程检索探针（needle-in-haystack）。

在超长 prompt 开头埋一条无语义密语加代码，塞入大量无关填充，末尾提问。
逐级加长，测：① 模型实际能吃下的最大上下文（prompt 超过 max-model-len
会干净报错）；② 长上下文里能否把开头的针召回（真正的长程注意力，而非
短期记忆）。调整上下文窗口配置前后可各跑一次，对比边界变化。

用法: docker exec medix-fix python3 /workspace/MediTriage/diag/stress/ctx_probe.py
输出: /workspace/log/ctx/results.jsonl + stdout
"""
import asyncio, sys, time, json, os
# 路径引导：向上定位 MediTriage 根（含 agent/meditriage/paths.py），从任意目录可运行。
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
# end 引导
from meditriage.core.llm_client import LLMClient

OUT = str(_paths.LOG_DIR / "ctx")
os.makedirs(OUT, exist_ok=True)
# ~33 字
FILLER = "这是一段用于填充上下文的无关医学背景说明，本句不含任何关键信息，请忽略。"
# 目标 token 量级
TARGETS = [2000, 8000, 16000, 30000, 50000, 100000, 200000, 250000]
NEEDLE = "夜莺在第七层灯塔下唱歌"   # 无语义针，杜绝靠常识猜
CODE = "X7Q9"


async def probe(tok_target, client):
    head = f"请牢记这条密语：『{NEEDLE}』，它对应的代码是 {CODE}。下面是大量无关内容，无需理会。\n\n"
    tail = "\n\n【问题】上文最开头让你牢记的密语是什么？对应代码是什么？只回答密语和代码。"
    n = max(1, int(tok_target * 1.4 / len(FILLER)))   # 中文粗估 ~1.4 char/token
    prompt = head + (FILLER * n) + tail
    approx_tok = int(len(prompt) / 1.4)
    t = time.time()
    try:
        a = await client.chat(
            [{"role": "user", "content": prompt}],
            max_tokens=128,
            temperature=0,
        )
        dt = time.time() - t
        recall = (NEEDLE in a) and (CODE in a)
        return {"target": tok_target, "chars": len(prompt),
                "approx_tok": approx_tok, "lat": round(dt, 1),
                "recall": recall, "ans": (a or "")[:90]}
    except Exception as e:
        return {"target": tok_target, "chars": len(prompt),
                "approx_tok": approx_tok, "lat": round(time.time() - t, 1),
                "err": str(e)[:200]}


async def main():
    c = LLMClient()
    res = open(f"{OUT}/results.jsonl", "w", encoding="utf-8")
    print("=== 上下文边界 + needle-in-haystack 探针 ===", flush=True)
    maxok = 0
    for tt in TARGETS:
        r = await probe(tt, c)
        res.write(json.dumps(r, ensure_ascii=False) + "\n")
        res.flush()
        if "err" in r:
            print(
                f"target~{tt:>7}tok  approx={r['approx_tok']:>7}  "
                f"❌ {r['err'][:90]}",
                flush=True,
            )
        else:
            if not r["err"] if "err" in r else True:
                maxok = max(maxok, r["approx_tok"])
            print(
                f"target~{tt:>7}tok  approx={r['approx_tok']:>7}  "
                f"recall={'✅' if r['recall'] else '❌'}  lat={r['lat']}s",
                flush=True,
            )
    print(f">>> 最大成功处理 ≈ {maxok} tok（近似）", flush=True)

asyncio.run(main())
