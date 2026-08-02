"""记忆边界 + 容量压测。

A 短期记忆深度：用"非语义关键词"回溯——埋一个词，隔 d-1 轮再问"d 轮前
  那个词是什么"。关键词无语义，长期记忆的语义召回不会帮忙，可干净测出短期
  记忆"记到第几轮"。扫 d=1..20。
B 跨会话长期记忆：多 session 各埋一个过敏事实，新 session 问"我之前对什么
  过敏"，测 agent_memory 跨会话召回（top-k）。
C 容量：补量到 N，逐条记延迟，比较 first100 vs last100 漂移 + agent_memory
  增长 + RSS。

用法: docker exec medix-fix python3 /workspace/MediTriage/diag/stress/stress_test.py [N]
输出: /workspace/log/stress/results.jsonl + stdout 报告
"""
import asyncio, time, json, sys, os
# 路径引导：向上定位 MediTriage 根（含 agent/meditriage/paths.py），从任意目录可运行
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
os.environ.setdefault("MEDIX_MEMORY_USER", "test:diag")  # 隔离：工程诊断脚本不写生产长期记忆
from meditriage.swarm import process_with_swarm
try:
    from pymilvus import MilvusClient
    _mc = MilvusClient(uri="http://medical-milvus:19530")
except Exception:
    _mc = None

N = int(sys.argv[1]) if len(sys.argv) > 1 else 500
OUT = str(_paths.LOG_DIR / "stress")
os.makedirs(OUT, exist_ok=True)
RES = open(f"{OUT}/results.jsonl", "w", encoding="utf-8")

WORDS = ["西瓜", "铁塔", "月亮", "风筝", "钢琴", "骆驼", "灯塔", "枫叶", "贝壳", "陀螺",
         "葡萄", "火车", "蝴蝶", "雪山", "竹子", "海豚", "稻草", "琥珀", "罗盘", "蜂蜜",
         "橡树", "瀑布", "石榴", "号角", "孔雀", "麦穗", "珊瑚", "蜡烛", "藤蔓", "铜鼓",
         "蜡梅", "星图", "陶罐", "苔藓", "鲸鱼", "篝火", "蒲公英", "齿轮", "灯笼", "砚台"]
ALLERGENS = ["青霉素", "头孢", "阿司匹林", "磺胺", "海鲜", "花粉", "芒果", "乳胶", "碘造影剂", "布洛芬",
             "链霉素", "花生", "鸡蛋", "酒精", "尘螨", "猫毛", "奎宁", "对乙酰氨基酚", "牛奶", "坚果"]
MED = ["高血压", "2型糖尿病", "慢阻肺", "冠心病", "哮喘", "痛风", "甲亢", "贫血", "胃溃疡", "偏头痛",
       "慢性肾病", "房颤", "心力衰竭", "脂肪肝", "过敏性鼻炎", "骨质疏松", "焦虑症", "带状疱疹", "颈椎病", "高血脂"]


def mem_rows():
    if not _mc:
        return -1
    try:
        _mc.flush("agent_memory")
        return _mc.get_collection_stats("agent_memory").get("row_count", -1)
    except Exception:
        return -1


def rss_mb():
    try:
        for l in open("/proc/self/status"):
            if l.startswith("VmRSS"):
                return int(l.split()[1]) // 1024
    except Exception:
        return -1


lat = []
t0 = time.time()
idx = [0]


async def ask(sid, q, kind, meta=None):
    rec = {"i": idx[0], "sid": sid, "kind": kind, "q": q[:42]}
    if meta:
        rec.update(meta)
    ts = time.time()
    a = ""
    try:
        r = await process_with_swarm(question=q, session_id=sid)
        a = ((r or {}).get("answer") or "")
        dt = time.time() - ts
        rec.update({"lat": round(dt, 2), "len": len(a)})
        lat.append(dt)
    except Exception as e:
        rec.update({"lat": round(time.time() - ts, 2), "err": str(e)[:140]})
    idx[0] += 1
    RES.write(json.dumps(rec, ensure_ascii=False) + "\n")
    RES.flush()
    return a


async def main():
    print(f"START N={N} mem_rows0={mem_rows()} rss0={rss_mb()}MB", flush=True)

    # ---------- A 短期记忆深度（关键词回溯扫描）----------
    depth = []
    sid = "mem-depth"
    wi = 0
    for d in [1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 16, 20]:
        W = WORDS[wi % len(WORDS)]
        wi += 1
        await ask(sid, f"请记住一个关键词：「{W}」。", "plant_kw",
                  {"word": W, "target_d": d})
        for _ in range(d - 1):
            fw = WORDS[wi % len(WORDS)]
            wi += 1
            await ask(sid, f"再记一个关键词：「{fw}」。", "filler_kw", {"word": fw})
        a = await ask(
            sid, f"请问 {d} 轮之前我最早让你记的那个关键词是什么？只回答那个词。",
            "probe_kw", {"target_d": d, "expected": W})
        hit = W in a
        depth.append((d, W, hit))
        print(f"[A] 回溯 {d} 轮 期望「{W}」 -> {'✅记得' if hit else '❌忘了'}",
              flush=True)

    # ---------- B 跨会话长期记忆（语义过敏事实）----------
    B = min(20, len(ALLERGENS))
    planted = []
    for s in range(B):
        al = ALLERGENS[s]
        planted.append(al)
        await ask(f"mem-b{s}", f"请记住：我对{al}过敏。", "plant_allergy",
                  {"allergen": al})
    a = await ask(
        "mem-probe",
        "我在之前的咨询里提到过我对哪些药物或东西过敏？请尽量全部列出。",
        "recall_cross", {"planted": B})
    cross_cov = sum(1 for x in planted if x in a)
    a2 = await ask("mem-probe2", f"我之前是不是说过我对{planted[0]}过敏？",
                   "recall_cross_point", {"expected": planted[0]})
    cross_pt = (planted[0] in a2)
    print(
        f"[B] 跨会话：埋 {B} 个过敏事实 -> 一次列出命中 {cross_cov}/{B}；"
        f"单点回忆'{planted[0]}' {'✅' if cross_pt else '❌'}", flush=True)

    # ---------- C 容量补量 ----------
    while idx[0] < N:
        d = MED[idx[0] % len(MED)]
        await ask(f"cap-s{idx[0] // 8}", f"{d}有哪些常见症状和日常注意事项？",
                  "capacity")
        if idx[0] % 40 == 0:
            r50 = lat[-50:] or [0]
            print(
                f"[C {idx[0]}/{N}] t={int(time.time()-t0)}s "
                f"avg50={sum(r50)/len(r50):.1f}s mem_rows={mem_rows()} "
                f"rss={rss_mb()}MB", flush=True)

    # ---------- 报告 ----------
    boundary = max([d for d, w, h in depth if h], default=0)
    ls = sorted(lat)
    n = len(ls) or 1
    pc = lambda q: ls[min(n - 1, int(n * q))] if ls else 0
    f100 = lat[:100] or [0]
    l100 = lat[-100:] or [0]
    print("=" * 58, flush=True)
    print("【A 短期记忆深度】", flush=True)
    for d, w, h in depth:
        print(f"   回溯 {d:>2} 轮 -> {'✅' if h else '❌'}", flush=True)
    print(f"   >>> 短期记忆深度边界 ≈ 最远可回溯 {boundary} 轮", flush=True)
    print(
        f"【B 跨会话长期记忆】埋 {B} 个 -> 一次列出 {cross_cov}/{B}；"
        f"单点 {'✅' if cross_pt else '❌'}", flush=True)
    print(
        f"【C 容量/吞吐】total={idx[0]} time={int(time.time()-t0)}s "
        f"p50={pc(.5):.1f}s p90={pc(.9):.1f}s p99={pc(.99):.1f}s "
        f"max={max(lat or [0]):.1f}s", flush=True)
    print(
        f"   延迟漂移 first100={sum(f100)/len(f100):.1f}s -> "
        f"last100={sum(l100)/len(l100):.1f}s | mem_rows={mem_rows()} "
        f"rss={rss_mb()}MB", flush=True)


asyncio.run(main())
