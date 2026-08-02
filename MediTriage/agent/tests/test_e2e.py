"""端到端验证：复杂问题 Swarm 路由 + 多轮短期记忆。"""
import asyncio
import time


from meditriage.swarm import SwarmCoordinator


def emit(etype, data):
    if etype in ("lead_routing", "tool_call_started", "subtask_started",
                 "subtask_completed"):
        extra = (data.get("tool_name") or data.get("route")
                 or data.get("subtask_type") or "")
        print(f"    [event] {etype} {data.get('agent_id', '')} {extra}")


async def main():
    coord = SwarmCoordinator(enable_swarm=True)

    # --- 验证 1：复杂问题 → Swarm 多 Agent 路由 ---
    print("=" * 70)
    print("[验证1] 复杂问题 Swarm 路由")
    print("=" * 70)
    q1 = "我最近头痛、视力模糊、血压偏高，应该怎么办？"
    print("Q:", q1)
    t0 = time.time()
    r1 = await coord.process(q1, session_id="e2e-complex", event_emitter=emit)
    print(f"  MODE: {'swarm' if r1.get('swarm_enabled') else 'single'}")
    print(f"  AGENTS: {r1.get('agents_involved', [])}")
    print(f"  TIME: {time.time()-t0:.1f}s")
    print(f"  ANSWER: {(r1.get('answer') or '')[:300]}")

    # --- 验证 2：多轮短期记忆 ---
    print("\n" + "=" * 70)
    print("[验证2] 多轮短期记忆（同一 session）")
    print("=" * 70)
    sid = "e2e-memory"
    print("Round1 Q: 我有高血压")
    await coord.process("我有高血压", session_id=sid)
    print("Round2 Q: 那饮食应该注意什么？（不重复提高血压，测试是否记住上下文）")
    r2 = await coord.process("那饮食应该注意什么？", session_id=sid)
    ans2 = (r2.get("answer") or "")
    print(f"  ANSWER: {ans2[:300]}")
    # 判断是否关联到高血压
    hit = any(k in ans2 for k in ["高血压", "血压", "钠", "盐"])
    print(f"  [记忆关联: {'命中高血压上下文' if hit else '未命中'}]")


if __name__ == "__main__":
    asyncio.run(main())


def test_e2e():
    """pytest smoke：整条链路跑通不抛即通过（需本地 vLLM/Milvus 已起）。"""
    asyncio.run(main())
