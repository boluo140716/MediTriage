"""Swarm 端到端 smoke test：路由 + tool calling + RAG。"""
import asyncio
import time


from meditriage.swarm import process_with_swarm


def emit(etype, data):
    # 打印 Agent 内部事件，验证可视化事件流
    tag = data.get("agent_id", "")
    if etype == "tool_call_started":
        print(
            f"    [event] {etype}: {data.get('tool_name')}"
            f"({data.get('arguments')})"
        )
    elif etype == "tool_call_completed":
        print(
            f"    [event] {etype}: {data.get('tool_name')} -> "
            f"{str(data.get('result_preview', ''))[:80]}"
        )
    elif etype in ("lead_routing", "agent_thinking", "final_answer",
                   "session_completed"):
        print(f"    [event] {etype} {tag}")


async def run_one(q):
    print(f"\n{'='*70}\nQ: {q}\n{'='*70}")
    t0 = time.time()
    result = await process_with_swarm(
        q, session_id=f"smoke-{int(t0)}", event_emitter=emit
    )
    dt = time.time() - t0
    print(f"\n  MODE: {'swarm' if result.get('swarm_enabled') else 'single'}")
    print(f"  AGENTS: {result.get('agents_involved', [])}")
    print(f"  TIME: {dt:.1f}s")
    print(f"  ANSWER: {(result.get('answer') or '')[:400]}")
    sugg = result.get("suggestions", [])
    if sugg:
        print(f"  SUGGESTIONS ({len(sugg)}): {sugg[:3]}")


async def main():
    # 简单问题（预期单 Agent 路由）
    await run_one("感冒了应该吃什么药？")


if __name__ == "__main__":
    asyncio.run(main())


def test_swarm_smoke():
    """pytest smoke：整条链路跑通不抛即通过（需本地 vLLM/Milvus 已起）。"""
    asyncio.run(main())
