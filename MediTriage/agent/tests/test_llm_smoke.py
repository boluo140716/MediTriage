"""最小连通测试：config.py -> llm_client -> 本地 vLLM"""
import asyncio


from meditriage.core.llm_client import LLMClient


async def main():
    c = LLMClient()
    print("base_url:", c.config["base_url"])
    print("model:", c.model_name)
    r = await c.chat(
        [
            {
                "role": "user",
                "content": "What is the primary cause of a common cold? "
                "Answer in one sentence.",
            }
        ],
        max_tokens=120,
    )
    print("LLM RESPONSE:", r[:300])


if __name__ == "__main__":
    asyncio.run(main())


def test_llm_smoke():
    """pytest smoke：整条链路跑通不抛即通过（需本地 vLLM/Milvus 已起）。"""
    asyncio.run(main())
