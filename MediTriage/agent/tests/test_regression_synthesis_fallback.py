"""回归：LeadAgent synthesize_results 空返回/异常时的兜底。

守护：
① LLM 正常返回 -> 直接采用；
② LLM 返回空/过短（<20 字符）-> 回退拼接各 Agent 已产出的正文 + 免责声明；
③ LLM 抛异常 -> 同上回退（不把"汇总结果时出错"错误文本直接暴露给用户）；
④ 无任何可用贡献时 -> 返回友好错误提示（不抛异常）。
"""
import asyncio

from meditriage.swarm.lead_agent import LeadAgent
from meditriage.swarm.shared_context import SharedContext, SubTask


class FakeLLM:
    def __init__(self, result=None, exc=None):
        self.result = result
        self.exc = exc

    async def chat(self, messages, **kwargs):
        if self.exc:
            raise self.exc
        return self.result


def _make_sc(answer="可能是腰肌劳损，建议热敷并观察，必要时就医。"):
    sc = SharedContext(session_id="s1")
    st = SubTask(
        id="st-1", type="diagnostic_agent_task",
        description="诊断", assigned_agent="diagnostic_agent",
    )
    sc.add_subtask(st)
    sc.complete_subtask("st-1", "diagnostic_agent", {"answer": answer})
    return sc


async def _run(llm, sc=None):
    lead = LeadAgent(llm_client=llm)
    return await lead.synthesize_results(
        question="我腰疼两天了", shared_context=sc or _make_sc(),
    )


def test_synthesize_uses_llm_when_valid():
    r = asyncio.run(_run(FakeLLM(
        "【风险评估】\n当前风险等级评估为中危，建议尽快就医评估，不要拖延。"
        "\n【核心建议】\n1. 注意休息。\n2. 症状加重及时就诊。"
    )))
    assert "【风险评估】" in r
    assert "腰肌劳损" not in r  # 正常路径不混入 Agent 贡献


def test_synthesize_fallback_on_empty():
    r = asyncio.run(_run(FakeLLM("")))
    assert "腰肌劳损" in r          # 回退拼接了 Agent 贡献正文
    assert "免责声明" in r


def test_synthesize_fallback_on_short():
    r = asyncio.run(_run(FakeLLM("好的")))
    assert "腰肌劳损" in r


def test_synthesize_fallback_on_exception():
    r = asyncio.run(_run(FakeLLM(exc=RuntimeError("api down"))))
    assert "腰肌劳损" in r
    assert "汇总结果时出错" not in r  # 不把内部错误文本暴露给用户


def test_synthesize_no_contributions_fallback():
    # 无任何 Agent 贡献时：synthesize_results 开头即返回既有降级提示（不抛异常）
    sc = SharedContext(session_id="s2")
    r = asyncio.run(_run(FakeLLM(""), sc=sc))
    assert "未能提供有效分析结果" in r


def test_synthesize_truncates_long_contributions():
    """长贡献（>600 字）在汇总/兜底时被截断，控制生成长度。"""
    long_answer = "可能是腰肌劳损。\n" + ("这是很长的细节描述。" * 80)
    sc = _make_sc(answer=long_answer)
    r = asyncio.run(_run(FakeLLM(""), sc=sc))
    assert "截断" in r                    # 兜底文本带截断标记
    assert len(r) < len(long_answer)      # 截断后更短
    assert "腰肌劳损" in r                # 关键结论保留
