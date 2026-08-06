"""回归：澄清式多轮问诊（信息不足时主动追问 2-3 个问题）。

守护：
① 规则预筛：危机不追问 / 科普咨询不追问 / 维度全齐不追问 / 模糊进追问链路；
② 判定规则化：缺关键维度必追问（不依赖 LLM 判定，行为稳定）；
③ 逐轮追问：一轮只问 1 个问题（最高优先级缺失维度），回答后继续追问；
④ LLM 只生成问题，失败/超时/非法/空 -> 模板兜底（信息不足必追问）；
⑤ 追问轮上限：默认 3 轮，达到上限不再追问；
⑥ 端到端：/api/ask 追问轮 result 带 clarify，且不建转诊单。
"""
import asyncio
import json
import sys
from pathlib import Path

import pytest

from meditriage.swarm.clarification import (
    _CLARIFY_MARKER,
    _fallback_questions,
    _missing_dimensions,
    count_clarify_rounds,
    maybe_clarify,
    rule_precheck,
)


class FakeLLM:
    def __init__(self, text=None, exc=None):
        self.text = text
        self.exc = exc

    async def chat(self, messages, **kwargs):
        if self.exc:
            raise self.exc
        return self.text


class FakeMemory:
    def __init__(self):
        self.msgs = []

    def get_recent_messages(self, session_id, limit=200):
        return self.msgs

    def add_message(self, session_id, role, content):
        self.msgs.append({"role": role, "content": content})


def _run(question="我最近经常头晕", answer="建议规律作息", llm_text=None,
         llm_exc=None, memory=None, max_rounds=3, enabled=True):
    llm = FakeLLM(text=llm_text, exc=llm_exc)
    mem = memory or FakeMemory()
    return asyncio.run(maybe_clarify(
        question=question, session_id="s1", final_answer=answer,
        llm_client=llm, memory=mem, max_rounds=max_rounds, enabled=enabled,
    ))


CLARIFY_JSON = ('{"should_clarify": true, '
                '"questions": ["症状持续多久了？", "有没有伴随胸闷或出冷汗？", '
                '"最近在吃什么药吗？"], "reason": "缺少时长与伴随症状"}')
NO_CLARIFY_JSON = '{"should_clarify": false, "questions": [], "reason": ""}'


# ---- 规则预筛 ----
def test_precheck_crisis_not_clarify():
    # 问题侧（患者自述）危机词 -> 不追问，交转诊拦截
    assert rule_precheck("我胸口疼得厉害，还冒冷汗，怎么办？", "") == "crisis"
    assert rule_precheck("我这两天总是头晕，还晕倒过一次", "") == "crisis"
    # 答案侧科普"拨打120"等危险信号提示不误伤（_CRISIS_PAT 含 "120"；
    # 答案侧风险由转诊评估独立兜底，不参与澄清预筛）
    assert rule_precheck("我最近经常头晕", "请立即急诊或拨打120，不要等待") == "unclear"


def test_precheck_info_query_not_clarify():
    assert rule_precheck("高血压饮食注意什么？", "") == "info_query"
    # 泛指科普（无个人主体、无明确症状词）也直接回答
    assert rule_precheck("2型糖尿病吃什么水果合适？", "") == "info_query"
    assert rule_precheck("什么是高血压？", "") == "info_query"


def test_precheck_sufficient_not_clarify():
    assert rule_precheck("头痛一周了越来越严重，怎么办？", "") == "sufficient"


def test_precheck_unclear_triggers_llm():
    assert rule_precheck("我最近经常头晕", "") == "unclear"


# ---- LLM 判定（逐轮单问）----
def test_clarify_returns_one_question():
    # LLM 正常生成 -> 只返回 1 个问题（逐轮追问，一次只问一个）
    r = _run(llm_text=CLARIFY_JSON)
    assert r is not None
    assert len(r["questions"]) == 1
    assert r["questions"][0] == "症状持续多久了？"


def test_clarify_uses_llm_question_when_valid():
    # LLM 给 1 个有效问题 -> 直接采用
    r = _run(llm_text='{"should_clarify": true, "questions": ["疼痛程度如何？"], "reason": "x"}')
    assert r is not None
    assert r["questions"] == ["疼痛程度如何？"]


def test_clarify_takes_first_question_from_llm():
    # LLM 给多个 -> 只取第一个（一轮只问 1 个）
    r = _run(llm_text=('{"should_clarify": true, "questions": '
                       '["症状持续多久了？", "有没有伴随其他症状？"], "reason": "x"}'))
    assert r is not None
    assert len(r["questions"]) == 1
    assert r["questions"][0] == "症状持续多久了？"


def test_clarify_no_need():
    # 判定是规则驱动：LLM 说 should_clarify=false 不改变结果（信息不足必追问）
    r = _run(llm_text=NO_CLARIFY_JSON)
    assert r is not None
    assert len(r["questions"]) == 1
    # 关键维度已全部补全（时长/程度/伴随/既往）-> 不追问
    r2 = _run(question="头痛一周了越来越严重，还伴随恶心，我以前血压高在吃降压药",
              llm_text=CLARIFY_JSON)
    assert r2 is None


def test_clarify_llm_error_falls_back():
    # LLM 故障 -> 模板兜底 1 个问题，不阻断、不落空
    r = _run(llm_exc=RuntimeError("api down"))
    assert r is not None
    assert len(r["questions"]) == 1


def test_clarify_bad_json_falls_back():
    # LLM 输出非法 -> 模板兜底 1 个问题
    r = _run(llm_text="抱歉，我无法评估。")
    assert r is not None
    assert len(r["questions"]) == 1


def test_clarify_empty_questions_falls_back():
    # LLM 输出空数组 -> 模板兜底
    r = _run(llm_text='{"should_clarify": true, "questions": [], "reason": "x"}')
    assert r is not None
    assert len(r["questions"]) == 1


def test_clarify_asks_highest_priority_missing_dimension():
    # 逐轮追问：按优先级（时长>程度>伴随>既往）问最关键的缺失维度
    r = _run(question="我最近头晕", llm_text=CLARIFY_JSON)
    assert r is not None
    assert r["questions"] == ["症状持续多久了？"]
    assert "症状持续时间" in r["reason"]


def test_clarify_uses_accumulated_history():
    # 逐轮追问：评估基于会话累计信息（历史+当前），避免重复问已回答维度
    mem = FakeMemory()
    mem.add_message("s1", "user", "我这两天总是腰疼")
    r = _run(memory=mem, llm_text=CLARIFY_JSON, question="中度疼，有点影响睡觉")
    assert r is not None
    assert "症状持续时间" not in r["reason"]  # 时长已由历史提供
    assert "严重程度" not in r["reason"]      # 程度已由当前输入提供
    assert "伴随症状" in r["reason"]          # 下一个缺失维度


def test_clarify_skips_dimension_on_noninformative_answer():
    # 用户纯回应（"有点影响日常活动"不含"严重/轻度"等程度词）-> 跳过程度维度，
    # 继续问下一个，不卡死在同一问题
    mem = FakeMemory()
    mem.add_message("s1", "user", "我头痛三天了")  # 时长已答
    mem.add_message("s1", "assistant", f"{_CLARIFY_MARKER}\n症状有多严重？")  # 上轮已追问
    r = _run(memory=mem, llm_text=CLARIFY_JSON, question="有点影响日常活动")
    assert r is not None
    assert "严重程度" not in r["reason"]  # 已跳过程度
    assert "伴随症状" in r["reason"]      # 推进到伴随


def test_clarify_noninformative_single_word_answer():
    # 单字回应"会"同样不卡死
    mem = FakeMemory()
    mem.add_message("s1", "user", "我头痛三天了")
    mem.add_message("s1", "assistant", f"{_CLARIFY_MARKER}\n症状有多严重？")  # 上轮已追问
    r = _run(memory=mem, llm_text=CLARIFY_JSON, question="会")
    assert r is not None
    assert "严重程度" not in r["reason"]
    assert "伴随症状" in r["reason"]


def test_precheck_short_circuits_llm():
    # 危机即使 LLM 说"要追问"也不追问（危机优先，直接转人工）
    r = _run(question="我胸口疼得厉害，还冒冷汗，怎么办？", llm_text=CLARIFY_JSON)
    assert r is None


# ---- 缺失维度 / 兜底 ----
def test_missing_dimensions():
    assert "症状持续时间" not in _missing_dimensions("我这两天总是腰疼")
    assert len(_missing_dimensions("我最近头晕")) >= 3
    assert len(_missing_dimensions("头痛一周越来越严重，还伴随恶心")) < 2


def test_missing_dimensions_negative_answer():
    # 否定式回答（"没有发热麻木"）应视为已澄清伴随维度
    m = _missing_dimensions("中度疼，没有发热也没有麻木")
    assert "伴随症状" not in m
    assert "严重程度" not in m
    assert "既往史/用药" in m
    assert "无伴随症状" not in _missing_dimensions("腰疼两天，无伴随症状")


def test_fallback_questions_cover_missing():
    missing = ["症状持续时间", "严重程度", "伴随症状"]
    qs = _fallback_questions(missing)
    assert len(qs) == 3
    assert all(isinstance(q, str) and len(q) > 5 for q in qs)


# ---- 主动转人工跳过澄清 ----
def test_ask_doctor_skips_clarification():
    mem = FakeMemory()
    r = _run(question="帮我转人工，我要找医生", llm_text=CLARIFY_JSON, memory=mem)
    assert r is None  # 主动要求转人工：不追问


# ---- 追问轮上限 ----
def test_clarify_round_limit():
    mem = FakeMemory()
    mem.add_message("s1", "assistant", f"{_CLARIFY_MARKER}\n问题1")
    mem.add_message("s1", "assistant", f"{_CLARIFY_MARKER}\n问题2")
    assert count_clarify_rounds(mem.msgs) == 2
    r = _run(memory=mem, llm_text=CLARIFY_JSON, max_rounds=2)
    assert r is None  # 已达上限


def test_clarify_default_round_limit_is_three():
    # 默认上限 3 轮：2 条标记后仍可再追问 1 次
    mem = FakeMemory()
    mem.add_message("s1", "assistant", f"{_CLARIFY_MARKER}\n问题1")
    mem.add_message("s1", "assistant", f"{_CLARIFY_MARKER}\n问题2")
    r = _run(memory=mem, llm_text=CLARIFY_JSON)
    assert r is not None
    # 3 条标记 -> 达到上限
    mem.add_message("s1", "assistant", f"{_CLARIFY_MARKER}\n问题3")
    r2 = _run(memory=mem, llm_text=CLARIFY_JSON)
    assert r2 is None


def test_clarify_disabled():
    r = _run(llm_text=CLARIFY_JSON, enabled=False)
    assert r is None


# ---- 端到端：/api/ask 追问轮 ----
@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MEDITRIAGE_ESCALATION_DB", str(tmp_path / "esc.db"))
    monkeypatch.setenv("MEDITRIAGE_ESCALATION_LLM", "0")
    # 重置转诊服务进程单例：避免前面测试建的转诊单残留污染本 e2e
    # （get_escalation_service 是进程级缓存，monkeypatch env 不刷新它）
    import meditriage.workflows.escalation as _esc_mod
    monkeypatch.setattr(_esc_mod, "_service", None)
    web_dir = Path(__file__).resolve().parents[1] / "web"
    if str(web_dir) not in sys.path:
        sys.path.insert(0, str(web_dir))
    from meditriage.workflows import get_escalation_service
    import server

    class _FakeSwarmGate:
        async def clarify_if_needed(self, q, sid, uid=None):
            return None  # 轻量门放行，走 process_with_swarm

    import meditriage.swarm.langgraph_swarm as _lsw
    monkeypatch.setattr(_lsw, "_get_swarm", lambda: _FakeSwarmGate())

    async def fake_swarm(question, session_id, event_emitter, user_id=None):
        event_emitter("session_started", {"session_id": session_id})
        return {
            "answer": "初步分析：头晕可能与多种因素有关。",
            "suggestions": [], "disclaimer": "",
            "swarm_enabled": True, "agents_involved": ["consultation_agent"],
            "session_id": session_id,
            "clarify": {
                "questions": ["症状持续多久了？"],
                "reason": "缺少症状持续时间",
            },
        }

    monkeypatch.setattr(server, "process_with_swarm", fake_swarm)
    from fastapi.testclient import TestClient
    with TestClient(server.app) as c:
        yield c, get_escalation_service()


def test_ask_clarify_carries_questions_and_skips_escalation(client):
    c, svc = client
    r = c.post("/api/ask", json={
        "question": "我最近经常头晕", "session_id": "clarify-e2e",
    })
    assert r.status_code == 200
    result = None
    for frame in r.text.split("\n\n"):
        frame = frame.strip()
        if not frame or "result" not in frame:
            continue
        for line in frame.split("\n"):
            if line.startswith("data: "):
                p = json.loads(line[6:])
                if p.get("type") == "result":
                    result = p["data"]
    assert result is not None
    assert result["clarify"]["questions"]
    assert len(result["clarify"]["questions"]) == 1
    # 追问轮不建转诊单
    assert len(svc.store.list(status=None)) == 0


# ---- 轻量澄清门（clarify_if_needed）----
class _GateFakeLLM:
    async def chat(self, messages, **kwargs):
        return '{"questions": ["这个症状持续多久了？"]}'


class _GateFakeLead:
    llm_client = _GateFakeLLM()


def _make_swarm(monkeypatch):
    from meditriage.swarm.langgraph_swarm import LangGraphSwarm
    monkeypatch.setattr(
        "meditriage.swarm.langgraph_swarm.LeadAgent",
        lambda: _GateFakeLead(),
    )
    return LangGraphSwarm()


def test_gate_returns_question_and_lands_memory(monkeypatch):
    swarm = _make_swarm(monkeypatch)
    clarify = asyncio.run(swarm.clarify_if_needed("我这两天总是腰疼", "g1"))
    assert clarify is not None
    assert len(clarify["questions"]) == 1
    roles = [m["role"] for m in swarm.short_term_memory.get_recent_messages("g1")]
    assert "user" in roles                 # 轻量轮落用户消息
    assert "assistant" in roles            # [CLARIFY-ROUND] 标记


def test_gate_returns_none_when_sufficient(monkeypatch):
    swarm = _make_swarm(monkeypatch)
    q = "头痛一周越来越严重，还伴随恶心，我以前血压高在吃降压药"
    clarify = asyncio.run(swarm.clarify_if_needed(q, "g2"))
    assert clarify is None                 # 信息足够 -> 走完整流程


def test_gate_returns_none_on_crisis(monkeypatch):
    swarm = _make_swarm(monkeypatch)
    clarify = asyncio.run(swarm.clarify_if_needed(
        "我胸口疼得厉害，还冒冷汗", "g3"))
    assert clarify is None                 # 危机 -> 走完整流程（转诊拦截）


def test_gate_returns_none_on_info_query(monkeypatch):
    swarm = _make_swarm(monkeypatch)
    clarify = asyncio.run(swarm.clarify_if_needed("高血压饮食注意什么？", "g4"))
    assert clarify is None                 # 科普 -> 直接完整流程回答
