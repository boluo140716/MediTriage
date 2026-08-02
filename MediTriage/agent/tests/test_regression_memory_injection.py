"""回归：多轮记忆注入卫生（纯单元，不依赖服务）。

守护三件事（跨话题串扰的结构性根源）：
① format_user_input 不再把 recent_history 渲染进 user 消息——同一段
   历史已由 AgentLoop 消息级注入，双通道会随轮次递归膨胀；
② historical_cases 以"仅供参考、无关请忽略"的可读列表呈现，
   不再以 dict repr 混进背景信息；
③ 长期记忆检索排除当前会话自己的记忆。
"""
from meditriage.agents.consultation_agent import ConsultationAgent
from meditriage.memory.long_term import LongTermMemory


def _fmt(input_data):
    # format_user_input 不依赖 self，绕过重型构造直接调用
    return ConsultationAgent.format_user_input(object(), input_data)


def test_recent_history_not_rendered_into_user_message():
    out = _fmt({
        "question": "我最近肝区不舒服",
        "context": {
            "recent_history": [
                {"role": "user", "content": "之前的妇科问题描述"},
                {"role": "assistant", "content": "妇科相关回答"},
            ],
        },
    })
    assert "妇科" not in out
    assert "recent_history" not in out
    assert "我最近肝区不舒服" in out


def test_historical_cases_rendered_readably_with_ignore_note():
    out = _fmt({
        "question": "肝功能异常怎么办",
        "context": {
            "historical_cases": [
                {"summary": "问题：高血压饮食。回答：低盐饮食…", "score": 0.7},
            ],
        },
    })
    assert "相似历史案例" in out and "请忽略" in out
    assert "高血压饮食" in out
    assert "{'summary'" not in out  # 不再是 dict repr


def test_long_term_excludes_current_session():
    ltm = object.__new__(LongTermMemory)
    ltm.enabled = True
    ltm.user_id = "u"

    class _FakeMem:
        def search_memory(self, user_id, query, top_k):
            return [
                {"id": "1", "content": "本会话刚说过的话", "score": 0.95,
                 "metadata": {"session_id": "s1"}},
                {"id": "2", "content": "上周的相似案例", "score": 0.80,
                 "metadata": {"session_id": "s2"}},
            ]

    ltm._memory = _FakeMem()
    out = ltm.search_similar_sessions("q", limit=2, exclude_session="s1")
    assert [o["memory_id"] for o in out] == ["2"]
