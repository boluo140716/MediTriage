"""回归：长期记忆按登录身份隔离（纯单元，不依赖 Milvus）。

守护：检索与写入都用本请求的 user_id（Cloudflare Access 注入的邮箱），
无身份时退回单租户默认值，不再让所有人共享一个 user_id。
"""
from meditriage.memory.long_term import LongTermMemory, DEFAULT_USER_ID


class _SpyMem:
    def __init__(self):
        self.calls = []

    def search_memory(self, user_id, query, top_k):
        self.calls.append(user_id)
        return []

    def add_memory(self, user_id, content, mtype, session_id, source):
        self.calls.append(user_id)
        return True


def _ltm(spy):
    m = object.__new__(LongTermMemory)
    m.enabled = True
    m.user_id = DEFAULT_USER_ID
    m._memory = spy
    return m


def test_search_uses_request_user_id():
    spy = _SpyMem()
    _ltm(spy).search_similar_sessions("q", limit=3, user_id="alice@example.com")
    assert spy.calls == ["alice@example.com"]


def test_search_falls_back_to_default_when_no_identity():
    spy = _SpyMem()
    _ltm(spy).search_similar_sessions("q", limit=3, user_id=None)
    assert spy.calls == [DEFAULT_USER_ID]


def test_persist_uses_request_user_id():
    import meditriage.swarm.swarm_coordinator as sc

    captured = {}

    class _Mem:
        def add_memory(self, user_id, content, mtype, session_id, source):
            captured["user_id"] = user_id
            return True

    # 打桩 MedicalMemory，验证写入用的是传入的 user_id
    import meditriage.memory.medical_memory as mm
    orig = mm.MedicalMemory
    mm.MedicalMemory = lambda *a, **k: _Mem()
    try:
        sc._persist_long_term(
            "问题", "一个足够长的有效回答内容，便于通过可持久化判定。" * 3,
            "sess-1", "mode=test", {"answer": "x"},
            user_id="bob@example.com",
        )
    finally:
        mm.MedicalMemory = orig
    assert captured.get("user_id") == "bob@example.com"
