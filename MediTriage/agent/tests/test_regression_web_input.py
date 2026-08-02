"""回归：Web 入口输入校验与落盘路径安全（纯单元，不起服务）。

守护：
① 超长 question / 非法 session_id / 非 data:image 的 image 在请求模型层
   即被拒（不进推理链路，封 SSRF 与路径遍历入口）；
② session_summary 落盘路径对 session_id 防御性过滤（纵深第二层）。
"""
import pytest
from pydantic import ValidationError


def _ask_request():
    import sys
    from pathlib import Path
    web_dir = Path(__file__).resolve().parents[1] / "web"
    sys.path.insert(0, str(web_dir))
    try:
        from server import AskRequest
        return AskRequest
    finally:
        sys.path.remove(str(web_dir))


def test_valid_request_accepted():
    AskRequest = _ask_request()
    r = AskRequest(question="头疼怎么办", session_id="web-123456-abcdef")
    assert r.question


def test_overlong_question_rejected():
    AskRequest = _ask_request()
    with pytest.raises(ValidationError):
        AskRequest(question="头疼" * 5000)


def test_path_traversal_session_id_rejected():
    AskRequest = _ask_request()
    with pytest.raises(ValidationError):
        AskRequest(question="q", session_id="x-../../tmp/evil")


def test_non_data_uri_image_rejected():
    AskRequest = _ask_request()
    for bad in ["http://169.254.169.254/meta", "file:///etc/passwd"]:
        with pytest.raises(ValidationError):
            AskRequest(question="看图", image=bad)
    # 正常 data URI 通过
    ok = AskRequest(question="看图", image="data:image/png;base64,iVBOR")
    assert ok.image.startswith("data:image/")


def test_summary_path_sanitized(tmp_path):
    from meditriage.memory.session_summary import SessionSummaryManager
    m = SessionSummaryManager(base_dir=str(tmp_path / "summaries"))
    p = m._get_summary_path("x-../../escape")
    assert str(tmp_path) in str(p.resolve())
    assert ".." not in p.name
