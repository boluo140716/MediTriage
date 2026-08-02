"""测试集构建器单元测试。

只覆盖纯函数（解析逻辑），不触网/不读大文件，可独立快速跑：
    docker exec medix-fix bash -c \
      "cd /workspace/MediTriage/agent && \
       python -m pytest tests/test_benchmark.py -k 'cmexam or answer_letter' -q"
"""
import sys
from pathlib import Path

# --- 路径引导：向上定位 MediTriage 根（含 agent/meditriage/paths.py），从任意目录可运行 ---
import sys as _sys
from pathlib import Path as _Path
_ASK = next(
    p for p in _Path(__file__).resolve().parents
    if (p / "agent" / "meditriage" / "paths.py").is_file()
)
for _p in (str(_ASK / "agent"), str(_Path(__file__).resolve().parent)):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)
import meditriage.paths as _paths
# --- end 引导 ---

sys.path.insert(0, str(_paths.DATA_DIR / "benchmark"))
from build_eval_set import parse_cmexam_options, parse_mcq_answer_letter


def test_parse_cmexam_options():
    opts = parse_cmexam_options("A．头痛 B．恶心 C．发热 D．咳嗽 E．乏力")
    assert opts["A"].strip() == "头痛" and opts["E"].strip() == "乏力"


def test_parse_mcq_answer_letter():
    assert parse_mcq_answer_letter("答案是 C。") == "C"
    assert parse_mcq_answer_letter("C") == "C"
    assert parse_mcq_answer_letter("选 B 和 D") == "B"


# --- judge_lib 纯函数（不触网，可独立快速跑）---
sys.path.insert(0, str(_ASK / "diag"))
from judge_lib import wilson_ci, extract_score


def test_wilson_ci():
    lo, hi = wilson_ci(80, 100)
    assert 0.70 < lo < 0.72 and 0.86 < hi < 0.88


def test_extract_score():
    assert extract_score('{"score": 4, "reason":"x"}') == 4
    assert extract_score("评分：5/5") == 5


# --- RRF 融合纯函数（不触网，可独立快速跑）---
from meditriage.knowledge.langchain_rag import _rrf_fuse


def test_rrf_fuse():
    dense = [("d1", 0.9), ("d2", 0.8), ("d3", 0.1)]
    sparse = [("d2", 5.0), ("d3", 4.0), ("d1", 0.5)]
    out = _rrf_fuse(dense, sparse, k=60)
    ids = [x[0] for x in out]
    assert ids[0] == "d2"   # 两路都靠前 → 融合最高
    assert set(ids) == {"d1", "d2", "d3"}
