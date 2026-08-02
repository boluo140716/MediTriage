"""回归：证据接地 prompt 规则（结构性测试，不依赖服务）。

守护两条规则不被后续改动悄悄回退：
① 不再有"否则写'证据等级：检索未提供'"占位符指令；
② 三个 agent + lead 合成 + deep_research 合成器都带"禁泛化权威背书"规则。
测试直接读源码断言：只要规则仍在代码里即通过。
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TEXT_PRODUCERS = [
    ROOT / "meditriage/agents/consultation_agent.py",
    ROOT / "meditriage/agents/diagnostic_agent.py",
    ROOT / "meditriage/agents/research_agent.py",
    ROOT / "meditriage/swarm/lead_agent.py",
    ROOT / "meditriage/research/evidence_synthesizer.py",
]


def _effective_src(f: Path) -> str:
    """返回该文件的有效 prompt 源码。

    共享接地规则定义在 agents/_prompt_blocks.py；若某 agent 经 identity_and_grounding
    组合 prompt，则把共享块源码一并计入——规则仍在该 agent 的有效 prompt 源里。
    """
    src = f.read_text(encoding="utf-8")
    shared = f.parent / "_prompt_blocks.py"
    if "identity_and_grounding" in src and shared.exists():
        src += "\n" + shared.read_text(encoding="utf-8")
    return src


@pytest.mark.parametrize("f", TEXT_PRODUCERS, ids=lambda p: p.name)
def test_no_placeholder_instruction(f):
    """旧"否则写占位符"指令必须清零。"""
    src = _effective_src(f)
    assert '否则写"证据等级：检索未提供"' not in src, (
        f"{f.name} 仍在教模型写占位符"
    )
    assert '否则写"检索未提供"' not in src, f"{f.name} 仍在教模型写占位符"


@pytest.mark.parametrize("f", TEXT_PRODUCERS, ids=lambda p: p.name)
def test_ban_vague_authority_present(f):
    """每个产文本环节都必须带'禁泛化权威背书'规则。"""
    src = _effective_src(f)
    assert "专家共识高度一致" in src, (
        f"{f.name} 缺少'禁专家共识高度一致'背书规则"
    )
    assert "多项文献指出" in src, f"{f.name} 缺少'禁多项文献指出'背书规则"
