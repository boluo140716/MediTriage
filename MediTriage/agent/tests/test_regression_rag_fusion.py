"""回归：BM25 分词与 RRF 加权融合（纯单元，不依赖 Milvus/模型）。

守护三件事：
① 医疗缩写/复合术语经用户词典整体成词（"慢阻肺" 不再被切成 "慢阻"+"肺"，
   "CHA2DS2-VASc" 不再被 "-" 拆碎），英文统一小写、纯标点 token 丢弃——
   否则 BM25 精确术语匹配失效，相关文档难以排进融合候选前列；
② _rrf_fuse 支持 per-ranking 权重，稀疏路降权后稀疏命中的融合贡献按比例下降
   （实测本语料 BM25 等权会注入相邻疾病噪声拉低精排）；
③ 缺省不传 weights 时维持等权，向后兼容旧调用。
"""
from meditriage.knowledge.langchain_rag import _tokenize, _rrf_fuse


def test_tokenize_keeps_medical_abbrev_whole():
    toks = _tokenize("慢阻肺急性加重怎么处理")
    assert "慢阻肺" in toks          # 整体成词，不再是 "慢阻"+"肺"
    assert "慢阻" not in toks


def test_tokenize_compound_term_not_split_by_hyphen():
    toks = _tokenize("CHA2DS2-VASc 评分")
    assert "cha2ds2-vasc" in toks     # 小写化 + 连字符术语整体保留
    assert "-" not in toks            # 纯标点 token 被丢弃


def test_tokenize_lowercases_english_and_drops_punct():
    toks = _tokenize("COPD and GOLD? 治疗；方案")
    assert "copd" in toks and "gold" in toks
    assert "?" not in toks and "；" not in toks
    assert "COPD" not in toks         # 已小写


def test_rrf_weights_downweight_sparse():
    dense = [("a", 0.9), ("b", 0.8)]
    sparse = [("c", 5.0), ("a", 4.0)]   # c 是稀疏独有命中
    # 等权：c 排名应较高
    eq = dict(_rrf_fuse(dense, sparse, k=60))
    # 稀疏降权 0.2：c 的贡献被压低
    wt = dict(_rrf_fuse(dense, sparse, k=60, weights=[1.0, 0.2]))
    assert wt["c"] < eq["c"]            # 稀疏独有项被降权
    assert wt["a"] > wt["c"]            # dense+稀疏共同命中的 a 应高于稀疏独有 c


def test_rrf_default_equal_weight_backward_compatible():
    r1 = [("x", 1.0), ("y", 0.5)]
    r2 = [("y", 1.0), ("z", 0.5)]
    out = dict(_rrf_fuse(r1, r2))       # 不传 weights
    # y 在两路都靠前，融合分应最高
    assert out["y"] == max(out.values())
