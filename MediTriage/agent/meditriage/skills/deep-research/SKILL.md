---
name: deep-research
description: Conduct deep research over a local medical knowledge base (Milvus RAG) and PubMed literature with evidence synthesis. Use for complex medical questions requiring literature review.
---

# Deep Research (深度研究)

综合本地医学知识库（Milvus RAG）与 PubMed 文献的深度研究能力。

## When to Use

- 复杂的医学问题，需要多源信息综合
- 需要最新研究进展和文献综述
- 需要高置信度的证据支持

## 底层实现

- 工作流: `DeepResearchWorkflow`
- 数据源: Milvus 向量数据库 + PubMed 文献 + 证据综合
- 技术: 并行搜索和检索 + LLM 证据综合

## 调用方式

```bash
/deep-research 糖尿病的最新治疗方法
```
