---
name: disease-code
description: 查询疾病的 ICD-10 编码与分类。基于本地 ICD-10 中文码表（1586 条三位类目码）确定性查表，支持标准病名与常见口语（如冠心病、乙肝、慢阻肺）；命中多条并列返回，无匹配时明确告知未收录、绝不臆造编码。Use when the user asks for a disease's ICD-10 code or classification.
---

# Disease Code (疾病编码)

查询疾病的 ICD-10 编码与所属章节。

- 数据：本地 ICD-10 中文三位类目码表（1586 条）+ 章节目录，见 `MediTriage/data/icd10/`（来源见该目录 SOURCE.md）。
- 匹配：口语归一 → 精确 → 双向子串 → 字符模糊；命中多条并列返回。
- 无把握时返回“未收录”，不返回不相关的最近邻。
- 确定性查表，不依赖向量库 / LLM，离线可用。
