---
name: recommend-lifestyle
description: 给出疾病的生活方式建议（饮食/运动/作息/戒烟限酒/监测）。基于本地确定性处方表（高血压、2型糖尿病、血脂异常/冠心病、慢阻肺、慢性肾病、感冒、通用健康），支持常见别名；未收录则明确告知，不返回无关泛化文案。Use when the user asks for lifestyle/diet/exercise advice for a condition.
---

# Recommend Lifestyle (生活方式建议)

按疾病给出结构化生活方式建议（饮食、运动、作息、戒烟限酒、监测）。

## When to Use

- 用户问“高血压患者饮食注意什么”“糖尿病如何运动”
- 需要某疾病的生活方式调整建议

## 底层实现

- 数据：本地确定性处方表 `MediTriage/data/lifestyle/lifestyle_table.json`（来源见该目录 SOURCE.md）。
- 覆盖：高血压、2型糖尿病、血脂异常/冠心病、慢性阻塞性肺病、慢性肾病、感冒、通用健康。
- 匹配：病名归一（标准名 + 别名）→ 精确 → 子串；未收录明确告知，不返回无关泛化文案。
- 确定性查表，不依赖向量库 / LLM，离线可用。

## 调用方式

```bash
/recommend-lifestyle 高血压
```
