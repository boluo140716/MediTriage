# 生活方式处方表数据来源

`lifestyle_table.json` 供 `recommend-lifestyle` skill 做确定性查表。每条建议忠实于
以下来源整理，未手编：

- 高血压 / 2型糖尿病 / 感冒 / 通用健康：本地整理的中文生活方式资料
  （`data/rag_corpus/local_zh/0X_lifestyle_*.txt`，依据中国防治指南与
  《中国居民膳食指南》结构化）。
- 慢性阻塞性肺病：GOLD（全球慢性阻塞性肺病倡议）/ MedlinePlus。
- 慢性肾病：KDIGO / NKF KDOQI / ADA Standards of Care。
- 血脂异常与动脉粥样硬化性心血管病：ACC/AHA 血脂管理指南 / ADA Standards of Care 10.14。

指南类条目为对相应指南生活方式建议的忠实摘要，仅供参考，不替代专业诊疗。
