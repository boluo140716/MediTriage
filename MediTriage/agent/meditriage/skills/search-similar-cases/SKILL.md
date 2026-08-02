---
name: search-similar-cases
description: Search similar historical cases from long-term memory (本地 Milvus + BGE-M3). Use when user asks "有类似的案例吗", "之前有人问过", "相关病例", or when context from past sessions would be helpful.
---

# Search Similar Cases (搜索相似案例)

搜索长期记忆（本地 Milvus + BGE-M3）中的相似历史案例。

## When to Use

- 用户问"有类似的案例吗""之前有人问过这个问题吗"
- 需要参考历史病例或类似问题的处理经验
- 跨会话的知识检索

## 调用方式

```bash
/search-similar-cases 高血压患者的饮食建议
```

## 返回格式

```json
{
  "answer": "格式化的相似案例",
  "total_found": 3,
  "query": "高血压患者的饮食建议"
}
```

## Note

This skill searches **across all historical sessions** using vector similarity (本地 Milvus + BGE-M3). For current session history, use `/search-history`.
