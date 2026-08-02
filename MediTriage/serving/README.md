# serving/ — Agent 推理底座服务

负责把模型权重起成 Agent 用的 vLLM 服务。

- **`serve_vllm_official.sh`** — vLLM serve 官方 `models/MediX-R1-8B`（Qwen3-VL backbone，VLM），
  OpenAI 兼容 API 于 `:8000`，GPU1，`max-model-len 262144`。**当前 Agent 在用的就是它。**

## 用法（容器内）

```bash
docker exec -d medix-fix bash -c "cd /workspace && bash MediTriage/serving/serve_vllm_official.sh"
```

> Agent 与底座的**唯一耦合**就是这层 `:8000` 的 OpenAI 兼容 HTTP 契约；`config.py` 的 `base_url` 指向它。
> 换模型只需重起本服务，Agent 侧零改动。
