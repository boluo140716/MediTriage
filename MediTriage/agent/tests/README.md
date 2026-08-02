# tests/

MediTriage/agent 的测试，分两层：

- **纯单元回归**（`test_regression_*.py` + `test_benchmark.py`）：不依赖任何服务，开箱即跑。
  覆盖分块与表格抽取、引用与证据面板、护栏与危机干预、检索融合/打分/改写、记忆注入防御、
  用户隔离、Web 输入清洗等回归点；`test_benchmark.py` 覆盖测试集构建器的解析逻辑。
- **集成 / 冒烟**（下表）：会**真实调用**本地 vLLM(`:8000`) 与 Milvus(`medical-milvus:19530`)，
  需在容器 `medix-fix` 内、相关服务已启动时运行。

| 用例 | 覆盖 |
|---|---|
| `test_llm_smoke.py` | LLMClient → 本地 vLLM 基础对话 |
| `test_rag_quality.py` | 知识库检索（Milvus + BGE-M3）召回质量 |
| `test_swarm_smoke.py` | `process_with_swarm` 路由 + 多 Agent 协作（默认 LangGraph 引擎） |
| `test_vision_smoke.py` | 医学图像问答（VLM 视觉链路） |
| `test_e2e.py` | SwarmCoordinator（legacy 引擎）端到端 + 多轮短期记忆 |
| `test_memory_governance.py` | 长期记忆治理（Milvus 命名空间隔离，独立 test user_id 不污染生产记忆） |

## 运行

```bash
# 全部（pytest）
docker exec medix-fix bash -c "cd /workspace/MediTriage/agent && python -m pytest -ra"

# 仅纯单元回归（无需起服务）
docker exec medix-fix bash -c "cd /workspace/MediTriage/agent && python -m pytest tests/test_regression_*.py tests/test_benchmark.py -q"

# 单个（冒烟用例也支持直接 python 跑）
docker exec medix-fix bash -c "cd /workspace/MediTriage/agent && python tests/test_llm_smoke.py"
```

> 更重型的容量/上下文/对抗压测等工程诊断脚本在 `MediTriage/diag/`（独立运行，非 pytest 用例）。
