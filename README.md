# MediTriage — 医疗多智能体问答系统

一个端到端的医疗多模态智能体系统：以 **MediX-R1-8B**（Qwen3-VL backbone，视觉+语言）为推理底座，叠加**多 Agent 协作**、**RAG 指南证据增强**、**短期/长期记忆**、**医学影像问答**与 **Web 可视化**。Agent 与推理底座只经 vLLM `:8000`（OpenAI 兼容 HTTP）解耦，换底座重起 vLLM 即可、应用侧零改动。

> 研究/演示用途，**不能替代专业医生诊断**。

## 架构总览

```
                          ┌─────────────────────────────────────────┐
   浏览器 ── SSH隧道 ──►   │ Web (FastAPI + SSE)  MediTriage/agent/web │
                          └───────────────┬─────────────────────────┘
                                          │  文本 → Swarm 路由        图像 → 单轮 VLM
                          ┌───────────────▼─────────────────────────┐
                          │ LangGraph 编排：LeadAgent 分解 → 并行扇出  │
                          │ consultation/diagnostic/research worker   │
                          │（9 skills；ReAct 循环同为 StateGraph 图）  │
                          │ + 安全层(约束校验/自动修复) + 接地/身份锁定 │
                          └──────┬───────────────┬──────────────┬────┘
                          RAG    │       记忆     │     推理      │
                  ┌──────────────▼──┐  ┌──────────▼───┐  ┌───────▼────────┐
                  │ Milvus + BGE-M3 │  │ 短期(会话)    │  │ vLLM :8000      │
                  │ (medical-milvus)│  │ 长期(Milvus)  │  │ MediX-R1-8B 256K│
                  └─────────────────┘  └──────────────┘  └────────────────┘
```

## 目录结构

| 路径 | 说明 |
|---|---|
| **`MediTriage/agent/`** | 可安装包 `meditriage`：`core`（Agent Loop / LLM 客户端 / 视觉链路）、`agents`、`swarm`（LangGraph 编排）、`knowledge`（混合检索 RAG）、`memory`（短/长期）、`research`、`guardrails`（约束校验 + 自动修复）+ 9 个医疗 skills；外层 `web/`（FastAPI + SSE）与 `tests/`。详见 [agent README](MediTriage/agent/README.md) |
| **`MediTriage/serving/`** | 起 vLLM 推理底座（OpenAI 兼容 `:8000`） |
| **`MediTriage/infra/`** | 起 Milvus / Web 演示 / 公网隧道的脚本 |
| **`MediTriage/diag/`** | 工程诊断脚本：容量压测、上下文边界、对抗、检索调优 |
| **`MediTriage/models/`** | 模型权重（不入库，需自行下载，见「前置准备」） |
| **`MediTriage/data/`** | RAG 语料 manifest 与构建脚本、ICD/lifestyle 表、评测集脚本（大文件不入库） |
| **`config.py`** | LLM 接入契约（推理 + 评审端点）；密钥从 `~/.config/` 读，不入库 |
| **`docker/`** | Dockerfile（运行环境） |

> 资产路径由 `MediTriage/agent/meditriage/paths.py` 统一锚定（`MODELS_DIR / DATA_DIR / LOG_DIR / MILVUS_URI`，均可经环境变量覆盖），可在任意部署路径运行。

## 前置准备

- **模型权重**（放到 `MediTriage/models/`，均从 HuggingFace 下载）：
  - `MediX-R1-8B` — 推理底座（MBZUAI 开源医疗多模态 VLM）
  - `bge-m3` — 多语言嵌入；`bge-reranker-v2-m3` — 交叉编码器重排
- **运行环境**：`docker build -f docker/Dockerfile -t meditriage .` 构建镜像，挂载本目录后在容器内起服务（依赖 NVIDIA GPU）。
- **密钥**（可选，仅评测/质量打分用）：`~/.config/` 下放 `gemini_api_key` / `deepseek_api_key`，或经同名环境变量注入。

## 快速上手

```bash
# 1) 起向量库（Milvus Standalone）
bash MediTriage/infra/start_medical_milvus.sh

# 2) 起推理服务（vLLM serve MediX-R1-8B，256K 上下文）
bash MediTriage/serving/serve_vllm_official.sh

# 3) 装 Agent 包并起 Web 演示
cd MediTriage/agent && pip install -e . && cd -
bash MediTriage/infra/start_web_demo.sh
#   浏览器经 SSH 隧道访问：ssh -L 8080:127.0.0.1:8080 <server> → http://localhost:8080

# 测试（纯单元回归不依赖服务；集成/冒烟需上述服务已起）
cd MediTriage/agent && python -m pytest -ra

# 工程诊断（容量/上下文边界/对抗/检索质量）
python3 MediTriage/diag/stress/ctx_probe.py
```

## 能力边界（当前）

- 文本问诊（单 Agent / 多 Agent Swarm 协作，LangGraph 编排）、RAG 指南证据、ICD/风险/症状等 9 skills
- 医学影像问答（VLM 视觉链路，Web 支持上传/拖拽/粘贴）
- 短期（会话）+ 长期（跨会话 Milvus 语义）记忆
- 提示注入 / 越界防御、接地防幻觉、安全层（免责/高危就医提示）
- 上下文窗口 256K（needle-in-haystack 250K 召回验证）
- 底座可替换：Agent 仅经 vLLM OpenAI 兼容 HTTP 与模型耦合，换底座重起 vLLM 即可，Agent 侧零改动

---

## 轻量运行方案

默认全量方案（本地 vLLM + MediX-R1-8B）保持不变。若机器资源有限，
可用以下两种方式运行——**Agent / RAG / 记忆 / Web 业务代码零改动**，
因为推理底座与业务完全解耦（仅经 OpenAI 兼容 HTTP 契约，见 `config.py`）。

### 方案 A：云端 API 推理（普通电脑即可）

1. 启动 Milvus 向量库（Docker，纯 CPU 即可）：
   ```bash
   bash MediTriage/infra/start_medical_milvus.sh
   ```
   本地 Docker 部署时，向量库地址需指向本机：
   ```bash
   export MILVUS_URI=http://127.0.0.1:19530
   ```
2. 选择推理底座并设置环境变量（完整清单见 `config.py` 顶部注释）：

   **① 文本问诊（DeepSeek）—— 图像问答自动降级为提示信息：**
   ```bash
   export MEDITRIAGE_LLM_BASE_URL=https://api.deepseek.com
   export MEDITRIAGE_LLM_API_KEY=sk-你的key
   export MEDITRIAGE_LLM_MODEL=deepseek-chat
   export MEDITRIAGE_LLM_MULTIMODAL=0
   ```

   **② 保留影像问答（阿里云百炼 Qwen-VL，OpenAI 兼容）：**
   ```bash
   export MEDITRIAGE_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
   export MEDITRIAGE_LLM_API_KEY=sk-你的key
   export MEDITRIAGE_LLM_MODEL=qwen-vl-max
   export MEDITRIAGE_LLM_MULTIMODAL=1
   ```
3. 启动 Web（无需 medix-fix 容器，直接跑）：
   ```bash
   cd MediTriage/agent
   pip install -r requirements.txt
   pip install -e .
   python web/server.py        # 浏览器打开 http://127.0.0.1:8080
   ```

### 方案 B：低显存 GPU（16~24GB 消费级显卡）

新增脚本 `MediTriage/serving/serve_vllm_low_vram.sh`（与全量脚本同一契约）：
- 默认把上下文窗口从 256K 缩到 32K（显存大头是 KV cache）
- 可选量化（AWQ/FP8，需先准备对应权重）与 CPU offload

```bash
# 默认：32K 上下文
bash MediTriage/serving/serve_vllm_low_vram.sh

# 更小显存：8K 上下文 / 指定 GPU
GPU=0 MAX_MODEL_LEN=8192 bash MediTriage/serving/serve_vllm_low_vram.sh

# 量化权重 + CPU offload 组合
MODEL_PATH=models/MediX-R1-8B-AWQ QUANTIZATION=awq bash MediTriage/serving/serve_vllm_low_vram.sh
CPU_OFFLOAD_GB=8 MAX_MODEL_LEN=16384 bash MediTriage/serving/serve_vllm_low_vram.sh
```

### 验证（不需要 GPU，也不需要任何服务）

```bash
cd MediTriage/agent
python -m pytest tests/test_regression_*.py tests/test_benchmark.py -q
```
### 方案 A2：阿里云百炼一站式在线（嵌入 + 重排 + 对话）

不想下载任何本地模型？把嵌入 / 重排也切到百炼在线 API，一个 API Key 全搞定：

```bash
# 1) 百炼 Key（https://bailian.console.aliyun.com 开通后获取）
export MEDITRIAGE_DASHSCOPE_API_KEY=sk-你的key

# 2) 嵌入 + 重排走在线 API（本地零模型下载）
export MEDITRIAGE_EMBED_PROVIDER=dashscope      # text-embedding-v4，默认 1024 维
export MEDITRIAGE_RERANK_PROVIDER=dashscope     # gte-rerank-v2

# 3) 对话也走百炼（qwen）或继续用 DeepSeek：
export MEDITRIAGE_LLM_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
export MEDITRIAGE_LLM_API_KEY=sk-你的key
export MEDITRIAGE_LLM_MODEL=qwen-plus
export MEDITRIAGE_LLM_MULTIMODAL=0

# 4) 灌库 + 启动（本地只跑 Milvus + Web）
cd MediTriage/agent && pip install -r requirements.txt && pip install -e .
python ../data/rag_corpus/build_rag_index_v2.py    # 向量化知识库
python web/server.py                                # http://127.0.0.1:8080
```

> 嵌入/重排接口改动在 `MediTriage/agent/meditriage/knowledge/langchain_rag.py`：
> 新增 `_ApiEmbeddings`（OpenAI 兼容 /embeddings）与 `_ApiReranker`（gte-rerank-v2 原生接口），
> 不设置 provider 时仍走本地 sentence-transformers，行为不变。