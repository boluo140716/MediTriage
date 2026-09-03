# MediTriage — 医疗多智能体问答系统

一个端到端的医疗多模态智能体系统：以 **MediX-R1-8B**（Qwen3-VL backbone，视觉+语言）为推理底座，叠加**多 Agent 协作**,**RAG 指南证据增强**,**短期/长期记忆**,**医学影像问答**,**自动转人工问诊**与**Web 可视化**。Agent 与推理底座只经 vLLM `:8000`（OpenAI 兼容 HTTP）解耦，换底座重起 vLLM 即可、应用侧零改动。

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
## 系统页面
<table>
  <tr>
    <td align="center"><img src="https://github.com/user-attachments/assets/a4302575-f0f9-4ed4-84ee-9f19e2eeb924" width="130" alt="患者端" /></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/6250efe9-cdd3-4a34-aca6-b08161c6b840" width="130" alt="医生端" /></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/e84bdc0a-c1b4-48ac-a906-b102dc9880d2" width="130" alt="链路推理示例" /></td>
    <td align="center"><img src="https://github.com/user-attachments/assets/ced024a1-4e0e-4e3d-9931-bdc2b9f2789e" width="130" alt="链路推理示例" />
</td>

  </tr>
  <tr>
    <td align="center"><b>患者端</b></td>
    <td align="center"><b>医生端</b></td>
    <td align="center"><b>链路推理示例1</b></td>
    <td align="center"><b>链路推理示例2</b></td>
  </tr>
</table>


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

