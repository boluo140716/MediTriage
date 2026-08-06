# MediTriage 医疗多模态智能体系统

多智能体医疗分诊助手：**澄清式多轮问诊** + **多 Agent 协作** + **RAG 证据增强** + **视觉问答** + **转诊闭环**。

## 核心特性

- **澄清式多轮问诊**：信息不足时逐轮追问（一次只问 1 个问题，按优先级补齐缺失维度）；轻量门让追问秒回（3~4s），信息足够后才跑完整诊断流程
- **多 Agent 协作**：LeadAgent 动态分解任务，diagnostic / consultation / research 三 Agent 经 LangGraph Send 扇出并行执行，共享黑板汇总
- **RAG 证据增强**：Milvus 医学知识库（临床指南/科普）语义检索 + 在线嵌入/重排，回答可溯源
- **工具并行化**：同一轮多个 Skill 并行执行 + 保序回灌（模型看到的上下文与串行一致，质量不变，整批提速约 21%）
- **转诊闭环**：仅"高危 / 运行异常 / 用户主动要求"转人工（危机 100% 拦截，普通咨询不误转）；医生端回复后状态回传
- **流式输出**：思考完成后打字机效果逐字输出
- **会话持久化**：刷新页面对话不丢
- **多模态**：医学影像视觉问答（VLM 链路）

## 架构

```
用户输入
  └─ 轻量澄清门（信息不足 -> 追问 1 题，秒回）
       └─ 信息足够 / 危机 / 达上限 -> LangGraph 完整流程
            ├─ LeadAgent 路由分解（LLM）
            ├─ diagnostic / consultation / research 并行
            │    └─ Agent Loop（ReAct）+ 9 Skills + RAG 检索（工具并行）
            ├─ LeadAgent 汇总（synthesize + 出口护栏）
            ├─ 澄清判定（信息不足则追问，否则出最终诊断）
            └─ 转诊评估（危机/高危/异常/主动 -> 转人工闭环）
```

- **Agent Loop**：LLM 驱动的 ReAct 循环（LangGraph StateGraph），Agent 自主规划、调用 Skills、观察结果
- **记忆系统**：短期记忆（会话历史 + 追问轮标记）+ 长期记忆（Milvus `agent_memory` 跨会话检索）
- **Skills**：9 个医疗 Skill（症状分析 / 风险评估 / 指南检索 / 知识检索 / 生活方式 / 疾病编码 / 深度研究等），统一注册为 function calling
- **安全层**：约束校验 + 自动修复 + 输出护栏 + 危机优先转诊

## 快速启动（Windows）

### 1. 启动向量库（Docker）

```bash
docker start medical-milvus   # Milvus Standalone，端口 19530
```

### 2. 配置 API Key

编辑 `run_windows.ps1`，填入：
- **对话推理**：DeepSeek（`MEDITRIAGE_LLM_API_KEY` / `MEDITRIAGE_LLM_MODEL`）
- **嵌入 / 重排**：阿里云百炼（`MEDITRIAGE_DASHSCOPE_API_KEY`，`text-embedding-v4` / `gte-rerank-v2`）

### 3. 启动 Web

```powershell
cd D:\workProject\Asklepios
.\run_windows.ps1
```

打开 http://127.0.0.1:8090 即可使用。

> 环境变量以 `MEDITRIAGE_` 为前缀；`MEDITRIAGE_CLARIFY=0` 关闭澄清追问，`MEDITRIAGE_TOOL_PARALLEL=0` 关闭工具并行（可一键回退）。

## 测试

仓库内回归测试（pytest，纯单元，不依赖服务）：

```bash
cd MediTriage/agent
python -m pytest tests -k regression -q
```

覆盖：澄清逐轮追问 / 流式输出 / 转诊判定 / 工具并行 / RAG 缓存 / query 改写熔断 / 合成兜底等。

## 目录结构

```
MediTriage/agent/
├── meditriage/
│   ├── agents/          # consultation / diagnostic / research Agent
│   ├── core/            # Agent Loop、LLM 客户端
│   ├── knowledge/       # RAG 检索、query 改写、Milvus
│   ├── memory/          # 短期/长期记忆
│   ├── skills/          # 9 个医疗 Skill
│   ├── swarm/           # LangGraph 编排、澄清判定、LeadAgent
│   └── workflows/       # 转诊闭环（escalation）
├── web/                 # FastAPI + SSE + 前端
└── tests/               # 回归测试
```
