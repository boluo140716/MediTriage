# MediTriage 多智能体医疗助手

基于 Skills-Agent 两层架构的多智能体协作医疗助手系统，融合 Agent Loop、Agent Swarm、记忆管理和 Milvus 知识库。

## 项目概述

本项目采用 **Skills-Agent 两层架构**，通过 9 个自包含 Skills（8 原子 + 1 复合）和 3 个专业 Agent 协同工作，提供智能、专业的医疗服务。

### 核心特性

- **Skills 直达架构**: 9个 Skills 自包含，直接转换为 OpenAI function calling 格式
- **Agent Loop**: LLM 驱动的 ReAct 循环（LangGraph StateGraph 建模），Agent 自主规划、调用 Skills 并完成任务
- **Agent Swarm**: 多 Agent 协作（LangGraph StateGraph 编排：LeadAgent 中心指派 + Send 并行扇出 + 共享黑板汇总）
- **记忆系统**: 短期记忆（会话级对话历史）+ 长期记忆（本地 Milvus + BGE-M3 跨会话检索）+ 多轮对话上下文利用
- **Milvus 知识库**: 统一知识管理，语义检索，支持模糊查询（"血压高" → "高血压"）
- **Skills 体系**: 9个预定义技能（8个原子 + 1个复合 deep-research），一键调用医疗能力
- **Harness 安全层**: 约束驱动 + 记忆治理的输出校验/自动修复，内嵌于 Agent Loop

## Skills 直达架构

### 架构设计

```
Skills (函数) → 直接转换 → OpenAI Format → LLM 调用
         ↓
    Milvus/业务逻辑
```

### 关键特性

1. **Skills 直达 LLM**
   - Skill 函数直接转换为 OpenAI function calling 格式
   - SkillRegistry 统一管理：注册、执行、格式转换

2. **简化的注册流程**
   ```python
   skill → OpenAI Format
   ```

3. **Agent 灵活选择**
   - 每个 Agent 注册全部9个 Skills
   - Agent Loop 根据任务自主选择合适的 Skills
   - 一个 Agent 可以跨领域调用 Skills

4. **用户友好入口**
   - 8个原子 Skills：快速查询，立即响应
   - 1个复合 Skill（deep-research）：多查询规划 + 检索 + 证据综合
   - 用户无需理解 Agent 架构

5. **多轮对话支持**
   - 短期记忆：会话级对话历史（默认取最近 10 轮，去重 + 预算窗口）
   - 长期记忆：本地 Milvus + BGE-M3 跨会话检索（collection `agent_memory`）
   - 追问能基于历史对话上下文作答

### 测试验证

仓库内提供 smoke / 集成测试用例（`tests/`，pytest），覆盖范围：

**核心功能**：
- Agent Loop 和 Skill 调用
- Agent Swarm 多 Agent 协作
- 记忆系统（短期 + 长期）
- 多轮对话上下文利用
- Skills 自主选择
- Milvus 知识库集成

**Harness 安全层**：
- 约束系统（Skill 调用验证、输出验证、自动修复）
- 记忆卫生（去重 + 预算窗口）
- 约束在 Agent Loop 中的注入链路

运行测试（集成/冒烟，需容器内 vLLM:8000 + Milvus 已起）：
```bash
python -m pytest -ra              # 全部
python tests/test_llm_smoke.py    # 单个（也可直接 python 跑）
```

## 从零开始运行

### 1. 环境准备

```bash
conda create -n medix-swarm python=3.12 -y
conda activate medix-swarm
cd MediTriage/agent
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
# 可编辑安装为包（meditriage.*），免运行期 sys.path 拼接
pip install -e .
```

### 3. 配置 API

创建 `../../config.py`（仓库根）：

```python
# 当前实际：接本地 vLLM（serve 官方 MediX-R1-8B）
LLM_CONFIG = {
    "api_key": "not-needed",                 # vLLM 不校验 key
    "model_name": "medix-r1-8b",
    "base_url": "http://localhost:8000/v1",
    "temperature": 0.7,
    "max_tokens": 4096,                      # 远 < max-model-len(262144)
}
# 长期记忆：本地 Milvus + BGE-M3（meditriage/memory/medical_memory.py）
```

### 4. 构建知识库索引

```bash
# 中文语料 + 已解析指南(_parsed/) → BGE-M3 向量 → Milvus（幂等重建；需先起 medical-milvus）
python ../data/rag_corpus/build_rag_index_v2.py
```

### 5. 运行测试

```bash
python -m pytest -ra
```

### 6. 开始使用

```bash
python main.py
```

## 项目结构

```
MediTriage/agent/                         # 可安装包根（pip install -e .）
├── meditriage/                           # 库包（import meditriage.*）
│   ├── __init__.py
│   ├── paths.py                         # 资产路径锚点（模型/数据/日志，均可 env 覆盖）
│   ├── core/                            # 运行时内核
│   │   ├── agent_loop.py                # ReAct 循环（StateGraph 建模 + 约束验证）
│   │   ├── llm_client.py                # vLLM(OpenAI 兼容) 客户端
│   │   ├── skill_loader.py              # 动态发现 skills/
│   │   ├── skill_registry.py            # Skill 注册表（直转 OpenAI format）
│   │   ├── state_manager.py             # 执行状态
│   │   └── vision_handler.py            # 图像 VQA 单轮链路
│   ├── agents/                          # Worker 角色（模板方法 + Mixin）
│   │   ├── base_agent.py                # Agent 基类
│   │   ├── consultation_agent.py        # 健康咨询
│   │   ├── diagnostic_agent.py          # 症状诊断
│   │   ├── research_agent.py            # 医学研究
│   │   ├── skill_registry_mixin.py      # Skill 注册（共享）
│   │   ├── extractors.py                # 答案结构化抽取（共享）
│   │   └── _prompt_blocks.py            # 接地铁律等共享提示词块
│   ├── swarm/                           # 多 Agent 编排
│   │   ├── langgraph_swarm.py           # LangGraph StateGraph 编排（默认引擎）
│   │   ├── lead_agent.py                # 任务分解 + 结果合成
│   │   ├── swarm_coordinator.py         # 旧版协调器（SWARM_ENGINE=legacy 回退）
│   │   ├── shared_context.py            # 黑板（SubTask / Contribution）
│   │   └── events.py                    # 事件类型
│   ├── knowledge/                       # RAG
│   │   ├── milvus_kb.py                 # 知识库门面（单例）
│   │   ├── langchain_rag.py             # 检索引擎（BGE-M3 + BM25 + RRF + reranker；文件名为历史兼容）
│   │   ├── query_rewrite.py             # 检索前 query 改写
│   │   ├── citations.py                 # 检索结果 → 可读来源引用
│   │   └── badcase.py                   # 低质检索事件落盘（rag_misses.jsonl）
│   ├── memory/                          # 记忆分层
│   │   ├── short_term.py                # 短期（会话级）
│   │   ├── long_term.py                 # 长期检索门面
│   │   ├── medical_memory.py            # 长期实现（Milvus + 信任/时间加权）
│   │   ├── session_summary.py           # 会话总结（Markdown 落盘）
│   │   └── hygiene.py                   # 记忆卫生（去重 + 预算窗口）
│   ├── research/                        # 深度研究
│   │   ├── deep_research_workflow.py    # 编排
│   │   ├── evidence_synthesizer.py      # 证据综合
│   │   └── pubmed.py                    # PubMed 检索（E-utilities，无网降级）
│   ├── guardrails/                      # 运行时护栏（检测 → 修复）
│   │   ├── validator.py                 # 约束验证
│   │   ├── auto_fixer.py                # 违规自动修复
│   │   ├── _keywords.py                 # 高危症状 / 就医判定（共享）
│   │   └── agent_constraints.yaml       # Agent 能力边界
│   └── skills/                          # 9 个医疗 Skill（SKILL.md + script/）
│       ├── search-knowledge/    assess-risk/         analyze-symptoms/
│       ├── recommend-lifestyle/ disease-code/        clinical-guideline/
│       └── deep-research/       search-history/      search-similar-cases/
│
├── web/                                 # FastAPI 后端（SSE）+ 单文件前端
│   ├── server.py
│   └── index.html
├── tests/                               # 集成 / 冒烟 / 回归测试
├── main.py                              # CLI 入口（交互式对话）
├── setup.py                             # 打包（pip install -e .）
├── pytest.ini
├── requirements.txt
└── README.md
```

**架构说明**：
- **直达架构**：Skills → OpenAI Format
- **Skills 自包含**：每个 Skill 在 `script/` 目录下实现，直接调用知识库
- **动态加载**：`skill_loader.py` 扫描 `skills/` 目录动态加载
- **SkillRegistry**：统一管理 Skill 注册、执行、格式转换
- **统一配置**：使用项目根目录的 `config.py`
- **记忆分离**：会话总结以 Markdown 落盘，长期记忆走 Milvus 向量库（`agent_memory` collection）

## Skills 和 Agent 清单

### 9 个 Skills（8 原子 + 1 复合）

**所有 Agent 共享以下 Skills**：

| Skill | 功能 | 数据源 | 特点 |
|-------|------|--------|------|
| `search_knowledge` | 搜索医学知识库 | Milvus | 语义检索 |
| `recommend_lifestyle` | 生活方式和用药建议 | Milvus | 个性化建议 |
| `assess_risk` | 风险等级评估 | 规则引擎 | 高危症状识别 |
| `analyze_symptoms` | 症状模式分析 | 规则引擎 | 多系统分析 |
| `disease_code` | ICD-10疾病编码 | Milvus | 标准编码 |
| `clinical_guideline` | 临床指南检索 | Milvus | 权威指南 |
| `search_history` | 搜索当前会话历史 | 短期记忆 | 会话内检索 |
| `search_similar_cases` | 搜索相似历史案例 | 长期记忆 Milvus | 跨会话检索 |
| `deep_research` | 深度研究（复合） | Milvus + PubMed | 多查询规划 + 检索 + 证据综合 |

### 3个专业 Agent（自主选择 Skills）

#### 1. ConsultationAgent（健康咨询）
- **能力**: 通用健康咨询和生活方式指导
- **注册 Skills**: 全部9个（自主选择合适的 Skills）
- **常用 Skills**: `search_knowledge`, `recommend_lifestyle`

#### 2. DiagnosticAgent（症状诊断）
- **能力**: 症状分析、风险评估和鉴别诊断
- **注册 Skills**: 全部9个（自主选择合适的 Skills）
- **常用 Skills**: `assess_risk`, `analyze_symptoms`, `disease_code`

#### 3. ResearchAgent（医学研究）
- **能力**: 循证医学证据和权威指南检索
- **注册 Skills**: 全部9个（自主选择合适的 Skills）
- **常用 Skills**: `clinical_guideline`, `deep_research`

### 协调与编排

- **LeadAgent**: 任务分解和结果汇总（非编排器）
- **编排引擎**: 默认 LangGraph（`langgraph_swarm.py`，StateGraph 建模
  记忆注入→路由→Send 并行扇出→汇总→持久化）；`SWARM_ENGINE=legacy`
  回退旧版 SwarmCoordinator（行为对齐，保留对照与回退）。单 Agent 内部的
  ReAct 循环（core/agent_loop.py）同样以 StateGraph 显式建模。

### Skills 架构特点

- **直达架构**: Skills → OpenAI Format
- **Skills 自包含**: 直接调用 Milvus 或内置逻辑
- **Agent 灵活性**: 每个 Agent 注册全部9个 Skills，根据任务自主选择
- **SkillRegistry**: 统一管理注册、执行、格式转换
- **统一知识库**: 医学知识统一存储在 Milvus 向量数据库，支持语义检索
- **易于扩展**: 添加新 Skill 或新知识无需修改 Agent 代码


## 配置说明

项目使用项目根目录的统一配置文件：`config.py`

### 配置内容

```python
# LLM 端点（本地 vLLM，OpenAI 兼容；model_name 与 vLLM --served-model-name 一致）
LLM_CONFIG = {
    "api_key": "not-needed",                 # vLLM 不校验 key
    "model_name": "medix-r1-8b",
    "base_url": "http://localhost:8000/v1",
    "temperature": 0.7,
    "max_tokens": 4096,
}
```

> 说明：长期记忆为本地 Milvus + BGE-M3，无需任何云服务或额外 API Key。
> 短期记忆的 Redis 后端（可选）经 `ShortTermMemory(storage_type="redis", redis_config={...})` 传入连接信息。

### 记忆系统配置

本系统支持两层记忆机制：**短期记忆（会话级）**和 **长期记忆（跨会话）**。

#### 短期记忆（ShortTermMemory）

**作用**：存储当前会话的对话历史，支持多轮对话上下文理解。

**配置**：
```python
from meditriage.memory.short_term import ShortTermMemory
memory = ShortTermMemory(storage_type="memory")  # 或 "redis"
```

**使用示例**：
```python
sid = "user_123"
memory.add_message(session_id=sid, role="user", content="我有高血压")
memory.add_message(session_id=sid, role="assistant", content="高血压需要...")

# 获取会话历史（最近若干条）
history = memory.get_history(sid, limit=10)
```

**存储方式**：
- **内存**（默认）：进程内存储，无需配置
- **Redis**（可选）：经 `redis_config` 参数传入连接信息，键 1 小时过期，跨进程可用
- 读取窗口：默认最近 10 轮对话，经去重 + 预算窗口防上下文膨胀

#### 长期记忆（本地 Milvus + BGE-M3）

**作用**：跨会话记忆，通过向量相似度检索历史案例和经验。

> 实现：本地 Milvus（`medical-milvus:19530`，collection `agent_memory`）+ BGE-M3，
> 向量化与相似度检索均在本地完成，无需任何云服务或额外 API Key。

**使用示例**：
```python
from meditriage.memory.long_term import LongTermMemory
memory = LongTermMemory()

# 检索相似历史会话（写入由 swarm 落库统一处理）
results = memory.search_similar_sessions("高血压患者如何管理？")
# → 返回历史相似案例
```

**存储方式**：
- **本地 Milvus + BGE-M3**：向量化与相似度检索均在本地完成
- 存储范围：跨会话持久化（collection `agent_memory`）
- 存储内容：会话总结
- 无需任何云服务，无需额外 API Key

#### 记忆系统如何融入对话

**流程**：

```
1. 会话开始
   ↓
2. 从本地 Milvus（agent_memory）检索相关长期记忆（历史案例）
   ↓
3. 初始化短期记忆（对话历史）
   ↓
4. Agent 执行
   - 读取短期记忆：获取当前会话上下文
   - 写入短期记忆：记录本轮对话
   - 参考长期记忆：利用历史经验
   ↓
5. 会话结束
   ↓
6. 短期记忆转换为结构化数据 → 存入本地 Milvus 长期记忆（agent_memory）
   ↓
7. 清空短期记忆
```

**多轮对话示例**：

```python
# 第1轮
用户: "我有高血压"
系统: [短期记忆添加用户消息]
系统: [Agent 处理] "高血压需要注意..."
系统: [短期记忆添加助手消息]

# 第2轮
用户: "那我应该吃什么药？"  # 追问
系统: [读取短期记忆] → 获取上一轮"高血压"上下文
系统: [Agent 处理] "根据您的高血压情况，建议..."  # 正确理解追问
```

**注意事项**：
- 长期记忆依赖本地 Milvus（`medical-milvus`）+ BGE-M3；Milvus 不可用时优雅降级，仅使用短期记忆继续工作
- 短期记忆默认使用内存存储，无需配置 Redis
- 无需任何云服务或外部 API Key

## Harness 安全层

安全层让 Agent 在明确约束下自主工作、自我修正：约束以 YAML 声明，运行时校验输出并自动修复违规。

### 安全层机制

| 机制 | MediTriage 实现 | 位置 |
|------|-----------|------|
| **约束驱动**| YAML 定义 Agent 能力边界，运行时验证 | `guardrails/` |
| **自动修复**| 输出违规自动添加免责声明、高危警告 | `guardrails/` |
| **记忆卫生**| 短期记忆去重 + 预算窗口，防上下文膨胀 | `memory/hygiene.py` |

### 核心功能

**1. 约束验证**（`guardrails/agent_constraints.yaml`）
- 定义每个 Agent 允许的 Skills 和禁止的行为
- 运行时自动验证 Skill 调用和输出内容
- 违规时记录警告日志

**2. 自动修复**（`guardrails/auto_fixer.py`）
- 缺少免责声明 → 自动添加
- 高危症状（胸痛、呼吸困难等）→ 自动添加就医提醒

**3. 记忆卫生**（`memory/hygiene.py`）
- 精确去重重复消息（基于 (role, content) 的 MD5 哈希）
- 预算窗口：按字符预算保留最近若干条历史，防多轮上下文膨胀溢出

### 验证

运行测试套件（含 Harness 安全层用例）：
```bash
python -m pytest -ra
```

---

## 统一知识库

- **向量数据库**: Milvus Standalone（容器 `medical-milvus`，`medical-milvus:19530`）
- **Embedding 模型**: BGE-M3（多语言，1024维，COSINE）
- **数据存储**: `data/rag_corpus/`（指南 PDF 解析产物 `_parsed/` + 中文语料 `local_zh/`）
- **当前规模**: collection `medical_knowledge_m3`，约 10251 个 chunks（向量 FLAT + COSINE 精确索引，`mtype` INVERTED 标量索引）
- **检索编排**: jieba-BM25 + 向量混合召回、加权 RRF 融合、查询改写、bge-reranker 重排、父节聚合、低相关阈值弃权；仅复用 BGE-M3 嵌入与 bge-reranker，不依赖 LangChain，见 `knowledge/langchain_rag.py`
- **badcase 沉淀**: 低相关 / 空检索 / 灰区命中以结构化 JSON 行落盘 `rag_misses.jsonl`，供周期复盘与语料扩展选题（`knowledge/badcase.py`）

## 技术架构

### Agent Loop (Think-Act-Observe)

```
┌─────────┐     ┌────────┐     ┌──────────┐
│  Think  │ ──> │  Act   │ ──> │  Observe │
└─────────┘     └────────┘     └──────────┘
     ↑                               │
     └───────────────────────────────┘
```

### Skills 直达架构

```
用户问题
   ↓
【原子查询】→ 直接调用 Skills → Milvus/业务逻辑
   │                ↓
   │         OpenAI Format
   │
   └─【复杂问题】
          ↓
   编排引擎（默认 LangGraph 图；SWARM_ENGINE=legacy 走 SwarmCoordinator）
          ↓
     LeadAgent（分解任务）
          ↓
    发布到 SharedContext（共享环境）
          ↓
    ┌─────┴─────┬────────┐
    ↓           ↓        ↓
ConsultAgent DiagAgent ResearchAgent
（SkillRegistry）（直达 LLM）（并行执行）
    │           │        │
    └───────────┴────────┘
          ↓
    LeadAgent（汇总结果）
          ↓
   SessionSummary（学习）
```

**核心原理**：
- Skills → OpenAI Format
- SkillRegistry 统一管理（注册、执行、转换）
- Agent 注册所有 Skills，根据任务自主选择
- Agent 通过共享黑板（SharedContext）交换中间结果
- LeadAgent 中心指派子任务，Worker 并行产出后由 LeadAgent 汇总

### Agent Swarm 群体智能

**关键特性**：中心指派、并行执行、共享黑板汇总

**工作流程**：
1. 简单问题 → 单 Agent（快速响应）
2. 复杂问题 → LeadAgent 分解任务
3. LeadAgent 按类型把子任务指派给对应 Worker
4. 并行执行（每个 Agent 自主选择 Skills）
5. LeadAgent 汇总结果
6. SessionSummary 学习总结

### 记忆系统架构

```
┌────────────────────────────────────┐
│  短期记忆（会话级，内存/Redis）     │
│  - 对话历史（messages）            │
│  - 当前会话上下文                  │
│  - 去重 + 预算窗口                 │
│  存储：内存（默认）或 Redis        │
└────────────────────────────────────┘
           ↕ (会话结束时)
┌────────────────────────────────────┐
│  长期记忆（跨会话，本地 Milvus）   │
│  - 会话总结                        │
│  存储：Milvus + BGE-M3            │
│  (collection: agent_memory)        │
└────────────────────────────────────┘
```

## 免责声明

本系统仅供学习和研究使用，不能替代专业医生的诊断和治疗。所有医疗建议仅供参考，如有健康问题请及时就医。

## 许可证

MIT License

## 致谢

- 基于 [MediX-R1-8B](https://huggingface.co/MBZUAI/MediX-R1-8B) 医学多模态模型（MBZUAI）
- 推理后端 [vLLM](https://github.com/vllm-project/vllm)（OpenAI 兼容服务），编排基于 [LangGraph](https://github.com/langchain-ai/langgraph)
- 向量检索与记忆基于 [Milvus](https://milvus.io/) + [BGE-M3](https://huggingface.co/BAAI/bge-m3)

---
