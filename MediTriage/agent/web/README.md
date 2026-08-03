# MediTriage Agent Swarm · Web 前端

FastAPI 后端 + SSE（Server-Sent Events）流式输出 + 单文件 HTML/JS 前端，
用于实时可视化 Swarm 内部的 Agent 协作过程：路由 → 思考 → 工具调用 → 综合 → 最终答案。

## 启动

```bash
cd MediTriage/agent
pip install fastapi uvicorn               # 或 pip install -r requirements.txt
python -m uvicorn web.server:app --host 0.0.0.0 --port 8080
```

浏览器访问：`http://<server-ip>:8080`

## 前置条件

1. `config.py` 已配置 LLM_CONFIG（指向本地 vLLM，serve 官方 MediX-R1-8B）
2. vLLM 服务运行中（用 `serving/serve_vllm_official.sh`）

## API

### `POST /api/ask`

请求：
```json
{
  "question": "我头痛、视力模糊、血压偏高怎么办",
  "session_id": "可选",
  "image": "可选，data:image/* 格式的 data URI",
  "image_name": "可选"
}
```

带 `image` 走单轮 VLM 视觉链路（`vision_handler`），不带则走多 Agent swarm；
纯文本追问若指代图片（"图中""这张 CT"等），自动复用会话中最近上传的影像。

响应：`text/event-stream` (SSE)，每行：
```
event: <event_type>
data: {"type":"<event_type>","data":{...},"ts":"..."}
```

### 事件类型

| event_type | 来源 | 触发时机 |
|------------|------|---------|
| `ack` | server | SSE 连接建立 |
| `session_started` | LangGraphSwarm | 接到问题 |
| `lead_routing` | LangGraphSwarm | LeadAgent 决定走单 Agent 还是 Swarm |
| `task_decomposed` | SharedContext | LeadAgent 分解出的子任务注册进黑板 |
| `swarm_started` | LangGraphSwarm | Swarm 模式启动 |
| `subtask_started` | SharedContext | 某个子任务被某个 Agent 认领 |
| `agent_thinking` | AgentLoop | Agent 开始第 N 轮 ReAct |
| `llm_response` | AgentLoop / vision_handler | LLM 返回（含 tool_calls 与否；视觉链路补发一条上报 token 用量） |
| `tool_call_started` | AgentLoop | Agent 调用某个 skill |
| `tool_call_completed` | AgentLoop | Skill 返回结果 |
| `final_answer` | AgentLoop | 单个 Agent 给出最终答案 |
| `subtask_completed` | SharedContext | 子任务完成 |
| `swarm_completed` | LangGraphSwarm | 所有子任务结束 |
| `vision_started` | vision_handler | 带图请求进入视觉链路 |
| `vision_answer` | vision_handler | VLM 给出图像分析结果 |
| `result` | server | 完整结果（含 suggestions、disclaimer） |
| `session_completed` | LangGraphSwarm / vision_handler | 整个会话结束 |
| `error` | server / vision_handler | 任何异常 |
| `end` | server | SSE 流结束（前端可以关闭连接） |

`SWARM_ENGINE=legacy` 回退旧版 `SwarmCoordinator` 时发同名事件，前端无感。

### `GET /health` · `GET /api/meta`

`/health` 健康检查；`/api/meta` 直连 vLLM `/v1/models` 返回当前模型名与
`max_model_len`，供前端展示真实上下文窗口。

## 转诊闭环（P1-2）：AI 初诊 + 人工复核

`/api/ask` 返回的 `result` 事件会附带 `escalation` 字段（未命中为 `null`）。
命中即自动在 SQLite 建转诊单（`data/escalations.db`，状态机：
`ai_processing -> escalated -> doctor_replied`），患者端显示转诊卡片，
医生端在 `/doctor` 处理队列并回复。

触发三层（`meditriage/workflows/escalation.py`）：
1. 强制转诊：危机关键词（自伤/心脑血管急症等）、回答超时/报错/空回答、用户主动要求转人工；
2. 信息咨询短路：怎么/如何/注意事项类科普提问直接不转（省一次 LLM 调用）；
3. 模糊场景：LLM 置信度打分（0~1），低于阈值（默认 0.4）转诊；
   LLM 失败/超时安全兜底转人工（宁过不欠）。

环境变量（全部 `MEDITRIAGE_` 前缀）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `MEDITRIAGE_ESCALATION_ENABLED` | 1 | 总开关 |
| `MEDITRIAGE_ESCALATION_THRESHOLD` | 0.4 | 置信度阈值（越低越严） |
| `MEDITRIAGE_ESCALATION_DB` | `data/escalations.db` | SQLite 路径 |
| `MEDITRIAGE_ESCALATION_LLM` | 1 | 0 = 纯规则模式（不调 LLM，省成本/延迟） |

### 转诊 API

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/doctor` | 医生端页面（模拟身份，无登录；认证在 P2-5） |
| GET | `/api/escalations?status=` | 转诊队列（status 可选 `escalated` / `doctor_replied`） |
| GET | `/api/escalations/{id}` | 详情（含结构化交接摘要） |
| POST | `/api/escalations/{id}/reply` | 医生回复：`{"reply": "..."}`，escalated -> doctor_replied |

## 事件流实现

SSE 事件由 `event_emitter` 回调贯穿。编排引擎是 LangGraph StateGraph
（`meditriage/swarm/langgraph_swarm.py`：enrich 记忆注入 → route 分解 →
Send 扇出并行 worker → synthesize 汇总 → persist 落库），会话与路由事件在
图节点发出；`AgentLoop.run()`（ReAct 循环，同样以 StateGraph 建模）在
LLM / tool 关键点发出细粒度事件；`SharedContext.subscribe()` 把 swarm
黑板事件转发给前端。`event_emitter` 可选——不传则退化为纯 CLI，无 SSE 开销。
