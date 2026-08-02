# MediTriage 服务运维手册（启停 / 健康检查 / 排错）

本手册覆盖 MediTriage Agent 全部运行时服务的启动、停止、状态查看与常见排错。
所有服务跑在容器 `medix-fix`（挂载本仓库 → `/workspace`）；宿主机无 sudo，
root 所属操作经 `docker exec medix-fix` 完成。

---

## 一、服务拓扑与依赖顺序

```
浏览器 ──HTTPS──> Cloudflare 边缘(Access 登录门) ──隧道──> 宿主机 127.0.0.1:8080
                                                              │ (TCP 代理 _web_proxy.py)
                                                              ▼
                                              medix-fix:8080  Web(FastAPI + SSE)
                                                              │
                                              ┌───────────────┴───────────────┐
                                              ▼                               ▼
                                  vLLM :8000 (medix-r1-8b, GPU1)   Milvus :19530
                                  推理底座，256K 上下文            RAG 语料 + 长期记忆向量库
```

| 服务 | 端口 | 角色 | 启动脚本 |
|---|---|---|---|
| Milvus Standalone | 19530 | RAG 语料 + 长期记忆向量库 | `infra/start_medical_milvus.sh` |
| vLLM | 8000 | 推理底座（OpenAI 兼容，medix-r1-8b） | `serving/serve_vllm_official.sh` |
| Web 演示页 | 8080 | FastAPI + SSE 前端 | `infra/start_web_demo.sh` |
| Cloudflare Tunnel | — | 公网发布（可选） | `infra/start_cloudflare_tunnel.sh` |

**依赖顺序**：Milvus 与 vLLM 相互独立，可并行起；Web 依赖二者；隧道依赖 Web。
**启动顺序**：`Milvus + vLLM` → `Web` → （可选）`Tunnel`。

---

## 二、启动（按顺序）

### 0. 资源检查（起 vLLM 前）

```bash
nvidia-smi                       # 确认 GPU1 空闲（vLLM 用 GPU1，约需 72GB）
ps aux --sort=-rss | head        # 确认 host RAM 充足
```

### 1. Milvus（向量库）

```bash
bash MediTriage/infra/start_medical_milvus.sh          # 启动（幂等）
bash MediTriage/infra/start_medical_milvus.sh status   # 查看
```

### 2. vLLM（推理底座，GPU1，256K 上下文）

```bash
docker exec -d medix-fix bash -c "cd /workspace && bash MediTriage/serving/serve_vllm_official.sh"
# 首次加载权重约 1–2 分钟；就绪判据见下方健康检查
```

### 3. Web 演示页（容器内 FastAPI + 宿主机 TCP 代理）

```bash
bash MediTriage/infra/start_web_demo.sh                 # 启动（幂等，发布前探 /health）
bash MediTriage/infra/start_web_demo.sh status
# 就绪后本机访问 http://127.0.0.1:8080
# 远程访问（无隧道时）：ssh -L 8080:127.0.0.1:8080 <server> 后开 http://localhost:8080
```

### 4. 公网发布（可选，Cloudflare Tunnel）

服务器零入站端口：cloudflared 只出站连 Cloudflare 边缘，公网流量过
HTTPS/WAF，并由 Cloudflare Access 做邮箱白名单登录门。一次性前置
（约 15 分钟，详见脚本头部注释）：Cloudflare 账号 + 托管域名 →
`cloudflared tunnel login` → `tunnel create` → `tunnel route dns` 绑子域 →
Zero Trust 控制台给子域配 Access 邮箱白名单。

```bash
bash MediTriage/infra/start_cloudflare_tunnel.sh        # 启动（幂等）
bash MediTriage/infra/start_cloudflare_tunnel.sh status
# 公网入口：https://demo.<你的域名>（先过 Access 邮箱登录门）
# 长期记忆按 Access 登录邮箱隔离（web/server.py 读 cf-access-authenticated-user-email 头）
```

---

## 三、健康检查（确认每层就绪）

```bash
# Milvus
bash MediTriage/infra/start_medical_milvus.sh status

# vLLM（返回 model 列表与 max_model_len 即就绪）
curl -s http://127.0.0.1:8000/v1/models

# Web（返回 {"status":"ok"} 即就绪；/api/meta 返回 vLLM 上下文窗口）
curl -s http://127.0.0.1:8080/health
curl -s http://127.0.0.1:8080/api/meta

# 隧道（注册到 Cloudflare 边缘的连接 + Access 门是否生效）
grep -a "Registered tunnel connection" /tmp/medix_cf_tunnel.log | tail
curl -s -o /dev/null -w "%{http_code}\n" https://demo.<你的域名>/health   # 期望 302（被 Access 拦）
```

一条命令快速体检全栈：

```bash
curl -s http://127.0.0.1:8000/v1/models | head -c 80; echo
curl -s http://127.0.0.1:8080/api/meta; echo
```

---

## 四、停止

```bash
bash MediTriage/infra/start_cloudflare_tunnel.sh stop   # 先停对外入口
bash MediTriage/infra/start_web_demo.sh stop            # 停 Web（代理 + 容器内服务）
docker exec medix-fix bash -c "pkill -f 'vllm serve'"  # 停 vLLM，释放 GPU1
bash MediTriage/infra/start_medical_milvus.sh stop      # 停 Milvus（数据持久在 data/milvus_data）
```

---

## 五、常见运维操作

**改了 Agent 代码后重载 Web**（无需动 vLLM/Milvus）：
```bash
bash MediTriage/infra/start_web_demo.sh stop && bash MediTriage/infra/start_web_demo.sh
```

**跑回归测试**（先装包，纯单元回归不依赖服务）：
```bash
docker exec medix-fix bash -c "cd /workspace/MediTriage/agent && pip install -e . -q && python -m pytest tests/ -q -k regression"
```

**清理运行期产物**（会话缓存 / 日志 / 字节码 / pytest 缓存）：
```bash
bash MediTriage/infra/clean_runtime.sh --dry-run        # 先预览
bash MediTriage/infra/clean_runtime.sh                  # 实清
```

**前端改动看不到**：Web 服务端带 no-cache 头，浏览器强刷即可；仍不对则重启 Web。

---

## 六、排错速查

| 现象 | 排查 / 处理 |
|---|---|
| Web `/health` 不通 | 容器内服务或宿主代理未起；看 `/tmp/medix_web_proxy.log` 与容器内 `log/web/server.log`；重跑 `start_web_demo.sh` |
| `/api/ask` 报错或卡住 | 查 vLLM 是否就绪（`curl :8000/v1/models`）、GPU1 是否 OOM（`nvidia-smi`）；后端日志 `grep -a ERROR log/web/server.log` |
| vLLM 起不来 | GPU1 被占（`nvidia-smi`）或上次进程残留（`pkill -f 'vllm serve'` 后重起）；权重路径见 `serving/serve_vllm_official.sh` |
| RAG 检索为空 / 报连接异常 | Milvus 未起或地址错（容器内应为 `medical-milvus:19530`，非 127.0.0.1）；`start_medical_milvus.sh status` |
| 隧道 502 | 本地 Web 未就绪就起了隧道；脚本已在发布前探 `/health`，确认 Web 先起 |
| 公网访问不弹登录页 | Cloudflare Access 策略未绑定该子域；到 Zero Trust 控制台给子域加 Self-hosted Application + Emails 白名单 |
| 机器重启后服务全没 | 容器内服务与隧道均非开机自启，按本手册第二节重新拉起 |

---

## 七、相关文档

- `serving/README.md` — vLLM 底座细节
- 仓库根 `README.md` — 架构总览与快速上手
- `MediTriage/agent/README.md` — Agent 内部架构（core/agents/swarm/knowledge/memory/skills）
