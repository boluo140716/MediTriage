#!/usr/bin/env bash
# start_web_demo.sh — 启动 MediTriage Agent 演示页面
#
# 数据流:
#   浏览器 --> 宿主机 ${LISTEN}:${PORT} --(stdlib TCP代理 MediTriage/infra/_web_proxy.py)--> medix-fix:8080 (uvicorn FastAPI+SSE)
#   容器内 swarm --> vLLM(localhost:8000, 官方 MediX-R1-8B) + RAG(medical-milvus:19530)
#
# 为什么要代理：medix-fix 容器只发布了 8000(vLLM)，演示页跑在容器 8080 未对宿主机暴露；
#   不重建容器(会杀掉已加载的 vLLM)、不装 socat、不用 root —— 用 Python 裸 TCP 透传(SSE 友好)。
#
# 用法:
#   bash MediTriage/infra/start_web_demo.sh          # 启动(幂等)
#   bash MediTriage/infra/start_web_demo.sh stop     # 停止代理 + 容器内服务
#   bash MediTriage/infra/start_web_demo.sh status   # 状态
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"          # = MediTriage/infra
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"                   # MediTriage/infra -> 仓库根
CONTAINER="medix-fix"
CONTAINER_ROOT="${MEDIX_CONTAINER_ROOT:-/workspace}"               # 仓库在容器内的挂载点（medix-fix 默认 /workspace；换挂载点设此 env）
PORT=8080
LISTEN="127.0.0.1"        # 安全默认：仅绑本机(与 medical-milvus 一致)，浏览器走 SSH 隧道；要 LAN 直连改 0.0.0.0
PID_FILE="/tmp/medix_web_proxy.pid"
PROXY_LOG="/tmp/medix_web_proxy.log"

ACTION="${1:-start}"

stop_proxy() {
  [ -f "$PID_FILE" ] && { kill "$(cat "$PID_FILE")" 2>/dev/null || true; rm -f "$PID_FILE"; }
  pkill -f "_web_proxy.py" 2>/dev/null || true
}

case "$ACTION" in
  start)
    # 1) 解析容器 IP（首个网络）
    CIP="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}' "$CONTAINER" | awk '{print $1}')"
    [ -z "$CIP" ] && { echo "ERROR: cannot resolve ${CONTAINER} IP"; exit 1; }

    # 2) 容器内 web server（幂等；注意 ss 在该容器内不报 LISTEN，改用 /health 探活）
    if curl -sf -m 3 "http://${CIP}:${PORT}/health" >/dev/null 2>&1; then
      echo "==> web server already running in ${CONTAINER}:${PORT}"
    else
      echo "==> starting web server in ${CONTAINER}..."
      docker exec -d "$CONTAINER" bash -c \
        "mkdir -p ${CONTAINER_ROOT}/log/web && cd ${CONTAINER_ROOT}/MediTriage/agent && python3 web/server.py > ${CONTAINER_ROOT}/log/web/server.log 2>&1"
      for i in $(seq 1 15); do
        sleep 1
        curl -sf -m 3 "http://${CIP}:${PORT}/health" >/dev/null 2>&1 && break
      done
    fi

    # 3) 宿主机 TCP 代理（setsid 完全脱离，survive 终端退出）
    stop_proxy
    setsid python3 "${SCRIPT_DIR}/_web_proxy.py" "$LISTEN" "$PORT" "$CIP" "$PORT" \
      > "$PROXY_LOG" 2>&1 &
    echo $! > "$PID_FILE"
    sleep 1

    # 4) 验证
    if curl -sf -m 5 "http://127.0.0.1:${PORT}/health" >/dev/null; then
      echo "==> READY  (proxy pid=$(cat "$PID_FILE"), -> ${CIP}:${PORT}, listen=${LISTEN})"
      echo "    本机:     http://127.0.0.1:${PORT}"
      if [ "$LISTEN" = "0.0.0.0" ]; then
        HOST_IP="$(hostname -I 2>/dev/null | awk '{print $1}')"
        [ -n "$HOST_IP" ] && echo "    局域网:   http://${HOST_IP}:${PORT}"
      fi
      echo "    SSH隧道:  ssh -L ${PORT}:127.0.0.1:${PORT} <server>  然后开 http://localhost:${PORT}"
    else
      echo "==> proxy started but /health failed — check ${PROXY_LOG} 和 容器日志 ${CONTAINER_ROOT}/log/web/server.log"
      exit 1
    fi
    ;;
  stop)
    stop_proxy
    docker exec "$CONTAINER" bash -c "pkill -f 'web/server.py'" 2>/dev/null || true
    # 等进程真正退出再返回，避免紧接着 start 时探活探到「正在死亡仍占端口」的旧进程(竞态)
    for i in $(seq 1 10); do
      docker exec "$CONTAINER" pgrep -f 'web/server.py' >/dev/null 2>&1 || break
      sleep 0.5
    done
    echo "==> stopped web proxy + container web server"
    ;;
  status)
    echo "--- host proxy ---"
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "running (pid=$(cat "$PID_FILE"))"
    else
      echo "not running"
    fi
    echo "--- container web server ---"
    CIP="$(docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}} {{end}}' "$CONTAINER" 2>/dev/null | awk '{print $1}')"
    if [ -n "$CIP" ] && curl -sf -m 3 "http://${CIP}:${PORT}/health" >/dev/null 2>&1; then
      echo "running (${CIP}:${PORT})"
    else
      echo "not running"
    fi
    echo "--- host health ---"
    curl -sf -m 5 "http://127.0.0.1:${PORT}/health" || echo "  (no response)"
    ;;
  *)
    echo "Usage: $0 {start|stop|status}"; exit 1;;
esac
