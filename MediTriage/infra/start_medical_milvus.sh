#!/usr/bin/env bash
# start_medical_milvus.sh — 启动/停止/查看本项目专属的 Milvus Standalone
#
# 本脚本、镜像、数据目录全部归属本项目。
#   - 容器名:   medical-milvus
#   - 镜像:     milvusdb/milvus:v2.5.27（含 CVE-2026-26190 安全修复）
#   - 数据卷: meditriage-milvus-data（Docker 命名卷，容器删除后数据保留）
#   - 网络:     rag-net（与 medix-fix 容器互通，Agent 用容器名 medical-milvus:19530 访问）
#   - 端口:     绑 127.0.0.1（共享服务器，不暴露 LAN）
#
# 用法:
#   bash MediTriage/infra/start_medical_milvus.sh          # 启动（幂等）
#   bash MediTriage/infra/start_medical_milvus.sh stop     # 停止并删除容器（数据保留）
#   bash MediTriage/infra/start_medical_milvus.sh status   # 状态
#   bash MediTriage/infra/start_medical_milvus.sh logs     # 日志
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"   # MediTriage/infra/ -> 项目根上溯两层
NAME="medical-milvus"
IMAGE="milvusdb/milvus:v2.5.27"  # 2.5系列最新(含CVE-2026-26190安全修复)；2.6取消了embedded-etcd单容器模式
VOLUME_NAME="meditriage-milvus-data"
NETWORK="rag-net"
PORT_GRPC="19530"
PORT_HTTP="9091"

ACTION="${1:-start}"

ensure_network() {
  docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK" >/dev/null
}

case "$ACTION" in
  start)
    ensure_network
    if docker ps --filter "name=^${NAME}$" --format '{{.Names}}' | grep -qx "$NAME"; then
      echo "==> ${NAME} already running."
      docker ps --filter "name=^${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
      exit 0
    fi
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker volume inspect "$VOLUME_NAME" >/dev/null 2>&1 || docker volume create "$VOLUME_NAME" >/dev/null
    echo "==> Starting ${NAME}  (image=${IMAGE}, data=${VOLUME_NAME}, net=${NETWORK})"
    docker run -d \
      --name "$NAME" \
      --network "$NETWORK" \
      --restart unless-stopped \
      --security-opt seccomp:unconfined \
      -e ETCD_USE_EMBED=true \
      -e ETCD_DATA_DIR=/var/lib/milvus/etcd \
      -e COMMON_STORAGETYPE=local \
      -p "127.0.0.1:${PORT_GRPC}:19530" \
      -p "127.0.0.1:${PORT_HTTP}:9091" \
      -v "${VOLUME_NAME}:/var/lib/milvus" \
      --health-cmd="curl -f http://localhost:9091/healthz || exit 1" \
      --health-interval=30s \
      --health-start-period=90s \
      --health-timeout=20s \
      --health-retries=3 \
      "$IMAGE" \
      milvus run standalone

    echo "==> Waiting for healthy (up to 120s)..."
    for i in $(seq 1 24); do
      sleep 5
      status="$(docker inspect --format='{{.State.Health.Status}}' "$NAME" 2>/dev/null || echo unknown)"
      printf "  [%3ds] health=%s\n" $((i*5)) "$status"
      [[ "$status" == "healthy" ]] && break
    done
    docker ps --filter "name=^${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    ;;
  stop)
    docker rm -f "$NAME" >/dev/null 2>&1 && echo "==> ${NAME} stopped (data kept in ${VOLUME_NAME})" || echo "==> ${NAME} not running"
    ;;
  status)
    docker ps -a --filter "name=^${NAME}$" --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}'
    ;;
  logs)
    docker logs --tail 50 "$NAME"
    ;;
  *)
    echo "Usage: $0 {start|stop|status|logs}"; exit 1
    ;;
esac
