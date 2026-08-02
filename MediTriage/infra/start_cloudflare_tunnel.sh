#!/usr/bin/env bash
# start_cloudflare_tunnel.sh — 把本机演示页（127.0.0.1:8080）经 Cloudflare Tunnel 发布到公网
#
# 为什么用它：服务器一个入站端口都不开（cloudflared 只出站连 Cloudflare 边缘），
#   公网流量过 Cloudflare 的 HTTPS/WAF/限流，并由 Cloudflare Access 做登录门——
#   不暴露源站发布服务的低风险成熟方案。
#
# 前置（一次性，在 Cloudflare 控制台 + 浏览器完成，约 15 分钟）：
#   1) 有 Cloudflare 账号，且有一个已托管在 Cloudflare 的域名
#   2) cloudflared tunnel login            # 浏览器授权，凭证落 ~/.cloudflared/
#   3) cloudflared tunnel create meditriage # 建命名隧道，得到 <TUNNEL_ID>
#   4) cloudflared tunnel route dns meditriage demo.<你的域名>  # 绑公网子域
#   5) 在 Zero Trust 控制台给 demo.<域名> 配 Access 策略（邮箱白名单）
# 完成后本脚本据 ~/.cloudflared/config.yml 拉起隧道（幂等）。
#
# 用法：
#   bash MediTriage/infra/start_cloudflare_tunnel.sh           # 启动（幂等）
#   bash MediTriage/infra/start_cloudflare_tunnel.sh stop      # 停止
#   bash MediTriage/infra/start_cloudflare_tunnel.sh status    # 状态
#   bash MediTriage/infra/start_cloudflare_tunnel.sh quick     # 临时隧道（无需账号/域名，
#                                                            #   随机 *.trycloudflare.com，无 Access 门，仅自测用）
set -euo pipefail

CFD="${CLOUDFLARED:-$HOME/.local/bin/cloudflared}"
CONFIG="${CF_TUNNEL_CONFIG:-$HOME/.cloudflared/config.yml}"
LOCAL_URL="${CF_LOCAL_URL:-http://127.0.0.1:8080}"
PID_FILE="/tmp/medix_cf_tunnel.pid"
LOG="/tmp/medix_cf_tunnel.log"

command -v "$CFD" >/dev/null 2>&1 || { echo "ERROR: 找不到 cloudflared（$CFD）"; exit 1; }

stop_tunnel() {
  [ -f "$PID_FILE" ] && { kill "$(cat "$PID_FILE")" 2>/dev/null || true; rm -f "$PID_FILE"; }
  pkill -f "cloudflared tunnel" 2>/dev/null || true
}

case "${1:-start}" in
  start)
    # 先确认本地演示页活着，避免发布一个 502 的隧道
    curl -sf -m 5 "${LOCAL_URL}/health" >/dev/null \
      || { echo "ERROR: 本地演示页 ${LOCAL_URL} 未就绪，先跑 start_web_demo.sh"; exit 1; }
    [ -f "$CONFIG" ] || { echo "ERROR: 缺 $CONFIG（先完成本脚本头部注释的前置 1-5）"; exit 1; }
    stop_tunnel
    setsid "$CFD" tunnel --config "$CONFIG" run > "$LOG" 2>&1 &
    echo $! > "$PID_FILE"; sleep 3
    if kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "==> tunnel up (pid=$(cat "$PID_FILE"))，公网地址见 Access 策略绑定的子域；日志：$LOG"
    else
      echo "==> 启动失败，看日志：$LOG"; tail -5 "$LOG"; exit 1
    fi
    ;;
  quick)
    curl -sf -m 5 "${LOCAL_URL}/health" >/dev/null \
      || { echo "ERROR: 本地演示页未就绪"; exit 1; }
    echo "==> 临时隧道（无 Access 门，仅自测；随机 URL 见下方日志，Ctrl-C 结束）"
    echo "    安全提示：此模式任何拿到 URL 的人都能用，勿长期开放。"
    exec "$CFD" tunnel --url "$LOCAL_URL"
    ;;
  stop)
    stop_tunnel; echo "==> tunnel stopped"
    ;;
  status)
    if [ -f "$PID_FILE" ] && kill -0 "$(cat "$PID_FILE")" 2>/dev/null; then
      echo "running (pid=$(cat "$PID_FILE"))"
    else
      echo "not running"
    fi
    ;;
  *) echo "Usage: $0 {start|quick|stop|status}"; exit 1;;
esac
