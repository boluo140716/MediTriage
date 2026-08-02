#!/usr/bin/env bash
# clean_runtime.sh — 一键清理 MediTriage 运行期产物（可安全删除、随时可重生）
#
# 清理范围（全部 .gitignore、不影响代码与数据）：
#   - 会话缓存.tmp/      会话总结缓存（paths.CACHE_DIR）
#   - log/               运行日志（paths.LOG_DIR；保留目录骨架）
#   - **/__pycache__     Python 字节码
#   - .pytest_cache/     pytest 缓存
# 默认不动 *.egg-info（删了需重新 pip install -e；加 --deep 才清，见下）。
#
# 用法：
#   bash MediTriage/infra/clean_runtime.sh           # 清运行期产物
#   bash MediTriage/infra/clean_runtime.sh --deep    # 额外清 egg-info（需随后重装包）
#   bash MediTriage/infra/clean_runtime.sh --dry-run # 只列出将删什么，不动手
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"      # MediTriage/infra
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"                  # 仓库根（含 config.py）

DRY=0; DEEP=0
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    --deep)    DEEP=1 ;;
    *) echo "未知参数: $a"; exit 1 ;;
  esac
done

# 部分运行期产物为容器内 root 所属（host 无 sudo），经容器删除
run() {  # run <描述> <在容器内执行的命令>
  echo "==> $1"
  [ "$DRY" = 1 ] && { echo "    [dry-run] $2"; return; }
  docker exec medix-fix bash -lc "$2" 2>/dev/null || true
}

CR="${MEDIX_CONTAINER_ROOT:-/workspace}"   # 仓库在容器内的挂载点

run "会话缓存.tmp/（会话总结缓存）" \
    "rm -rf '${CR}/会话缓存.tmp' '${CR}/MediTriage/agent/会话缓存.tmp'"
run "log/ 内容（保留目录）" \
    "find '${CR}/log' -mindepth 1 -delete 2>/dev/null || true"
run "**/__pycache__" \
    "find '${CR}/MediTriage' -type d -name __pycache__ -prune -exec rm -rf {} +"
run ".pytest_cache" \
    "find '${CR}/MediTriage' -type d -name .pytest_cache -prune -exec rm -rf {} +"

if [ "$DEEP" = 1 ]; then
  run "*.egg-info（删后需重新 pip install -e .）" \
      "find '${CR}/MediTriage' -type d -name '*.egg-info' -prune -exec rm -rf {} +"
  echo "    提示：egg-info 已清，运行测试前请在 MediTriage/agent 下重新 'pip install -e .'"
fi

if [ "$DRY" = 1 ]; then
  echo "==> done (dry-run，未实际删除)"
else
  echo "==> done"
fi
