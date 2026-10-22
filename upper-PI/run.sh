#!/usr/bin/env bash
# 一键启动上位机。用法：
#   ./run.sh                整车运行
#   ./run.sh --plan-only    只算路线，不碰硬件
#   FULL_MAP_LINE_SPEED=6 ./run.sh   临时覆盖某个参数
set -euo pipefail
cd "$(dirname "$0")"

# 加载 .env（如果存在）
if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

# src/ 是平铺模块目录，主程序内部用 from X import 直接引用
export PYTHONPATH="$PWD/src:${PYTHONPATH:-}"

exec python3 src/full_map_walk_test.py "$@"
