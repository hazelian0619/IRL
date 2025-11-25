#!/bin/bash
###############################################################################
# start_town_backend.sh - 启动Stanford Town backend (Reverie server)
#
# 用法:
#   bash scripts/start_town_backend.sh [--python PATH] [--no-log]
#
# 功能:
#   1. 校验external_town/reverie/backend_server是否就绪
#   2. 检查python版本与utils.py配置
#   3. 启动Reverie backend并将日志写入logs/backend.log (默认)
###############################################################################

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
BACKEND_DIR="$PROJECT_ROOT/external_town/reverie/backend_server"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/backend.log"
PYTHON_BIN="python3"
ENABLE_LOG=1

usage() {
    cat <<'USAGE'
Usage: bash scripts/start_town_backend.sh [options]

Options:
  --python PATH   使用指定的python解释器(默认: python3)
  --no-log        不写入logs/backend.log
  -h, --help      显示帮助

说明:
  该脚本会切换到external_town/reverie/backend_server目录并执行
  `python reverie.py`。请在另一个终端保证scripts/start_town.sh已启动
  frontend server，然后按照提示输入forked simulation名称。
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --python)
            shift
            [[ $# -gt 0 ]] || { echo "ERROR: --python 需要参数"; exit 1; }
            PYTHON_BIN="$1"
            ;;
        --no-log)
            ENABLE_LOG=0
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
    shift
done

if [[ ! -d "$BACKEND_DIR" ]]; then
    echo "ERROR: backend directory not found: $BACKEND_DIR" >&2
    exit 1
fi

if [[ ! -f "$BACKEND_DIR/reverie.py" ]]; then
    echo "ERROR: reverie.py missing in $BACKEND_DIR" >&2
    exit 1
fi

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "ERROR: $PYTHON_BIN not found" >&2
    exit 1
fi

if [[ ! -f "$BACKEND_DIR/utils.py" ]]; then
    cat <<'WARN' >&2
WARN: utils.py not found in backend_server.
Please create it (refer to external_town/README.md) before启动backend。
WARN
    exit 1
fi

mkdir -p "$LOG_DIR"

echo "════════════════════════════════════════════════════════════════"
echo "  Stanford Town Reverie Backend"
echo "════════════════════════════════════════════════════════════════"
echo "Project root : $PROJECT_ROOT"
echo "Backend dir  : $BACKEND_DIR"
echo "Python       : $PYTHON_BIN"
if [[ $ENABLE_LOG -eq 1 ]]; then
    echo "Log file     : $LOG_FILE"
else
    echo "Log file     : (disabled)"
fi
echo "────────────── Startup instructions ──────────────"
echo "1. Frontend server必须先通过 scripts/start_town.sh 运行"
echo "2. 脚本将启动reverie.py 并进入交互式提示"
echo "3. 在提示中依次输入 forked simulation 名称、新的 simulation 名称"
echo "4. 之后可使用 run <steps> / fin / exit 等命令"
echo "════════════════════════════════════════════════════════════════"

cd "$BACKEND_DIR"

if [[ $ENABLE_LOG -eq 1 ]]; then
    echo "[start_town_backend] Logging to $LOG_FILE"
    "$PYTHON_BIN" reverie.py | tee "$LOG_FILE"
else
    "$PYTHON_BIN" reverie.py
fi
