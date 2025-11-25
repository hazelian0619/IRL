#!/bin/bash
###############################################################################
# stop_town.sh - 停止斯坦福小镇环境
###############################################################################

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
PID_FILE="$PROJECT_ROOT/logs/frontend.pid"

GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}Stopping Town environment...${NC}"

if [ -f "$PID_FILE" ]; then
    PID=$(cat "$PID_FILE")
    if ps -p $PID > /dev/null 2>&1; then
        kill $PID
        echo -e "${GREEN}✓ Stopped frontend server (PID: $PID)${NC}"
    else
        echo -e "${RED}Process $PID not found (already stopped?)${NC}"
    fi
    rm "$PID_FILE"
else
    echo -e "${RED}No PID file found${NC}"
    # 尝试通过端口查找
    if lsof -ti:8000 >/dev/null 2>&1; then
        lsof -ti:8000 | xargs kill -9
        echo -e "${GREEN}✓ Killed process on port 8000${NC}"
    fi
fi

echo -e "${GREEN}Done${NC}"
