#!/bin/bash
###############################################################################
# start_town.sh - 启动斯坦福小镇环境
# 用途：同时启动frontend和backend服务器
# 依赖：Python 3.9+, Django 2.2, 已配置的utils.py
###############################################################################

set -e  # 遇到错误立即退出

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 项目根目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
EXTERNAL_TOWN="$PROJECT_ROOT/external_town"

echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  Stanford Town Environment Startup${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"

# 1. 检查目录存在性
echo -e "${YELLOW}[1/5] Checking directories...${NC}"
if [ ! -d "$EXTERNAL_TOWN" ]; then
    echo -e "${RED}ERROR: external_town directory not found${NC}"
    exit 1
fi

FRONTEND_DIR="$EXTERNAL_TOWN/environment/frontend_server"
BACKEND_DIR="$EXTERNAL_TOWN/reverie/backend_server"

if [ ! -f "$FRONTEND_DIR/manage.py" ]; then
    echo -e "${RED}ERROR: manage.py not found in frontend_server${NC}"
    exit 1
fi

if [ ! -f "$BACKEND_DIR/reverie.py" ]; then
    echo -e "${RED}ERROR: reverie.py not found in backend_server${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Directories validated${NC}"

# 2. 检查依赖
echo -e "${YELLOW}[2/5] Checking dependencies...${NC}"
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}ERROR: python3 not found${NC}"
    exit 1
fi

# 检查Django是否安装
if ! python3 -c "import django" 2>/dev/null; then
    echo -e "${RED}ERROR: Django not installed. Run: pip install -r requirements.txt${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Dependencies OK${NC}"

# 3. 检查utils.py配置
echo -e "${YELLOW}[3/5] Checking utils.py configuration...${NC}"
if [ ! -f "$BACKEND_DIR/utils.py" ]; then
    echo -e "${RED}ERROR: utils.py not found. Please create it first.${NC}"
    exit 1
fi

# 检查是否配置了OpenAI key（可选警告）
if grep -q 'openai_api_key = ""' "$BACKEND_DIR/utils.py"; then
    echo -e "${YELLOW}WARNING: OpenAI API key is empty in utils.py${NC}"
    echo -e "${YELLOW}You can add it later for LLM interactions${NC}"
fi

echo -e "${GREEN}✓ utils.py exists${NC}"

# 4. 启动Frontend Server
echo -e "${YELLOW}[4/5] Starting Frontend Server (Django)...${NC}"
cd "$FRONTEND_DIR"

# 检查端口8000是否被占用
if lsof -Pi :8000 -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}WARNING: Port 8000 already in use${NC}"
    read -p "Kill existing process? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        lsof -ti:8000 | xargs kill -9 2>/dev/null || true
        sleep 2
    else
        echo -e "${RED}Aborted${NC}"
        exit 1
    fi
fi

# 后台启动Django服务器
nohup python3 manage.py runserver > "$PROJECT_ROOT/logs/frontend.log" 2>&1 &
FRONTEND_PID=$!

echo -e "${GREEN}✓ Frontend server started (PID: $FRONTEND_PID)${NC}"
echo -e "  Log: $PROJECT_ROOT/logs/frontend.log"
sleep 3  # 等待服务器启动

# 验证frontend是否成功启动
if curl -s http://localhost:8000/ > /dev/null 2>&1; then
    echo -e "${GREEN}✓ Frontend server responding at http://localhost:8000${NC}"
else
    echo -e "${RED}ERROR: Frontend server failed to start${NC}"
    kill $FRONTEND_PID 2>/dev/null || true
    exit 1
fi

# 5. 记录PID
echo -e "${YELLOW}[5/5] Recording process info...${NC}"
echo "$FRONTEND_PID" > "$PROJECT_ROOT/logs/frontend.pid"

echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo -e "${GREEN}✓ Town environment started successfully${NC}"
echo -e "${GREEN}═══════════════════════════════════════════════════${NC}"
echo ""
echo "Frontend Server: http://localhost:8000"
echo "Frontend PID: $FRONTEND_PID"
echo ""
echo "To stop the server, run:"
echo "  kill $FRONTEND_PID"
echo "  # or use: bash scripts/stop_town.sh"
echo ""
echo "Next steps:"
echo "  1. Navigate to http://localhost:8000/ to verify"
echo "  2. Use TownBridge API to interact programmatically"
