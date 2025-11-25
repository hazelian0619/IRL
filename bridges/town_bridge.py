"""
TownBridge - 斯坦福小镇环境的Python接口

功能：
1. 启动/停止Town环境
2. 获取Agent状态（位置、记忆、行为）
3. 读取simulation数据
4. 触发事件（预留接口）

依赖：
- Town frontend server运行在 http://localhost:8000
- Storage目录包含simulation数据
"""

import os
import json
import requests
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Optional


class TownBridge:
    """斯坦福小镇环境桥接接口"""

    def __init__(self,
                 frontend_url: str = "http://localhost:8000",
                 storage_path: Optional[str] = None):
        """
        初始化TownBridge

        Args:
            frontend_url: Frontend服务器URL
            storage_path: Storage目录路径（如果不提供，自动推断）
        """
        self.frontend_url = frontend_url

        # 自动推断storage路径
        if storage_path is None:
            script_dir = Path(__file__).parent.parent
            self.storage_path = script_dir / "external_town" / "environment" / "frontend_server" / "storage"
        else:
            self.storage_path = Path(storage_path)

        self.compressed_storage_path = self.storage_path.parent / "compressed_storage"
        self._frontend_pid = None

    def check_server_health(self) -> bool:
        """检查frontend服务器是否运行"""
        try:
            response = requests.get(self.frontend_url, timeout=2)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    def start_frontend_server(self) -> bool:
        """
        启动frontend服务器

        Returns:
            bool: 是否成功启动
        """
        script_path = Path(__file__).parent.parent / "scripts" / "start_town.sh"

        if not script_path.exists():
            raise FileNotFoundError(f"启动脚本不存在: {script_path}")

        print("🚀 Starting Town frontend server...")
        result = subprocess.run(["bash", str(script_path)],
                              capture_output=True, text=True)

        if result.returncode != 0:
            print(f"❌ 启动失败:\n{result.stderr}")
            return False

        # 等待服务器启动
        for i in range(10):
            time.sleep(1)
            if self.check_server_health():
                print("✅ Frontend server started successfully")
                return True

        print("❌ Server failed to respond after 10 seconds")
        return False

    def stop_frontend_server(self) -> bool:
        """停止frontend服务器"""
        script_path = Path(__file__).parent.parent / "scripts" / "stop_town.sh"

        if not script_path.exists():
            raise FileNotFoundError(f"停止脚本不存在: {script_path}")

        print("🛑 Stopping Town frontend server...")
        result = subprocess.run(["bash", str(script_path)],
                              capture_output=True, text=True)

        if result.returncode == 0:
            print("✅ Frontend server stopped")
            return True
        else:
            print(f"❌ 停止失败:\n{result.stderr}")
            return False

    def list_simulations(self) -> List[str]:
        """列出所有可用的simulation"""
        if not self.storage_path.exists():
            return []

        return [d.name for d in self.storage_path.iterdir() if d.is_dir()]

    def get_simulation_meta(self, sim_code: str) -> Optional[Dict]:
        """
        获取simulation的meta信息

        Args:
            sim_code: simulation名称

        Returns:
            meta信息字典，如果不存在返回None
        """
        meta_path = self.storage_path / sim_code / "reverie" / "meta.json"

        if not meta_path.exists():
            return None

        with open(meta_path, 'r') as f:
            return json.load(f)

    def get_agent_list(self, sim_code: str) -> List[str]:
        """
        获取simulation中的所有agent名称

        Args:
            sim_code: simulation名称

        Returns:
            agent名称列表
        """
        personas_path = self.storage_path / sim_code / "personas"

        if not personas_path.exists():
            return []

        return [d.name for d in personas_path.iterdir() if d.is_dir()]

    def get_agent_scratch(self, sim_code: str, agent_name: str) -> Optional[Dict]:
        """
        获取agent的scratch数据（当前状态）

        Args:
            sim_code: simulation名称
            agent_name: agent名称

        Returns:
            scratch数据字典
        """
        scratch_path = (self.storage_path / sim_code / "personas" /
                       agent_name / "bootstrap_memory" / "scratch.json")

        if not scratch_path.exists():
            return None

        with open(scratch_path, 'r') as f:
            return json.load(f)

    def get_agent_memory(self, sim_code: str, agent_name: str) -> Optional[Dict]:
        """
        获取agent的记忆节点

        Args:
            sim_code: simulation名称
            agent_name: agent名称

        Returns:
            记忆节点数据
        """
        nodes_path = (self.storage_path / sim_code / "personas" /
                     agent_name / "bootstrap_memory" / "associative_memory" / "nodes.json")

        if not nodes_path.exists():
            return None

        with open(nodes_path, 'r') as f:
            return json.load(f)

    def get_environment_state(self, sim_code: str, step: int) -> Optional[Dict]:
        """
        获取指定step的环境状态

        Args:
            sim_code: simulation名称
            step: 时间步

        Returns:
            环境状态数据
        """
        env_path = self.storage_path / sim_code / "environment" / f"{step}.json"

        if not env_path.exists():
            return None

        with open(env_path, 'r') as f:
            return json.load(f)

    def get_movement_data(self, sim_code: str, step: int) -> Optional[Dict]:
        """
        获取指定step的移动数据

        Args:
            sim_code: simulation名称
            step: 时间步

        Returns:
            移动数据
        """
        move_path = self.storage_path / sim_code / "movement" / f"{step}.json"

        if not move_path.exists():
            return None

        with open(move_path, 'r') as f:
            return json.load(f)

    def get_agent_state_via_http(self, sim_code: str, step: int,
                                  agent_name: str) -> Optional[Dict]:
        """
        通过HTTP API获取agent状态（需要frontend服务器运行）

        Args:
            sim_code: simulation名称
            step: 时间步
            agent_name: agent名称

        Returns:
            agent状态数据
        """
        if not self.check_server_health():
            raise RuntimeError("Frontend server is not running")

        url = f"{self.frontend_url}/replay_persona_state/{sim_code}/{step}/{agent_name}/"

        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                return response.json()
            else:
                return None
        except requests.exceptions.RequestException as e:
            print(f"HTTP request failed: {e}")
            return None

    def __repr__(self):
        status = "🟢 Online" if self.check_server_health() else "🔴 Offline"
        return f"TownBridge(server={status}, storage={self.storage_path})"


if __name__ == "__main__":
    # 测试代码
    bridge = TownBridge()
    print(bridge)
    print(f"\nAvailable simulations: {bridge.list_simulations()}")
