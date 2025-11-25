"""
测试脚本：验证TownBridge接口功能

测试内容：
1. 检查服务器健康状态
2. 列出可用simulations
3. 读取simulation meta信息
4. 获取agent列表
5. 读取agent的scratch和memory数据
6. 读取环境状态

运行方式：
    python scripts/test_town_bridge.py
"""

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from bridges.town_bridge import TownBridge


def print_section(title):
    """打印分隔线"""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def test_town_bridge():
    """执行完整测试"""

    print_section("TownBridge 接口测试")

    # 初始化
    bridge = TownBridge()
    print(f"\n✓ TownBridge initialized")
    print(f"  {bridge}")

    # 测试1: 检查服务器状态
    print_section("Test 1: 服务器健康检查")
    is_healthy = bridge.check_server_health()
    if is_healthy:
        print("✅ Frontend server is ONLINE")
    else:
        print("⚠️  Frontend server is OFFLINE")
        print("   Tip: Run 'bash scripts/start_town.sh' to start the server")

    # 测试2: 列出所有simulations
    print_section("Test 2: 列出可用Simulations")
    simulations = bridge.list_simulations()
    print(f"Found {len(simulations)} simulation(s):")
    for i, sim in enumerate(simulations, 1):
        print(f"  {i}. {sim}")

    if not simulations:
        print("❌ No simulations found")
        print("   Check storage path:", bridge.storage_path)
        return False

    # 选择一个simulation进行测试
    test_sim = None
    for sim in simulations:
        if "isabella" in sim.lower():
            test_sim = sim
            break
    if test_sim is None:
        test_sim = simulations[0]

    print(f"\n➡️  Using simulation: {test_sim}")

    # 测试3: 读取meta信息
    print_section("Test 3: 读取Simulation Meta")
    meta = bridge.get_simulation_meta(test_sim)
    if meta:
        print("✅ Meta data loaded successfully")
        print(f"  Start date: {meta.get('start_date', 'N/A')}")
        print(f"  Current time: {meta.get('curr_time', 'N/A')}")
        print(f"  Seconds per step: {meta.get('sec_per_step', 'N/A')}")
        print(f"  Maze name: {meta.get('maze_name', 'N/A')}")
    else:
        print("❌ Failed to load meta data")
        return False

    # 测试4: 获取agent列表
    print_section("Test 4: 获取Agent列表")
    agents = bridge.get_agent_list(test_sim)
    print(f"Found {len(agents)} agent(s):")
    for i, agent in enumerate(agents, 1):
        print(f"  {i}. {agent}")

    if not agents:
        print("❌ No agents found")
        return False

    test_agent = agents[0]
    print(f"\n➡️  Using agent: {test_agent}")

    # 测试5: 读取agent scratch
    print_section("Test 5: 读取Agent Scratch数据")
    scratch = bridge.get_agent_scratch(test_sim, test_agent)
    if scratch:
        print("✅ Scratch data loaded successfully")
        print(f"  Name: {scratch.get('name', 'N/A')}")
        print(f"  Age: {scratch.get('age', 'N/A')}")
        print(f"  Current time: {scratch.get('curr_time', 'N/A')}")
        print(f"  Current tile: {scratch.get('curr_tile', 'N/A')}")
        print(f"  Innate traits: {scratch.get('innate', 'N/A')[:80]}...")
    else:
        print("❌ Failed to load scratch data")
        return False

    # 测试6: 读取agent memory
    print_section("Test 6: 读取Agent Memory节点")
    memory = bridge.get_agent_memory(test_sim, test_agent)
    if memory:
        print("✅ Memory data loaded successfully")
        print(f"  Total memory nodes: {len(memory)}")
        if len(memory) > 0:
            print(f"  Sample memory (first 3):")
            for i, (node_id, node_data) in enumerate(list(memory.items())[:3]):
                created = node_data.get('created', 'N/A')
                description = node_data.get('description', 'N/A')
                print(f"    {i+1}. [{created}] {description[:60]}...")
    else:
        print("❌ Failed to load memory data")
        return False

    # 测试7: 读取环境状态（step 0）
    print_section("Test 7: 读取环境状态")
    env_state = bridge.get_environment_state(test_sim, 0)
    if env_state:
        print("✅ Environment state loaded (step 0)")
        print(f"  Keys: {list(env_state.keys())[:5]}")
    else:
        print("⚠️  Environment state not found (step 0)")
        print("   This is OK if simulation hasn't been run yet")

    # 测试8: HTTP API（如果服务器在线）
    if is_healthy:
        print_section("Test 8: HTTP API测试")
        try:
            state = bridge.get_agent_state_via_http(test_sim, 0, test_agent)
            if state:
                print("✅ HTTP API working")
                print(f"  Response type: {type(state)}")
            else:
                print("⚠️  HTTP API returned no data")
        except Exception as e:
            print(f"❌ HTTP API error: {e}")

    # 总结
    print_section("测试总结")
    print("✅ TownBridge基础功能测试通过")
    print("\n下一步：")
    print("  1. 确保frontend server运行: bash scripts/start_town.sh")
    print("  2. 在浏览器访问: http://localhost:8000")
    print("  3. 使用TownBridge进行数据采集")

    return True


if __name__ == "__main__":
    try:
        success = test_town_bridge()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
