#!/usr/bin/env python3
"""
修复Alice的nodes.json结构错误

问题：
1. 多了一层 "node_details" 包装
2. type字段是数组而非字符串
3. type_count是字符串而非数字

作者: Claude Code
日期: 2025-11-10
"""

import json
from pathlib import Path
import sys

def fix_nodes_structure(nodes_path: Path) -> bool:
    """修复nodes.json的结构"""

    # 备份原文件
    backup_path = nodes_path.with_suffix('.json.backup')

    try:
        # 读取原文件
        with open(nodes_path, 'r', encoding='utf-8') as f:
            nodes_data = json.load(f)

        # 备份
        with open(backup_path, 'w', encoding='utf-8') as f:
            json.dump(nodes_data, f, indent=2)
        print(f"✓ 备份保存: {backup_path}")

        # 修复结构
        fixed_nodes = {}
        for node_id, node_value in nodes_data.items():
            # 检查是否有多余的node_details包装
            if "node_details" in node_value:
                # 去掉包装层
                node_content = node_value["node_details"]
            else:
                node_content = node_value

            # 修复type字段（从数组变字符串）
            if isinstance(node_content.get("type"), list):
                node_content["type"] = node_content["type"][0]

            # 修复type_count字段（从字符串变数字）
            if isinstance(node_content.get("type_count"), str):
                try:
                    node_content["type_count"] = int(node_content["type_count"])
                except ValueError:
                    node_content["type_count"] = 1

            # 确保必需字段存在
            required_fields = {
                "node_count": node_content.get("node_count", 1),
                "type_count": node_content.get("type_count", 1),
                "type": node_content.get("type", "event"),
                "depth": node_content.get("depth", 0),
                "created": node_content.get("created", "2023-01-01 00:00:00"),
                "expiration": node_content.get("expiration"),
                "subject": node_content.get("subject", "Alice Chen"),
                "predicate": node_content.get("predicate", "is"),
                "object": node_content.get("object", "unknown"),
                "description": node_content.get("description", ""),
                "embedding_key": node_content.get("embedding_key", ""),
                "poignancy": node_content.get("poignancy", 5),
                "keywords": node_content.get("keywords", []),
                "filling": node_content.get("filling", [])
            }

            fixed_nodes[node_id] = required_fields

        # 写入修复后的文件
        with open(nodes_path, 'w', encoding='utf-8') as f:
            json.dump(fixed_nodes, f, indent=2, ensure_ascii=False)

        print(f"✓ 修复完成: {nodes_path}")
        print(f"  - 修复了 {len(fixed_nodes)} 个节点")
        return True

    except Exception as e:
        print(f"❌ 修复失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def fix_embeddings(embeddings_path: Path, nodes_data: dict) -> bool:
    """确保embeddings.json的键与nodes匹配"""
    try:
        # 读取embeddings
        with open(embeddings_path, 'r', encoding='utf-8') as f:
            embeddings = json.load(f)

        # 获取所有embedding_key
        all_keys = set()
        for node_data in nodes_data.values():
            if "embedding_key" in node_data:
                all_keys.add(node_data["embedding_key"])

        # 确保所有key都存在（即使是空数组）
        for key in all_keys:
            if key not in embeddings:
                embeddings[key] = []

        # 写回
        with open(embeddings_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f, indent=2, ensure_ascii=False)

        print(f"✓ embeddings.json已同步 ({len(all_keys)} 个键)")
        return True

    except Exception as e:
        print(f"⚠️  embeddings同步失败: {e}")
        return False


def main():
    """主函数"""
    print("="*60)
    print("Alice nodes.json 结构修复工具")
    print("="*60)

    # 查找所有Alice的nodes.json
    base_path = Path(__file__).parent.parent / "external_town" / "environment" / "frontend_server" / "storage"

    alice_paths = []
    for sim_dir in base_path.iterdir():
        if sim_dir.is_dir():
            alice_node_path = sim_dir / "personas" / "Alice Chen" / "bootstrap_memory" / "associative_memory" / "nodes.json"
            if alice_node_path.exists():
                alice_paths.append(alice_node_path)

    if not alice_paths:
        print("❌ 未找到Alice的nodes.json文件")
        return 1

    print(f"\n找到 {len(alice_paths)} 个Alice simulation:")
    for path in alice_paths:
        print(f"  - {path.parent.parent.parent.parent.name}")

    # 修复所有找到的文件
    success_count = 0
    for nodes_path in alice_paths:
        print(f"\n处理: {nodes_path.parent.parent.parent.parent.name}")

        # 修复nodes.json
        if fix_nodes_structure(nodes_path):
            success_count += 1

            # 读取修复后的nodes
            with open(nodes_path, 'r') as f:
                fixed_nodes = json.load(f)

            # 同步embeddings.json
            embeddings_path = nodes_path.parent / "embeddings.json"
            if embeddings_path.exists():
                fix_embeddings(embeddings_path, fixed_nodes)

    print(f"\n{'='*60}")
    print(f"修复完成: {success_count}/{len(alice_paths)} 成功")
    print(f"{'='*60}")

    return 0 if success_count == len(alice_paths) else 1


if __name__ == "__main__":
    sys.exit(main())
