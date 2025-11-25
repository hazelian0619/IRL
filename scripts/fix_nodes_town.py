import json
import os

# Alice路径
bootstrap_path = 'external_town/environment/frontend_server/storage/alice_bfi_test/personas/Alice Chen/bootstrap_memory'
assoc_path = os.path.join(bootstrap_path, 'associative_memory')
os.makedirs(assoc_path, exist_ok=True)

# initial_memories (6条，基于你的数据)
initial_memories = [
    {"subject": "Alice Chen", "predicate": "born", "object": "Seattle", "description": "Alice was born in Seattle on March 15, 1995.", "embedding_key": "Alice Chen born Seattle 1995", "type": "event", "created": "1995-03-15 10:30:00"},
    {"subject": "Alice Chen", "predicate": "graduated", "object": "University of Washington", "description": "Alice graduated with Environmental Science degree.", "embedding_key": "Alice graduated University Washington environmental", "type": "event", "created": "2017-06-01 10:30:00"},
    {"subject": "Alice Chen", "predicate": "started", "object": "environmental consultant", "description": "Alice started career as environmental consultant.", "embedding_key": "Alice started career consultant environmental", "type": "event", "created": "2018-01-01 10:30:00"},
    {"subject": "Alice Chen", "predicate": "enjoys", "object": "hiking reading sci-fi", "description": "Alice enjoys hiking and reading sci-fi novels.", "embedding_key": "Alice enjoys hiking reading sci-fi", "type": "event", "created": "2023-02-01 10:30:00"},
    {"subject": "Alice Chen", "predicate": "planning", "object": "mountains trip", "description": "Alice is planning a weekend trip to the mountains.", "embedding_key": "Alice planning trip mountains", "type": "event", "created": "2023-02-13 10:30:00"},
    {"subject": "Alice Chen", "predicate": "values", "object": "creativity openness", "description": "Alice values creativity and openness in life.", "embedding_key": "Alice values creativity openness", "type": "trait", "created": "2023-02-13 10:30:00"}
]

# 生成nodes (匹配Town格式：键数字字符串，node_count=1, type_count=str, depth=0等[web:31])
nodes = {}
for i, mem in enumerate(initial_memories, start=1):
    key = str(i)
    nodes[key] = {
        "node_count": 1,
        "type_count": "1",
        "type": [mem["type"]],
        "depth": 0,
        "created": mem["created"],
        "expiration": None,
        "subject": mem["subject"],
        "predicate": mem["predicate"],
        "object": mem["object"],
        "description": mem["description"],
        "embedding_key": mem["embedding_key"][:50]  # 截断如输出
    }

# 保存
nodes_path = os.path.join(assoc_path, 'nodes.json')
with open(nodes_path, 'w') as f:
    json.dump(nodes, f, indent=2)

# 空embeddings/kw_strength
with open(os.path.join(bootstrap_path, 'embeddings.json'), 'w') as f:
    json.dump({}, f)
with open(os.path.join(bootstrap_path, 'kw_strength.json'), 'w') as f:
    json.dump({}, f)

print(f"✓ 修正nodes.json: {len(nodes)}节点，格式匹配Town")
print("Sample '1':", nodes['1'])
