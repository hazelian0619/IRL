"""
Alice Chen初始化脚本

功能：
1. 基于base simulation创建新的simulation（包含Alice）
2. 注入Alice的人格参数、传记、初始记忆
3. 生成完整的bootstrap_memory结构
4. 验证注入成功

用法：
    python scripts/init_alice.py
"""

import json
import shutil
from pathlib import Path
from datetime import datetime


class AliceInitializer:
    """Alice Chen注入到Town环境"""

    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.town_storage = self.project_root / "external_town" / "environment" / "frontend_server" / "storage"

        # 加载Alice的配置数据
        self.load_alice_data()

    def load_alice_data(self):
        """加载Alice的预设数据"""
        # 预设人格参数
        preset_path = self.project_root / "data" / "personas" / "preset_personality.json"
        with open(preset_path, 'r') as f:
            self.preset = json.load(f)

        # 传记Prompt
        bio_path = self.project_root / "data" / "personas" / "alice_biography_prompt.txt"
        with open(bio_path, 'r', encoding='utf-8') as f:
            bio_content = f.read()

        # 从传记中提取关键字段
        self.extract_biography_fields(bio_content)

        # 初始记忆
        memory_path = self.project_root / "data" / "personas" / "initial_memory.json"
        with open(memory_path, 'r') as f:
            self.initial_memory = json.load(f)

        print("✓ Alice数据加载完成")

    def extract_biography_fields(self, bio_content: str):
        """从传记中提取字段"""
        # 这里简化处理，直接使用预定义的文本
        self.innate = "curious, sociable, optimistic, creative, adaptable"

        self.learned = "Alice Chen is a coffee shop owner and freelance illustrator who loves bringing people together through coffee and art. She is always looking for ways to make her cafe a creative hub where people can discover new experiences and connect authentically."

        self.currently = "Alice is currently planning an 'Art Wall' project at her cafe to showcase local artists monthly, while also working on illustrations for a children's book. She is excited to make her cafe a cultural gathering spot in the town."

        self.lifestyle = "Alice typically wakes up around 7am, goes for a morning run, and opens her cafe by 9am. She closes around 6pm but stays flexible depending on inspiration or events. She spends evenings illustrating or reading, and goes to bed around 11pm."

        self.daily_plan_req = "Alice opens her cafe at 9am and serves customers until 6pm. She enjoys chatting with regulars and experimenting with new coffee recipes. In the evenings, she works on illustration projects."

    def create_new_simulation(self, sim_name: str = None) -> Path:
        """
        创建新的simulation（基于base）

        Args:
            sim_name: simulation名称，默认为 alice_experiment_YYYYMMDD

        Returns:
            新simulation的路径
        """
        if sim_name is None:
            sim_name = f"alice_experiment_{datetime.now().strftime('%Y%m%d')}"

        # 源：base simulation
        base_sim = self.town_storage / "base_the_ville_isabella_maria_klaus"
        if not base_sim.exists():
            raise FileNotFoundError(f"Base simulation not found: {base_sim}")

        # 目标：新simulation
        new_sim = self.town_storage / sim_name

        if new_sim.exists():
            print(f"⚠️  Simulation {sim_name} already exists. Using existing one.")
            return new_sim

        # 复制base simulation
        print(f"Creating new simulation: {sim_name}")
        shutil.copytree(base_sim, new_sim)

        # 更新meta.json
        meta_path = new_sim / "reverie" / "meta.json"
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        meta['fork_sim_code'] = "base_the_ville_isabella_maria_klaus"
        meta['sim_code'] = sim_name

        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

        print(f"✓ New simulation created: {new_sim}")
        return new_sim

    def create_alice_scratch(self) -> dict:
        """创建Alice的scratch.json"""
        scratch = {
            "vision_r": 8,
            "att_bandwidth": 8,
            "retention": 8,
            "curr_time": None,
            "curr_tile": None,
            "daily_plan_req": self.daily_plan_req,
            "name": "Alice Chen",
            "first_name": "Alice",
            "last_name": "Chen",
            "age": 28,
            "innate": self.innate,
            "learned": self.learned,
            "currently": self.currently,
            "lifestyle": self.lifestyle,
            "living_area": "the Ville:Alice Chen's apartment:main room",
            "concept_forget": 100,
            "daily_reflection_time": 180,
            "daily_reflection_size": 5,
            "overlap_reflect_th": 4,
            "kw_strg_event_reflect_th": 10,
            "kw_strg_thought_reflect_th": 9,
            "recency_w": 1,
            "relevance_w": 1,
            "importance_w": 1,
            "recency_decay": 0.995,
            "importance_trigger_max": 150,
            "importance_trigger_curr": 150,
            "importance_ele_n": 0,
            "thought_count": 5,
            "daily_req": [],
            "f_daily_schedule": [],
            "f_daily_schedule_hourly_org": [],
            "act_address": None,
            "act_start_time": None,
            "act_duration": None,
            "act_description": None,
            "act_pronunciatio": None,
            "act_event": ["Alice Chen", None, None],
            "act_obj_description": None,
            "act_obj_pronunciatio": None,
            "act_obj_event": [None, None, None],
            "chatting_with": None,
            "chat": None,
            "chatting_with_buffer": {},
            "chatting_end_time": None,
            "act_path_set": False,
            "planned_path": []
        }

        return scratch

    def create_alice_memories(self) -> dict:
        """创建Alice的associative_memory节点"""
        nodes = {}

        for i, mem in enumerate(self.initial_memory['initial_memories'], start=1):
            node_id = str(i)
            nodes[node_id] = {
                "node_count": i,
                "type_count": "1",
                "type": ["event"],
                "depth": 0,
                "created": mem['created'],
                "expiration": None,
                "subject": "Alice Chen",
                "predicate": "experienced",
                "object": mem['description'].split('.')[0],  # 第一句作为object
                "description": mem['description'],
                "embedding_key": f"Alice Chen {mem['description'][:50]}",
                "poignancy": mem['importance'],
                "keywords": mem['keywords'],
                "filling": []
            }

        return nodes

    def create_alice_spatial_memory(self) -> dict:
        """创建Alice的spatial_memory（使用Town的默认空间）"""
        # 这里使用一个简化的spatial memory，实际应该与Town的地图匹配
        spatial_memory = {
            "the Ville": {
                "Alice Chen's apartment": {
                    "main room": ["bed", "desk", "easel", "bookshelf", "coffee maker"]
                },
                "Alice's Cafe": {
                    "main area": ["counter", "espresso machine", "tables", "chairs", "art wall"],
                    "kitchen": ["sink", "refrigerator", "storage"]
                }
            }
        }

        return spatial_memory

    def inject_alice(self, sim_path: Path):
        """将Alice注入到simulation中"""
        # 创建Alice的personas文件夹
        alice_folder = sim_path / "personas" / "Alice Chen"
        alice_folder.mkdir(parents=True, exist_ok=True)

        # 创建bootstrap_memory文件夹
        bootstrap = alice_folder / "bootstrap_memory"
        bootstrap.mkdir(exist_ok=True)

        # 1. 生成scratch.json
        scratch = self.create_alice_scratch()
        with open(bootstrap / "scratch.json", 'w') as f:
            json.dump(scratch, f, indent=2)
        print("✓ scratch.json created")

        # 2. 生成spatial_memory.json
        spatial_memory = self.create_alice_spatial_memory()
        with open(bootstrap / "spatial_memory.json", 'w') as f:
            json.dump(spatial_memory, f, indent=2)
        print("✓ spatial_memory.json created")

        # 3. 创建associative_memory文件夹
        assoc_mem = bootstrap / "associative_memory"
        assoc_mem.mkdir(exist_ok=True)

        # nodes.json
        nodes = self.create_alice_memories()
        with open(assoc_mem / "nodes.json", 'w') as f:
            json.dump(nodes, f, indent=2)
        print("✓ nodes.json created")

        # embeddings.json (初始为空)
        with open(assoc_mem / "embeddings.json", 'w') as f:
            json.dump({}, f)

        # kw_strength.json (初始为空)
        with open(assoc_mem / "kw_strength.json", 'w') as f:
            json.dump({"kw_strength": {}}, f)

        print(f"✓ Alice Chen injected into: {alice_folder}")

    def verify_injection(self, sim_path: Path) -> bool:
        """验证Alice注入成功"""
        alice_folder = sim_path / "personas" / "Alice Chen"

        checks = {
            "Alice folder exists": alice_folder.exists(),
            "scratch.json exists": (alice_folder / "bootstrap_memory" / "scratch.json").exists(),
            "spatial_memory.json exists": (alice_folder / "bootstrap_memory" / "spatial_memory.json").exists(),
            "nodes.json exists": (alice_folder / "bootstrap_memory" / "associative_memory" / "nodes.json").exists()
        }

        print("\n【验证结果】")
        for check, passed in checks.items():
            print(f"  {'✅' if passed else '❌'} {check}")

        return all(checks.values())

    def run(self, sim_name: str = None) -> str:
        """
        完整执行初始化流程

        Returns:
            新simulation的名称
        """
        print("="*60)
        print("Alice Chen 初始化流程")
        print("="*60)

        # 1. 创建新simulation
        sim_path = self.create_new_simulation(sim_name)

        # 2. 注入Alice
        self.inject_alice(sim_path)

        # 3. 验证
        success = self.verify_injection(sim_path)

        if success:
            print("\n✅ Alice初始化完成！")
            print(f"Simulation: {sim_path.name}")

            # 记录到日志
            log_path = self.project_root / "logs" / "step1.2_artifacts" / "alice_injection_log.txt"
            with open(log_path, 'w') as f:
                f.write(f"Alice Chen Injection Log\n")
                f.write(f"Timestamp: {datetime.now().isoformat()}\n")
                f.write(f"Simulation: {sim_path.name}\n")
                f.write(f"Preset Parameters: {self.preset['big_five_parameters']}\n")
                f.write(f"Status: Success\n")

            return sim_path.name
        else:
            print("\n❌ 初始化失败，请检查错误")
            return None


if __name__ == "__main__":
    initializer = AliceInitializer()
    sim_name = initializer.run()

    if sim_name:
        print(f"\n下一步:")
        print(f"  1. 启动Town Backend with simulation: {sim_name}")
        print(f"  2. 运行BFI-44前测")
