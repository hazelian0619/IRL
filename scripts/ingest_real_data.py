"""Ingest Town simulation artifacts into a controlled REAL dataset folder.

Usage:
    python scripts/ingest_real_data.py --simulation alice_experiment_20251109 \
        --agent "Alice Chen" --sections personas,environment --output data/real_runs

The script copies the requested sections from
external_town/environment/frontend_server/storage/<simulation> into a
timestamped folder under ``--output`` and records metadata for downstream
auditing.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bridges.town_bridge import TownBridge


def parse_sections(raw: str) -> Iterable[str]:
    if not raw:
        return ()
    return {part.strip().lower() for part in raw.split(',') if part.strip()}


def count_files(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob('*') if _.is_file())


def copy_tree(src: Path, dst: Path, dry_run: bool) -> int:
    if not src.exists():
        return 0
    if dry_run:
        return count_files(src)
    shutil.copytree(src, dst)
    return count_files(dst)


def ensure_parent(path: Path, dry_run: bool) -> None:
    if dry_run:
        return
    path.mkdir(parents=True, exist_ok=True)


def ingest(args: argparse.Namespace) -> Dict[str, int]:
    bridge = TownBridge()
    available = set(bridge.list_simulations())
    if args.simulation not in available:
        raise SystemExit(f"Simulation '{args.simulation}' not found under {bridge.storage_path}")

    ingestion_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    dest_root = Path(args.output).expanduser().resolve() / args.simulation / ingestion_id
    source_root = bridge.storage_path / args.simulation

    sections = parse_sections(args.sections)
    if not sections:
        sections = {'environment', 'personas', 'reverie'}

    ensure_parent(dest_root, args.dry_run)

    stats: Dict[str, int] = {}

    if 'environment' in sections:
        stats['environment_files'] = copy_tree(
            source_root / 'environment', dest_root / 'environment', args.dry_run
        )

    if 'reverie' in sections:
        stats['reverie_files'] = copy_tree(
            source_root / 'reverie', dest_root / 'reverie', args.dry_run
        )

    if 'personas' in sections:
        personas_src = source_root / 'personas'
        if args.agent:
            personas_src = personas_src / args.agent
            if not personas_src.exists():
                raise SystemExit(f"Agent '{args.agent}' not found in {personas_src.parent}")
            personas_dst = dest_root / 'personas' / args.agent
        else:
            personas_dst = dest_root / 'personas'
        stats['personas_files'] = copy_tree(personas_src, personas_dst, args.dry_run)

    metadata = {
        'simulation': args.simulation,
        'agent': args.agent,
        'sections': sorted(sections),
        'source_path': str(source_root),
        'destination_path': str(dest_root),
        'created_at_utc': ingestion_id,
        'data_type': args.label,
        'notes': args.notes,
        'dry_run': args.dry_run,
    }
    metadata['file_counts'] = stats

    if not args.dry_run:
        ensure_parent(dest_root, False)
        meta_path = dest_root / 'metadata.json'
        meta_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding='utf-8')

    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Ingest REAL Town data snapshots.')
    parser.add_argument('--simulation', required=True, help='simulation目录名称 (storage下的子目录)')
    parser.add_argument('--agent', help='仅复制指定agent persona (默认复制全部)')
    parser.add_argument('--sections', default='environment,personas,reverie',
                        help='逗号分隔: environment,personas,reverie')
    parser.add_argument('--output', default='data/real_ingest', help='输出根目录')
    parser.add_argument('--label', default='REAL', help='metadata中的data_type标签')
    parser.add_argument('--notes', default='', help='可选备注写入metadata')
    parser.add_argument('--dry-run', action='store_true', help='只打印不会拷贝文件')
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    stats = ingest(args)
    print("Ingestion summary:")
    for key, value in stats.items():
        print(f"  - {key}: {value}")


if __name__ == '__main__':
    main()
